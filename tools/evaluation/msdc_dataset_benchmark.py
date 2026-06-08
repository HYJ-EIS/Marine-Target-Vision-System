"""
Utilities for running single-sequence MS-DC-ELT dataset evaluations.

The online tracker must not read ground truth. This module only uses GT for
offline MOTChallenge normalization and diagnostics after tracker export.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS  # noqa: E402


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class SingleSequenceDataset:
    dataset_root: Path
    dataset_name: str
    gt_file: Path
    labels_file: Path | None
    video_ref_file: Path | None
    video_path: Path | None
    seq_name: str


def windows_path_to_wsl_path(path_text: str | Path) -> Path:
    """Convert common Windows and WSL UNC path strings to a local POSIX path."""
    text = str(path_text).strip().strip("\"'")
    text = text.replace("\ufeff", "").strip()
    if not text:
        return Path("")

    normalized = text.replace("/", "\\")
    lower = normalized.lower()
    wsl_prefixes = (
        "\\\\wsl.localhost\\ubuntu-d\\",
        "\\\\wsl$\\ubuntu-d\\",
        "\\\\wsl.localhost\\ubuntu\\",
        "\\\\wsl$\\ubuntu\\",
    )
    for prefix in wsl_prefixes:
        if lower.startswith(prefix):
            rest = normalized[len(prefix):].replace("\\", "/")
            return Path("/") / rest

    match = re.match(r"^([A-Za-z]):\\(.*)$", normalized)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}") / rest

    return Path(text)


def _read_first_nonempty_line(path: Path) -> str:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _find_video_ref_file(dataset_root: Path) -> Path | None:
    candidates = [
        dataset_root / "原视频地址.txt",
        dataset_root / "原视频文件地址.txt",
        dataset_root / "video.txt",
        dataset_root / "video_path.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for candidate in sorted(dataset_root.glob("*视频*地址*.txt")):
        if candidate.is_file():
            return candidate
    return None


def _default_seq_name(dataset_root: Path, video_path: Path | None) -> str:
    if video_path is not None and video_path.stem:
        return video_path.stem
    return dataset_root.name


def resolve_single_sequence_dataset(
    dataset_root: str | Path,
    video: str | Path | None = None,
    seq_name: str | None = None,
) -> SingleSequenceDataset:
    """Resolve one copied MOT annotation directory into a normalized spec."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    gt_candidates = [root / "gt" / "gt.txt", root / "gt.txt"]
    gt_file = next((path for path in gt_candidates if path.is_file()), None)
    if gt_file is None:
        raise FileNotFoundError(f"Missing single-sequence gt.txt under: {root}")

    label_candidates = [root / "gt" / "labels.txt", root / "labels.txt"]
    labels_file = next((path for path in label_candidates if path.is_file()), None)

    video_ref_file = _find_video_ref_file(root)
    if video is not None:
        video_path = windows_path_to_wsl_path(video)
    elif video_ref_file is not None:
        video_path = windows_path_to_wsl_path(_read_first_nonempty_line(video_ref_file))
    else:
        video_path = None

    resolved_seq_name = seq_name or _default_seq_name(root, video_path)
    return SingleSequenceDataset(
        dataset_root=root,
        dataset_name=root.name,
        gt_file=gt_file,
        labels_file=labels_file,
        video_ref_file=video_ref_file,
        video_path=video_path,
        seq_name=resolved_seq_name,
    )


def read_video_info(video_path: str | Path) -> VideoInfo:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return VideoInfo(fps=fps, width=width, height=height, frame_count=frame_count)


def write_motchallenge_gt_sequence(
    dataset: SingleSequenceDataset,
    output_gt_root: str | Path,
    video_info: VideoInfo,
    max_frames: int = 0,
) -> Path:
    """Create <gt-root>/<seq>/gt/gt.txt plus seqinfo.ini for TrackEval."""
    output_gt_root = Path(output_gt_root)
    seq_dir = output_gt_root / dataset.seq_name
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    seq_length = int(max_frames) if int(max_frames) > 0 else int(video_info.frame_count)
    gt_out = gt_dir / "gt.txt"
    if int(max_frames) > 0:
        with dataset.gt_file.open("r", encoding="utf-8-sig", newline="") as src, gt_out.open(
            "w", encoding="utf-8", newline=""
        ) as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            for row in reader:
                if len(row) < 6:
                    continue
                if int(float(row[0])) <= int(max_frames):
                    writer.writerow(row)
    else:
        shutil.copyfile(dataset.gt_file, gt_out)
    seqinfo = "\n".join([
        "[Sequence]",
        f"name={dataset.seq_name}",
        "imDir=img1",
        f"frameRate={video_info.fps:g}",
        f"seqLength={seq_length}",
        f"imWidth={int(video_info.width)}",
        f"imHeight={int(video_info.height)}",
        "imExt=.jpg",
        "",
    ])
    (seq_dir / "seqinfo.ini").write_text(seqinfo, encoding="utf-8")
    return seq_dir


def _load_mot_rows(path: str | Path) -> dict[int, list[tuple[int, tuple[float, float, float, float], float]]]:
    rows_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float], float]]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 6:
                continue
            frame_id = int(float(row[0]))
            obj_id = int(float(row[1]))
            box = tuple(float(value) for value in row[2:6])
            conf = float(row[6]) if len(row) > 6 else 1.0
            rows_by_frame.setdefault(frame_id, []).append((obj_id, box, conf))
    return rows_by_frame


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def _segments(frames: Iterable[int]) -> str:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return ""
    parts: list[str] = []
    start = prev = ordered[0]
    for frame in ordered[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        parts.append(f"{start}-{prev}")
        start = prev = frame
    parts.append(f"{start}-{prev}")
    return ";".join(parts)


def compute_per_gt_diagnostics(
    gt_file: str | Path,
    tracker_file: str | Path,
    iou_threshold: float = 0.1,
    center_threshold: float = 120.0,
) -> list[dict]:
    """Summarize coverage and identity fragmentation for each GT trajectory."""
    gt_rows = _load_mot_rows(gt_file)
    tracker_rows = _load_mot_rows(tracker_file)
    gt_ids = sorted({obj_id for rows in gt_rows.values() for obj_id, _, _ in rows})
    diagnostics: list[dict] = []

    for gt_id in gt_ids:
        gt_frames: set[int] = set()
        matched_frames: set[int] = set()
        matched_ids: set[int] = set()
        for frame_id, rows in gt_rows.items():
            gt_box = next((box for obj_id, box, _ in rows if obj_id == gt_id), None)
            if gt_box is None:
                continue
            gt_frames.add(frame_id)
            for pred_id, pred_box, _ in tracker_rows.get(frame_id, []):
                if _iou(gt_box, pred_box) >= iou_threshold or _center_dist(gt_box, pred_box) <= center_threshold:
                    matched_frames.add(frame_id)
                    matched_ids.add(pred_id)
        missed_frames = gt_frames - matched_frames
        gt_frame_count = len(gt_frames)
        tracked_frame_count = len(matched_frames)
        diagnostics.append({
            "gt_id": int(gt_id),
            "gt_frame_count": int(gt_frame_count),
            "tracked_frame_count": int(tracked_frame_count),
            "missed_frame_count": int(len(missed_frames)),
            "coverage_ratio": round(tracked_frame_count / gt_frame_count, 4) if gt_frame_count else 0.0,
            "predicted_id_count": int(len(matched_ids)),
            "predicted_ids": ";".join(str(obj_id) for obj_id in sorted(matched_ids)),
            "matched_segments": _segments(matched_frames),
            "missed_segments": _segments(missed_frames),
        })
    return diagnostics


def compute_tracker_box_stats(
    gt_file: str | Path,
    tracker_file: str | Path,
    duplicate_iou_threshold: float = 0.5,
    far_iou_threshold: float = 0.1,
    far_center_threshold: float = 120.0,
) -> dict:
    """Compute simple duplicate, empty-box, and far-from-GT tracker diagnostics."""
    gt_rows = _load_mot_rows(gt_file)
    tracker_rows = _load_mot_rows(tracker_file)
    all_tracker = [(frame_id, obj_id, box) for frame_id, rows in tracker_rows.items() for obj_id, box, _ in rows]
    duplicate_pairs = 0
    far_count = 0
    zero_area = 0
    for frame_id, rows in tracker_rows.items():
        for idx, (_, box_a, _) in enumerate(rows):
            if box_a[2] <= 0 or box_a[3] <= 0:
                zero_area += 1
            for _, box_b, _ in rows[idx + 1:]:
                if _iou(box_a, box_b) >= duplicate_iou_threshold:
                    duplicate_pairs += 1
            gt_boxes = [box for _, box, _ in gt_rows.get(frame_id, [])]
            if not gt_boxes:
                continue
            best_iou = max((_iou(box_a, gt_box) for gt_box in gt_boxes), default=0.0)
            best_dist = min((_center_dist(box_a, gt_box) for gt_box in gt_boxes), default=float("inf"))
            if best_iou < far_iou_threshold and best_dist > far_center_threshold:
                far_count += 1
    ids = [obj_id for _, obj_id, _ in all_tracker]
    return {
        "rows": int(len(all_tracker)),
        "unique_ids": int(len(set(ids))),
        "max_id": int(max(ids)) if ids else 0,
        "duplicate_pairs_iou05": int(duplicate_pairs),
        "zero_area_count": int(zero_area),
        "far_from_gt_count": int(far_count),
        "far_ratio": round(far_count / len(all_tracker), 4) if all_tracker else 0.0,
    }


def _load_stage_rows(path: str | Path) -> dict[int, list[dict]]:
    rows_by_frame: dict[int, list[dict]] = {}
    stage_path = Path(path)
    if not stage_path.is_file():
        return rows_by_frame
    with stage_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            frame_id = int(item.get("frame_idx", 0)) + 1
            rows_by_frame[frame_id] = list(item.get("boxes", []))
    return rows_by_frame


def _stage_box_tuple(box: dict) -> tuple[float, float, float, float]:
    return (
        float(box.get("x", 0.0)),
        float(box.get("y", 0.0)),
        float(box.get("w", 0.0)),
        float(box.get("h", 0.0)),
    )


def compute_per_gt_stage_coverage(
    gt_file: str | Path,
    stage_observations: str | Path,
    iou_threshold: float = 0.1,
    center_threshold: float = 120.0,
) -> list[dict]:
    """Compute per-GT coverage by detection/debug stage from stage_observations.jsonl."""
    gt_rows = _load_mot_rows(gt_file)
    stage_rows = _load_stage_rows(stage_observations)
    gt_ids = sorted({obj_id for rows in gt_rows.values() for obj_id, _, _ in rows})
    stages = ["high_det", "low_det", "low_only", "roi_low_det", "output"]
    output_ids: dict[int, set[int]] = {gt_id: set() for gt_id in gt_ids}
    matched_by_stage: dict[int, dict[str, set[int]]] = {
        gt_id: {stage: set() for stage in stages}
        for gt_id in gt_ids
    }

    for frame_id, rows in gt_rows.items():
        boxes = stage_rows.get(frame_id, [])
        for gt_id, gt_box, _ in rows:
            for box in boxes:
                stage = str(box.get("stage", ""))
                if stage not in stages:
                    continue
                stage_box = _stage_box_tuple(box)
                if _iou(gt_box, stage_box) >= iou_threshold or _center_dist(gt_box, stage_box) <= center_threshold:
                    matched_by_stage[gt_id][stage].add(frame_id)
                    if stage == "output" and "track_id" in box:
                        output_ids[gt_id].add(int(box["track_id"]))

    diagnostics = []
    for gt_id in gt_ids:
        gt_frames = {
            frame_id
            for frame_id, rows in gt_rows.items()
            if any(obj_id == gt_id for obj_id, _, _ in rows)
        }
        gt_frame_count = len(gt_frames)
        any_after_roi = set()
        for stage in stages:
            any_after_roi.update(matched_by_stage[gt_id][stage])
        row = {
            "gt_id": int(gt_id),
            "gt_frame_count": int(gt_frame_count),
        }
        for stage in stages:
            count = len(matched_by_stage[gt_id][stage])
            key_prefix = stage
            row[f"{key_prefix}_frame_count"] = int(count)
        for stage in stages:
            count = len(matched_by_stage[gt_id][stage])
            row[f"{stage}_coverage"] = round(count / gt_frame_count, 4) if gt_frame_count else 0.0
        row["output_predicted_id_count"] = int(len(output_ids[gt_id]))
        row["output_predicted_ids"] = ";".join(str(obj_id) for obj_id in sorted(output_ids[gt_id]))
        row["missing_after_roi_segments"] = _segments(gt_frames - any_after_roi)
        diagnostics.append(row)
    return diagnostics


def write_stage_coverage_csv(
    gt_file: str | Path,
    stage_observations: str | Path,
    output_dir: str | Path,
    tracker_name: str,
) -> Path | None:
    stage_path = Path(stage_observations)
    if not stage_path.is_file():
        return None
    rows = compute_per_gt_stage_coverage(gt_file, stage_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{tracker_name}_per_gt_stage_coverage.csv"
    fields = [
        "gt_id",
        "gt_frame_count",
        "high_det_frame_count",
        "low_det_frame_count",
        "low_only_frame_count",
        "roi_low_det_frame_count",
        "output_frame_count",
        "high_det_coverage",
        "low_det_coverage",
        "low_only_coverage",
        "roi_low_det_coverage",
        "output_coverage",
        "output_predicted_id_count",
        "output_predicted_ids",
        "missing_after_roi_segments",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_diagnostics_for_tracker(
    gt_file: str | Path,
    tracker_file: str | Path,
    output_dir: str | Path,
    tracker_name: str,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_gt_rows = compute_per_gt_diagnostics(gt_file, tracker_file)
    stats = compute_tracker_box_stats(gt_file, tracker_file)

    per_gt_csv = output_dir / f"{tracker_name}_per_gt_diagnostics.csv"
    fields = [
        "gt_id",
        "gt_frame_count",
        "tracked_frame_count",
        "missed_frame_count",
        "coverage_ratio",
        "predicted_id_count",
        "predicted_ids",
        "matched_segments",
        "missed_segments",
    ]
    with per_gt_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_gt_rows)

    stats_csv = output_dir / f"{tracker_name}_box_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats))
        writer.writeheader()
        writer.writerow(stats)
    return per_gt_csv, stats_csv


def _quote_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def _env_prefix(env_delta: dict[str, str]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(env_delta.items()))


def iso_now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def build_run_metadata(
    run_id: str,
    mode: str,
    commit_hash: str,
    dataset_roots: list[str | Path],
    output_root: str | Path,
    trackers: list[str],
    variants: list[str],
    max_frames: int,
    duration_seconds: float,
    render: bool,
    formal: bool,
) -> dict:
    return {
        "run_id": run_id,
        "mode": mode,
        "commit_hash": commit_hash,
        "dataset_roots": [str(Path(path)) for path in dataset_roots],
        "output_root": str(Path(output_root)),
        "trackers": list(trackers),
        "variants": list(variants),
        "max_frames": int(max_frames),
        "duration_seconds": float(duration_seconds),
        "render": bool(render),
        "formal": bool(formal),
        "started_at": iso_now(),
    }


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_command_record(
    path: str | Path,
    label: str,
    command: list[str],
    env_delta: dict[str, str],
    status: str,
    start_time: str,
    end_time: str,
    returncode: int | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "label": label,
        "command": list(command),
        "env_delta": dict(env_delta),
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "returncode": returncode,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_failure_record(
    path: str | Path,
    stage: str,
    label: str,
    command: str,
    returncode: int | None,
    message: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8-sig") as fh:
        fields = ["stage", "label", "command", "returncode", "message"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "stage": stage,
            "label": label,
            "command": command,
            "returncode": "" if returncode is None else int(returncode),
            "message": message,
        })


def _build_export_command(
    input_video: Path,
    output_root: Path,
    tracker_type: str,
    tracker_name: str,
    seq_name: str,
    max_frames: int,
    progress_interval: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "export_mot_results.py"),
        "--input",
        str(input_video),
        "--output-root",
        str(output_root),
        "--tracker",
        tracker_type,
        "--tracker-name",
        tracker_name,
        "--seq-name",
        seq_name,
    ]
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    if progress_interval > 0:
        cmd.extend(["--progress-interval", str(progress_interval)])
    return cmd


def _build_render_command(
    input_video: Path,
    tracker_file: Path,
    output_file: Path,
    max_frames: int,
    progress_interval: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "render_mot_video.py"),
        "--input",
        str(input_video),
        "--mot-results",
        str(tracker_file),
        "--output",
        str(output_file),
        "--class-source",
        "detector",
    ]
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    if progress_interval > 0:
        cmd.extend(["--progress-interval", str(progress_interval)])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or print a dual-dataset MS-DC-ELT benchmark plan")
    parser.add_argument("--dataset-root", nargs="+", required=True, help="Single-sequence annotation roots")
    parser.add_argument("--output-root", default="results/msdc_dataset_benchmark")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["botsort", "ocsort", "msdc_elt"],
        choices=["botsort", "ocsort", "msdc_elt"],
    )
    parser.add_argument("--variants", nargs="+", default=["Ours-full"], choices=list(ABLATION_VARIANTS))
    parser.add_argument("--max-frames", type=int, default=0, help="Smoke/debug only; 0 means full video")
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="Run the first N seconds; 0 disables")
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Render annotated videos after successful exports")
    parser.add_argument("--run", action="store_true", help="Execute exports/eval/diagnostics; default prints commands")
    parser.add_argument("--run-id", default="", help="Timestamp/run folder name; default uses current time")
    parser.add_argument("--mode", default="benchmark", choices=["benchmark", "main", "ablation", "smoke"])
    parser.add_argument("--commit-hash", default="", help="Commit hash recorded in run metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_root) / run_id
    gt_root = root / "mot_gt"
    trackers_root = root / "trackers"
    eval_root = root / "eval"
    diagnostics_root = root / "diagnostics"
    metadata_dir = root / "metadata"
    commands_path = metadata_dir / "commands.jsonl"
    failures_path = metadata_dir / "failures.csv"
    metadata = build_run_metadata(
        run_id=run_id,
        mode=args.mode,
        commit_hash=args.commit_hash,
        dataset_roots=[Path(path) for path in args.dataset_root],
        output_root=root,
        trackers=list(args.trackers),
        variants=list(args.variants),
        max_frames=int(args.max_frames),
        duration_seconds=float(args.duration_seconds),
        render=bool(args.render),
        formal=not bool(args.max_frames) and not bool(args.duration_seconds),
    )
    write_json(metadata_dir / "run_metadata.json", metadata)
    sequence_names: list[str] = []

    for dataset_root in args.dataset_root:
        spec = resolve_single_sequence_dataset(dataset_root)
        sequence_names.append(spec.seq_name)
        if spec.video_path is None:
            raise FileNotFoundError(f"No video reference file found under: {spec.dataset_root}")
        if not spec.video_path.is_file():
            raise FileNotFoundError(f"Resolved video path is not accessible: {spec.video_path}")

        print(f"[DATASET] {spec.dataset_name} seq={spec.seq_name}")
        print(f"  GT: {spec.gt_file}")
        print(f"  video: {spec.video_path}")
        video_accessible = spec.video_path.is_file()
        if not video_accessible:
            print(f"  [WARN] resolved video path is not accessible: {spec.video_path}")
        export_max_frames = int(args.max_frames)
        info = None
        if video_accessible and float(args.duration_seconds) > 0:
            info = read_video_info(spec.video_path)
            export_max_frames = max(1, int(round(float(args.duration_seconds) * float(info.fps))))
        if args.run:
            if not video_accessible:
                raise FileNotFoundError(f"Resolved video path is not accessible: {spec.video_path}")
            if info is None:
                info = read_video_info(spec.video_path)
            write_motchallenge_gt_sequence(spec, gt_root, info, max_frames=export_max_frames)

        planned_trackers: list[tuple[str, str, dict[str, str]]] = []
        for tracker in args.trackers:
            if tracker == "msdc_elt":
                for variant in args.variants:
                    planned_trackers.append(("msdc_elt", variant, dict(ABLATION_VARIANTS[variant])))
            else:
                planned_trackers.append((tracker, tracker, {}))

        for tracker_type, tracker_name, env_delta in planned_trackers:
            cmd = _build_export_command(
                input_video=spec.video_path,
                output_root=trackers_root,
                tracker_type=tracker_type,
                tracker_name=tracker_name,
                seq_name=spec.seq_name,
                max_frames=export_max_frames,
                progress_interval=args.progress_interval,
            )
            printable = _quote_command(cmd)
            prefix = _env_prefix(env_delta)
            if prefix:
                printable = f"{prefix} {printable}"
            print(f"[EXPORT:{tracker_name}] {printable}")
            if args.run:
                env = os.environ.copy()
                env.update(env_delta)
                started = iso_now()
                try:
                    completed = subprocess.run(cmd, cwd=str(_ROOT), env=env, check=True)
                    write_command_record(
                        commands_path,
                        f"EXPORT:{tracker_name}",
                        cmd,
                        env_delta,
                        "success",
                        started,
                        iso_now(),
                        completed.returncode,
                    )
                except subprocess.CalledProcessError as exc:
                    write_command_record(
                        commands_path,
                        f"EXPORT:{tracker_name}",
                        cmd,
                        env_delta,
                        "failed",
                        started,
                        iso_now(),
                        exc.returncode,
                    )
                    write_failure_record(
                        failures_path,
                        "export",
                        tracker_name,
                        _quote_command(cmd),
                        exc.returncode,
                        str(exc),
                    )
                    raise
            tracker_file = trackers_root / tracker_name / "data" / f"{spec.seq_name}.txt"
            if args.run:
                out_dir = diagnostics_root / spec.seq_name
                eval_gt_file = gt_root / spec.seq_name / "gt" / "gt.txt"
                write_diagnostics_for_tracker(eval_gt_file, tracker_file, out_dir, tracker_name)
                write_stage_coverage_csv(
                    eval_gt_file,
                    trackers_root / tracker_name / "diagnostics" / spec.seq_name / "stage_observations.jsonl",
                    out_dir,
                    tracker_name,
                )
            if args.render:
                render_cmd = _build_render_command(
                    input_video=spec.video_path,
                    tracker_file=tracker_file,
                    output_file=root / "visualizations" / spec.seq_name / f"{tracker_name}.mp4",
                    max_frames=export_max_frames,
                    progress_interval=args.progress_interval,
                )
                print(f"[RENDER:{tracker_name}] {_quote_command(render_cmd)}")
                if args.run:
                    started = iso_now()
                    try:
                        completed = subprocess.run(render_cmd, cwd=str(_ROOT), check=True)
                        write_command_record(
                            commands_path,
                            f"RENDER:{tracker_name}",
                            render_cmd,
                            {},
                            "success",
                            started,
                            iso_now(),
                            completed.returncode,
                        )
                    except subprocess.CalledProcessError as exc:
                        write_command_record(
                            commands_path,
                            f"RENDER:{tracker_name}",
                            render_cmd,
                            {},
                            "failed",
                            started,
                            iso_now(),
                            exc.returncode,
                        )
                        write_failure_record(
                            failures_path,
                            "render",
                            tracker_name,
                            _quote_command(render_cmd),
                            exc.returncode,
                            str(exc),
                        )
                        raise

    eval_cmd = [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "motchallenge_eval.py"),
        "--gt-root",
        str(gt_root),
        "--trackers-root",
        str(trackers_root),
        "--output-root",
        str(eval_root),
        "--sequences",
        *sequence_names,
        "--quiet",
    ]
    print(f"[EVAL] {_quote_command(eval_cmd)}")
    if args.run:
        started = iso_now()
        try:
            completed = subprocess.run(eval_cmd, cwd=str(_ROOT), check=True)
            write_command_record(
                commands_path,
                "EVAL",
                eval_cmd,
                {},
                "success",
                started,
                iso_now(),
                completed.returncode,
            )
        except subprocess.CalledProcessError as exc:
            write_command_record(
                commands_path,
                "EVAL",
                eval_cmd,
                {},
                "failed",
                started,
                iso_now(),
                exc.returncode,
            )
            write_failure_record(
                failures_path,
                "eval",
                "EVAL",
                _quote_command(eval_cmd),
                exc.returncode,
                str(exc),
            )
            raise

    if not args.run:
        print("[INFO] print-only mode; add --run to execute exports/evaluation/diagnostics.")
    if args.max_frames > 0:
        print("[WARN] --max-frames is for smoke/debug runs. Formal evaluation should use full videos.")
    if args.duration_seconds > 0:
        print("[WARN] --duration-seconds clips both tracker export and GT; use only for slice diagnostics.")
    if not args.render:
        print("[INFO] add --render with --run to create annotated videos.")
    print(f"[INFO] planned output root: {root}")


if __name__ == "__main__":
    main()
