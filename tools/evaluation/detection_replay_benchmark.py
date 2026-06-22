"""Replay cached detector boxes through trackers for tracker-only diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.constants import DATASET_EXPORT_TRACKER_CHOICES
from target_module.image_detect_module.utils.file_utils import get_file_type
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from tools.evaluation.export_mot_results import format_mot_result_line
from tools.evaluation.motchallenge_eval import run_motchallenge_eval
from tools.evaluation.msdc_dataset_benchmark import (
    read_video_info,
    resolve_single_sequence_dataset,
    write_diagnostics_for_tracker,
    write_motchallenge_gt_sequence,
    write_stage_coverage_csv,
)

SUMMARY_FIELDS = ["tracker", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "IDTP", "IDFP", "IDFN"]


def tracker_output_name(tracker_type: str) -> str:
    if tracker_type == "msdc_elt":
        return "msdc_elt_high_replay"
    return f"{tracker_type}_replay"


def write_detection_cache(path: str | Path, rows: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_detection_cache(path: str | Path, max_frames: int = 0) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if max_frames > 0 and int(row.get("frame_id", 0)) > max_frames:
                continue
            rows.append(row)
    return rows


def write_replay_summary(run_root: str | Path, summary: dict[str, dict]) -> Path:
    output_path = Path(run_root) / "summary" / "replay_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for tracker, values in summary.items():
            writer.writerow({"tracker": tracker, **values})
    return output_path


def _resolve_file_type(video_path: Path, requested: str = "") -> str:
    if requested:
        return requested
    resolved = get_file_type(video_path.stem)
    return "visible" if resolved == "unknown" else resolved


def dump_video_detections(
    *,
    input_video: str | Path,
    output_cache: str | Path,
    file_type: str,
    max_frames: int,
    progress_interval: int = 0,
) -> Path:
    input_video = Path(input_video)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    detector = td.get_detector()
    processor = getattr(detector, "processor", None)
    if processor is None or not hasattr(processor, "process_frame"):
        cap.release()
        raise RuntimeError("detector.processor.process_frame(frame, file_type) is unavailable")

    rows: list[dict] = []
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if max_frames > 0 and frame_id > max_frames:
                break
            stats = processor.process_frame(frame, file_type)
            if stats is None:
                raise RuntimeError(f"Detection failed: frame={frame_id}")
            rows.append({"frame_id": frame_id, "boxes": list(stats.get("boxes", []) or [])})
            if progress_interval > 0 and frame_id % progress_interval == 0:
                print(f"[PROGRESS] dump {input_video.stem}: {frame_id}/{max_frames}", flush=True)
    finally:
        cap.release()
    return write_detection_cache(output_cache, rows)


def _set_msdc_replay_config() -> dict[str, object]:
    keys = ["MSDC_DEBUG_EVENTS"]
    old_values = {key: getattr(Config, key) for key in keys if hasattr(Config, key)}
    Config.MSDC_DEBUG_EVENTS = True
    return old_values


def _restore_config(old_values: dict[str, object]) -> None:
    for key, value in old_values.items():
        setattr(Config, key, value)


def replay_tracker_from_cache(
    *,
    input_video: str | Path,
    detection_cache: str | Path,
    output_root: str | Path,
    tracker_type: str,
    seq_name: str,
    file_type: str,
    frame_rate: float,
    max_frames: int,
    progress_interval: int = 0,
) -> Path:
    if tracker_type not in DATASET_EXPORT_TRACKER_CHOICES:
        raise ValueError(f"Unsupported replay tracker: {tracker_type}")

    input_video = Path(input_video)
    tracker_name = tracker_output_name(tracker_type)
    output_file = Path(output_root) / tracker_name / "data" / f"{seq_name}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_rows = {int(row["frame_id"]): list(row.get("boxes", []) or []) for row in load_detection_cache(detection_cache, max_frames)}

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    old_config: dict[str, object] = {}
    try:
        if tracker_type == "msdc_elt":
            from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker

            old_config = _set_msdc_replay_config()
            tracker = None
            diagnostics_dir = Path(output_root) / tracker_name / "diagnostics" / seq_name
            lifecycle_tracker = MSDCLifecycleTracker(
                config=Config,
                file_type=file_type,
                debug_dir=diagnostics_dir,
                processor=None,
            )
        else:
            tracker = MultiObjectTracker(frame_rate=frame_rate, tracker_type=tracker_type)
            lifecycle_tracker = None

        frame_id = 0
        with output_file.open("w", encoding="utf-8") as fh:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                if max_frames > 0 and frame_id > max_frames:
                    break
                boxes = cache_rows.get(frame_id, [])
                if tracker_type == "msdc_elt":
                    tracked_boxes = lifecycle_tracker.update(
                        frame=frame,
                        frame_idx=frame_id - 1,
                        file_type=file_type,
                        high_boxes=boxes,
                        low_boxes=[],
                    )
                else:
                    tracked_boxes = tracker.update(boxes, frame.shape, frame=frame)
                for box in tracked_boxes:
                    fh.write(format_mot_result_line(frame_id, box) + "\n")
                if progress_interval > 0 and frame_id % progress_interval == 0:
                    fh.flush()
                    print(f"[PROGRESS] replay {tracker_name} {seq_name}: {frame_id}/{max_frames}", flush=True)
    finally:
        cap.release()
        if old_config:
            _restore_config(old_config)
    return output_file


def run_detection_replay_benchmark(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / run_id
    gt_root = run_root / "mot_gt"
    trackers_root = run_root / "trackers"
    eval_root = run_root / "eval"
    diagnostics_root = run_root / "diagnostics"
    detections_root = run_root / "detections"
    run_root.mkdir(parents=True, exist_ok=True)

    sequences: list[str] = []
    tracker_names = [tracker_output_name(tracker) for tracker in args.trackers]
    max_frames = int(args.max_frames)
    for dataset_root in args.dataset_root:
        spec = resolve_single_sequence_dataset(dataset_root, video=args.input or None, seq_name=args.seq_name or None)
        if spec.video_path is None:
            raise FileNotFoundError(f"No video reference file found under: {spec.dataset_root}")
        if not spec.video_path.is_file():
            raise FileNotFoundError(f"Resolved video path is not accessible: {spec.video_path}")
        info = read_video_info(spec.video_path)
        file_type = _resolve_file_type(spec.video_path, args.file_type)
        sequences.append(spec.seq_name)
        write_motchallenge_gt_sequence(spec, gt_root, info, max_frames=max_frames)
        cache_path = detections_root / f"{spec.seq_name}_high_detections.jsonl"
        dump_video_detections(
            input_video=spec.video_path,
            output_cache=cache_path,
            file_type=file_type,
            max_frames=max_frames,
            progress_interval=int(args.progress_interval),
        )
        for tracker_type in args.trackers:
            tracker_file = replay_tracker_from_cache(
                input_video=spec.video_path,
                detection_cache=cache_path,
                output_root=trackers_root,
                tracker_type=tracker_type,
                seq_name=spec.seq_name,
                file_type=file_type,
                frame_rate=info.fps,
                max_frames=max_frames,
                progress_interval=int(args.progress_interval),
            )
            diag_dir = diagnostics_root / spec.seq_name
            eval_gt_file = gt_root / spec.seq_name / "gt" / "gt.txt"
            tracker_name = tracker_output_name(tracker_type)
            write_diagnostics_for_tracker(eval_gt_file, tracker_file, diag_dir, tracker_name)
            write_stage_coverage_csv(
                eval_gt_file,
                trackers_root / tracker_name / "diagnostics" / spec.seq_name / "stage_observations.jsonl",
                diag_dir,
                tracker_name,
            )

    summary = run_motchallenge_eval(
        gt_root=gt_root,
        trackers_root=trackers_root,
        output_root=eval_root,
        trackers=tracker_names,
        sequences=sequences,
        print_results=False,
    )
    return write_replay_summary(run_root, summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay cached detector boxes through trackers")
    parser.add_argument("--dataset-root", nargs="+", required=True, help="Single-sequence annotation roots")
    parser.add_argument("--output-root", default="results/detection_replay_benchmark")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=DATASET_EXPORT_TRACKER_CHOICES,
        default=list(DATASET_EXPORT_TRACKER_CHOICES),
    )
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--input", default="", help="Optional direct input video path for a single dataset")
    parser.add_argument("--seq-name", default="", help="Optional sequence name override for a single dataset")
    parser.add_argument("--file-type", default="", choices=["", "visible", "infrared"])
    args = parser.parse_args(argv)
    if int(args.max_frames) <= 0:
        parser.error("--max-frames must be > 0 for replay diagnostics")
    if int(args.progress_interval) < 0:
        parser.error("--progress-interval must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    summary_path = run_detection_replay_benchmark(parse_args(argv))
    print(f"[OK] Replay summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
