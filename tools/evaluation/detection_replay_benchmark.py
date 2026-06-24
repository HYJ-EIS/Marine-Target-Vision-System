"""Replay cached detector boxes through trackers for tracker-only diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
import json
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterable

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.constants import (
    DATASET_EXPORT_TRACKER_CHOICES,
    FORMAL_FRAME_LIMIT,
    FORMAL_MSDC_VARIANT,
    METRIC_FIELDS,
)
from target_module.image_detect_module.utils.file_utils import get_file_type
from target_module.image_detect_module.utils.msdc_detection import (
    run_msdc_low_threshold_detection,
    split_msdc_high_from_low_boxes,
)
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from tools.evaluation.export_mot_results import format_mot_result_line
from tools.evaluation.motchallenge_eval import run_motchallenge_eval
from tools.evaluation.msdc_dataset_benchmark import (
    _build_render_command,
    read_video_info,
    resolve_single_sequence_dataset,
    write_diagnostics_for_tracker,
    write_motchallenge_gt_sequence,
    write_stage_coverage_csv,
)
from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS, FORMAL_V3_ENV

SUMMARY_FIELDS = ["tracker", "variant", "replay_detections", *METRIC_FIELDS]
_MSDC_FORMAL_REPLAY_CONFIG_BASELINE = {
    "MSDC_USE_REACQUIRE": True,
    "MSDC_DEBUG_EVENTS": False,
    "MSDC_DEBUG_TRACK_SNAPSHOT_LIMIT": 128,
    "MSDC_DEBUG_DETAIL_LIMIT": 8,
    "MSDC_EVIDENCE_MODE": "score",
    "MSDC_CONFIRM_SCORE": 2.5,
    "MSDC_CONFIRM_MIN_HITS": 4,
    "MSDC_CONFIRM_REQUIRE_DET": True,
    "MSDC_CONFIRM_MIN_DET_HITS": 4,
    "MSDC_CONFIRM_MIN_REAL_DET_HITS": 4,
    "MSDC_CONFIRM_REQUIRE_HIGH_DET": True,
    "MSDC_LOW_SPAWN_MIN_CONF": 0.30,
    "MSDC_LOW_CANDIDATE_ENABLE": True,
    "MSDC_LOW_CONFIRM_MIN_HITS": 5,
    "MSDC_LOW_CONFIRM_WINDOW": 8,
    "MSDC_LOW_CONFIRM_MIN_AVG_SCORE": 0.22,
    "MSDC_LOW_CONFIRM_MAX_MISSES": 1,
    "MSDC_LOW_CONFIRM_MAX_AREA_CHANGE": 1.8,
    "MSDC_LOW_CONFIRM_MAX_CENTER_STEP_FACTOR": 3.0,
    "MSDC_LOW_INHERIT_ENABLE": True,
    "MSDC_LOW_INHERIT_SCORE": 0.40,
    "MSDC_LOW_INHERIT_IOU_THRESH": 0.02,
    "MSDC_LOW_INHERIT_CENTER_DIST": 220.0,
    "MSDC_LOW_INHERIT_MAX_LOST_AGE": 120,
    "MSDC_LOW_INHERIT_USE_HISTORY_VELOCITY": True,
    "MSDC_LOW_INHERIT_MAX_PREDICT_AGE": 120,
    "MSDC_LOW_INHERIT_VELOCITY_MIN": 0.15,
    "MSDC_LOW_INHERIT_CLASS_MATCH": True,
    "MSDC_LOW_INHERIT_CLASS_MISMATCH_CENTER_DIST": 80.0,
    "MSDC_LOW_INHERIT_CLASS_MISMATCH_PENALTY": 0.0,
    "MSDC_LOW_INHERIT_WEIGHT_IOU": 0.35,
    "MSDC_LOW_INHERIT_WEIGHT_CENTER": 0.25,
    "MSDC_LOW_INHERIT_WEIGHT_VELOCITY": 0.20,
    "MSDC_LOW_INHERIT_WEIGHT_LOW_SCORE": 0.15,
    "MSDC_LOW_INHERIT_WEIGHT_RECENCY": 0.05,
    "MSDC_CANDIDATE_MAX_AGE": 5,
    "MSDC_LOST_MAX_AGE": 40,
    "MSDC_REACQUIRE_INTERVAL": 5,
    "MSDC_REACQUIRE_SCORE": 1.5,
    "MSDC_REACQUIRE_IOU_THRESH": 0.05,
    "MSDC_REACQUIRE_CENTER_DIST": 160.0,
    "MSDC_REACQUIRE_CENTER_SCALE_FACTOR": 4.0,
    "MSDC_REACQUIRE_MAX_CENTER_DIST": 240.0,
    "MSDC_REMOVED_GUARD_FRAMES": 80,
    "MSDC_REUSE_GUARD_ENABLE": True,
    "MSDC_MAX_ACTIVE_TRACKS": 64,
    "MSDC_MAX_LOST_TRACKS": 32,
    "MSDC_MAX_CANDIDATES": 32,
    "MSDC_MAX_LOW_CANDIDATES": 24,
    "MSDC_MAX_TOTAL_TRACKS": 128,
    "MSDC_SPAWN_SUPPRESS_ENABLE": True,
    "MSDC_SPAWN_SUPPRESS_IOU": 0.1,
    "MSDC_SPAWN_SUPPRESS_CENTER_DIST": 80.0,
    "MSDC_LOW_SPAWN_SUPPRESS_CENTER_DIST": 120.0,
    "MSDC_OUTPUT_NMS_ENABLE": True,
    "MSDC_OUTPUT_NMS_IOU": 0.3,
    "MSDC_OUTPUT_NMS_CENTER_DIST": 60.0,
    "MSDC_OUTPUT_NMS_FRAGMENT_AREA_RATIO": 0.35,
    "MSDC_OUTPUT_NMS_CONTAINMENT_RATIO": 0.50,
    "MSDC_OUTPUT_MAX_REAL_DET_AGE": 3,
    "MSDC_OUTPUT_MIN_BOX_SIZE": 12,
    "MSDC_LOW_OBS_TOPK": 32,
    "MSDC_LOW_OBS_GLOBAL_TOPK": 32,
    "MSDC_LOW_OBS_PER_TRACK_NEAREST": 1,
    "MSDC_LOW_OBS_MAX_PER_FRAME": 64,
    "MSDC_LOW_OBS_MIN_CONF": 0.25,
    "MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY": True,
    "MSDC_LOW_OBS_TRACK_PROXIMITY_CENTER_DIST": 240.0,
    "MSDC_LOW_OBS_TRACK_PROXIMITY_IOU": 0.01,
    "MSDC_LOW_OBS_MOTION_GATE_CENTER_DIST": 240.0,
    "MSDC_LOW_OBS_MOTION_GATE_IOU": 0.01,
}


def tracker_output_name(tracker_type: str, variant: str = "") -> str:
    if tracker_type == "msdc_elt" and variant:
        return f"{variant}_replay"
    return f"{tracker_type}_replay"


def high_boxes_from_cache_row(row: dict) -> list[dict]:
    return list(row.get("high_boxes", row.get("boxes", [])) or [])


def low_boxes_from_cache_row(row: dict) -> list[dict]:
    return list(row.get("low_boxes", row.get("boxes", [])) or [])


@contextmanager
def temporary_env(env_delta: dict[str, str], isolate_msdc: bool = False):
    managed_keys = set(env_delta)
    if isolate_msdc:
        managed_keys.update(key for key in os.environ if key.startswith("MSDC_"))
    old_values = {key: os.environ.get(key) for key in managed_keys}
    try:
        if isolate_msdc:
            for key in list(os.environ):
                if key.startswith("MSDC_") and key not in env_delta:
                    os.environ.pop(key, None)
        for key, value in env_delta.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
            row = {field: "" for field in SUMMARY_FIELDS}
            row.update({key: value for key, value in values.items() if key in SUMMARY_FIELDS})
            row["tracker"] = tracker
            row["variant"] = values.get("variant", _summary_variant_from_tracker(tracker))
            row["replay_detections"] = values.get("replay_detections", True)
            writer.writerow(row)
    return output_path


def _summary_variant_from_tracker(tracker_name: str) -> str:
    if not tracker_name.endswith("_replay"):
        return ""
    base_name = tracker_name[:-len("_replay")]
    baseline_names = set(DATASET_EXPORT_TRACKER_CHOICES) - {"msdc_elt"}
    return "" if base_name in baseline_names else base_name


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
            low_boxes = run_msdc_low_threshold_detection(detector, frame, file_type)
            high_boxes = split_msdc_high_from_low_boxes(low_boxes, file_type)
            rows.append({
                "frame_id": frame_id,
                "file_type": file_type,
                "replay_detections": True,
                "high_boxes": high_boxes,
                "low_boxes": low_boxes,
            })
            if progress_interval > 0 and frame_id % progress_interval == 0:
                print(f"[PROGRESS] dump {input_video.stem}: {frame_id}/{max_frames}", flush=True)
    finally:
        cap.release()
    return write_detection_cache(output_cache, rows)


def _coerce_config_env_value(value: str, old_value: object) -> object:
    if isinstance(old_value, bool):
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(str(value).strip())
    if isinstance(old_value, float):
        return float(str(value).strip())
    return str(value)


def _config_value_to_env(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _snapshot_msdc_config() -> dict[str, object]:
    return {
        key: getattr(Config, key)
        for key in dir(Config)
        if key.startswith("MSDC_") and not callable(getattr(Config, key))
    }


def _apply_config_env(env_delta: dict[str, str]) -> dict[str, object]:
    old_values: dict[str, object] = {}
    for key, value in env_delta.items():
        if not hasattr(Config, key):
            continue
        old_value = getattr(Config, key)
        old_values[key] = old_value
        setattr(Config, key, _coerce_config_env_value(str(value), old_value))
    return old_values


def _restore_config(old_values: dict[str, object]) -> None:
    for key, value in old_values.items():
        setattr(Config, key, value)


def _complete_msdc_variant_env(variant: str) -> dict[str, str]:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unsupported MS-DC replay variant: {variant}")
    env = {key: _config_value_to_env(value) for key, value in _MSDC_FORMAL_REPLAY_CONFIG_BASELINE.items()}
    env.update(FORMAL_V3_ENV)
    env.update(ABLATION_VARIANTS[variant])
    env.setdefault("MSDC_DEBUG_EVENTS", "1")
    return env


@contextmanager
def msdc_variant_context(variant: str):
    env_delta = _complete_msdc_variant_env(variant)
    old_config = _snapshot_msdc_config()
    try:
        with temporary_env(env_delta, isolate_msdc=True):
            _restore_config(_MSDC_FORMAL_REPLAY_CONFIG_BASELINE)
            _apply_config_env(env_delta)
            yield
    finally:
        _restore_config(old_config)


def effective_max_frames(args: argparse.Namespace) -> int:
    formal_frame_limit = int(getattr(args, "formal_frame_limit", 0))
    return formal_frame_limit if formal_frame_limit > 0 else int(getattr(args, "max_frames", 0))


def replay_tracker_from_cache(
    *,
    input_video: str | Path,
    detection_cache: str | Path,
    output_root: str | Path,
    tracker_type: str,
    variant: str = "",
    seq_name: str,
    file_type: str,
    frame_rate: float,
    max_frames: int,
    progress_interval: int = 0,
) -> Path:
    if tracker_type not in DATASET_EXPORT_TRACKER_CHOICES:
        raise ValueError(f"Unsupported replay tracker: {tracker_type}")
    if tracker_type == "msdc_elt" and variant and variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unsupported MS-DC replay variant: {variant}")

    input_video = Path(input_video)
    tracker_name = tracker_output_name(tracker_type, variant)
    output_file = Path(output_root) / tracker_name / "data" / f"{seq_name}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_rows = {
        int(row["frame_id"]): {
            "high_boxes": high_boxes_from_cache_row(row),
            "low_boxes": low_boxes_from_cache_row(row),
        }
        for row in load_detection_cache(detection_cache, max_frames)
    }

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    old_config: dict[str, object] = {}
    replay_context = nullcontext()
    if tracker_type == "msdc_elt" and variant:
        replay_context = msdc_variant_context(variant)
    elif tracker_type == "msdc_elt":
        replay_context = temporary_env({"MSDC_DEBUG_EVENTS": "1"})
    try:
        with replay_context:
            if tracker_type == "msdc_elt" and not variant:
                old_config = _apply_config_env({"MSDC_DEBUG_EVENTS": "1"})
            if tracker_type == "msdc_elt":
                from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker

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
                    cached = cache_rows.get(frame_id, {"high_boxes": [], "low_boxes": []})
                    high_boxes = cached["high_boxes"]
                    low_boxes = cached["low_boxes"]
                    if tracker_type == "msdc_elt":
                        tracked_boxes = lifecycle_tracker.update(
                            frame=frame,
                            frame_idx=frame_id - 1,
                            file_type=file_type,
                            high_boxes=high_boxes,
                            low_boxes=low_boxes,
                        )
                    else:
                        tracked_boxes = tracker.update(high_boxes, frame.shape, frame=frame)
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
    planned_trackers: list[tuple[str, str, str]] = []
    for tracker in args.trackers:
        if tracker == "msdc_elt":
            for variant in args.variants:
                planned_trackers.append((tracker, variant, tracker_output_name(tracker, variant)))
        else:
            planned_trackers.append((tracker, "", tracker_output_name(tracker)))
    tracker_names = [tracker_name for _, _, tracker_name in planned_trackers]
    max_frames = effective_max_frames(args)
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
        cache_path = detections_root / f"{spec.seq_name}_high_low_detections.jsonl"
        dump_video_detections(
            input_video=spec.video_path,
            output_cache=cache_path,
            file_type=file_type,
            max_frames=max_frames,
            progress_interval=int(args.progress_interval),
        )
        for tracker_type, variant, tracker_name in planned_trackers:
            tracker_file = replay_tracker_from_cache(
                input_video=spec.video_path,
                detection_cache=cache_path,
                output_root=trackers_root,
                tracker_type=tracker_type,
                variant=variant,
                seq_name=spec.seq_name,
                file_type=file_type,
                frame_rate=info.fps,
                max_frames=max_frames,
                progress_interval=int(args.progress_interval),
            )
            diag_dir = diagnostics_root / spec.seq_name
            eval_gt_file = gt_root / spec.seq_name / "gt" / "gt.txt"
            write_diagnostics_for_tracker(eval_gt_file, tracker_file, diag_dir, tracker_name)
            write_stage_coverage_csv(
                eval_gt_file,
                trackers_root / tracker_name / "diagnostics" / spec.seq_name / "stage_observations.jsonl",
                diag_dir,
                tracker_name,
            )
            if args.render:
                render_cmd = _build_render_command(
                    input_video=spec.video_path,
                    tracker_file=tracker_file,
                    output_file=run_root / "visualizations" / spec.seq_name / f"{tracker_name}.mp4",
                    max_frames=max_frames,
                    progress_interval=int(args.progress_interval),
                    class_source=str(args.render_class_source),
                )
                subprocess.run(render_cmd, cwd=str(_ROOT), check=True)

    summary = run_motchallenge_eval(
        gt_root=gt_root,
        trackers_root=trackers_root,
        output_root=eval_root,
        trackers=tracker_names,
        sequences=sequences,
        print_results=False,
    )
    for _, variant, tracker_name in planned_trackers:
        summary.setdefault(tracker_name, {})
        summary[tracker_name]["variant"] = variant
        summary[tracker_name]["replay_detections"] = True
    return write_replay_summary(run_root, summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay cached detector boxes through trackers")
    parser.add_argument("--dataset-root", nargs="+", required=True, help="Single-sequence annotation roots")
    parser.add_argument("--output-root", default="results/detection_replay_benchmark")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-frames", type=int, default=300, help="Smoke/debug limit used when --formal-frame-limit is 0")
    parser.add_argument(
        "--formal-frame-limit",
        type=int,
        default=0,
        help=f"Formal replay frame limit; 0 disables and uses --max-frames. Use {FORMAL_FRAME_LIMIT} for formal replay.",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=DATASET_EXPORT_TRACKER_CHOICES,
        default=list(DATASET_EXPORT_TRACKER_CHOICES),
    )
    parser.add_argument("--variants", nargs="+", default=[FORMAL_MSDC_VARIANT], choices=list(ABLATION_VARIANTS))
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Render annotated videos after replay exports")
    parser.add_argument("--render-class-source", choices=["detector", "none"], default="detector")
    parser.add_argument("--input", default="", help="Optional direct input video path for a single dataset")
    parser.add_argument("--seq-name", default="", help="Optional sequence name override for a single dataset")
    parser.add_argument("--file-type", default="", choices=["", "visible", "infrared"])
    args = parser.parse_args(argv)
    if int(args.max_frames) < 0:
        parser.error("--max-frames must be >= 0")
    if int(args.formal_frame_limit) < 0:
        parser.error("--formal-frame-limit must be >= 0")
    if int(args.formal_frame_limit) == 0 and int(args.max_frames) <= 0:
        parser.error("--max-frames must be > 0 when --formal-frame-limit is 0")
    if int(args.progress_interval) < 0:
        parser.error("--progress-interval must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    summary_path = run_detection_replay_benchmark(parse_args(argv))
    print(f"[OK] Replay summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
