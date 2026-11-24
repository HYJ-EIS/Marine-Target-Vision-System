"""Replay cached detector boxes through trackers for tracker-only diagnostics."""

from __future__ import annotations

import argparse
import configparser
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
import numpy as np

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
from tools.evaluation.cache_reuse import detection_cache_name, link_detection_cache_for_sequence
from tools.evaluation.stage_timing import StageTiming
from target_module.image_detect_module.utils.tracking_update import update_tracking_for_frame
from tools.evaluation.export_mot_results import format_mot_result_line
from tools.evaluation.motchallenge_eval import run_motchallenge_eval
from tools.evaluation.msdc_dataset_benchmark import (
    _build_render_command,
    VideoInfo,
    read_video_info,
    resolve_single_sequence_dataset,
    write_diagnostics_for_tracker,
    write_motchallenge_gt_sequence,
    write_stage_coverage_csv,
)
from tools.evaluation.msdc_diagnostic_metrics import (
    summarize_msdc_diagnostics,
    write_msdc_diagnostic_summary,
)
from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS, FORMAL_V3_ENV

SUMMARY_FIELDS = ["tracker", "variant", "replay_detections", *METRIC_FIELDS]

# Execution-only optimization: no additional tracker configuration is admitted.
CACHE_ONLY_MSDC_VARIANTS = frozenset({
    FORMAL_MSDC_VARIANT,
    "paper_no_a", "paper_count_e", "paper_no_r", "paper_count_e_no_r",
    "paper_full_s0", "paper_no_r_s0", "paper_lost_only_s0", "paper_removed_only_s0",
    *(f"paper_assoc_d{distance}" for distance in (40, 60, 100, 120)),
    *(f"paper_candidate_t{theta}" for theta in (15, 20, 30, 35)),
})


def _validate_replay_frame_source(frame_source, tracker_type, variant, max_frames, env_override=None):
    if frame_source not in {"decode", "cache_only"}:
        raise ValueError(f"Unsupported replay frame_source: {frame_source}")
    if frame_source == "cache_only":
        if tracker_type != "msdc_elt" or variant not in CACHE_ONLY_MSDC_VARIANTS:
            raise ValueError("cache_only is restricted to the seventeen fixed paper MSDC configurations")
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("cache_only requires an explicit positive frame count N")
        if env_override is not None:
            raise ValueError("cache_only does not accept dynamic configuration overrides")


def _validate_cache_only_rows(rows, length):
    # Do not clip first: duplicates, extra rows and reordered frames must all fail.
    ids = [row.get("frame_id") for row in rows]
    if any(type(frame) is not int for frame in ids) or ids != list(range(1, length + 1)):
        raise ValueError(f"cache_only requires cache frame IDs exactly 1..{length} in order")


def _cached_source_video_info(source_detection_root, sequence, length):
    """Read main's measured metadata without opening the source video again."""
    if not source_detection_root:
        raise ValueError("cache_only requires --source-detection-cache-root from the completed main run")
    path = Path(source_detection_root).parent / "mot_gt" / sequence / "seqinfo.ini"
    parser = configparser.ConfigParser()
    with path.open(encoding="utf-8-sig") as stream:
        parser.read_file(stream)
    section = parser["Sequence"]
    info = VideoInfo(
        fps=section.getfloat("frameRate"), width=section.getint("imWidth"),
        height=section.getint("imHeight"), frame_count=section.getint("seqLength"),
    )
    if info.frame_count != length or min(info.fps, info.width, info.height) <= 0:
        raise ValueError(f"cache_only source metadata does not describe the requested {length} frames: {path}")
    return info


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


def is_detection_cache_complete(path: str | Path, max_frames: int) -> bool:
    path = Path(path)
    if max_frames <= 0 or not path.is_file():
        return False
    try:
        max_seen = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                max_seen = max(max_seen, int(json.loads(line).get("frame_id", 0)))
        return max_seen >= max_frames
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def is_detection_cache_contiguous(path: str | Path, max_frames: int) -> bool:
    path = Path(path)
    if max_frames <= 0 or not path.is_file():
        return False
    try:
        seen_frames: set[int] = set()
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                frame_id = int(json.loads(line).get("frame_id", 0))
                if 1 <= frame_id <= max_frames:
                    seen_frames.add(frame_id)
        return len(seen_frames) == max_frames
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def is_mot_result_complete(path: str | Path, max_frames: int) -> bool:
    path = Path(path)
    if max_frames <= 0 or not path.is_file():
        return False
    try:
        timing = json.loads(path.with_suffix(".timing.json").read_text(encoding="utf-8"))
        if int(timing["processed_frames"]) != max_frames:
            return False
        processed = 0
        with path.with_suffix(".frames.jsonl").open(encoding="utf-8") as stream:
            for processed, line in enumerate(stream, 1):
                if int(json.loads(line)["frame_id"]) != processed:
                    return False
        if processed != max_frames:
            return False
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                if not 1 <= int(float(row[0])) <= max_frames:
                    return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def is_render_video_complete(path: str | Path, max_frames: int) -> bool:
    path = Path(path)
    if max_frames <= 0 or not path.is_file():
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return frame_count >= max_frames
    finally:
        cap.release()


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
    timing = StageTiming()
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    detector = td.get_detector()
    rows: list[dict] = []
    frame_id = 0
    try:
        while True:
            if max_frames > 0 and frame_id >= max_frames:
                break
            started = time.perf_counter()
            ok, frame = cap.read()
            timing.add("read_decode", started)
            if not ok:
                break
            frame_id += 1
            if max_frames > 0 and frame_id > max_frames:
                break
            started = time.perf_counter()
            low_boxes = run_msdc_low_threshold_detection(detector, frame, file_type)
            timing.add("low_detection", started)
            started = time.perf_counter()
            high_boxes = split_msdc_high_from_low_boxes(low_boxes, file_type)
            timing.add("high_split", started)
            timing.frames += 1
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
    started = time.perf_counter()
    result = write_detection_cache(output_cache, rows)
    timing.add("write_export", started)
    timing.write(Path(output_cache).with_suffix(".timing.json"), phase="detection", file_type=file_type)
    return result


def _coerce_config_env_value(value: str, old_value: object) -> object:
    if isinstance(old_value, bool):
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(str(value).strip())
    if isinstance(old_value, float):
        return float(str(value).strip())
    return str(value)


def _snapshot_msdc_config() -> dict[str, object]:
    return {
        key: getattr(Config, key)
        for key in dir(Config)
        if key.startswith("MSDC_") and not callable(getattr(Config, key))
    }


def write_effective_msdc_config(
    output_root: str | Path,
    tracker_name: str,
    config_snapshot: dict[str, object],
) -> Path:
    output_path = Path(output_root) / "effective_config" / f"{tracker_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


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
    env = dict(FORMAL_V3_ENV)
    env.update(ABLATION_VARIANTS[variant])
    env.setdefault("MSDC_DEBUG_EVENTS", "1")
    return env


@contextmanager
def msdc_variant_context(variant: str, env_override: dict[str, str] | None = None):
    if env_override is not None:
        env_delta = _complete_msdc_variant_env(FORMAL_MSDC_VARIANT)
        env_delta.update(env_override)
    else:
        env_delta = _complete_msdc_variant_env(variant)
    old_config = _snapshot_msdc_config()
    try:
        with temporary_env(env_delta, isolate_msdc=True):
            _apply_config_env(env_delta)
            yield
    finally:
        _restore_config(old_config)


def write_effective_msdc_variant_config(
    run_root: str | Path,
    tracker_name: str,
    variant: str = "",
    env_override: dict[str, str] | None = None,
) -> Path:
    if variant:
        with msdc_variant_context(variant, env_override=env_override):
            return write_effective_msdc_config(run_root, tracker_name, _snapshot_msdc_config())

    old_config: dict[str, object] = {}
    try:
        with temporary_env({"MSDC_DEBUG_EVENTS": "1"}):
            old_config = _apply_config_env({"MSDC_DEBUG_EVENTS": "1"})
            return write_effective_msdc_config(run_root, tracker_name, _snapshot_msdc_config())
    finally:
        if old_config:
            _restore_config(old_config)


def effective_max_frames(args: argparse.Namespace) -> int:
    formal_frame_limit = int(getattr(args, "formal_frame_limit", 0))
    return formal_frame_limit if formal_frame_limit > 0 else int(getattr(args, "max_frames", 0))


def expected_replay_frames(max_frames: int, video_frame_count: int, *, frame_source: str = "decode") -> int:
    if frame_source == "cache_only":
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("cache_only requires an explicit positive frame count N")
        return max_frames
    if max_frames <= 0:
        return max(0, int(video_frame_count))
    if video_frame_count <= 0:
        return int(max_frames)
    return min(int(max_frames), int(video_frame_count))


def baseline_constructor_parameters(tracker) -> dict:
    """Snapshot the constructed baseline and the project's applied overrides."""
    kind = str(getattr(tracker, "tracker_type", "unknown"))
    result = {
        "tracker_type": kind,
        "frame_rate": getattr(tracker, "frame_rate", None),
        "project_min_hits": getattr(tracker, "_min_hits", None),
    }
    if kind == "bytetrack":
        instance = tracker._bt
        keys = ("track_activation_threshold", "minimum_matching_threshold",
                "minimum_consecutive_frames", "det_thresh", "max_time_lost")
        result["constructor_parameters"] = {
            key: getattr(instance, key, None) for key in keys
        }
        result["constructor_parameters"]["lost_track_buffer"] = int(Config.BYTETRACK_LOST_BUFFER)
    else:
        adapter = getattr(tracker, "_official", None)
        instance = getattr(adapter, "_tracker", None)
        if kind == "official_botsort":
            args = getattr(instance, "args", None)
            if args is None and adapter is not None:
                args = adapter._botsort_args()
            result["constructor_parameters"] = dict(vars(args)) if args is not None else {}
            result["constructor_parameters"]["frame_rate"] = int(getattr(adapter, "frame_rate", 0))
            result["applied_track_buffer"] = getattr(instance, "buffer_size", None)
            result["applied_max_time_lost"] = getattr(instance, "max_time_lost", None)
        elif kind == "official_ocsort":
            keys = ("det_thresh", "max_age", "min_hits", "iou_threshold", "delta_t", "inertia", "use_byte")
            result["constructor_parameters"] = {key: getattr(instance, key, None) for key in keys}
            association = getattr(instance, "asso_func", None)
            result["constructor_parameters"]["asso_func"] = getattr(association, "__name__", str(association))
    return result


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
    env_override: dict[str, str] | None = None,
    baseline_input: str = "high",
    preserve_config: bool = False,
    frame_source: str = "decode",
) -> Path:
    _validate_replay_frame_source(frame_source, tracker_type, variant, max_frames, env_override)
    if baseline_input not in {"high", "all"}:
        raise ValueError(f"Unsupported baseline input: {baseline_input}")
    if tracker_type not in DATASET_EXPORT_TRACKER_CHOICES:
        raise ValueError(f"Unsupported replay tracker: {tracker_type}")
    if tracker_type == "msdc_elt" and variant and variant not in ABLATION_VARIANTS and env_override is None:
        raise ValueError(f"Unsupported MS-DC replay variant: {variant}")

    input_video = Path(input_video)
    tracker_name = tracker_output_name(tracker_type, variant)
    timing = StageTiming()
    output_file = Path(output_root) / tracker_name / "data" / f"{seq_name}.txt"
    loaded_rows = load_detection_cache(detection_cache, 0 if frame_source == "cache_only" else max_frames)
    if frame_source == "cache_only":
        _validate_cache_only_rows(loaded_rows, max_frames)
    cache_rows = {
        int(row["frame_id"]): {
            "high_boxes": high_boxes_from_cache_row(row),
            "low_boxes": low_boxes_from_cache_row(row),
        }
        for row in loaded_rows
    }
    del loaded_rows
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cap = None
    placeholder = None
    pixel_frames_decoded = 0
    if frame_source == "decode":
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video}")
    else:
        # Current MSDC update only validates ndarray/nonempty; no pixels or shape
        # enter its state transition. Keep the same update path for every cache row.
        placeholder = np.zeros((1, 1, 3), dtype=np.uint8)

    old_config: dict[str, object] = {}
    diagnostic_seconds = 0.0
    raw_update_seconds = 0.0
    replay_context = nullcontext()
    lifecycle_tracker = None
    tracker = None
    if preserve_config and variant:
        raise ValueError("preserve_config cannot be combined with a variant")
    if tracker_type == "msdc_elt" and variant:
        replay_context = msdc_variant_context(variant, env_override=env_override)
    elif tracker_type == "msdc_elt" and not preserve_config:
        replay_context = temporary_env({"MSDC_DEBUG_EVENTS": "1"})
    try:
        with replay_context:
            if tracker_type == "msdc_elt" and not variant and not preserve_config:
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
                if hasattr(lifecycle_tracker, "enable_paper_logging"):
                    lifecycle_tracker.enable_paper_logging(
                        Path(output_root) / tracker_name / "paper_logs" / seq_name,
                        seq_name, tracker_name,
                    )
                write_effective_msdc_config(
                    Path(output_root).parent,
                    tracker_name,
                    _snapshot_msdc_config(),
                )
            else:
                tracker = MultiObjectTracker(frame_rate=frame_rate, tracker_type=tracker_type)
                lifecycle_tracker = None

            frame_id = 0
            with output_file.open("w", encoding="utf-8") as fh, output_file.with_suffix(".frames.jsonl").open("w", encoding="utf-8") as frame_log:
                while True:
                    if max_frames > 0 and frame_id >= max_frames:
                        break
                    if frame_source == "decode":
                        started = time.perf_counter()
                        ok, frame = cap.read()
                        timing.add("read_decode", started)
                        if not ok:
                            break
                        pixel_frames_decoded += 1
                    else:
                        frame = placeholder
                    frame_id += 1
                    if max_frames > 0 and frame_id > max_frames:
                        break
                    if frame_id not in cache_rows:
                        raise ValueError(f"Missing detection cache frame: {frame_id}")
                    cached = cache_rows[frame_id]
                    high_boxes = cached["high_boxes"]
                    low_boxes = cached["low_boxes"]
                    input_boxes = low_boxes if tracker_type != "msdc_elt" and baseline_input == "all" else high_boxes
                    started = time.perf_counter()
                    tracked_boxes = update_tracking_for_frame(
                        tracker_type, tracker, lifecycle_tracker, None,
                        frame, frame_id - 1, file_type, input_boxes, low_boxes,
                    )
                    update_seconds = time.perf_counter() - started
                    debug_seconds = (
                        float(getattr(lifecycle_tracker, "last_timing_debug", {}).get("debug_s", 0.0))
                        if lifecycle_tracker is not None else 0.0
                    )
                    if not 0.0 <= debug_seconds <= update_seconds:
                        raise ValueError("Invalid lifecycle diagnostic timing")
                    raw_update_seconds += update_seconds
                    diagnostic_seconds += debug_seconds
                    timing.seconds["tracker"] += update_seconds - debug_seconds
                    timing.seconds["write_export"] += debug_seconds
                    timing.frames += 1
                    started = time.perf_counter()
                    for box in tracked_boxes:
                        fh.write(format_mot_result_line(frame_id, box) + "\n")
                    frame_log.write(json.dumps({
                        "frame_id": frame_id,
                        "update_wall_s": update_seconds,
                        "diagnostic_s": debug_seconds,
                        "tracker_s": update_seconds - debug_seconds,
                        "output_boxes": len(tracked_boxes),
                    }) + "\n")
                    timing.add("write_export", started)
                    if progress_interval > 0 and frame_id % progress_interval == 0:
                        fh.flush()
                        print(f"[PROGRESS] replay {tracker_name} {seq_name}: {frame_id}/{max_frames}", flush=True)
            if tracker is not None:
                started = time.perf_counter()
                metadata_path = Path(output_root) / tracker_name / "paper_metadata" / f"{seq_name}.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(json.dumps({
                    "sequence": seq_name, "processed_frames": frame_id,
                    **baseline_constructor_parameters(tracker),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                timing.add("write_export", started)
    finally:
        if lifecycle_tracker is not None and hasattr(lifecycle_tracker, "close_paper_logging"):
            started = time.perf_counter()
            lifecycle_tracker.close_paper_logging()
            timing.add("write_export", started)
        if cap is not None:
            cap.release()
        if old_config:
            _restore_config(old_config)
    timing.write(output_file.with_suffix(".timing.json"), phase="tracking_replay", tracker=tracker_name,
                 tracker_includes_debug_io=False,
                 raw_update_seconds=raw_update_seconds,
                 diagnostic_seconds=diagnostic_seconds,
                 frame_source=frame_source,
                 pixel_frames_decoded=pixel_frames_decoded,
                 timing_scope="update wall time minus measured lifecycle diagnostic generation/export; diagnostics charged to write_export",
                 baseline_input=baseline_input if tracker_type != "msdc_elt" else "high_and_low")
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
    if os.environ.get("PAPER_EXPERIMENT_SPEC"):
        from target_module.image_detect_module.paper_config import paper_environment
        (run_root / "paper_environment.json").write_text(
            json.dumps(paper_environment(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    baseline_input = getattr(args, "baseline_input", "high")
    policy_file = run_root / "baseline_input_policy.json"
    if policy_file.is_file():
        if json.loads(policy_file.read_text())["baseline_input"] != baseline_input:
            raise ValueError("Use a new run directory when changing baseline input")
    elif baseline_input != "high" and trackers_root.exists():
        raise ValueError("Cannot reuse legacy tracker results with different baseline input")
    policy_file.write_text(json.dumps({"baseline_input": baseline_input}) + "\n")

    sequences: list[str] = []
    planned_trackers: list[tuple[str, str, str]] = []
    dynamic_variant_envs = dict(getattr(args, "dynamic_variant_envs", {}) or {})
    for tracker in args.trackers:
        if tracker == "msdc_elt":
            variants = list(dynamic_variant_envs) if dynamic_variant_envs else list(args.variants)
            for variant in variants:
                planned_trackers.append((tracker, variant, tracker_output_name(tracker, variant)))
        else:
            planned_trackers.append((tracker, "", tracker_output_name(tracker)))
    tracker_names = [tracker_name for _, _, tracker_name in planned_trackers]
    max_frames = effective_max_frames(args)
    frame_source = getattr(args, "frame_source", "decode")
    for tracker_type, variant, _ in planned_trackers:
        _validate_replay_frame_source(frame_source, tracker_type, variant, max_frames, dynamic_variant_envs.get(variant))
    diagnostic_summary_rows: list[dict] = []
    for dataset_root in args.dataset_root:
        spec = resolve_single_sequence_dataset(dataset_root, video=args.input or None, seq_name=args.seq_name or None)
        if spec.video_path is None:
            raise FileNotFoundError(f"No video reference file found under: {spec.dataset_root}")
        if not spec.video_path.is_file():
            raise FileNotFoundError(f"Resolved video path is not accessible: {spec.video_path}")
        source_detection_cache_root = str(getattr(args, "source_detection_cache_root", "") or "")
        info = (
            _cached_source_video_info(source_detection_cache_root, spec.seq_name, max_frames)
            if frame_source == "cache_only" else read_video_info(spec.video_path)
        )
        file_type = _resolve_file_type(spec.video_path, args.file_type)
        expected_frames = expected_replay_frames(max_frames, int(info.frame_count), frame_source=frame_source)
        sequences.append(spec.seq_name)
        write_motchallenge_gt_sequence(spec, gt_root, info, max_frames=max_frames)
        if source_detection_cache_root:
            cache_path = link_detection_cache_for_sequence(
                source_detection_cache_root,
                detections_root,
                spec.seq_name,
            )
        else:
            cache_path = detections_root / detection_cache_name(spec.seq_name)
        cache_complete = (
            is_detection_cache_contiguous(cache_path, expected_frames)
            if source_detection_cache_root
            else is_detection_cache_complete(cache_path, expected_frames)
        )
        if cache_complete:
            print(f"[SKIP] detection cache complete: {cache_path}", flush=True)
        else:
            if source_detection_cache_root:
                raise RuntimeError(
                    "Incomplete source detection cache for "
                    f"{spec.seq_name}: {cache_path} does not cover {expected_frames} frames"
                )
            dump_video_detections(
                input_video=spec.video_path,
                output_cache=cache_path,
                file_type=file_type,
                max_frames=max_frames,
                progress_interval=int(args.progress_interval),
            )
        for tracker_type, variant, tracker_name in planned_trackers:
            env_override = dynamic_variant_envs.get(variant) if variant in dynamic_variant_envs else None
            tracker_file = trackers_root / tracker_name / "data" / f"{spec.seq_name}.txt"
            if tracker_type == "msdc_elt":
                write_effective_msdc_variant_config(run_root, tracker_name, variant, env_override=env_override)
            if is_mot_result_complete(tracker_file, expected_frames):
                print(f"[SKIP] replay MOT complete: {tracker_file}", flush=True)
            else:
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
                    env_override=env_override,
                    baseline_input=baseline_input,
                    frame_source=frame_source,
                )
            diag_dir = diagnostics_root / spec.seq_name
            eval_gt_file = gt_root / spec.seq_name / "gt" / "gt.txt"
            per_gt_csv, _ = write_diagnostics_for_tracker(eval_gt_file, tracker_file, diag_dir, tracker_name)
            tracker_diag_dir = trackers_root / tracker_name / "diagnostics" / spec.seq_name
            stage_observations_path = tracker_diag_dir / "stage_observations.jsonl"
            write_stage_coverage_csv(eval_gt_file, stage_observations_path, diag_dir, tracker_name)
            lifecycle_events_path = tracker_diag_dir / "lifecycle_events.jsonl"
            if per_gt_csv.is_file():
                row = summarize_msdc_diagnostics(lifecycle_events_path, stage_observations_path, per_gt_csv)
                row.update({"seq_name": spec.seq_name, "tracker": tracker_name})
                diagnostic_summary_rows.append(row)
            if args.render:
                render_output = run_root / "visualizations" / spec.seq_name / f"{tracker_name}.mp4"
                if is_render_video_complete(render_output, expected_frames):
                    print(f"[SKIP] rendered video complete: {render_output}", flush=True)
                    continue
                render_cmd = _build_render_command(
                    input_video=spec.video_path,
                    tracker_file=tracker_file,
                    output_file=render_output,
                    max_frames=max_frames,
                    progress_interval=int(args.progress_interval),
                    class_source=str(args.render_class_source),
                    detections_cache=cache_path if str(args.render_class_source) == "cache" else None,
                )
                if os.environ.get("PAPER_EXPERIMENT_SPEC"):
                    render_cmd.extend(["--max-output-width", "1920", "--video-codec", "h264"])
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
    write_msdc_diagnostic_summary(
        diagnostics_root / "msdc_diagnostic_summary.csv",
        diagnostic_summary_rows,
    )
    return write_replay_summary(run_root, summary)


def run_dynamic_msdc_replay_benchmark(
    *,
    dataset_root: list[str],
    output_root: str | Path,
    run_id: str,
    dynamic_variant_envs: dict[str, dict[str, str]],
    source_detection_cache_root: str | Path,
    formal_frame_limit: int = FORMAL_FRAME_LIMIT,
    progress_interval: int = 0,
    render_class_source: str = "cache",
    render: bool = False,
) -> Path:
    args = argparse.Namespace(
        dataset_root=list(dataset_root),
        output_root=str(output_root),
        run_id=run_id,
        max_frames=300,
        formal_frame_limit=int(formal_frame_limit),
        trackers=["msdc_elt"],
        variants=[],
        dynamic_variant_envs=dynamic_variant_envs,
        progress_interval=int(progress_interval),
        render=bool(render),
        render_class_source=render_class_source,
        input="",
        seq_name="",
        file_type="",
        source_detection_cache_root=str(source_detection_cache_root),
    )
    return run_detection_replay_benchmark(args)


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
    parser.add_argument("--frame-source", choices=("decode", "cache_only"), default="decode",
                        help="cache_only: fixed paper MSDC replay using completed main metadata/cache, with no pixel decoding")
    parser.add_argument("--baseline-input", choices=["high", "all"], default="high",
                        help="Baseline input: high boxes, or the complete low-threshold cache (includes high boxes)")
    parser.add_argument("--render", action="store_true", help="Render annotated videos after replay exports")
    parser.add_argument("--render-class-source", choices=["detector", "cache", "none"], default="cache")
    parser.add_argument("--input", default="", help="Optional direct input video path for a single dataset")
    parser.add_argument("--seq-name", default="", help="Optional sequence name override for a single dataset")
    parser.add_argument("--file-type", default="", choices=["", "visible", "infrared"])
    parser.add_argument(
        "--source-detection-cache-root",
        default="",
        help="Optional directory containing <seq>_high_low_detections.jsonl caches to reuse",
    )
    args = parser.parse_args(argv)
    if int(args.max_frames) < 0:
        parser.error("--max-frames must be >= 0")
    if int(args.formal_frame_limit) < 0:
        parser.error("--formal-frame-limit must be >= 0")
    if int(args.formal_frame_limit) == 0 and int(args.max_frames) <= 0:
        parser.error("--max-frames must be > 0 when --formal-frame-limit is 0")
    if int(args.progress_interval) < 0:
        parser.error("--progress-interval must be >= 0")
    if args.frame_source == "cache_only":
        if args.formal_frame_limit <= 0:
            parser.error("cache_only requires explicit --formal-frame-limit N > 0")
        if not args.source_detection_cache_root:
            parser.error("cache_only requires --source-detection-cache-root")
        if args.trackers != ["msdc_elt"] or any(name not in CACHE_ONLY_MSDC_VARIANTS for name in args.variants):
            parser.error("cache_only is restricted to the seventeen fixed paper MSDC configurations")
    return args


def main(argv: list[str] | None = None) -> None:
    if os.environ.get("PAPER_EXPERIMENT_SPEC"):
        from target_module.image_detect_module.paper_config import seed_paper_runtime
        seed_paper_runtime(20261030)
    summary_path = run_detection_replay_benchmark(parse_args(argv))
    print(f"[OK] Replay summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
