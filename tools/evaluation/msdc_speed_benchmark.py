"""Run detector/tracker speed measurements for MS-DC-ELT paper experiments."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


MSDC_INTERNAL_TIMING_KEYS = [
    "low_filter_s",
    "observation_build_s",
    "evidence_update_s",
    "output_s",
    "debug_s",
    "total_update_s",
]

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import (  # noqa: E402
    METHOD_LABELS,
    PAPER_TRACKER_CHOICES,
    SPEED_FIELDS,
)
from tools.experiments.run_msdc_ablation import V3_CANDIDATE_TOPK_NO_ROI_NO_MOTION_ENV  # noqa: E402


_FORMAL_V3_BOOL_KEYS = {
    "MSDC_USE_REACQUIRE",
    "MSDC_REUSE_GUARD_ENABLE",
    "MSDC_DEBUG_EVENTS",
    "MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY",
}
_FORMAL_V3_INT_KEYS = {
    "MSDC_OUTPUT_MAX_REAL_DET_AGE",
    "MSDC_OUTPUT_MIN_BOX_SIZE",
    "MSDC_LOW_CONFIRM_MIN_HITS",
    "MSDC_LOW_CONFIRM_WINDOW",
    "MSDC_CONFIRM_MIN_REAL_DET_HITS",
    "MSDC_CANDIDATE_MAX_AGE",
    "MSDC_LOW_OBS_TOPK",
    "MSDC_LOW_OBS_GLOBAL_TOPK",
    "MSDC_LOW_OBS_PER_TRACK_NEAREST",
    "MSDC_LOW_OBS_MAX_PER_FRAME",
    "MSDC_MAX_ACTIVE_TRACKS",
    "MSDC_MAX_LOST_TRACKS",
    "MSDC_MAX_CANDIDATES",
    "MSDC_MAX_LOW_CANDIDATES",
    "MSDC_MAX_TOTAL_TRACKS",
    "MSDC_REACQUIRE_INTERVAL",
}
_FORMAL_V3_FLOAT_KEYS = {
    "MSDC_LOW_OBS_MIN_CONF",
    "MSDC_REACQUIRE_CENTER_DIST",
    "MSDC_REACQUIRE_MAX_CENTER_DIST",
}


def _coerce_formal_v3_value(key: str, value: str) -> bool | int | float:
    if key in _FORMAL_V3_BOOL_KEYS:
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
    if key in _FORMAL_V3_INT_KEYS:
        return int(value)
    if key in _FORMAL_V3_FLOAT_KEYS:
        return float(value)
    raise KeyError(f"Unsupported formal v3 speed config key: {key}")


FORMAL_V3_CONFIG_OVERRIDES = {
    key: _coerce_formal_v3_value(key, V3_CANDIDATE_TOPK_NO_ROI_NO_MOTION_ENV[key])
    for key in [*_FORMAL_V3_BOOL_KEYS, *_FORMAL_V3_INT_KEYS, *_FORMAL_V3_FLOAT_KEYS]
}
FORMAL_V3_CONFIG_OVERRIDES["MSDC_DEBUG_EVENTS"] = False


def percentile(values: list[float] | tuple[float, ...], q: float) -> float:
    """Return a percentile using sorted linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    quantile = min(100.0, max(0.0, float(q))) / 100.0
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_latencies(latencies: list[float] | tuple[float, ...], processed_frames: int) -> dict[str, float | int]:
    """Summarize per-frame latencies expressed in seconds."""
    processed = int(processed_frames)
    latency_values = [float(value) for value in latencies]
    total_time_s = float(sum(latency_values))
    mean_latency_s = total_time_s / len(latency_values) if latency_values else 0.0
    mean_fps = float(processed) / total_time_s if total_time_s > 0.0 else 0.0
    latency_ms = [value * 1000.0 for value in latency_values]
    return {
        "processed_frames": processed,
        "total_time_s": total_time_s,
        "mean_fps": mean_fps,
        "mean_latency_ms": mean_latency_s * 1000.0,
        "p50_latency_ms": percentile(latency_ms, 50),
        "p95_latency_ms": percentile(latency_ms, 95),
    }


class CountingProcessor:
    """ImageProcessor wrapper that counts and times detector calls by stage."""

    def __init__(self, processor: Any):
        self._processor = processor
        self.stage = "unspecified"
        self.call_counts: dict[str, int] = {}
        self.stage_seconds: dict[str, float] = {}

    @contextmanager
    def use_stage(self, stage: str) -> Iterator["CountingProcessor"]:
        previous = self.stage
        self.stage = stage
        try:
            yield self
        finally:
            self.stage = previous

    def process_frame(self, frame, file_type, conf_override=None):
        stage = self.stage or "unspecified"
        start = time.perf_counter()
        try:
            return self._processor.process_frame(frame, file_type, conf_override=conf_override)
        finally:
            elapsed = time.perf_counter() - start
            self.call_counts[stage] = self.call_counts.get(stage, 0) + 1
            self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + elapsed

    def __getattr__(self, name: str):
        return getattr(self._processor, name)


def peak_memory_mb() -> float | str:
    """Return current process peak RSS in MB when the platform exposes it."""
    try:
        import resource

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            usage /= 1024.0 * 1024.0
        else:
            usage /= 1024.0
        return round(usage, 3)
    except Exception:
        return "N/A"


def _mean_ms(values: list[float]) -> float:
    return (sum(values) / len(values) * 1000.0) if values else 0.0


def _empty_msdc_internal_timing() -> dict[str, float]:
    return {key: 0.0 for key in MSDC_INTERNAL_TIMING_KEYS}


def _format_float(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.6f}"


def _default_run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "N/A"
    except Exception:
        return "N/A"


def _resolve_benchmark_input(args: argparse.Namespace) -> tuple[Path, str, str]:
    from tools.evaluation.msdc_dataset_benchmark import resolve_single_sequence_dataset
    from target_module.image_detect_module.utils.file_utils import get_file_type

    dataset = resolve_single_sequence_dataset(
        args.dataset_root,
        video=args.input or None,
        seq_name=args.seq_name or None,
    )
    if dataset.video_path is None:
        raise FileNotFoundError("No input video resolved; pass --input or provide a dataset video reference file")
    input_video = Path(dataset.video_path)
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")

    file_type = args.file_type or get_file_type(input_video.stem)
    if file_type == "unknown":
        file_type = "visible"
    return input_video, dataset.seq_name, file_type


def _reset_detector_singleton(td_module: Any) -> None:
    if hasattr(td_module, "_detector_instance"):
        td_module._detector_instance = None


@contextmanager
def _temporary_config_overrides(config: Any, overrides: dict[str, bool | int | float]) -> Iterator[None]:
    previous = {
        key: (hasattr(config, key), getattr(config, key, None))
        for key in overrides
    }
    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        yield
    finally:
        for key, (had_attr, value) in previous.items():
            if had_attr:
                setattr(config, key, value)
            else:
                delattr(config, key)


def _write_timing_row(timing_fh, row: dict[str, Any]) -> None:
    timing_fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _run_tracker_benchmark(
    *,
    input_video: Path,
    seq_name: str,
    file_type: str,
    tracker_type: str,
    requested_frames: int,
    run_id: str,
    commit_hash: str,
    timing_fh,
    progress_interval: int,
) -> dict[str, str | int]:
    import cv2
    import target_module.image_detect_module.target_detection as td
    from target_module.image_detect_module.config import Config
    from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker
    from target_module.image_detect_module.utils.msdc_detection import (
        run_msdc_low_threshold_detection,
        split_msdc_high_from_low_boxes,
    )
    from target_module.image_detect_module.utils.tracking_update import (
        is_msdc_tracker,
        update_tracking_for_frame,
    )
    from target_module.image_detect_module.utils.tracker import MultiObjectTracker

    _reset_detector_singleton(td)
    detector = td.get_detector()
    if getattr(detector, "processor", None) is None:
        raise RuntimeError("Detector processor is unavailable")
    processor = CountingProcessor(detector.processor)
    detector.processor = processor

    cap = cv2.VideoCapture(str(input_video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        resolution = f"{width}x{height}" if width > 0 and height > 0 else "N/A"

        if is_msdc_tracker(tracker_type):
            tracker = None
            lifecycle_tracker = MSDCLifecycleTracker(
                config=Config,
                file_type=file_type,
                debug_dir=None,
                processor=processor,
            )
        else:
            tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_type)
            lifecycle_tracker = None

        latencies: list[float] = []
        read_times: list[float] = []
        high_det_times: list[float] = []
        low_det_times: list[float] = []
        tracker_times: list[float] = []
        msdc_internal_times: dict[str, list[float]] = {key: [] for key in MSDC_INTERNAL_TIMING_KEYS}
        processed_frames = 0

        while processed_frames < requested_frames:
            frame_start = time.perf_counter()
            read_start = time.perf_counter()
            ret, frame = cap.read()
            read_s = time.perf_counter() - read_start
            if not ret:
                break

            processed_frames += 1
            high_det_s = 0.0
            low_det_s = 0.0
            tracker_s = 0.0
            msdc_timing = _empty_msdc_internal_timing()

            if is_msdc_tracker(tracker_type):
                stage_start = time.perf_counter()
                with processor.use_stage("low_det"):
                    low_boxes = run_msdc_low_threshold_detection(detector, frame, file_type)
                low_det_s = time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                boxes = split_msdc_high_from_low_boxes(low_boxes, file_type)
                high_det_s = time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                with processor.use_stage("tracker_update"):
                    update_tracking_for_frame(
                        tracker_type=tracker_type,
                        tracker=tracker,
                        lifecycle_tracker=lifecycle_tracker,
                        detector=detector,
                        frame=frame,
                        frame_idx=processed_frames - 1,
                        file_type=file_type,
                        boxes=boxes,
                        low_boxes=low_boxes,
                    )
                tracker_s = time.perf_counter() - stage_start
                if lifecycle_tracker is not None:
                    msdc_timing = {
                        key: float(value)
                        for key, value in dict(getattr(lifecycle_tracker, "last_timing_debug", {}) or {}).items()
                        if key in msdc_internal_times
                    }
                    for key in MSDC_INTERNAL_TIMING_KEYS:
                        msdc_timing.setdefault(key, 0.0)
            else:
                stage_start = time.perf_counter()
                with processor.use_stage("high_det"):
                    detection_stats = processor.process_frame(frame, file_type)
                high_det_s = time.perf_counter() - stage_start
                if detection_stats is None:
                    raise RuntimeError(f"Detection failed: tracker={tracker_type} frame={processed_frames}")
                boxes = detection_stats.get("boxes", [])

                stage_start = time.perf_counter()
                with processor.use_stage("tracker_update"):
                    tracker.update(boxes, frame.shape, frame=frame)
                tracker_s = time.perf_counter() - stage_start

            total_s = time.perf_counter() - frame_start
            latencies.append(total_s)
            read_times.append(read_s)
            high_det_times.append(high_det_s)
            low_det_times.append(low_det_s)
            tracker_times.append(tracker_s)
            for key, value in msdc_timing.items():
                msdc_internal_times[key].append(float(value))

            timing_row = {
                "frame_id": processed_frames,
                "tracker": tracker_type,
                "read_s": read_s,
                "high_det_s": high_det_s,
                "low_det_s": low_det_s,
                "tracker_s": tracker_s,
                "render_s": 0.0,
                "write_s": 0.0,
                "total_s": total_s,
            }
            timing_row.update({f"msdc_{key}": float(msdc_timing.get(key, 0.0)) for key in MSDC_INTERNAL_TIMING_KEYS})
            _write_timing_row(timing_fh, timing_row)
            if progress_interval > 0 and processed_frames % progress_interval == 0:
                print(f"[PROGRESS] {tracker_type} {seq_name}: {processed_frames}/{requested_frames}", flush=True)

        summary = summarize_latencies(latencies, processed_frames=processed_frames)
        call_counts = dict(processor.call_counts)
        return {
            "run_id": run_id,
            "commit_hash": commit_hash,
            "video_path": str(input_video),
            "seq_name": seq_name,
            "tracker": tracker_type,
            "method": METHOD_LABELS.get(tracker_type, tracker_type),
            "resolution": resolution,
            "requested_frames": int(requested_frames),
            "processed_frames": int(summary["processed_frames"]),
            "total_time_s": _format_float(summary["total_time_s"]),
            "mean_fps": _format_float(summary["mean_fps"]),
            "mean_latency_ms": _format_float(summary["mean_latency_ms"]),
            "p50_latency_ms": _format_float(summary["p50_latency_ms"]),
            "p95_latency_ms": _format_float(summary["p95_latency_ms"]),
            "peak_memory_mb": _format_float(peak_memory_mb()),
            "detector_calls_total": int(sum(call_counts.values())),
            "detector_calls_high_det": int(call_counts.get("high_det", 0)),
            "detector_calls_low_det": int(call_counts.get("low_det", 0)),
            "detector_calls_tracker_update": int(call_counts.get("tracker_update", 0)),
            "mean_read_decode_ms": _format_float(_mean_ms(read_times)),
            "mean_low_detection_ms": _format_float(_mean_ms(low_det_times)),
            "mean_high_split_ms": _format_float(_mean_ms(high_det_times) if is_msdc_tracker(tracker_type) else 0.0),
            "mean_low_filter_budget_ms": _format_float(_mean_ms(msdc_internal_times["low_filter_s"])),
            "mean_observation_build_ms": _format_float(_mean_ms(msdc_internal_times["observation_build_s"])),
            "mean_evidence_update_ms": _format_float(_mean_ms(msdc_internal_times["evidence_update_s"])),
            "mean_output_nms_ms": _format_float(_mean_ms(msdc_internal_times["output_s"])),
            "mean_render_write_ms": _format_float(0.0),
            "mean_read_ms": _format_float(_mean_ms(read_times)),
            "mean_high_det_ms": _format_float(_mean_ms(high_det_times)),
            "mean_low_det_ms": _format_float(_mean_ms(low_det_times)),
            "mean_tracker_ms": _format_float(_mean_ms(tracker_times)),
            "mean_msdc_low_filter_ms": _format_float(_mean_ms(msdc_internal_times["low_filter_s"])),
            "mean_msdc_observation_build_ms": _format_float(_mean_ms(msdc_internal_times["observation_build_s"])),
            "mean_msdc_evidence_update_ms": _format_float(_mean_ms(msdc_internal_times["evidence_update_s"])),
            "mean_msdc_output_ms": _format_float(_mean_ms(msdc_internal_times["output_s"])),
            "mean_msdc_debug_ms": _format_float(_mean_ms(msdc_internal_times["debug_s"])),
            "mean_msdc_total_update_ms": _format_float(_mean_ms(msdc_internal_times["total_update_s"])),
            "mean_render_ms": _format_float(0.0),
            "mean_write_ms": _format_float(0.0),
        }
    finally:
        cap.release()


def run_speed_benchmark(args: argparse.Namespace) -> tuple[Path, Path]:
    from target_module.image_detect_module.config import Config

    input_video, seq_name, file_type = _resolve_benchmark_input(args)
    run_id = args.run_id or _default_run_id()
    commit_hash = args.commit_hash or _git_commit_hash()
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    timings_path = output_dir / "speed_timings.jsonl"
    results_path = output_dir / "speed_results.csv"

    with _temporary_config_overrides(Config, FORMAL_V3_CONFIG_OVERRIDES):
        with timings_path.open("w", encoding="utf-8") as timing_fh:
            rows = [
                _run_tracker_benchmark(
                    input_video=input_video,
                    seq_name=seq_name,
                    file_type=file_type,
                    tracker_type=tracker_type,
                    requested_frames=int(args.frames),
                    run_id=run_id,
                    commit_hash=commit_hash,
                    timing_fh=timing_fh,
                    progress_interval=int(args.progress_interval),
                )
                for tracker_type in args.trackers
            ]

    with results_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=SPEED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return results_path, timings_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detector/tracker speed without rendering or TrackEval")
    parser.add_argument("--dataset-root", required=True, help="Single-sequence dataset root")
    parser.add_argument("--input", default="", help="Optional direct input video path")
    parser.add_argument("--seq-name", default="", help="Optional sequence name override")
    parser.add_argument("--output-root", default="results/msdc_speed_benchmark")
    parser.add_argument("--run-id", default="", help="Output run id; default is timestamp")
    parser.add_argument("--commit-hash", default="", help="Commit hash recorded in speed_results.csv")
    parser.add_argument("--frames", type=int, default=1000, help="Requested frame count")
    parser.add_argument("--trackers", nargs="+", choices=PAPER_TRACKER_CHOICES, default=list(PAPER_TRACKER_CHOICES))
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--file-type", default="", choices=["", "visible", "infrared"])
    args = parser.parse_args(argv)
    if int(args.frames) < 0:
        parser.error("--frames must be >= 0")
    if int(args.progress_interval) < 0:
        parser.error("--progress-interval must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_path, timings_path = run_speed_benchmark(args)
    print(f"[OK] Speed results: {results_path.resolve()}")
    print(f"[OK] Per-frame timings: {timings_path.resolve()}")


if __name__ == "__main__":
    main()
