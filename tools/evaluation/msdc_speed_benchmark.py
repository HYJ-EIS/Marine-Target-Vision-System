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


TRACKER_CHOICES = ["ocsort", "botsort", "msdc_elt"]
DEFAULT_TRACKERS = list(TRACKER_CHOICES)
SPEED_FIELDS = [
    "run_id",
    "commit_hash",
    "video_path",
    "seq_name",
    "tracker",
    "method",
    "resolution",
    "requested_frames",
    "processed_frames",
    "total_time_s",
    "mean_fps",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "peak_memory_mb",
    "detector_calls_total",
    "detector_calls_high_det",
    "detector_calls_low_det",
    "detector_calls_tracker_update",
    "detector_calls_roi_redetect",
    "mean_read_ms",
    "mean_high_det_ms",
    "mean_low_det_ms",
    "mean_roi_redetect_ms",
    "mean_tracker_ms",
]
METHOD_LABELS = {
    "ocsort": "FFCA-YOLO + OC-SORT",
    "botsort": "FFCA-YOLO + BoT-SORT",
    "msdc_elt": "FFCA-YOLO + MS-DC-ELT",
}

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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
    from target_module.image_detect_module.utils.tracker import MultiObjectTracker
    from tools.evaluation.export_mot_results import (
        _is_msdc_tracker,
        _run_msdc_high_threshold_detection,
        _run_msdc_low_threshold_detection,
        _split_msdc_high_from_low_boxes,
        _update_tracking_for_frame,
    )

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

        if _is_msdc_tracker(tracker_type):
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

            if _is_msdc_tracker(tracker_type):
                if bool(getattr(Config, "MSDC_EXPORT_SHARE_LOW_HIGH_DET", False)):
                    stage_start = time.perf_counter()
                    with processor.use_stage("low_det"):
                        low_boxes = _run_msdc_low_threshold_detection(detector, frame, file_type)
                    low_det_s = time.perf_counter() - stage_start

                    stage_start = time.perf_counter()
                    boxes = _split_msdc_high_from_low_boxes(low_boxes, file_type)
                    high_det_s = time.perf_counter() - stage_start
                else:
                    stage_start = time.perf_counter()
                    with processor.use_stage("high_det"):
                        boxes = _run_msdc_high_threshold_detection(detector, frame, file_type)
                    high_det_s = time.perf_counter() - stage_start

                    stage_start = time.perf_counter()
                    with processor.use_stage("low_det"):
                        low_boxes = _run_msdc_low_threshold_detection(detector, frame, file_type)
                    low_det_s = time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                with processor.use_stage("tracker_update"):
                    _update_tracking_for_frame(
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

            _write_timing_row(
                timing_fh,
                {
                    "frame_id": processed_frames,
                    "tracker": tracker_type,
                    "read_s": read_s,
                    "high_det_s": high_det_s,
                    "low_det_s": low_det_s,
                    "tracker_s": tracker_s,
                    "total_s": total_s,
                },
            )
            if progress_interval > 0 and processed_frames % progress_interval == 0:
                print(f"[PROGRESS] {tracker_type} {seq_name}: {processed_frames}/{requested_frames}", flush=True)

        summary = summarize_latencies(latencies, processed_frames=processed_frames)
        call_counts = dict(processor.call_counts)
        roi_calls = int(processor.call_counts.get("roi_redetect", 0))
        roi_seconds = float(processor.stage_seconds.get("roi_redetect", 0.0))
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
            "detector_calls_roi_redetect": roi_calls,
            "mean_read_ms": _format_float(_mean_ms(read_times)),
            "mean_high_det_ms": _format_float(_mean_ms(high_det_times)),
            "mean_low_det_ms": _format_float(_mean_ms(low_det_times)),
            "mean_roi_redetect_ms": _format_float((roi_seconds / roi_calls * 1000.0) if roi_calls else 0.0),
            "mean_tracker_ms": _format_float(_mean_ms(tracker_times)),
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

    had_debug_attr = hasattr(Config, "MSDC_DEBUG_EVENTS")
    old_debug_events = getattr(Config, "MSDC_DEBUG_EVENTS", None)
    Config.MSDC_DEBUG_EVENTS = False
    try:
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
    finally:
        if had_debug_attr:
            Config.MSDC_DEBUG_EVENTS = old_debug_events
        else:
            delattr(Config, "MSDC_DEBUG_EVENTS")

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
    parser.add_argument("--trackers", nargs="+", choices=TRACKER_CHOICES, default=DEFAULT_TRACKERS)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--file-type", default="", choices=["", "visible", "infrared"])
    args = parser.parse_args(argv)
    if int(args.frames) < 0:
        parser.error("--frames must be >= 0")
    if int(args.progress_interval) < 0:
        parser.error("--progress-interval must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    results_path, timings_path = run_speed_benchmark(parse_args(argv))
    print(f"[OK] Speed results: {results_path.resolve()}")
    print(f"[OK] Per-frame timings: {timings_path.resolve()}")


if __name__ == "__main__":
    main()
