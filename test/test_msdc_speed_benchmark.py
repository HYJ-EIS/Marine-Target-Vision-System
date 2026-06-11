import argparse
import csv
import io
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

import tools.evaluation.msdc_speed_benchmark as benchmark
from tools.evaluation.msdc_speed_benchmark import (
    CountingProcessor,
    percentile,
    summarize_latencies,
)


class DummyProcessor:
    def process_frame(self, frame, file_type, conf_override=None):
        return {"boxes": [], "processing_time": 0.01}


def test_percentile_uses_sorted_nearest_rank():
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 95) == 38.5


def test_summarize_latencies_reports_mean_p50_p95():
    summary = summarize_latencies([0.01, 0.02, 0.03], processed_frames=3)
    assert summary["processed_frames"] == 3
    assert round(summary["mean_latency_ms"], 4) == 20.0
    assert round(summary["p50_latency_ms"], 4) == 20.0
    assert round(summary["p95_latency_ms"], 4) == 29.0


def test_counting_processor_records_calls_by_stage():
    processor = CountingProcessor(DummyProcessor())
    processor.stage = "high_det"
    processor.process_frame(frame=object(), file_type="visible", conf_override=None)
    processor.stage = "low_det"
    processor.process_frame(frame=object(), file_type="visible", conf_override=0.18)
    assert processor.call_counts == {"high_det": 1, "low_det": 1}


def test_counting_processor_restores_outer_stage_after_nested_roi_stage():
    processor = CountingProcessor(DummyProcessor())

    with processor.use_stage("tracker_update"):
        processor.process_frame(frame=object(), file_type="visible")
        with processor.use_stage("roi_redetect"):
            processor.process_frame(frame=object(), file_type="visible")
        processor.process_frame(frame=object(), file_type="visible")

    assert processor.call_counts == {"tracker_update": 2, "roi_redetect": 1}


def test_direct_script_reaches_dataset_validation_without_import_error(tmp_path):
    script = _ROOT / "tools" / "evaluation" / "msdc_speed_benchmark.py"
    missing_dataset = tmp_path / "missing-dataset"
    result = subprocess.run(
        [sys.executable, str(script), "--dataset-root", str(missing_dataset), "--frames", "0"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stderr
    assert "Dataset root does not exist" in result.stderr


def test_run_speed_benchmark_writes_zero_frame_outputs(tmp_path, monkeypatch):
    def fake_run_tracker_benchmark(**kwargs):
        assert kwargs["requested_frames"] == 0
        return {
            "run_id": kwargs["run_id"],
            "commit_hash": kwargs["commit_hash"],
            "video_path": str(kwargs["input_video"]),
            "seq_name": kwargs["seq_name"],
            "tracker": kwargs["tracker_type"],
            "method": "fake",
            "resolution": "0x0",
            "requested_frames": 0,
            "processed_frames": 0,
            "total_time_s": "0.000000",
            "mean_fps": "0.000000",
            "mean_latency_ms": "0.000000",
            "p50_latency_ms": "0.000000",
            "p95_latency_ms": "0.000000",
            "peak_memory_mb": "N/A",
            "detector_calls_total": 0,
            "detector_calls_high_det": 0,
            "detector_calls_low_det": 0,
            "detector_calls_tracker_update": 0,
            "detector_calls_roi_redetect": 0,
            "mean_read_ms": "0.000000",
            "mean_high_det_ms": "0.000000",
            "mean_low_det_ms": "0.000000",
            "mean_roi_redetect_ms": "0.000000",
            "mean_tracker_ms": "0.000000",
        }

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    monkeypatch.setattr(benchmark, "_resolve_benchmark_input", lambda args: (video, "seq", "visible"))
    monkeypatch.setattr(benchmark, "_run_tracker_benchmark", fake_run_tracker_benchmark)

    args = argparse.Namespace(
        output_root=str(tmp_path / "out"),
        run_id="zero",
        commit_hash="abc123",
        frames=0,
        trackers=["ocsort", "msdc_elt"],
        progress_interval=0,
    )
    results_path, timings_path = benchmark.run_speed_benchmark(args)

    assert timings_path.read_text(encoding="utf-8") == ""
    with results_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["tracker"] for row in rows] == ["ocsort", "msdc_elt"]
    assert {row["processed_frames"] for row in rows} == {"0"}
    assert "detector_calls_roi_redetect" in rows[0]
    assert "mean_roi_redetect_ms" in rows[0]


def test_run_speed_benchmark_restores_debug_events_after_exception(tmp_path, monkeypatch):
    from target_module.image_detect_module.config import Config

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    monkeypatch.setattr(benchmark, "_resolve_benchmark_input", lambda args: (video, "seq", "visible"))
    monkeypatch.setattr(
        benchmark,
        "_run_tracker_benchmark",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(Config, "MSDC_DEBUG_EVENTS", True, raising=False)

    args = argparse.Namespace(
        output_root=str(tmp_path / "out"),
        run_id="raises",
        commit_hash="abc123",
        frames=0,
        trackers=["ocsort"],
        progress_interval=0,
    )
    with pytest.raises(RuntimeError, match="boom"):
        benchmark.run_speed_benchmark(args)
    assert Config.MSDC_DEBUG_EVENTS is True


def test_tracker_benchmark_releases_unopened_capture(tmp_path, monkeypatch):
    import cv2
    import target_module.image_detect_module.target_detection as td

    class FakeProcessor:
        def process_frame(self, frame, file_type, conf_override=None):
            return {"boxes": []}

    class FakeCapture:
        released = False

        def __init__(self, path):
            self.path = path

        def isOpened(self):
            return False

        def release(self):
            type(self).released = True

    monkeypatch.setattr(td, "get_detector", lambda: SimpleNamespace(processor=FakeProcessor()))
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    with pytest.raises(RuntimeError, match="Failed to open video"):
        benchmark._run_tracker_benchmark(
            input_video=tmp_path / "missing.mp4",
            seq_name="seq",
            file_type="visible",
            tracker_type="ocsort",
            requested_frames=1,
            run_id="run",
            commit_hash="abc123",
            timing_fh=io.StringIO(),
            progress_interval=0,
        )
    assert FakeCapture.released is True


def test_msdc_shared_low_high_detection_uses_one_detector_call(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import target_module.image_detect_module.target_detection as td
    import tools.evaluation.export_mot_results as export_helpers
    import target_module.image_detect_module.utils.lifecycle_tracker as lifecycle_module
    from target_module.image_detect_module.config import Config

    class FakeProcessor:
        def process_frame(self, frame, file_type, conf_override=None):
            return {
                "boxes": [
                    {"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.99, "class": "USV"},
                ],
            }

    class FakeCapture:
        def __init__(self, path):
            self._reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_WIDTH: 4,
                cv2.CAP_PROP_FRAME_HEIGHT: 3,
            }
            return values.get(prop, 0)

        def read(self):
            if self._reads > 0:
                return False, None
            self._reads += 1
            return True, np.zeros((3, 4, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(Config, "MSDC_EXPORT_SHARE_LOW_HIGH_DET", True, raising=False)
    monkeypatch.setattr(td, "get_detector", lambda: SimpleNamespace(processor=FakeProcessor()))
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(lifecycle_module, "MSDCLifecycleTracker", lambda **kwargs: object())
    monkeypatch.setattr(export_helpers, "_update_tracking_for_frame", lambda **kwargs: [])

    row = benchmark._run_tracker_benchmark(
        input_video=tmp_path / "input.mp4",
        seq_name="seq",
        file_type="visible",
        tracker_type="msdc_elt",
        requested_frames=1,
        run_id="run",
        commit_hash="abc123",
        timing_fh=io.StringIO(),
        progress_interval=0,
    )

    assert row["processed_frames"] == 1
    assert row["detector_calls_total"] == 1
    assert row["detector_calls_high_det"] == 0
    assert row["detector_calls_low_det"] == 1
    assert row["detector_calls_roi_redetect"] == 0
    assert row["mean_roi_redetect_ms"] == "0.000000"


def test_tracker_update_detector_calls_do_not_subtract_nested_roi_calls(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import target_module.image_detect_module.target_detection as td
    import tools.evaluation.export_mot_results as export_helpers
    import target_module.image_detect_module.utils.lifecycle_tracker as lifecycle_module
    from target_module.image_detect_module.config import Config

    captured_call_counts = {}

    class FakeProcessor:
        def process_frame(self, frame, file_type, conf_override=None):
            return {
                "boxes": [
                    {"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.99, "class": "USV"},
                ],
            }

    class FakeCapture:
        def __init__(self, path):
            self._reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_WIDTH: 4,
                cv2.CAP_PROP_FRAME_HEIGHT: 3,
            }
            return values.get(prop, 0)

        def read(self):
            if self._reads > 0:
                return False, None
            self._reads += 1
            return True, np.zeros((3, 4, 3), dtype=np.uint8)

        def release(self):
            pass

    def fake_update_tracking_for_frame(**kwargs):
        processor = kwargs["detector"].processor
        processor.process_frame(kwargs["frame"], kwargs["file_type"])
        with processor.use_stage("roi_redetect"):
            processor.process_frame(kwargs["frame"], kwargs["file_type"])
        processor.process_frame(kwargs["frame"], kwargs["file_type"])
        captured_call_counts.update(processor.call_counts)
        return []

    monkeypatch.setattr(Config, "MSDC_EXPORT_SHARE_LOW_HIGH_DET", True, raising=False)
    monkeypatch.setattr(td, "get_detector", lambda: SimpleNamespace(processor=FakeProcessor()))
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(lifecycle_module, "MSDCLifecycleTracker", lambda **kwargs: object())
    monkeypatch.setattr(export_helpers, "_update_tracking_for_frame", fake_update_tracking_for_frame)

    row = benchmark._run_tracker_benchmark(
        input_video=tmp_path / "input.mp4",
        seq_name="seq",
        file_type="visible",
        tracker_type="msdc_elt",
        requested_frames=1,
        run_id="run",
        commit_hash="abc123",
        timing_fh=io.StringIO(),
        progress_interval=0,
    )

    assert captured_call_counts["tracker_update"] == 2
    assert captured_call_counts["roi_redetect"] == 1
    assert row["detector_calls_total"] == 4
    assert row["detector_calls_tracker_update"] == 2
    assert row["detector_calls_roi_redetect"] == 1
