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
            "mean_render_ms": "0.000000",
            "mean_write_ms": "0.000000",
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
    assert "mean_render_ms" in rows[0]
    assert "mean_write_ms" in rows[0]


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


def test_run_speed_benchmark_applies_and_restores_formal_v3_config(tmp_path, monkeypatch):
    from target_module.image_detect_module.config import Config

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    seen_config = []

    def fake_run_tracker_benchmark(**kwargs):
        seen_config.append(
            {
                "motion": Config.MSDC_USE_MOTION,
                "roi": Config.MSDC_USE_ROI_REDETECT,
                "template": Config.MSDC_USE_TEMPLATE,
                "template_enable": Config.MSDC_TEMPLATE_ENABLE,
                "share": Config.MSDC_EXPORT_SHARE_LOW_HIGH_DET,
                "debug": Config.MSDC_DEBUG_EVENTS,
                "topk": Config.MSDC_LOW_OBS_TOPK,
                "min_conf": Config.MSDC_LOW_OBS_MIN_CONF,
            }
        )
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
            "mean_render_ms": "0.000000",
            "mean_write_ms": "0.000000",
        }

    monkeypatch.setattr(benchmark, "_resolve_benchmark_input", lambda args: (video, "seq", "visible"))
    monkeypatch.setattr(benchmark, "_run_tracker_benchmark", fake_run_tracker_benchmark)
    monkeypatch.setattr(Config, "MSDC_USE_MOTION", True, raising=False)
    monkeypatch.setattr(Config, "MSDC_USE_ROI_REDETECT", True, raising=False)
    monkeypatch.setattr(Config, "MSDC_USE_TEMPLATE", True, raising=False)
    monkeypatch.setattr(Config, "MSDC_TEMPLATE_ENABLE", True, raising=False)
    monkeypatch.setattr(Config, "MSDC_EXPORT_SHARE_LOW_HIGH_DET", False, raising=False)
    monkeypatch.setattr(Config, "MSDC_DEBUG_EVENTS", True, raising=False)
    monkeypatch.setattr(Config, "MSDC_LOW_OBS_TOPK", 0, raising=False)
    monkeypatch.setattr(Config, "MSDC_LOW_OBS_MIN_CONF", 0.0, raising=False)

    args = argparse.Namespace(
        output_root=str(tmp_path / "out"),
        run_id="formal-v3",
        commit_hash="abc123",
        frames=0,
        trackers=["msdc_elt"],
        progress_interval=0,
    )
    benchmark.run_speed_benchmark(args)

    assert seen_config == [
        {
            "motion": False,
            "roi": False,
            "template": False,
            "template_enable": False,
            "share": True,
            "debug": False,
            "topk": 32,
            "min_conf": 0.25,
        }
    ]
    assert Config.MSDC_USE_MOTION is True
    assert Config.MSDC_USE_ROI_REDETECT is True
    assert Config.MSDC_USE_TEMPLATE is True
    assert Config.MSDC_TEMPLATE_ENABLE is True
    assert Config.MSDC_EXPORT_SHARE_LOW_HIGH_DET is False
    assert Config.MSDC_DEBUG_EVENTS is True
    assert Config.MSDC_LOW_OBS_TOPK == 0
    assert Config.MSDC_LOW_OBS_MIN_CONF == 0.0


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
    assert row["mean_render_ms"] == "0.000000"
    assert row["mean_write_ms"] == "0.000000"


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


def test_write_msdc_share_low_high_compare_reports_speedup(tmp_path):
    before = {
        "tracker": "msdc_elt",
        "processed_frames": 10,
        "total_time_s": "20.000000",
        "mean_fps": "0.500000",
        "mean_latency_ms": "2000.000000",
        "detector_calls_total": 20,
        "detector_calls_high_det": 10,
        "detector_calls_low_det": 10,
        "mean_high_det_ms": "900.000000",
        "mean_low_det_ms": "800.000000",
        "mean_tracker_ms": "300.000000",
    }
    after = {
        "tracker": "msdc_elt",
        "processed_frames": 10,
        "total_time_s": "10.000000",
        "mean_fps": "1.000000",
        "mean_latency_ms": "1000.000000",
        "detector_calls_total": 10,
        "detector_calls_high_det": 0,
        "detector_calls_low_det": 10,
        "mean_high_det_ms": "0.100000",
        "mean_low_det_ms": "700.000000",
        "mean_tracker_ms": "299.000000",
    }

    path = benchmark.write_msdc_share_low_high_compare(tmp_path, before, after, label_prefix="case")

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert path == tmp_path / "speed_compare.csv"
    assert [row["label"] for row in rows] == ["case_before_two_pass", "case_after_shared_low_high"]
    assert rows[0]["share_low_high"] == "False"
    assert rows[1]["share_low_high"] == "True"
    assert rows[1]["fps_speedup_vs_before"] == "2.000000"
    assert rows[1]["total_time_delta_s_vs_before"] == "-10.000000"
    assert rows[1]["detector_call_delta_vs_before"] == "-10"


def test_run_msdc_share_low_high_comparison_toggles_config_and_writes_rows(tmp_path, monkeypatch):
    from target_module.image_detect_module.config import Config

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    calls = []

    def fake_run_tracker_benchmark(**kwargs):
        share = bool(Config.MSDC_EXPORT_SHARE_LOW_HIGH_DET)
        calls.append(share)
        return {
            "run_id": kwargs["run_id"],
            "commit_hash": kwargs["commit_hash"],
            "video_path": str(kwargs["input_video"]),
            "seq_name": kwargs["seq_name"],
            "tracker": kwargs["tracker_type"],
            "method": "fake",
            "resolution": "0x0",
            "requested_frames": kwargs["requested_frames"],
            "processed_frames": kwargs["requested_frames"],
            "total_time_s": "10.000000" if share else "20.000000",
            "mean_fps": "1.000000" if share else "0.500000",
            "mean_latency_ms": "1000.000000" if share else "2000.000000",
            "p50_latency_ms": "0.000000",
            "p95_latency_ms": "0.000000",
            "peak_memory_mb": "N/A",
            "detector_calls_total": 10 if share else 20,
            "detector_calls_high_det": 0 if share else 10,
            "detector_calls_low_det": 10,
            "detector_calls_tracker_update": 0,
            "detector_calls_roi_redetect": 0,
            "mean_read_ms": "0.000000",
            "mean_high_det_ms": "0.000000" if share else "900.000000",
            "mean_low_det_ms": "700.000000",
            "mean_roi_redetect_ms": "0.000000",
            "mean_tracker_ms": "300.000000",
            "mean_render_ms": "0.000000",
            "mean_write_ms": "0.000000",
        }

    monkeypatch.setattr(benchmark, "_resolve_benchmark_input", lambda args: (video, "seq", "visible"))
    monkeypatch.setattr(benchmark, "_run_tracker_benchmark", fake_run_tracker_benchmark)
    monkeypatch.setattr(Config, "MSDC_EXPORT_SHARE_LOW_HIGH_DET", True, raising=False)

    args = argparse.Namespace(
        output_root=str(tmp_path / "out"),
        run_id="compare",
        commit_hash="abc123",
        frames=5,
        trackers=["msdc_elt"],
        progress_interval=0,
        compare_label_prefix="sample",
    )
    results_path, timings_path, compare_path = benchmark.run_msdc_share_low_high_comparison(args)

    assert calls == [False, True]
    assert Config.MSDC_EXPORT_SHARE_LOW_HIGH_DET is True
    assert timings_path.is_file()
    with results_path.open("r", encoding="utf-8-sig", newline="") as fh:
        result_rows = list(csv.DictReader(fh))
    with compare_path.open("r", encoding="utf-8-sig", newline="") as fh:
        compare_rows = list(csv.DictReader(fh))
    assert [row["tracker"] for row in result_rows] == ["msdc_elt_before_two_pass", "msdc_elt_after_shared_low_high"]
    assert compare_rows[1]["fps_speedup_vs_before"] == "2.000000"
