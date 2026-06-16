import sys
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import video_main
from target_module.image_detect_module.config import Config


class DummyTracker:
    def __init__(self):
        self.calls = []

    def update(self, boxes, frame_shape, frame=None):
        self.calls.append((boxes, frame_shape, frame is not None))
        return [{"track_id": 10, "x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.9, "class": "UAV"}]


class DummyProcessor:
    def __init__(self):
        self.calls = []

    def process_frame(self, frame, file_type, conf_override=None):
        self.calls.append((frame.shape, file_type, conf_override))
        return {"boxes": [{"x": 5, "y": 6, "w": 7, "h": 8, "confidence": 0.2, "class": "UAV"}]}


class DummyDetector:
    def __init__(self):
        self.processor = DummyProcessor()


class DummyLifecycleTracker:
    def __init__(self):
        self.calls = []

    def update(self, frame, frame_idx, file_type, high_boxes=None, low_boxes=None):
        self.calls.append((frame.shape, frame_idx, file_type, high_boxes, low_boxes))
        return [{"track_id": 1, "x": 5, "y": 6, "w": 7, "h": 8, "confidence": 2.5, "class": "UAV", "class_confidence": 2.5, "lifecycle_state": "active"}]


def _frame():
    return np.zeros((16, 16, 3), dtype=np.uint8)


def test_parse_args_accepts_msdc_elt(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["video_main.py", "--tracker", "msdc_elt"])

    args = video_main.parse_args()

    assert args.tracker == "msdc_elt"


def test_baseline_tracking_does_not_call_low_threshold_detection():
    detector = DummyDetector()
    tracker = DummyTracker()
    lifecycle = DummyLifecycleTracker()
    result = {"data": {"boxes": [{"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.9, "class": "UAV"}]}}

    tracked_boxes = video_main._update_tracking_for_frame(
        tracker_type="botsort",
        tracker=tracker,
        lifecycle_tracker=lifecycle,
        detector=detector,
        frame=_frame(),
        frame_idx=3,
        file_type="visible",
        result=result,
    )

    assert tracked_boxes == result["data"]["boxes"]
    assert tracked_boxes[0]["track_id"] == 10
    assert tracker.calls
    assert detector.processor.calls == []
    assert lifecycle.calls == []
    assert result["data"]["boxes"][0]["track_id"] == 10


def test_msdc_tracking_calls_low_threshold_and_lifecycle_tracker():
    detector = DummyDetector()
    tracker = DummyTracker()
    lifecycle = DummyLifecycleTracker()
    result = {"data": {"boxes": [{"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.9, "class": "UAV"}]}}

    tracked_boxes = video_main._update_tracking_for_frame(
        tracker_type="msdc_elt",
        tracker=tracker,
        lifecycle_tracker=lifecycle,
        detector=detector,
        frame=_frame(),
        frame_idx=4,
        file_type="visible",
        result=result,
    )

    assert tracker.calls == []
    assert detector.processor.calls == [(_frame().shape, "visible", Config.MSDC_LOW_CONF_VISIBLE)]
    assert lifecycle.calls[0][1:] == (
        4,
        "visible",
        [{"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.9, "class": "UAV"}],
        [{"x": 5, "y": 6, "w": 7, "h": 8, "confidence": 0.2, "class": "UAV"}],
    )
    assert tracked_boxes == result["data"]["boxes"]
    assert result["data"]["boxes"][0]["lifecycle_state"] == "active"


def test_msdc_tracking_uses_supplied_high_low_boxes_without_detector_call():
    detector = DummyDetector()
    tracker = DummyTracker()
    lifecycle = DummyLifecycleTracker()
    result = {"data": {"boxes": []}}
    high_boxes = [{"x": 11, "y": 12, "w": 13, "h": 14, "confidence": 0.8, "class": "UAV"}]
    low_boxes = [{"x": 21, "y": 22, "w": 23, "h": 24, "confidence": 0.2, "class": "UAV"}]

    tracked_boxes = video_main._update_tracking_for_frame(
        tracker_type="msdc_elt",
        tracker=tracker,
        lifecycle_tracker=lifecycle,
        detector=detector,
        frame=_frame(),
        frame_idx=5,
        file_type="visible",
        result=result,
        high_boxes=high_boxes,
        low_boxes=low_boxes,
    )

    assert tracker.calls == []
    assert detector.processor.calls == []
    assert lifecycle.calls[0][1:] == (5, "visible", high_boxes, low_boxes)
    assert tracked_boxes == result["data"]["boxes"]


def test_msdc_output_path_defaults_to_independent_outputs_dir(tmp_path):
    run_dir, output_path, debug_dir = video_main._resolve_msdc_paths(
        input_path=str(tmp_path / "sample.mp4"),
        output_arg="",
    )

    assert Path(run_dir).name.startswith("sample_")
    assert Path(output_path).name == "annotated.mp4"
    assert Path(output_path).parent == Path(run_dir)
    assert Path(debug_dir) == Path(run_dir)
    assert "outputs/msdc_elt" in str(Path(run_dir).as_posix())
