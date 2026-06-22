import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.utils import tracking_update
from target_module.image_detect_module.utils.tracking_update import (
    is_msdc_tracker,
    update_tracking_for_frame,
)


class DummyTracker:
    def __init__(self):
        self.calls = []

    def update(self, boxes, frame_shape, frame=None):
        self.calls.append((boxes, frame_shape, frame))
        return [{"track_id": 7, "x": 1, "y": 2, "w": 3, "h": 4}]


class DummyLifecycleTracker:
    def __init__(self):
        self.calls = []

    def update(self, frame, frame_idx, file_type, high_boxes=None, low_boxes=None):
        self.calls.append((frame, frame_idx, file_type, high_boxes, low_boxes))
        return [{"track_id": 3, "lifecycle_state": "active"}]


def test_is_msdc_tracker_only_matches_msdc_elt():
    assert is_msdc_tracker("msdc_elt") is True
    assert is_msdc_tracker("ocsort") is False
    assert is_msdc_tracker(None) is False


def test_runtime_helper_is_the_only_tracking_update_entrypoint():
    files_without_private_runtime_helpers = [
        _ROOT / "video_main.py",
        _ROOT / "tools" / "evaluation" / "export_mot_results.py",
    ]
    for path in files_without_private_runtime_helpers:
        text = path.read_text(encoding="utf-8")
        assert "def _is_msdc_tracker" not in text
        assert "def _update_tracking_for_frame" not in text
        assert "is_msdc_tracker as _is_msdc_tracker" not in text
        assert "update_tracking_for_frame as _update_tracking_for_frame" not in text

    for relative_path in [
        Path("tools/evaluation/msdc_speed_benchmark.py"),
    ]:
        text = (_ROOT / relative_path).read_text(encoding="utf-8")
        assert "target_module.image_detect_module.utils.tracking_update" in text
        assert "_update_tracking_for_frame" not in text
        assert "_is_msdc_tracker" not in text


def test_baseline_update_calls_tracker_with_frame_for_gmc():
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    tracker = DummyTracker()
    boxes = [{"x": 1, "y": 2, "w": 3, "h": 4}]

    tracked = update_tracking_for_frame(
        tracker_type="botsort",
        tracker=tracker,
        lifecycle_tracker=None,
        detector=object(),
        frame=frame,
        frame_idx=5,
        file_type="visible",
        boxes=boxes,
    )

    assert tracked == [{"track_id": 7, "x": 1, "y": 2, "w": 3, "h": 4}]
    assert tracker.calls == [(boxes, frame.shape, frame)]


def test_msdc_update_uses_lifecycle_tracker_and_supplied_low_boxes():
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    lifecycle_tracker = DummyLifecycleTracker()
    high_boxes = [{"confidence": 0.9}]
    low_boxes = [{"confidence": 0.2}]

    tracked = update_tracking_for_frame(
        tracker_type="msdc_elt",
        tracker=None,
        lifecycle_tracker=lifecycle_tracker,
        detector=object(),
        frame=frame,
        frame_idx=5,
        file_type="infrared",
        boxes=high_boxes,
        low_boxes=low_boxes,
    )

    assert tracked == [{"track_id": 3, "lifecycle_state": "active"}]
    assert lifecycle_tracker.calls == [(frame, 5, "infrared", high_boxes, low_boxes)]


def test_msdc_update_runs_low_threshold_detection_when_low_boxes_missing(monkeypatch):
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    lifecycle_tracker = DummyLifecycleTracker()
    detector = object()
    high_boxes = [{"confidence": 0.9}]
    low_boxes = [{"confidence": 0.2}]
    calls = []

    def fake_low_detection(seen_detector, seen_frame, seen_file_type):
        calls.append((seen_detector, seen_frame, seen_file_type))
        return low_boxes

    monkeypatch.setattr(tracking_update, "run_msdc_low_threshold_detection", fake_low_detection)

    update_tracking_for_frame(
        tracker_type="msdc_elt",
        tracker=None,
        lifecycle_tracker=lifecycle_tracker,
        detector=detector,
        frame=frame,
        frame_idx=5,
        file_type="visible",
        boxes=high_boxes,
    )

    assert calls == [(detector, frame, "visible")]
    assert lifecycle_tracker.calls == [(frame, 5, "visible", high_boxes, low_boxes)]


def test_update_raises_when_required_tracker_is_missing():
    frame = np.zeros((12, 20, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Baseline tracker"):
        update_tracking_for_frame(
            tracker_type="ocsort",
            tracker=None,
            lifecycle_tracker=None,
            detector=object(),
            frame=frame,
            frame_idx=0,
            file_type="visible",
            boxes=[],
        )

    with pytest.raises(RuntimeError, match="MS-DC-ELT"):
        update_tracking_for_frame(
            tracker_type="msdc_elt",
            tracker=None,
            lifecycle_tracker=None,
            detector=object(),
            frame=frame,
            frame_idx=0,
            file_type="visible",
            boxes=[],
        )
