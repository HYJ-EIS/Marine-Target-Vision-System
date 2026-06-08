import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation import export_mot_results
from tools.evaluation.export_mot_results import format_mot_result_line


def test_format_mot_result_line_uses_motchallenge_result_columns():
    box = {
        "track_id": 7,
        "x": 10,
        "y": 20,
        "w": 30,
        "h": 40,
        "confidence": 0.87654,
    }

    line = format_mot_result_line(frame_id=3, box=box)

    assert line == "3,7,10.00,20.00,30.00,40.00,0.8765,-1,-1,-1"


def test_format_mot_result_line_ignores_msdc_extra_fields():
    box = {
        "track_id": 5,
        "x": 1,
        "y": 2,
        "w": 3,
        "h": 4,
        "confidence": 2.34567,
        "class": "UAV",
        "class_confidence": 2.34567,
        "lifecycle_state": "active",
        "evidence_score": 2.34567,
    }

    line = format_mot_result_line(frame_id=9, box=box)

    assert line == "9,5,1.00,2.00,3.00,4.00,2.3457,-1,-1,-1"
    assert len(line.split(",")) == 10


def test_exporter_accepts_msdc_elt_tracker_choice():
    assert "msdc_elt" in export_mot_results.TRACKER_CHOICES


class DummyProcessor:
    def __init__(self):
        self.calls = []

    def process_frame(self, frame, file_type, conf_override=None):
        self.calls.append((frame.shape, file_type, conf_override))
        return {"boxes": [{"x": 4, "y": 5, "w": 6, "h": 7, "confidence": 0.2, "class": "UAV"}]}


class DummyDetector:
    def __init__(self):
        self.processor = DummyProcessor()


class DummyLifecycleTracker:
    def __init__(self):
        self.calls = []

    def update(self, frame, frame_idx, file_type, high_boxes=None, low_boxes=None):
        self.calls.append((frame.shape, frame_idx, file_type, high_boxes, low_boxes))
        return [{"track_id": 1, "x": 4, "y": 5, "w": 6, "h": 7, "confidence": 2.5, "lifecycle_state": "active"}]


def test_msdc_export_branch_uses_lifecycle_tracker_contract():
    detector = DummyDetector()
    lifecycle_tracker = DummyLifecycleTracker()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    high_boxes = [{"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.9, "class": "UAV"}]

    tracked = export_mot_results._update_tracking_for_frame(
        tracker_type="msdc_elt",
        tracker=None,
        lifecycle_tracker=lifecycle_tracker,
        detector=detector,
        frame=frame,
        frame_idx=3,
        file_type="visible",
        boxes=high_boxes,
    )

    assert tracked[0]["lifecycle_state"] == "active"
    assert detector.processor.calls
    assert lifecycle_tracker.calls[0][1:] == (
        3,
        "visible",
        high_boxes,
        [{"x": 4, "y": 5, "w": 6, "h": 7, "confidence": 0.2, "class": "UAV"}],
    )


def test_msdc_high_threshold_detection_uses_in_memory_frame():
    detector = DummyDetector()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    boxes = export_mot_results._run_msdc_high_threshold_detection(detector, frame, "visible")

    assert boxes == [{"x": 4, "y": 5, "w": 6, "h": 7, "confidence": 0.2, "class": "UAV"}]
    assert detector.processor.calls == [(frame.shape, "visible", None)]


def test_msdc_shared_low_detection_can_be_split_without_second_low_call():
    detector = DummyDetector()
    lifecycle_tracker = DummyLifecycleTracker()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    low_boxes = [
        {"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.7, "class": "UAV"},
        {"x": 4, "y": 5, "w": 6, "h": 7, "confidence": 0.2, "class": "UAV"},
    ]
    high_boxes = export_mot_results._split_msdc_high_from_low_boxes(low_boxes, "visible")

    export_mot_results._update_tracking_for_frame(
        tracker_type="msdc_elt",
        tracker=None,
        lifecycle_tracker=lifecycle_tracker,
        detector=detector,
        frame=frame,
        frame_idx=4,
        file_type="visible",
        boxes=high_boxes,
        low_boxes=low_boxes,
    )

    assert high_boxes == [{"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.7, "class": "UAV"}]
    assert detector.processor.calls == []
    assert lifecycle_tracker.calls[0][3] == high_boxes
    assert lifecycle_tracker.calls[0][4] == low_boxes
