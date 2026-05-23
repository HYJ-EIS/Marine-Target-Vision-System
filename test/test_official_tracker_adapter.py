import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.utils.tracker import MultiObjectTracker


def _detections_at(x: int) -> list[dict]:
    return [
        {
            "x": x,
            "y": 40,
            "w": 30,
            "h": 20,
            "confidence": 0.9,
            "class": "UAV",
        }
    ]


def _assert_project_box(box: dict) -> None:
    assert {"track_id", "x", "y", "w", "h", "confidence", "class", "class_confidence"} <= set(box)
    assert box["track_id"] > 0
    assert box["w"] > 0
    assert box["h"] > 0
    assert box["class"] == "UAV"


def test_official_ocsort_keeps_project_output_contract():
    tracker = MultiObjectTracker(frame_rate=25, tracker_type="official_ocsort", min_hits=1)

    first = tracker.update(_detections_at(20), (120, 160, 3), frame=None)
    second = tracker.update(_detections_at(24), (120, 160, 3), frame=None)

    assert first
    assert second
    _assert_project_box(second[0])
    assert second[0]["track_id"] == first[0]["track_id"]


def test_official_botsort_keeps_project_output_contract_without_reid():
    tracker = MultiObjectTracker(frame_rate=25, tracker_type="official_botsort", min_hits=1)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    first = tracker.update(_detections_at(20), frame.shape, frame=frame)
    second = tracker.update(_detections_at(24), frame.shape, frame=frame)

    assert first
    assert second
    _assert_project_box(second[0])
    assert second[0]["track_id"] == first[0]["track_id"]


def test_official_adapters_accept_empty_detection_frames():
    for tracker_type in ("official_ocsort", "official_botsort"):
        tracker = MultiObjectTracker(frame_rate=25, tracker_type=tracker_type, min_hits=1)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        tracker.update(_detections_at(20), frame.shape, frame=frame)
        empty = tracker.update([], frame.shape, frame=frame)

        assert empty == []
