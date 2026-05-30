import numpy as np
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.utils.tracker import MultiObjectTracker


def _det(x: int, y: int = 40, cls: str = "UAV", conf: float = 0.9) -> dict:
    return {
        "x": x,
        "y": y,
        "w": 12,
        "h": 10,
        "confidence": conf,
        "class": cls,
    }


def _assert_project_box(box: dict, cls: str = "UAV") -> None:
    assert {"track_id", "x", "y", "w", "h", "confidence", "class", "class_confidence"} <= set(box)
    assert box["track_id"] > 0
    assert box["w"] > 0
    assert box["h"] > 0
    assert box["class"] == cls


def test_dist_tracker_keeps_id_for_small_target_motion():
    tracker = MultiObjectTracker(frame_rate=25, tracker_type="dist_tracker", min_hits=1)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    first = tracker.update([_det(20)], frame.shape, frame=frame)
    second = tracker.update([_det(24)], frame.shape, frame=frame)

    assert first
    assert second
    _assert_project_box(second[0])
    assert second[0]["track_id"] == first[0]["track_id"]


def test_dist_tracker_uses_l2_iou_to_bridge_zero_overlap_motion():
    tracker = MultiObjectTracker(frame_rate=25, tracker_type="dist_tracker", min_hits=1)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    first = tracker.update([_det(20)], frame.shape, frame=frame)
    second = tracker.update([_det(36)], frame.shape, frame=frame)

    assert first
    assert second
    assert second[0]["track_id"] == first[0]["track_id"]


def test_dist_tracker_handles_empty_frames_and_resets_ids():
    tracker = MultiObjectTracker(frame_rate=25, tracker_type="dist_tracker", min_hits=1)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    first = tracker.update([_det(20)], frame.shape, frame=frame)
    empty = tracker.update([], frame.shape, frame=frame)

    assert first
    assert empty == []

    tracker.reset()
    restarted = tracker.update([_det(20)], frame.shape, frame=frame)

    assert restarted
    assert restarted[0]["track_id"] == 1


def test_dist_tracker_prefers_nearest_small_target_when_iou_is_ambiguous():
    tracker = MultiObjectTracker(frame_rate=25, tracker_type="dist_tracker", min_hits=1)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    first = tracker.update([_det(20), _det(80)], frame.shape, frame=frame)
    second = tracker.update([_det(84), _det(24)], frame.shape, frame=frame)

    assert len(first) == 2
    assert len(second) == 2
    by_x = {box["x"]: box["track_id"] for box in second}
    assert by_x[24] == first[0]["track_id"]
    assert by_x[84] == first[1]["track_id"]
