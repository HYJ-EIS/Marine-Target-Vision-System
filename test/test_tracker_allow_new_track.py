import numpy as np
import pytest
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.utils.tracker import MultiObjectTracker, OCSortTracker


def test_ocsort_rejects_allow_new_track_length_mismatch():
    tracker = OCSortTracker(min_hits=1)
    dets = np.array([[10, 10, 20, 20]], dtype=np.float64)
    confs = np.array([0.5], dtype=np.float64)
    cls_ids = np.array([0], dtype=int)

    with pytest.raises(ValueError):
        tracker.update(dets, confs, cls_ids, allow_new_track=np.array([], dtype=bool))


def test_ocsort_does_not_initialize_when_new_tracks_disallowed():
    tracker = OCSortTracker(min_hits=1)
    dets = np.array([[10, 10, 20, 20]], dtype=np.float64)
    confs = np.array([0.5], dtype=np.float64)
    cls_ids = np.array([0], dtype=int)

    output = tracker.update(dets, confs, cls_ids, allow_new_track=np.array([False]))

    assert output == []
    assert tracker.trackers == []


def test_multi_object_tracker_passes_allow_new_track_to_ocsort():
    tracker = MultiObjectTracker(frame_rate=25, tracker_type="ocsort", min_hits=1)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    low_track_only_detection = {
        "x": 10,
        "y": 10,
        "w": 10,
        "h": 10,
        "confidence": 0.4,
        "class": "UAV",
        "allow_new_track": False,
    }

    output = tracker.update([low_track_only_detection], frame.shape, frame=frame)

    assert output == []
    assert tracker._ocsort.trackers == []
