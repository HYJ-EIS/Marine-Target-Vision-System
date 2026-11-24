"""Exercise the installed ByteTrack implementation through the project adapter."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from target_module.image_detect_module.utils.tracking_update import update_tracking_for_frame


FRAME = np.zeros((200, 300, 3), dtype=np.uint8)


def step(tracker, score=None, file_type="visible"):
    boxes = [] if score is None else [
        {"x": 30, "y": 40, "w": 50, "h": 50, "confidence": score, "class": "UAV"}
    ]
    return update_tracking_for_frame(
        "bytetrack", tracker, None, None, FRAME, 0, file_type, boxes
    )


@pytest.mark.parametrize("file_type,score", [("visible", 0.55), ("infrared", 0.70)])
@pytest.mark.parametrize("leading_empty", [0, 2])
def test_birth_threshold_and_four_detection_confirmation(file_type, score, leading_empty):
    tracker = MultiObjectTracker(tracker_type="bytetrack")
    for _ in range(leading_empty):
        assert step(tracker, file_type=file_type) == []
    for _ in range(3):
        assert step(tracker, score, file_type) == []
    result = step(tracker, score, file_type)
    assert len(result) == 1
    assert result[0]["track_id"] > 0
    # A low-score observation maintains an existing track but cannot create one.
    assert step(tracker, 0.4, file_type)[0]["track_id"] == result[0]["track_id"]


def test_infrared_rejects_visible_only_birth_and_reset_keeps_profile():
    tracker = MultiObjectTracker(tracker_type="bytetrack", frame_rate=25)
    for _ in range(6):
        assert step(tracker, 0.55, "infrared") == []
    tracker.reset()
    assert tracker._bt.track_activation_threshold == Config.INFRARED_CONF_THRESH
    assert tracker._bt.det_thresh == Config.INFRARED_CONF_THRESH
    assert tracker._bt.minimum_matching_threshold == pytest.approx(0.8)
    assert tracker._bt.max_time_lost == 40
    for _ in range(3):
        assert step(tracker, 0.7, "infrared") == []
    assert len(step(tracker, 0.7, "infrared")) == 1


@pytest.mark.parametrize("fps", [1, 25, 29.97, 60])
def test_lost_retention_is_forty_actual_frames(fps):
    tracker = MultiObjectTracker(tracker_type="bytetrack", frame_rate=fps)
    for _ in range(4):
        result = step(tracker, 0.9)
    track_id = result[0]["track_id"]
    for _ in range(39):
        assert step(tracker) == []
    assert step(tracker, 0.9)[0]["track_id"] == track_id
    for _ in range(41):
        step(tracker)
    assert any(t.external_track_id == track_id for t in tracker._bt.removed_tracks)


def test_explicit_image_min_hits_retains_immediate_first_frame_output():
    tracker = MultiObjectTracker(tracker_type="bytetrack", min_hits=1)
    assert len(step(tracker, 0.55)) == 1
