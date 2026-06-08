import json
import sys
from pathlib import Path

import cv2
import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, TrackState


class LifecycleTestConfig(Config):
    MSDC_MOTION_ENABLE = False
    MSDC_TEMPLATE_ENABLE = False
    MSDC_DEBUG_EVENTS = True
    MSDC_OUTPUT_CANDIDATES = False
    MSDC_ACTIVE_MISSING_PATIENCE = 1
    MSDC_LOST_MAX_AGE = 2
    MSDC_REACQUIRE_INTERVAL = 5
    MSDC_REACQUIRE_SCORE = 1.0
    MSDC_OUTPUT_MIN_BOX_SIZE = 8


class CandidateOutputConfig(LifecycleTestConfig):
    MSDC_OUTPUT_CANDIDATES = True


class LowCandidateLifecycleConfig(CandidateOutputConfig):
    MSDC_LOW_CANDIDATE_ENABLE = True
    MSDC_LOW_SPAWN_MIN_CONF = 0.30
    MSDC_LOW_CONFIRM_MIN_HITS = 5
    MSDC_LOW_CONFIRM_WINDOW = 8
    MSDC_LOW_CONFIRM_MIN_AVG_SCORE = 0.22


class TemplateLifecycleConfig(LifecycleTestConfig):
    MSDC_TEMPLATE_ENABLE = True
    MSDC_TEMPLATE_UPDATE_THRESH = 0.7
    MSDC_TEMPLATE_SEARCH_SCALE = 3.0
    MSDC_TEMPLATE_MIN_SIZE = 8


class ReacquireLifecycleConfig(LifecycleTestConfig):
    MSDC_LOST_MAX_AGE = 10
    MSDC_REACQUIRE_INTERVAL = 5
    MSDC_REACQUIRE_SCORE = 1.0


class ROILifecycleConfig(ReacquireLifecycleConfig):
    MSDC_USE_ROI_REDETECT = True
    MSDC_ROI_REDETECT_LOW_CONF = 0.12
    MSDC_ROI_REDETECT_ACTIVE_INTERVAL = 1000
    MSDC_ROI_REDETECT_LOST_INTERVAL = 5
    MSDC_ROI_REDETECT_SEARCH_SCALE = 2.0
    MSDC_ROI_REDETECT_UPSCALE = 1.0
    MSDC_ROI_REDETECT_MIN_CROP_SIZE = 16
    MSDC_ROI_REDETECT_MAX_TRACKS = 4
    MSDC_ROI_REDETECT_EXISTING_IOU = 0.5
    MSDC_ROI_REACQUIRE_CENTER_DIST = 600.0


class DummyROIProcessor:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def process_frame(self, frame, file_type, conf_override=None):
        self.calls.append((frame.shape, file_type, conf_override))
        return {"boxes": list(self.boxes)}


def _frame():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _pattern_patch(size=12):
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            patch[y, x] = ((x * 17 + y * 3) % 255, (x * 5 + y * 23) % 255, (x * 11 + y * 7) % 255)
    cv2.circle(patch, (size // 2, size // 2), 3, (255, 255, 255), -1)
    return patch


def _frame_with_patch(x=10, y=10, size=12):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[y:y + size, x:x + size] = _pattern_patch(size)
    return frame


def _box(confidence=0.9, x=10, y=10, w=10, h=10, cls="UAV"):
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "confidence": confidence,
        "class": cls,
        "class_confidence": confidence,
    }


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_lifecycle_tracker_confirms_active_and_writes_debug_jsonl(tmp_path):
    tracker = MSDCLifecycleTracker(config=LifecycleTestConfig, debug_dir=tmp_path)

    outputs = []
    for frame_idx in range(4):
        outputs = tracker.update(
            frame=_frame(),
            frame_idx=frame_idx,
            file_type="visible",
            high_boxes=[_box()],
            low_boxes=[_box(confidence=0.2)],
        )

    assert len(outputs) == 1
    assert outputs[0] == {
        "track_id": 1,
        "x": 10,
        "y": 10,
        "w": 10,
        "h": 10,
        "confidence": outputs[0]["confidence"],
        "class": "UAV",
        "class_confidence": outputs[0]["class_confidence"],
        "gid": 1,
        "public_id": 1,
        "lifecycle_state": "active",
        "real_det_age": 0,
    }
    assert outputs[0]["confidence"] >= LifecycleTestConfig.MSDC_CONFIRM_SCORE

    events = _jsonl(tmp_path / "lifecycle_events.jsonl")
    assert [event["event_type"] for event in events] == ["NEW_CANDIDATE", "CONFIRM_ACTIVE"]
    assert events[-1]["from_state"] == "candidate"
    assert events[-1]["to_state"] == "active"

    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    assert len(frames) == 4
    assert frames[-1]["frame_idx"] == 3
    assert frames[-1]["num_high"] == 1
    assert frames[-1]["num_low"] == 1
    assert frames[-1]["num_low_only"] == 0
    assert frames[-1]["num_motion"] == 0
    assert frames[-1]["active_track_count"] == 1

    pool_stats = _jsonl(tmp_path / "candidate_pool_stats.jsonl")
    assert len(pool_stats) == 4
    assert pool_stats[-1]["frame_idx"] == 3
    assert pool_stats[-1]["active_track_count"] == 1
    assert pool_stats[-1]["candidate_track_count"] == 0


def test_lifecycle_tracker_can_output_candidates_for_debug(tmp_path):
    tracker = MSDCLifecycleTracker(config=CandidateOutputConfig, debug_dir=tmp_path)

    outputs = tracker.update(
        frame=_frame(),
        frame_idx=0,
        file_type="visible",
        high_boxes=[_box(confidence=0.9)],
        low_boxes=[],
    )

    assert len(outputs) == 1
    assert outputs[0]["lifecycle_state"] == "candidate"
    assert outputs[0]["track_id"] == 1
    assert outputs[0]["public_id"] is None
    assert outputs[0]["class"] == "UAV"


def test_lifecycle_tracker_does_not_output_hidden_low_candidates_for_debug(tmp_path):
    tracker = MSDCLifecycleTracker(config=LowCandidateLifecycleConfig, debug_dir=tmp_path)

    outputs = tracker.update(
        frame=_frame(),
        frame_idx=0,
        file_type="visible",
        high_boxes=[],
        low_boxes=[_box(confidence=0.35)],
    )

    assert outputs == []
    assert tracker.tracks[0].state == TrackState.LOW_CANDIDATE
    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    assert frames[-1]["low_candidate_track_count"] == 1
    pool_stats = _jsonl(tmp_path / "candidate_pool_stats.jsonl")
    assert pool_stats[-1]["low_candidate_track_count"] == 1


def test_lifecycle_tracker_calls_motion_seed_generator(tmp_path):
    class DummyMotionSeed:
        def __init__(self):
            self.calls = []

        def update(self, frame, frame_idx=None):
            self.calls.append((frame.shape, frame_idx))
            return [{
                "x": 30,
                "y": 30,
                "w": 8,
                "h": 8,
                "score": 0.5,
                "source": "motion",
                "frame_idx": frame_idx,
            }], {"frame_idx": frame_idx, "num_motion_boxes": 1}

    tracker = MSDCLifecycleTracker(config=LifecycleTestConfig, debug_dir=tmp_path)
    dummy = DummyMotionSeed()
    tracker.motion_seed = dummy

    tracker.update(
        frame=_frame(),
        frame_idx=4,
        file_type="visible",
        high_boxes=[],
        low_boxes=[],
    )

    assert dummy.calls == [(_frame().shape, 4)]
    assert tracker.tracks == []
    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    assert frames[-1]["num_motion"] == 1
    assert frames[-1]["num_observations"] == 1


def test_empty_inputs_do_not_crash_and_active_can_enter_lost(tmp_path):
    tracker = MSDCLifecycleTracker(config=LifecycleTestConfig, debug_dir=tmp_path)
    for frame_idx in range(4):
        tracker.update(_frame(), frame_idx, "visible", high_boxes=[_box()], low_boxes=[])

    first_empty = tracker.update(_frame(), 4, "visible", high_boxes=[], low_boxes=[])
    second_empty = tracker.update(_frame(), 5, "visible", high_boxes=[], low_boxes=[])

    assert first_empty
    assert second_empty == []
    assert tracker.tracks[0].state.value == "lost"
    events = _jsonl(tmp_path / "lifecycle_events.jsonl")
    assert events[-1]["event_type"] == "ACTIVE_TO_LOST"


def test_lifecycle_tracker_reacquires_lost_track_and_writes_debug(tmp_path):
    tracker = MSDCLifecycleTracker(config=ReacquireLifecycleConfig, debug_dir=tmp_path)
    for frame_idx in range(4):
        tracker.update(_frame(), frame_idx, "visible", high_boxes=[_box()], low_boxes=[])

    tracker.update(_frame(), 4, "visible", high_boxes=[], low_boxes=[])
    tracker.update(_frame(), 5, "visible", high_boxes=[], low_boxes=[])
    assert tracker.tracks[0].state.value == "lost"

    outputs = tracker.update(
        _frame(),
        10,
        "visible",
        high_boxes=[],
        low_boxes=[_box(confidence=1.0, x=10, y=10)],
    )

    assert len(outputs) == 1
    assert outputs[0]["track_id"] == 1
    assert outputs[0]["lifecycle_state"] == "active"
    events = _jsonl(tmp_path / "lifecycle_events.jsonl")
    assert events[-1]["event_type"] == "LOST_REACQUIRED"

    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    assert frames[-1]["reacquire_debug"]["attempted"] is True
    assert frames[-1]["reacquire_debug"]["num_matches"] == 1


def test_lifecycle_tracker_uses_roi_redetect_for_lost_reacquire(tmp_path):
    processor = DummyROIProcessor([
        {"x": 6, "y": 5, "w": 10, "h": 10, "confidence": 0.9, "class": "UAV"},
    ])
    tracker = MSDCLifecycleTracker(config=ROILifecycleConfig, debug_dir=tmp_path, processor=processor)
    for frame_idx in range(4):
        tracker.update(_frame(), frame_idx, "visible", high_boxes=[_box()], low_boxes=[])

    tracker.update(_frame(), 4, "visible", high_boxes=[], low_boxes=[])
    tracker.update(_frame(), 5, "visible", high_boxes=[], low_boxes=[])
    assert tracker.tracks[0].state.value == "lost"

    outputs = tracker.update(_frame(), 10, "visible", high_boxes=[], low_boxes=[])

    assert outputs
    assert outputs[0]["track_id"] == 1
    assert outputs[0]["lifecycle_state"] == "active"
    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    assert frames[-1]["num_roi_low"] == 1
    assert frames[-1]["roi_redetect_debug"]["num_roi_boxes"] == 1
    assert frames[-1]["reacquire_debug"]["num_matches"] == 1


def test_lifecycle_tracker_writes_stage_observations_jsonl(tmp_path):
    tracker = MSDCLifecycleTracker(config=LifecycleTestConfig, debug_dir=tmp_path)

    tracker.update(
        frame=_frame(),
        frame_idx=0,
        file_type="visible",
        high_boxes=[_box(x=10, y=10)],
        low_boxes=[_box(confidence=0.2, x=40, y=40)],
    )

    rows = _jsonl(tmp_path / "stage_observations.jsonl")
    assert rows[0]["frame_idx"] == 0
    assert {box["stage"] for box in rows[0]["boxes"]} == {"high_det", "low_det", "low_only"}


def test_lifecycle_tracker_lost_timeout_goes_removed_without_output(tmp_path):
    tracker = MSDCLifecycleTracker(config=LifecycleTestConfig, debug_dir=tmp_path)
    for frame_idx in range(4):
        tracker.update(_frame(), frame_idx, "visible", high_boxes=[_box()], low_boxes=[])

    tracker.update(_frame(), 4, "visible", high_boxes=[], low_boxes=[])
    tracker.update(_frame(), 5, "visible", high_boxes=[], low_boxes=[])
    outputs = tracker.update(_frame(), 6, "visible", high_boxes=[], low_boxes=[])

    assert outputs == []
    assert tracker.tracks[0].state.value == "removed"
    events = _jsonl(tmp_path / "lifecycle_events.jsonl")
    assert events[-1]["event_type"] == "LOST_TO_removed"
    assert tracker.tracks[0].retired_signature["last_box"] == [10.0, 10.0, 20.0, 20.0]


def test_low_only_filter_keeps_non_overlapping_low_boxes(tmp_path):
    tracker = MSDCLifecycleTracker(config=LifecycleTestConfig, debug_dir=tmp_path)

    tracker.update(
        frame=_frame(),
        frame_idx=0,
        file_type="visible",
        high_boxes=[_box(x=10, y=10)],
        low_boxes=[
            _box(confidence=0.2, x=10, y=10),
            _box(confidence=0.2, x=40, y=40),
        ],
    )

    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    assert frames[-1]["num_high"] == 1
    assert frames[-1]["num_low"] == 2
    assert frames[-1]["num_low_only"] == 1
    assert len(tracker.tracks) == 1


def test_lifecycle_tracker_template_lock_is_active_only_and_writes_debug(tmp_path):
    tracker = MSDCLifecycleTracker(config=TemplateLifecycleConfig, debug_dir=tmp_path)

    # First frame creates a candidate; candidate tracks must not receive templates.
    tracker.update(
        frame=_frame_with_patch(x=10, y=10),
        frame_idx=0,
        file_type="visible",
        high_boxes=[_box(x=10, y=10, w=12, h=12)],
        low_boxes=[],
    )
    assert tracker.tracks[0].state.value == "candidate"
    assert tracker.tracks[0].template is None
    assert tracker.template_lock.template_count == 0

    # Repeated high evidence confirms active and initializes the template.
    tracker.update(
        frame=_frame_with_patch(x=10, y=10),
        frame_idx=1,
        file_type="visible",
        high_boxes=[_box(x=10, y=10, w=12, h=12)],
        low_boxes=[],
    )
    tracker.update(
        frame=_frame_with_patch(x=10, y=10),
        frame_idx=2,
        file_type="visible",
        high_boxes=[_box(x=10, y=10, w=12, h=12)],
        low_boxes=[],
    )
    tracker.update(
        frame=_frame_with_patch(x=10, y=10),
        frame_idx=3,
        file_type="visible",
        high_boxes=[_box(x=10, y=10, w=12, h=12)],
        low_boxes=[],
    )
    assert tracker.tracks[0].state.value == "active"
    assert tracker.tracks[0].template["initialized"] is True
    assert tracker.template_lock.template_count == 1

    # With no detector boxes, the active template can add auxiliary evidence.
    outputs = tracker.update(
        frame=_frame_with_patch(x=13, y=12),
        frame_idx=4,
        file_type="visible",
        high_boxes=[],
        low_boxes=[],
    )

    assert outputs
    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    assert frames[-1]["num_template"] == 1
    assert frames[-1]["template_score"] >= TemplateLifecycleConfig.MSDC_TEMPLATE_UPDATE_THRESH
    assert frames[-1]["template_updated"] is False
    assert frames[-1]["template_match_box"] is not None


def test_lifecycle_tracker_output_nms_suppresses_overlapping_active_boxes(tmp_path):
    tracker = MSDCLifecycleTracker(config=LifecycleTestConfig, debug_dir=tmp_path)
    tracker.tracks = [
        EvidenceTrack(
            gid=1,
            public_id=1,
            state=TrackState.ACTIVE,
            box=[10, 10, 30, 30],
            evidence_score=3.0,
            hits=3,
            last_seen=2,
            last_real_det_frame=2,
            real_det_hits=4,
            class_name="UAV",
        ),
        EvidenceTrack(
            gid=2,
            public_id=2,
            state=TrackState.ACTIVE,
            box=[11, 11, 31, 31],
            evidence_score=2.8,
            hits=3,
            last_seen=2,
            last_real_det_frame=2,
            real_det_hits=4,
            class_name="UAV",
        ),
    ]

    outputs = tracker._tracks_to_output_boxes(tracker.tracks)

    assert len(outputs) == 1
    assert outputs[0]["track_id"] == 1
    assert tracker.last_output_nms_debug["num_suppressed"] == 1


def test_lifecycle_tracker_does_not_output_stale_active_without_recent_real_detection(tmp_path):
    tracker = MSDCLifecycleTracker(config=LifecycleTestConfig, debug_dir=tmp_path)
    tracker.tracks = [
        EvidenceTrack(
            gid=1,
            public_id=1,
            state=TrackState.ACTIVE,
            box=[10, 10, 30, 30],
            velocity=[2.0, 0.0],
            evidence_score=3.0,
            hits=4,
            last_seen=2,
            last_real_det_frame=2,
            real_det_hits=4,
            class_name="UAV",
        )
    ]

    recent = tracker._tracks_to_output_boxes(tracker.tracks, frame_idx=4)
    stale = tracker._tracks_to_output_boxes(tracker.tracks, frame_idx=7)

    assert len(recent) == 1
    assert recent[0]["x"] == 14
    assert stale == []


def test_lifecycle_tracker_output_nms_suppresses_contained_fragments(tmp_path):
    tracker = MSDCLifecycleTracker(config=LifecycleTestConfig, debug_dir=tmp_path)
    tracker.tracks = [
        EvidenceTrack(
            gid=1,
            public_id=1,
            state=TrackState.ACTIVE,
            box=[10, 10, 110, 110],
            evidence_score=3.0,
            hits=4,
            last_seen=2,
            last_real_det_frame=2,
            real_det_hits=4,
            class_name="UAV",
        ),
        EvidenceTrack(
            gid=2,
            public_id=2,
            state=TrackState.ACTIVE,
            box=[65, 65, 85, 85],
            evidence_score=2.9,
            hits=4,
            last_seen=2,
            last_real_det_frame=2,
            real_det_hits=4,
            class_name="UAV",
        ),
    ]

    outputs = tracker._tracks_to_output_boxes(tracker.tracks, frame_idx=2)

    assert len(outputs) == 1
    assert outputs[0]["track_id"] == 1
    assert tracker.last_output_nms_debug["suppressed"][0]["fragment"] is True
