import json
import sys
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, TrackState


class LifecycleTestConfig(Config):
    MSDC_DEBUG_EVENTS = True
    MSDC_OUTPUT_CANDIDATES = False
    MSDC_ACTIVE_MISSING_PATIENCE = 1
    MSDC_LOST_MAX_AGE = 2
    MSDC_REACQUIRE_INTERVAL = 5
    MSDC_REACQUIRE_SCORE = 1.0
    MSDC_OUTPUT_MIN_BOX_SIZE = 8
    MSDC_LOW_OBS_TOPK = 0
    MSDC_LOW_OBS_MIN_CONF = 0.0
    MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY = False


class CandidateOutputConfig(LifecycleTestConfig):
    MSDC_OUTPUT_CANDIDATES = True


class LowCandidateLifecycleConfig(CandidateOutputConfig):
    MSDC_LOW_CANDIDATE_ENABLE = True
    MSDC_LOW_SPAWN_MIN_CONF = 0.30
    MSDC_LOW_CONFIRM_MIN_HITS = 5
    MSDC_LOW_CONFIRM_WINDOW = 8
    MSDC_LOW_CONFIRM_MIN_AVG_SCORE = 0.22


class ReacquireLifecycleConfig(LifecycleTestConfig):
    MSDC_LOST_MAX_AGE = 10
    MSDC_REACQUIRE_INTERVAL = 5
    MSDC_REACQUIRE_SCORE = 1.0


class LowObservationBudgetConfig(LifecycleTestConfig):
    MSDC_LOW_CANDIDATE_ENABLE = True
    MSDC_LOW_OBS_TOPK = 2
    MSDC_LOW_OBS_GLOBAL_TOPK = 2
    MSDC_LOW_OBS_MAX_PER_FRAME = 2
    MSDC_LOW_OBS_MIN_CONF = 0.25
    MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY = False


class LowObservationProximityConfig(LifecycleTestConfig):
    MSDC_LOW_CANDIDATE_ENABLE = True
    MSDC_LOW_OBS_TOPK = 10
    MSDC_LOW_OBS_GLOBAL_TOPK = 10
    MSDC_LOW_OBS_MAX_PER_FRAME = 10
    MSDC_LOW_OBS_MIN_CONF = 0.25
    MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY = True
    MSDC_LOW_OBS_TRACK_PROXIMITY_CENTER_DIST = 35.0
    MSDC_LOW_OBS_TRACK_PROXIMITY_IOU = 0.01
    MSDC_LOW_OBS_MOTION_GATE_CENTER_DIST = 35.0
    MSDC_LOW_OBS_MOTION_GATE_IOU = 0.01


class LowObservationNearestConfig(LifecycleTestConfig):
    MSDC_LOW_CANDIDATE_ENABLE = True
    MSDC_LOW_OBS_TOPK = 1
    MSDC_LOW_OBS_GLOBAL_TOPK = 1
    MSDC_LOW_OBS_MAX_PER_FRAME = 4
    MSDC_LOW_OBS_MIN_CONF = 0.25
    MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY = True
    MSDC_LOW_OBS_PER_TRACK_NEAREST = 1
    MSDC_LOW_OBS_MOTION_GATE_CENTER_DIST = 30.0
    MSDC_LOW_OBS_MOTION_GATE_IOU = 0.01
    MSDC_LOW_OBS_TRACK_PROXIMITY_CENTER_DIST = 30.0
    MSDC_LOW_OBS_TRACK_PROXIMITY_IOU = 0.01


class DebugDisabledConfig(LifecycleTestConfig):
    MSDC_DEBUG_EVENTS = False


def _frame():
    return np.zeros((64, 64, 3), dtype=np.uint8)


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


def test_lifecycle_tracker_limits_low_only_observations_before_tracker_update(tmp_path):
    tracker = MSDCLifecycleTracker(config=LowObservationBudgetConfig, debug_dir=tmp_path)
    low_boxes = [
        _box(confidence=0.20, x=1, y=1),
        _box(confidence=0.45, x=10, y=10),
        _box(confidence=0.35, x=22, y=10),
        _box(confidence=0.30, x=34, y=10),
    ]

    tracker.update(
        frame=_frame(),
        frame_idx=0,
        file_type="visible",
        high_boxes=[],
        low_boxes=low_boxes,
    )

    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    debug = frames[-1]["low_observation_budget_debug"]
    assert frames[-1]["num_low_only"] == 2
    assert debug["enabled"] is True
    assert debug["num_input"] == 4
    assert debug["num_after_min_conf"] == 3
    assert debug["num_output"] == 2
    assert debug["num_filtered_min_conf"] == 1
    assert debug["num_filtered_topk"] == 1
    assert [box["confidence"] for box in frames[-1]["_low_only_boxes"]] == [0.45, 0.35]


def test_lifecycle_tracker_filters_low_only_observations_far_from_tracks(tmp_path):
    tracker = MSDCLifecycleTracker(config=LowObservationProximityConfig, debug_dir=tmp_path)
    for frame_idx in range(4):
        tracker.update(
            frame=_frame(),
            frame_idx=frame_idx,
            file_type="visible",
            high_boxes=[_box(x=10, y=10)],
            low_boxes=[],
        )

    tracker.update(
        frame=_frame(),
        frame_idx=4,
        file_type="visible",
        high_boxes=[],
        low_boxes=[
            _box(confidence=0.35, x=12, y=10),
            _box(confidence=0.50, x=52, y=52),
        ],
    )

    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    debug = frames[-1]["low_observation_budget_debug"]
    assert frames[-1]["num_low_only"] == 1
    assert debug["track_proximity_enabled"] is True
    assert debug["track_proximity_track_count"] == 1
    assert debug["num_filtered_track_proximity"] == 1
    assert frames[-1]["_low_only_boxes"][0]["x"] == 12


def test_lifecycle_tracker_keeps_nearest_low_det_for_each_track_beyond_global_topk(tmp_path):
    tracker = MSDCLifecycleTracker(config=LowObservationNearestConfig, debug_dir=tmp_path)
    tracker.tracks = [
        EvidenceTrack(
            gid=1,
            public_id=1,
            state=TrackState.ACTIVE,
            box=[10, 10, 20, 20],
            velocity=[0, 0],
            evidence_score=3.0,
            hits=4,
            last_seen=3,
            last_real_det_frame=3,
            real_det_hits=4,
            class_id=2,
            class_name="UAV",
        ),
        EvidenceTrack(
            gid=2,
            public_id=2,
            state=TrackState.LOST,
            box=[40, 10, 50, 20],
            velocity=[0, 0],
            evidence_score=2.5,
            hits=4,
            last_seen=3,
            last_real_det_frame=3,
            real_det_hits=4,
            class_id=2,
            class_name="UAV",
        ),
    ]

    tracker.update(
        frame=_frame(),
        frame_idx=4,
        file_type="visible",
        high_boxes=[],
        low_boxes=[
            _box(confidence=0.90, x=11, y=10),
            _box(confidence=0.30, x=41, y=10),
            _box(confidence=0.80, x=63, y=52),
        ],
    )

    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    debug = frames[-1]["low_observation_budget_debug"]
    kept_x = [box["x"] for box in frames[-1]["_low_only_boxes"]]
    assert kept_x == [11, 41]
    assert debug["num_kept_global_topk"] == 1
    assert debug["num_kept_per_track_nearest"] == 2
    assert debug["num_filtered_motion_gate"] == 1
    assert debug["num_output"] == 2


def test_lifecycle_tracker_drops_low_dets_outside_motion_gate(tmp_path):
    tracker = MSDCLifecycleTracker(config=LowObservationNearestConfig, debug_dir=tmp_path)
    tracker.tracks = [
        EvidenceTrack(
            gid=1,
            public_id=1,
            state=TrackState.ACTIVE,
            box=[10, 10, 20, 20],
            velocity=[0, 0],
            evidence_score=3.0,
            hits=4,
            last_seen=3,
            last_real_det_frame=3,
            real_det_hits=4,
            class_id=2,
            class_name="UAV",
        )
    ]

    tracker.update(
        frame=_frame(),
        frame_idx=4,
        file_type="visible",
        high_boxes=[],
        low_boxes=[
            _box(confidence=0.99, x=52, y=52),
            _box(confidence=0.35, x=12, y=10),
        ],
    )

    frames = _jsonl(tmp_path / "msdc_tracks.jsonl")
    debug = frames[-1]["low_observation_budget_debug"]
    assert [box["x"] for box in frames[-1]["_low_only_boxes"]] == [12]
    assert debug["num_filtered_motion_gate"] == 1
    assert debug["num_output"] == 1


def test_lifecycle_tracker_skips_debug_jsonl_when_disabled(tmp_path):
    tracker = MSDCLifecycleTracker(config=DebugDisabledConfig, debug_dir=tmp_path)

    tracker.update(
        frame=_frame(),
        frame_idx=0,
        file_type="visible",
        high_boxes=[_box()],
        low_boxes=[],
    )

    assert not (tmp_path / "lifecycle_events.jsonl").exists()
    assert not (tmp_path / "msdc_tracks.jsonl").exists()
    assert tracker.last_debug_info["debug_events"] is False
    assert tracker.last_debug_info["frame_idx"] == 0


def test_lifecycle_tracker_records_internal_timing_when_debug_disabled(tmp_path):
    tracker = MSDCLifecycleTracker(config=DebugDisabledConfig, debug_dir=tmp_path)

    tracker.update(
        frame=_frame(),
        frame_idx=0,
        file_type="visible",
        high_boxes=[_box()],
        low_boxes=[],
    )

    timing = tracker.last_timing_debug
    expected_keys = {
        "low_filter_s",
        "observation_build_s",
        "evidence_update_s",
        "output_s",
        "debug_s",
        "total_update_s",
    }
    assert expected_keys <= set(timing)
    assert timing["total_update_s"] >= 0.0


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
