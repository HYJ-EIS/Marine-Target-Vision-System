import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.evidence_state import EvidenceStateUpdater
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, Observation, TrackState


def _obs(frame_idx, source="low_det", score=1.0, box=None, class_name="UAV"):
    return Observation(
        box=box or [10, 10, 20, 20],
        source=source,
        score=score,
        reliability=1.0,
        modality="visible",
        frame_idx=frame_idx,
        class_id=2,
        class_name=class_name,
    )


class FastLifecycleConfig(Config):
    MSDC_ACTIVE_MISSING_PATIENCE = 1
    MSDC_LOST_MAX_AGE = 2
    MSDC_REACQUIRE_INTERVAL = 5
    MSDC_REACQUIRE_SCORE = 1.0


class ReacquireLifecycleConfig(Config):
    MSDC_LOST_MAX_AGE = 10
    MSDC_REACQUIRE_INTERVAL = 5
    MSDC_REACQUIRE_SCORE = 1.5


class ROIReacquireLifecycleConfig(ReacquireLifecycleConfig):
    MSDC_ROI_REACQUIRE_CENTER_DIST = 600.0
    MSDC_REACQUIRE_SCORE = 1.0


class GuardLifecycleConfig(Config):
    MSDC_REUSE_GUARD_ENABLE = True
    MSDC_REMOVED_GUARD_FRAMES = 120
    MSDC_removed_GUARD_FRAMES = 120
    MSDC_REMOVED_GUARD_CENTER_DIST = 50.0
    MSDC_REMOVED_GUARD_IOU_THRESH = 0.1


class GuardDisabledConfig(GuardLifecycleConfig):
    MSDC_REUSE_GUARD_ENABLE = False


class SpawnSuppressConfig(Config):
    MSDC_ASSOC_IOU_THRESH = 0.8
    MSDC_ASSOC_CENTER_DIST = 10.0
    MSDC_SPAWN_SUPPRESS_ENABLE = True
    MSDC_SPAWN_SUPPRESS_IOU = 0.1
    MSDC_SPAWN_SUPPRESS_CENTER_DIST = 80.0


class LowSpawnSuppressConfig(SpawnSuppressConfig):
    MSDC_LOW_SPAWN_SUPPRESS_CENTER_DIST = 120.0


class LowCandidateConfig(Config):
    MSDC_LOW_CANDIDATE_ENABLE = True
    MSDC_CONFIRM_REQUIRE_HIGH_DET = True
    MSDC_LOW_SPAWN_MIN_CONF = 0.30
    MSDC_LOW_CONFIRM_MIN_HITS = 5
    MSDC_LOW_CONFIRM_WINDOW = 8
    MSDC_LOW_CONFIRM_MIN_AVG_SCORE = 0.22
    MSDC_LOW_CONFIRM_MAX_MISSES = 1
    MSDC_LOW_CONFIRM_MAX_AREA_CHANGE = 1.8


class LowCandidateInheritConfig(LowCandidateConfig):
    MSDC_LOW_INHERIT_ENABLE = True
    MSDC_LOW_INHERIT_SCORE = 0.5
    MSDC_LOW_INHERIT_IOU_THRESH = 0.0
    MSDC_LOW_INHERIT_CENTER_DIST = 80.0
    MSDC_LOW_INHERIT_MAX_LOST_AGE = 60
    MSDC_LOW_INHERIT_CLASS_MATCH = True


class LowCandidateInheritHistoryVelocityConfig(LowCandidateInheritConfig):
    MSDC_LOW_INHERIT_CENTER_DIST = 25.0
    MSDC_LOW_INHERIT_MAX_LOST_AGE = 40


class LowCandidateInheritClassDriftConfig(LowCandidateInheritConfig):
    MSDC_LOW_INHERIT_SCORE = 0.4
    MSDC_LOW_INHERIT_CENTER_DIST = 60.0
    MSDC_LOW_INHERIT_CLASS_MATCH = True


class SupportedMissingConfig(Config):
    MSDC_ACTIVE_MISSING_PATIENCE = 1
    MSDC_ACTIVE_SUPPORTED_MISSING_PATIENCE = 4
    MSDC_ACTIVE_SUPPORT_RECENT_REAL_WINDOW = 8
    MSDC_ACTIVE_SUPPORT_MIN_AUX_SCORE = 0.2


def _low_history(start_frame, boxes, score=0.35):
    history = []
    for offset, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        history.append({
            "frame_idx": start_frame + offset,
            "score": score,
            "box": [float(v) for v in box],
            "area": float(max(0, x2 - x1) * max(0, y2 - y1)),
            "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
        })
    return history


def test_high_observation_accumulates_into_active_track():
    updater = EvidenceStateUpdater(Config)
    tracks = []

    tracks, events = updater.update_tracks(tracks, [_obs(0, source="high_det")], frame_idx=0)
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.CANDIDATE
    assert tracks[0].hits == 1
    assert [event.event_type for event in events] == ["NEW_CANDIDATE"]

    tracks, events = updater.update_tracks(tracks, [_obs(1, source="high_det")], frame_idx=1)
    assert tracks[0].state == TrackState.CANDIDATE
    assert tracks[0].hits == 2
    assert events == []

    tracks, events = updater.update_tracks(tracks, [_obs(2, source="high_det")], frame_idx=2)
    assert tracks[0].state == TrackState.CANDIDATE
    assert events == []

    tracks, events = updater.update_tracks(tracks, [_obs(3, source="high_det")], frame_idx=3)
    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].public_id == 1
    assert tracks[0].hits == Config.MSDC_CONFIRM_MIN_HITS
    assert tracks[0].real_det_hits == Config.MSDC_CONFIRM_MIN_REAL_DET_HITS
    assert [event.event_type for event in events] == ["CONFIRM_ACTIVE"]

    active_boxes = updater.active_tracks_to_project_boxes(tracks)
    assert active_boxes == [{
        "track_id": tracks[0].public_id,
        "x": 10,
        "y": 10,
        "w": 10,
        "h": 10,
        "confidence": round(tracks[0].evidence_score, 4),
        "class": "UAV",
        "class_confidence": round(tracks[0].evidence_score, 4),
        "gid": tracks[0].gid,
        "public_id": tracks[0].public_id,
        "lifecycle_state": "active",
    }]


def test_motion_only_candidate_does_not_confirm_active():
    updater = EvidenceStateUpdater(Config)
    tracks = []

    for frame_idx in range(3):
        tracks, events = updater.update_tracks(
            tracks,
            [_obs(frame_idx, source="motion", score=1.0, box=[10, 10, 20, 20], class_name="unknown")],
            frame_idx=frame_idx,
        )

    assert tracks == []
    assert "CONFIRM_ACTIVE" not in [event.event_type for event in events]


def test_low_confidence_observation_does_not_spawn_candidate():
    updater = EvidenceStateUpdater(Config)

    tracks, events = updater.update_tracks(
        [],
        [_obs(0, source="low_det", score=0.25, box=[10, 10, 20, 20])],
        frame_idx=0,
    )

    assert tracks == []
    assert events == []


def test_low_only_observation_spawns_hidden_low_candidate_not_public_candidate():
    updater = EvidenceStateUpdater(LowCandidateConfig)

    tracks, events = updater.update_tracks(
        [],
        [_obs(0, source="low_det", score=0.35, box=[10, 10, 20, 20])],
        frame_idx=0,
    )

    assert len(tracks) == 1
    assert tracks[0].state == TrackState.LOW_CANDIDATE
    assert tracks[0].public_id is None
    assert [event.event_type for event in events] == ["NEW_LOW_CANDIDATE"]


def test_stable_low_candidate_confirms_active_after_temporal_gate():
    updater = EvidenceStateUpdater(LowCandidateConfig)
    tracks = []
    events = []

    for frame_idx in range(5):
        tracks, events = updater.update_tracks(
            tracks,
            [_obs(frame_idx, source="low_det", score=0.35, box=[10 + frame_idx, 10, 20 + frame_idx, 20])],
            frame_idx=frame_idx,
        )

    assert len(tracks) == 1
    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].public_id == 1
    assert tracks[0].hits == 5
    assert tracks[0].real_det_hits == 5
    assert [event.event_type for event in events] == ["CONFIRM_LOW_ACTIVE"]


def test_stable_low_candidate_inherits_nearby_lost_public_id_before_new_id():
    updater = EvidenceStateUpdater(LowCandidateInheritConfig)
    lost = EvidenceTrack(
        gid=3,
        public_id=3,
        state=TrackState.LOST,
        box=[100, 100, 120, 120],
        velocity=[1.0, 0.0],
        evidence_score=1.1,
        hits=6,
        misses=3,
        age=18,
        last_seen=10,
        last_real_det_frame=10,
        real_det_hits=6,
        class_id=2,
        class_name="UAV",
    )
    low_candidate = EvidenceTrack(
        gid=9,
        public_id=None,
        state=TrackState.LOW_CANDIDATE,
        box=[103, 100, 123, 120],
        velocity=[1.0, 0.0],
        evidence_score=1.8,
        hits=4,
        misses=0,
        age=4,
        last_seen=14,
        last_real_det_frame=14,
        real_det_hits=4,
        low_det_history=_low_history(11, [
            [100, 100, 120, 120],
            [101, 100, 121, 120],
            [102, 100, 122, 120],
            [103, 100, 123, 120],
        ]),
        source_history=["low_det"] * 4,
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [lost, low_candidate],
        [_obs(15, source="low_det", score=0.35, box=[104, 100, 124, 120])],
        frame_idx=15,
    )

    event_types = [event.event_type for event in events]
    inherited = tracks[0]
    assert "LOW_CANDIDATE_INHERITED_LOST" in event_types
    assert "LOW_CANDIDATE_MERGED" in event_types
    assert "CONFIRM_LOW_ACTIVE" not in event_types
    assert inherited.gid == 3
    assert inherited.public_id == 3
    assert inherited.state == TrackState.ACTIVE
    assert inherited.last_seen == 15
    assert inherited.misses == 0
    assert len(tracks) == 1
    active_boxes = updater.active_tracks_to_project_boxes(tracks)
    assert [box["track_id"] for box in active_boxes] == [3]


def test_low_candidate_inheritance_uses_low_history_velocity_for_lost_prediction():
    updater = EvidenceStateUpdater(LowCandidateInheritHistoryVelocityConfig)
    lost = EvidenceTrack(
        gid=3,
        public_id=3,
        state=TrackState.LOST,
        box=[120, 10, 130, 20],
        velocity=[0.0, 0.0],
        evidence_score=1.1,
        hits=6,
        misses=5,
        age=15,
        last_seen=5,
        last_real_det_frame=5,
        real_det_hits=6,
        low_det_history=_low_history(1, [
            [100, 10, 110, 20],
            [105, 10, 115, 20],
            [110, 10, 120, 20],
            [115, 10, 125, 20],
            [120, 10, 130, 20],
        ]),
        source_history=["low_det"] * 5,
        class_id=2,
        class_name="UAV",
    )
    low_candidate = EvidenceTrack(
        gid=9,
        state=TrackState.LOW_CANDIDATE,
        box=[190, 10, 200, 20],
        velocity=[5.0, 0.0],
        evidence_score=1.8,
        hits=4,
        misses=0,
        age=4,
        last_seen=18,
        last_real_det_frame=18,
        real_det_hits=4,
        low_det_history=_low_history(15, [
            [175, 10, 185, 20],
            [180, 10, 190, 20],
            [185, 10, 195, 20],
            [190, 10, 200, 20],
        ]),
        source_history=["low_det"] * 4,
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [lost, low_candidate],
        [_obs(19, source="low_det", score=0.35, box=[195, 10, 205, 20])],
        frame_idx=19,
    )

    event_types = [event.event_type for event in events]
    assert "LOW_CANDIDATE_INHERITED_LOST" in event_types
    assert "CONFIRM_LOW_ACTIVE" not in event_types
    assert len(tracks) == 1
    assert tracks[0].gid == 3
    assert tracks[0].public_id == 3
    assert tracks[0].state == TrackState.ACTIVE


def test_low_candidate_inheritance_tolerates_detector_class_drift_when_geometry_is_close():
    updater = EvidenceStateUpdater(LowCandidateInheritClassDriftConfig)
    lost = EvidenceTrack(
        gid=3,
        public_id=3,
        state=TrackState.LOST,
        box=[100, 100, 120, 120],
        velocity=[1.0, 0.0],
        evidence_score=1.1,
        hits=6,
        misses=4,
        age=18,
        last_seen=10,
        last_real_det_frame=10,
        real_det_hits=6,
        class_id=0,
        class_name="USV",
    )
    low_candidate = EvidenceTrack(
        gid=9,
        state=TrackState.LOW_CANDIDATE,
        box=[103, 100, 123, 120],
        velocity=[1.0, 0.0],
        evidence_score=1.8,
        hits=4,
        misses=0,
        age=4,
        last_seen=14,
        last_real_det_frame=14,
        real_det_hits=4,
        low_det_history=_low_history(11, [
            [100, 100, 120, 120],
            [101, 100, 121, 120],
            [102, 100, 122, 120],
            [103, 100, 123, 120],
        ]),
        source_history=["low_det"] * 4,
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [lost, low_candidate],
        [_obs(15, source="low_det", score=0.35, box=[104, 100, 124, 120])],
        frame_idx=15,
    )

    event_types = [event.event_type for event in events]
    assert "LOW_CANDIDATE_INHERITED_LOST" in event_types
    assert len(tracks) == 1
    assert tracks[0].public_id == 3
    assert tracks[0].class_name == "UAV"


def test_unstable_low_candidate_does_not_confirm_active():
    updater = EvidenceStateUpdater(LowCandidateConfig)
    tracks = []

    boxes = [
        [10, 10, 20, 20],
        [11, 10, 21, 20],
        [12, 10, 60, 60],
        [13, 10, 23, 20],
        [14, 10, 24, 20],
    ]
    for frame_idx, box in enumerate(boxes):
        tracks, events = updater.update_tracks(
            tracks,
            [_obs(frame_idx, source="low_det", score=0.35, box=box)],
            frame_idx=frame_idx,
        )

    assert tracks[0].state == TrackState.LOW_CANDIDATE
    assert tracks[0].public_id is None
    assert "CONFIRM_LOW_ACTIVE" not in [event.event_type for event in events]


def test_template_only_match_does_not_refresh_active_track():
    updater = EvidenceStateUpdater(Config)
    track = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.ACTIVE,
        box=[10, 10, 30, 30],
        velocity=[2.0, 0.0],
        evidence_score=3.0,
        hits=4,
        misses=0,
        age=4,
        last_seen=4,
        last_real_det_frame=4,
        real_det_hits=4,
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [track],
        [_obs(5, source="template", score=0.95, box=[12, 10, 32, 30])],
        frame_idx=5,
    )

    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].box == [10, 10, 30, 30]
    assert tracks[0].last_seen == 4
    assert tracks[0].last_real_det_frame == 4
    assert tracks[0].misses == 1
    assert tracks[0].template_only_streak == 1
    assert events == []


def test_roi_low_detection_refreshes_active_as_real_detection():
    updater = EvidenceStateUpdater(Config)
    track = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.ACTIVE,
        box=[10, 10, 30, 30],
        velocity=[0.0, 0.0],
        evidence_score=3.0,
        hits=4,
        misses=0,
        age=4,
        last_seen=4,
        last_real_det_frame=4,
        real_det_hits=4,
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [track],
        [_obs(5, source="roi_low_det", score=0.8, box=[11, 10, 31, 30])],
        frame_idx=5,
    )

    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].last_seen == 5
    assert tracks[0].last_real_det_frame == 5
    assert tracks[0].real_det_hits == 5
    assert tracks[0].misses == 0
    assert "roi_low_det" in tracks[0].source_history
    assert events == []


def test_roi_low_unmatched_observation_does_not_spawn_new_candidate():
    updater = EvidenceStateUpdater(Config)

    tracks, events = updater.update_tracks(
        [],
        [_obs(0, source="roi_low_det", score=0.9, box=[10, 10, 20, 20])],
        frame_idx=0,
    )

    assert tracks == []
    assert events == []


def test_low_only_observation_near_active_uses_wider_spawn_suppression():
    updater = EvidenceStateUpdater(LowSpawnSuppressConfig)
    active = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.ACTIVE,
        box=[100, 100, 160, 160],
        velocity=[0.0, 0.0],
        evidence_score=3.0,
        hits=4,
        misses=0,
        age=4,
        last_seen=4,
        last_real_det_frame=4,
        real_det_hits=4,
        class_id=0,
        class_name="USV",
    )

    tracks, events = updater.update_tracks(
        [active],
        [_obs(5, source="low_det", score=0.35, box=[205, 100, 265, 160], class_name="USV")],
        frame_idx=5,
    )

    assert len([track for track in tracks if track.state == TrackState.LOW_CANDIDATE]) == 0
    assert [event.event_type for event in events] == []
    assert updater.last_spawn_suppression_debug["num_suppressed_spawn_groups"] == 1


def test_auxiliary_support_extends_active_missing_only_with_recent_low_history():
    updater = EvidenceStateUpdater(SupportedMissingConfig)
    track = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.ACTIVE,
        box=[10, 10, 30, 30],
        velocity=[1.0, 0.0],
        evidence_score=3.0,
        hits=4,
        misses=0,
        age=4,
        last_seen=4,
        last_real_det_frame=4,
        real_det_hits=4,
        low_det_history=_low_history(4, [[10, 10, 30, 30]], score=0.4),
        source_history=["high_det", "low_det"],
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [track],
        [_obs(5, source="motion", score=0.6, box=[11, 10, 31, 30], class_name="unknown")],
        frame_idx=5,
    )
    tracks, events = updater.update_tracks(
        tracks,
        [_obs(6, source="template", score=0.7, box=[12, 10, 32, 30])],
        frame_idx=6,
    )

    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].misses == 2
    assert "ACTIVE_TO_LOST" not in [event.event_type for event in events]
    assert tracks[0].box == [10, 10, 30, 30]


def test_auxiliary_support_without_recent_low_history_still_enters_lost():
    updater = EvidenceStateUpdater(SupportedMissingConfig)
    track = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.ACTIVE,
        box=[10, 10, 30, 30],
        velocity=[1.0, 0.0],
        evidence_score=3.0,
        hits=4,
        misses=0,
        age=4,
        last_seen=4,
        last_real_det_frame=4,
        real_det_hits=4,
        source_history=["high_det"],
        class_id=2,
        class_name="UAV",
    )

    tracks, _ = updater.update_tracks(
        [track],
        [_obs(5, source="motion", score=0.6, box=[11, 10, 31, 30], class_name="unknown")],
        frame_idx=5,
    )
    tracks, events = updater.update_tracks(
        tracks,
        [_obs(6, source="template", score=0.7, box=[12, 10, 32, 30])],
        frame_idx=6,
    )

    assert tracks[0].state == TrackState.LOST
    assert [event.event_type for event in events] == ["ACTIVE_TO_LOST"]


def test_one_frame_candidate_is_pruned_without_repeated_evidence():
    updater = EvidenceStateUpdater(Config)
    tracks, _ = updater.update_tracks([], [_obs(0, source="high_det")], frame_idx=0)

    tracks, events = updater.update_tracks(tracks, [], frame_idx=1)
    assert tracks[0].state == TrackState.CANDIDATE
    assert events == []

    tracks, events = updater.update_tracks(tracks, [], frame_idx=2)
    assert tracks[0].state == TrackState.REMOVED
    assert [event.event_type for event in events] == ["PRUNED_CANDIDATE"]


def test_unmatched_active_moves_to_lost_then_removed():
    updater = EvidenceStateUpdater(FastLifecycleConfig)
    tracks = [
        EvidenceTrack(
            gid=5,
            state=TrackState.ACTIVE,
            box=[30, 30, 50, 50],
            evidence_score=3.0,
            hits=3,
            misses=0,
            age=3,
            last_seen=0,
            class_id=2,
            class_name="UAV",
        )
    ]

    tracks, events = updater.update_tracks(tracks, [], frame_idx=1)
    assert tracks[0].state == TrackState.ACTIVE
    assert events == []

    tracks, events = updater.update_tracks(tracks, [], frame_idx=2)
    assert tracks[0].state == TrackState.LOST
    assert [event.event_type for event in events] == ["ACTIVE_TO_LOST"]

    tracks, events = updater.update_tracks(tracks, [], frame_idx=3)
    assert tracks[0].state == TrackState.REMOVED
    assert [event.event_type for event in events] == ["LOST_TO_removed"]
    assert tracks[0].retired_signature["last_box"] == [30.0, 30.0, 50.0, 50.0]


def test_lost_track_reacquires_only_on_interval_with_low_and_motion():
    updater = EvidenceStateUpdater(ReacquireLifecycleConfig)
    tracks = [
        EvidenceTrack(
            gid=7,
            state=TrackState.LOST,
            box=[100, 100, 120, 120],
            evidence_score=1.0,
            hits=3,
            misses=2,
            age=6,
            last_seen=4,
            class_id=2,
            class_name="UAV",
        )
    ]

    tracks, events = updater.update_tracks(
        tracks,
        [
            _obs(6, source="low_det", score=1.0, box=[101, 101, 121, 121]),
            _obs(6, source="motion", score=1.0, box=[101, 101, 121, 121], class_name="unknown"),
        ],
        frame_idx=6,
    )

    assert tracks[0].state == TrackState.LOST
    assert "LOST_REACQUIRED" not in [event.event_type for event in events]
    assert updater.last_reacquire_debug["attempted"] is False

    tracks, events = updater.update_tracks(
        tracks,
        [
            _obs(10, source="low_det", score=1.0, box=[102, 101, 122, 121]),
            _obs(10, source="motion", score=1.0, box=[102, 101, 122, 121], class_name="unknown"),
        ],
        frame_idx=10,
    )

    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].misses == 0
    assert tracks[0].last_seen == 10
    assert [event.event_type for event in events] == ["LOST_REACQUIRED"]
    assert updater.last_reacquire_debug["attempted"] is True
    assert updater.last_reacquire_debug["num_matches"] == 1


def test_lost_track_reacquires_from_same_gid_roi_low_detection_with_wide_gate():
    updater = EvidenceStateUpdater(ROIReacquireLifecycleConfig)
    tracks = [
        EvidenceTrack(
            gid=7,
            public_id=3,
            state=TrackState.LOST,
            box=[100, 100, 120, 120],
            velocity=[0.0, 0.0],
            evidence_score=1.0,
            hits=4,
            misses=2,
            age=8,
            last_seen=4,
            last_real_det_frame=4,
            real_det_hits=4,
            class_id=2,
            class_name="UAV",
        )
    ]
    obs = _obs(10, source="roi_low_det", score=0.9, box=[420, 100, 440, 120])
    obs.raw["roi_track_gid"] = 7

    tracks, events = updater.update_tracks(tracks, [obs], frame_idx=10)

    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].public_id == 3
    assert tracks[0].last_real_det_frame == 10
    assert [event.event_type for event in events] == ["LOST_REACQUIRED"]
    assert updater.last_reacquire_debug["num_matches"] == 1


def test_removed_guard_prevents_old_gid_reuse_and_creates_new_id():
    updater = EvidenceStateUpdater(GuardLifecycleConfig)
    removed = EvidenceTrack(
        gid=9,
        state=TrackState.REMOVED,
        box=[40, 40, 60, 60],
        velocity=[1.0, 0.0],
        evidence_score=0.0,
        hits=4,
        misses=5,
        age=12,
        last_seen=5,
        retired_signature={
            "last_box": [40.0, 40.0, 60.0, 60.0],
            "last_seen": 5,
            "removed_frame_idx": 5,
            "last_velocity": [1.0, 0.0],
        },
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [removed],
        [_obs(6, source="low_det", score=1.0, box=[41, 40, 61, 60])],
        frame_idx=6,
    )

    event_types = [event.event_type for event in events]
    assert "PREVENT_removed_ID_REUSE" in event_types
    assert "NEW_ID_CREATED" in event_types
    assert "NEW_LOW_CANDIDATE" in event_types
    assert tracks[0].gid == 9
    assert tracks[1].gid == 10
    assert tracks[1].state == TrackState.LOW_CANDIDATE
    assert updater.last_removed_guard_debug["num_vetoes"] == 1


def test_removed_guard_can_be_disabled():
    updater = EvidenceStateUpdater(GuardDisabledConfig)
    removed = EvidenceTrack(
        gid=3,
        state=TrackState.REMOVED,
        box=[10, 10, 30, 30],
        retired_signature={
            "last_box": [10.0, 10.0, 30.0, 30.0],
            "last_seen": 4,
            "removed_frame_idx": 4,
        },
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [removed],
        [_obs(5, source="low_det", score=1.0, box=[11, 11, 31, 31])],
        frame_idx=5,
    )

    assert "PREVENT_removed_ID_REUSE" not in [event.event_type for event in events]
    assert tracks[1].gid == 4
    assert updater.last_removed_guard_debug["enabled"] is False


def test_overlapping_observations_are_merged_into_one_candidate():
    updater = EvidenceStateUpdater(Config)

    tracks, events = updater.update_tracks(
        [],
        [
            _obs(0, source="high_det", score=0.8, box=[10, 10, 20, 20]),
            _obs(0, source="motion", score=0.5, box=[11, 11, 21, 21], class_name="unknown"),
        ],
        frame_idx=0,
    )

    assert len(tracks) == 1
    assert tracks[0].state == TrackState.CANDIDATE
    assert tracks[0].source_history == ["high_det", "motion"]
    assert tracks[0].evidence_score == 1.34
    assert [event.event_type for event in events] == ["NEW_CANDIDATE"]


def test_unmatched_observation_near_active_is_not_spawned_as_duplicate_candidate():
    updater = EvidenceStateUpdater(SpawnSuppressConfig)
    active = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.ACTIVE,
        box=[10, 10, 30, 30],
        evidence_score=3.0,
        hits=3,
        misses=0,
        age=3,
        last_seen=0,
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [active],
        [_obs(1, source="low_det", score=1.0, box=[50, 10, 70, 30])],
        frame_idx=1,
    )

    assert len(tracks) == 1
    assert tracks[0].gid == 1
    assert "NEW_CANDIDATE" not in [event.event_type for event in events]
    assert updater.last_spawn_suppression_debug["num_suppressed_spawn_groups"] == 1
