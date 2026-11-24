"""Synthetic acceptance contract for the frozen paper experiment matrix."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from target_module.image_detect_module.config import MSDC_V3_CONFIG
from target_module.image_detect_module.utils.evidence_state import EvidenceStateUpdater, assign_state
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, Observation, TrackState
from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS


def config(**overrides):
    return SimpleNamespace(**{**MSDC_V3_CONFIG, **overrides})


def observation(frame, score=0.9, source="high_det", x=0):
    return Observation([x, 0, x + 20, 20], source, score, 1.0, "visible", frame, 0, "USV")


def track(gid=1, state=TrackState.CANDIDATE, public_id=None, x=0, **overrides):
    values = dict(
        gid=gid, state=state, public_id=public_id, box=[x, 0, x + 20, 20],
        evidence_score=3.0, hits=4, real_det_hits=4, age=4, last_seen=3,
        last_real_det_frame=3, source_history=["high_det"], class_id=0, class_name="USV",
    )
    values.update(overrides)
    return EvidenceTrack(**values)


def decoded(value, expected):
    if isinstance(expected, bool):
        return str(value).lower() in {"1", "true", "yes", "on"}
    return type(expected)(value)


EXPECTED_DELTAS = {
    "paper_no_a": {"MSDC_ASSOC_USE_CENTER_GATE": False},
    "paper_count_e": {"MSDC_CANDIDATE_CONFIRM_MODE": "count"},
    "paper_no_r": {"MSDC_USE_REACQUIRE": False, "MSDC_REMOVED_RECOVERY_ENABLE": False},
    "paper_count_e_no_r": {"MSDC_CANDIDATE_CONFIRM_MODE": "count", "MSDC_USE_REACQUIRE": False, "MSDC_REMOVED_RECOVERY_ENABLE": False},
    "paper_full_s0": {"MSDC_LOST_SPAWN_SUPPRESS_ENABLE": False},
    "paper_no_r_s0": {"MSDC_USE_REACQUIRE": False, "MSDC_REMOVED_RECOVERY_ENABLE": False, "MSDC_LOST_SPAWN_SUPPRESS_ENABLE": False},
    "paper_lost_only_s0": {"MSDC_REMOVED_RECOVERY_ENABLE": False, "MSDC_LOST_SPAWN_SUPPRESS_ENABLE": False},
    "paper_removed_only_s0": {"MSDC_USE_REACQUIRE": False, "MSDC_LOST_SPAWN_SUPPRESS_ENABLE": False},
    "paper_assoc_d40": {"MSDC_ASSOC_CENTER_DIST": 40.0},
    "paper_assoc_d60": {"MSDC_ASSOC_CENTER_DIST": 60.0},
    "paper_assoc_d100": {"MSDC_ASSOC_CENTER_DIST": 100.0},
    "paper_assoc_d120": {"MSDC_ASSOC_CENTER_DIST": 120.0},
    "paper_candidate_t15": {"MSDC_CANDIDATE_CONFIRM_SCORE": 1.5},
    "paper_candidate_t20": {"MSDC_CANDIDATE_CONFIRM_SCORE": 2.0},
    "paper_candidate_t30": {"MSDC_CANDIDATE_CONFIRM_SCORE": 3.0},
    "paper_candidate_t35": {"MSDC_CANDIDATE_CONFIRM_SCORE": 3.5},
}


@pytest.mark.parametrize("name,delta", EXPECTED_DELTAS.items())
def test_paper_registry_changes_only_the_prescribed_fields(name, delta):
    entry = ABLATION_VARIANTS[name]
    assert set(entry) == set(MSDC_V3_CONFIG)
    resolved = {key: decoded(entry[key], value) for key, value in MSDC_V3_CONFIG.items()}
    assert resolved == {**MSDC_V3_CONFIG, **delta}
    assert resolved["MSDC_EVIDENCE_MODE"] == "score"
    assert resolved["MSDC_LOW_CANDIDATE_ENABLE"] is False
    assert resolved["MSDC_PENDING_RECOVERY_ENABLE"] is False


def test_a_removes_distance_admission_but_keeps_ranking(monkeypatch):
    recorded = []

    def assignment(scores, valid_pairs, track_indices):
        recorded.append((scores.copy(), valid_pairs.copy()))
        return EvidenceStateUpdater._greedy_gated_assignment(scores, valid_pairs, track_indices)

    a1 = EvidenceStateUpdater(config())
    a0 = EvidenceStateUpdater(config(MSDC_ASSOC_USE_CENTER_GATE=False))
    monkeypatch.setattr(a1, "_greedy_gated_assignment", assignment)
    monkeypatch.setattr(a0, "_greedy_gated_assignment", assignment)
    tracks = [track()]
    groups = a1.merge_observations([observation(4, x=50)])
    assert a1.associate_tracks_to_observations(tracks, groups)[0] == [(0, 0)]
    assert a0.associate_tracks_to_observations(tracks, groups)[0] == []
    np.testing.assert_array_equal(recorded[0][0], recorded[1][0])
    assert recorded[0][1][0, 0] and not recorded[1][1][0, 0]


def test_count_four_low_detections_preserves_true_score_updates():
    full = EvidenceStateUpdater(config())
    count = EvidenceStateUpdater(config(MSDC_CANDIDATE_CONFIRM_MODE="count"))
    full_tracks, count_tracks, full_scores, count_scores = [], [], [], []
    for frame in range(4):
        obs = [observation(frame, score=0.4, source="low_det")]
        full_tracks, _ = full.update_tracks(full_tracks, obs, frame)
        count_tracks, _ = count.update_tracks(count_tracks, obs, frame)
        full_scores.append(full_tracks[0].evidence_score)
        count_scores.append(count_tracks[0].evidence_score)
    assert full_scores == count_scores == [0.4, 0.74, 1.029, 1.2746]
    assert full_tracks[0].state == TrackState.CANDIDATE
    assert count_tracks[0].state == TrackState.ACTIVE
    assert count_tracks[0].hits == count_tracks[0].real_det_hits == 4
    assert "high_det" not in count_tracks[0].source_history


@pytest.mark.parametrize("mode", ["evidence", "count"])
@pytest.mark.parametrize("hits,real_hits,matched", [(3, 4, True), (4, 3, True), (4, 4, False)])
def test_confirmation_requires_both_four_hits_and_current_detection(mode, hits, real_hits, matched):
    updater = EvidenceStateUpdater(config(MSDC_CANDIDATE_CONFIRM_MODE=mode))
    candidate = track(hits=hits, real_det_hits=real_hits)
    updater._transition_track(candidate, 3, matched, 1.0, [candidate])
    assert candidate.state != TrackState.ACTIVE


def test_candidate_theta_is_local_and_strict():
    updater = EvidenceStateUpdater(config(MSDC_CANDIDATE_CONFIRM_SCORE=1.5))
    candidate = track(evidence_score=2.0)
    updater._transition_track(candidate, 3, True, 1.0, [candidate])
    assert candidate.state == TrackState.ACTIVE
    assert assign_state(2.0, 1.0, 0.0, 0, updater.config) == TrackState.CANDIDATE
    full_candidate = track(evidence_score=2.5)
    full = EvidenceStateUpdater(config())
    full._transition_track(full_candidate, 3, True, 1.0, [full_candidate])
    assert full_candidate.state == TrackState.CANDIDATE


def test_count_cancels_candidate_score_pruning_only():
    updaters = [EvidenceStateUpdater(config()), EvidenceStateUpdater(config(MSDC_CANDIDATE_CONFIRM_MODE="count"))]
    results = []
    for updater in updaters:
        tracks, _ = updater.update_tracks([], [observation(0, score=0.5)], 0)
        tracks, _ = updater.update_tracks(tracks, [], 1)
        results.append(tracks[0])
    assert results[0].evidence_score == results[1].evidence_score == 0.0525
    assert results[0].state == TrackState.REMOVED
    assert results[1].state == TrackState.CANDIDATE


def test_count_keeps_age_limit_and_confirmation_before_age_pruning():
    updater = EvidenceStateUpdater(config(MSDC_CANDIDATE_CONFIRM_MODE="count"))
    ready = track(age=6, evidence_score=0.0, source_history=["low_det"])
    updater._transition_track(ready, 3, True, 0.4, [ready])
    assert ready.state == TrackState.ACTIVE
    not_ready = track(gid=2, age=6, hits=3, real_det_hits=3)
    updater._transition_track(not_ready, 3, True, 0.4, [not_ready])
    assert not_ready.state == TrackState.REMOVED


def test_s_is_independent_of_lost_recovery_and_active_suppression():
    lost = track(state=TrackState.LOST, public_id=7)
    full_s = EvidenceStateUpdater(config(MSDC_USE_REACQUIRE=False))
    no_s = EvidenceStateUpdater(config(MSDC_USE_REACQUIRE=False, MSDC_LOST_SPAWN_SUPPRESS_ENABLE=False))
    groups = full_s.merge_observations([observation(4)])
    assert full_s._filter_spawn_groups_near_lost([lost], groups, [0]) == []
    assert no_s._filter_spawn_groups_near_lost([lost], groups, [0]) == [0]
    active = track(state=TrackState.ACTIVE, public_id=7)
    assert full_s._filter_spawn_groups_near_active([active], groups, [0]) == []
    assert no_s._filter_spawn_groups_near_active([active], groups, [0]) == []


@pytest.mark.parametrize("lost_enabled,removed_enabled", [(True, True), (False, False), (True, False), (False, True)])
def test_four_s0_recovery_branch_combinations(lost_enabled, removed_enabled):
    updater = EvidenceStateUpdater(config(MSDC_USE_REACQUIRE=lost_enabled, MSDC_REMOVED_RECOVERY_ENABLE=removed_enabled, MSDC_LOST_SPAWN_SUPPRESS_ENABLE=False))
    lost = track(state=TrackState.LOST, public_id=7, misses=3)
    removed = track(gid=2, state=TrackState.REMOVED, public_id=8, x=500, misses=41)
    removed.retired_signature = updater._make_removed_signature(removed, 3)
    result, events = updater.update_tracks([lost, removed], [observation(4), observation(4, x=500)], 4)
    assert any(event.event_type == "LOST_REACQUIRED" for event in events) is lost_enabled
    assert any(event.event_type == "REMOVED_ID_RECOVERY_CANDIDATE" for event in events) is removed_enabled
    if not removed_enabled:
        born = [item for item in result if item.gid not in {1, 2} and item.box[0] == 500]
        assert len(born) == 1 and born[0].public_id is None
        assert removed.state == TrackState.REMOVED


def test_trace_all_observations_actions_and_states_without_debug_truncation():
    updater = EvidenceStateUpdater(config(MSDC_MAX_CANDIDATES=256))
    updater.paper_trace_enabled = True
    result, events = updater.update_tracks([], [observation(0, x=500 * idx) for idx in range(140)], 0)
    trace = updater.last_paper_trace
    assert len(trace["observations"]) == len(trace["states"]) == len(result) == 140
    assert sum(event["event_type"] == "NEW_CANDIDATE" for event in trace["events"]) == len(events) == 140
    assert updater.paper_trace_s > 0
    assert all(row["last_real_det_frame"] == 0 for row in trace["states"])
    assert all(row["model_class_id"] == 0 for row in trace["observations"])
    updater.update_tracks(result, [], 1)
    assert updater.last_paper_trace["observations"] == []
    assert len(updater.last_paper_trace["states"]) == 140


def test_trace_records_capacity_deletions_before_removal():
    updater = EvidenceStateUpdater(config(MSDC_MAX_ACTIVE_TRACKS=1, MSDC_MAX_LOST_TRACKS=1, MSDC_MAX_TOTAL_TRACKS=1))
    updater.paper_trace_enabled = True
    tracks = [track(gid=idx + 1, state=TrackState.ACTIVE, public_id=idx + 1, x=500 * idx, evidence_score=3.0 + idx) for idx in range(3)]
    result, _ = updater.update_tracks(tracks, [], 4)
    states = updater.last_paper_trace["states"]
    assert len(result) == 1 and len(states) == 3
    deleted = [row for row in states if not row["retained"]]
    assert len(deleted) == 2
    assert all(row["deletion_reason"] == "total_pool_capacity" for row in deleted)
    assert len([row for row in updater.last_paper_trace["events"] if row["event_type"] == "CAPACITY_DELETED"]) == 2


def test_trace_selected_associations_confirmation_and_reused_identity():
    updater = EvidenceStateUpdater(config(MSDC_CANDIDATE_CONFIRM_MODE="count"))
    updater.paper_trace_enabled = True
    candidate = track(hits=3, real_det_hits=3, evidence_score=0.7, source_history=["low_det"], last_seen=2, last_real_det_frame=2)
    result, _ = updater.update_tracks([candidate], [observation(3, score=0.4, source="low_det")], 3)
    association = next(row for row in updater.last_paper_trace["associations"] if row["phase"] == "ordinary" and row["outcome"] == "matched")
    assert association["group_id"] == 0 and association["state_before"] == "candidate"
    assert association["IoU"] == 1.0 and association["distance"] == 0.0
    confirm = next(row for row in updater.last_paper_trace["events"] if row["event_type"] == "CONFIRM_ACTIVE")
    assert confirm["id_origin"] == "new" and confirm["old_public_id"] is None
    assert confirm["new_public_id"] == result[0].public_id
    assert confirm["group_id"] == 0 and confirm["box_xyxy"] == [0.0, 0.0, 20.0, 20.0]
    assert confirm["model_class_id"] == 0
    removed = track(gid=2, state=TrackState.REMOVED, public_id=12, x=500)
    removed.retired_signature = updater._make_removed_signature(removed, 3)
    updater.update_tracks([removed], [observation(4, x=500)], 4)
    recovery = next(row for row in updater.last_paper_trace["events"] if row["event_type"] == "REMOVED_ID_RECOVERY_CANDIDATE")
    assert recovery["branch"] == "removed" and recovery["old_public_id"] == recovery["new_public_id"] == 12
    assert recovery["accepted"] is True and recovery["group_id"] == 0


def test_trace_does_not_change_lifecycle_results():
    plain, traced = EvidenceStateUpdater(config()), EvidenceStateUpdater(config())
    traced.paper_trace_enabled = True
    a, b = [], []
    for frame in range(7):
        observations = [observation(frame)] if frame < 4 else []
        a, events_a = plain.update_tracks(a, deepcopy(observations), frame)
        b, events_b = traced.update_tracks(b, deepcopy(observations), frame)
        assert [item.to_dict() for item in a] == [item.to_dict() for item in b]
        assert [item.to_dict() for item in events_a] == [item.to_dict() for item in events_b]
