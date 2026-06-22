"""
MS-DC-ELT-lite evidence accumulation and lifecycle state updates.

This module is intentionally standalone: it does not connect to video_main.py or
the baseline OC-SORT / BoT-SORT path.  The local geometry helpers mirror the
tracker helpers so importing this evidence layer does not pull in tracker
adapters that are unrelated to isolated lifecycle tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.msdc_types import (
    EvidenceTrack,
    LifecycleEvent,
    Observation,
    TrackState,
    track_to_output_box,
)


_SOURCE_PRIORITY = {
    "high_det": 0,
    "low_det": 1,
    "reacquire": 2,
}


def _box_array(box: Any) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float64).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"Expected 4 box values, got {arr.size}")
    return arr


def _iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    m, n = len(bb_test), len(bb_gt)
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)

    xx1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0].reshape(1, -1))
    yy1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1].reshape(1, -1))
    xx2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2].reshape(1, -1))
    yy2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3].reshape(1, -1))

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = np.maximum(0.0, bb_test[:, 2] - bb_test[:, 0]) * np.maximum(0.0, bb_test[:, 3] - bb_test[:, 1])
    area_b = np.maximum(0.0, bb_gt[:, 2] - bb_gt[:, 0]) * np.maximum(0.0, bb_gt[:, 3] - bb_gt[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-10)


def _center_distance_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    m, n = len(bb_test), len(bb_gt)
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)

    cx_a = (bb_test[:, 0] + bb_test[:, 2]) / 2.0
    cy_a = (bb_test[:, 1] + bb_test[:, 3]) / 2.0
    cx_b = (bb_gt[:, 0] + bb_gt[:, 2]) / 2.0
    cy_b = (bb_gt[:, 1] + bb_gt[:, 3]) / 2.0

    dx = cx_a[:, None] - cx_b[None, :]
    dy = cy_a[:, None] - cy_b[None, :]
    return np.sqrt(dx ** 2 + dy ** 2)


def _state_value(state: TrackState | str | None) -> str | None:
    if state is None:
        return None
    if isinstance(state, TrackState):
        return state.value
    return str(state)


def _as_state(state: TrackState | str) -> TrackState:
    if isinstance(state, TrackState):
        return state
    return TrackState(str(state))


def _config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config or Config, name, default)


def assign_state(
    p_t: float,
    v_t: float,
    m_t: float,
    dt: int,
    config=Config,
) -> TrackState:
    """Return exactly one lifecycle state from priority-ordered evidence terms."""
    theta_p = float(_config_value(config, "MSDC_CONFIRM_SCORE", 2.5))
    theta_p_low = float(_config_value(config, "MSDC_PRUNE_SCORE", 0.1))
    theta_v = float(_config_value(config, "MSDC_STATE_THETA_V", 0.5))
    theta_m = float(_config_value(config, "MSDC_STATE_THETA_M", 1.0))
    tau_max = int(_config_value(config, "MSDC_LOST_MAX_AGE", 5))

    observed = float(v_t) >= theta_v
    if not observed and float(p_t) < theta_p_low and int(dt) > tau_max:
        return TrackState.REMOVED
    if not observed and float(m_t) < theta_m:
        return TrackState.LOST
    if observed and float(p_t) > theta_p and float(m_t) < theta_m:
        return TrackState.ACTIVE
    return TrackState.CANDIDATE


@dataclass
class _ObservationGroup:
    observations: list[Observation] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    source_scores: dict[str, float] = field(default_factory=dict)
    source_history: list[str] = field(default_factory=list)
    class_id: int = -1
    class_name: str = "unknown"

    @property
    def box(self) -> np.ndarray:
        if not self.boxes:
            return np.zeros(4, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if np.sum(weights) <= 1e-9:
            return self.boxes[0].copy()
        return np.average(np.vstack(self.boxes), axis=0, weights=weights)


@dataclass
class _ReacquireMatch:
    track_idx: int
    group_idx: int
    score: float
    debug: dict = field(default_factory=dict)


class EvidenceStateUpdater:
    def __init__(self, config=Config):
        self.config = config or Config
        self.next_gid = 1
        self.next_public_id = 1
        self.last_reacquire_debug = self._empty_reacquire_debug()
        self.last_low_inherit_debug = self._empty_low_inherit_debug()
        self.last_removed_guard_debug = self._empty_removed_guard_debug()
        self.last_spawn_suppression_debug = self._empty_spawn_suppression_debug()
        self._removed_guard_cache_key = None
        self._removed_guard_cache = None

    def update(
        self,
        tracks: list[EvidenceTrack],
        observations: list[Observation],
        frame_idx: int,
        frame_shape: tuple | None = None,
    ) -> tuple[list[EvidenceTrack], list[LifecycleEvent]]:
        return self.update_tracks(tracks=tracks, observations=observations, frame_idx=frame_idx)

    def update_tracks(
        self,
        tracks: list[EvidenceTrack],
        observations: list[Observation],
        frame_idx: int,
    ) -> tuple[list[EvidenceTrack], list[LifecycleEvent]]:
        """
        Update evidence tracks with current observations.

        The current runtime supports high_det, low_det, and reacquire evidence.
        """
        tracks = list(tracks or [])
        observations = list(observations or [])
        self._ensure_next_gid(tracks)
        self._ensure_next_public_id(tracks)

        self.last_reacquire_debug = self._empty_reacquire_debug(frame_idx)
        self.last_low_inherit_debug = self._empty_low_inherit_debug(frame_idx)
        self.last_removed_guard_debug = self._empty_removed_guard_debug()
        self.last_spawn_suppression_debug = self._empty_spawn_suppression_debug(frame_idx)
        self._removed_guard_cache_key = None
        self._removed_guard_cache = None

        groups = self.merge_observations(observations)
        normal_track_indices = [
            idx
            for idx, track in enumerate(tracks)
            if _as_state(track.state) in {TrackState.LOW_CANDIDATE, TrackState.CANDIDATE, TrackState.ACTIVE}
        ]
        matches, unmatched_track_indices, unmatched_group_indices = self.associate_tracks_to_observations(
            tracks,
            groups,
            track_indices=normal_track_indices,
        )
        reacquire_matches = self._reacquire_lost_track_indices(
            tracks=tracks,
            groups=groups,
            available_group_indices=unmatched_group_indices,
            frame_idx=frame_idx,
        )
        if reacquire_matches:
            reacquired_groups = {match.group_idx for match in reacquire_matches}
            unmatched_group_indices = [idx for idx in unmatched_group_indices if idx not in reacquired_groups]
            matches.extend((match.track_idx, match.group_idx) for match in reacquire_matches)

        events: list[LifecycleEvent] = []
        real_matched_track_indices = {
            track_idx
            for track_idx, group_idx in matches
            if self._group_has_real_detection(groups[group_idx])
        }
        reacquire_score_by_track = {match.track_idx: match.score for match in reacquire_matches}

        for idx, track in enumerate(tracks):
            if _as_state(track.state) == TrackState.REMOVED:
                continue
            track.age = int(track.age) + 1

        for track_idx, group_idx in matches:
            track = tracks[track_idx]
            if _as_state(track.state) == TrackState.REMOVED:
                continue
            positive_score = self._evidence_increment(groups[group_idx])
            self._apply_matched_observation(track, groups[group_idx], frame_idx, positive_score)

        lost_indices = [
            idx
            for idx, track in enumerate(tracks)
            if _as_state(track.state) == TrackState.LOST and idx not in real_matched_track_indices
        ]
        aux_only_track_indices = [
            track_idx
            for track_idx, group_idx in matches
            if not self._group_has_real_detection(groups[group_idx])
        ]
        negative_track_indices = list(dict.fromkeys(list(unmatched_track_indices) + lost_indices + aux_only_track_indices))
        for track_idx in negative_track_indices:
            track = tracks[track_idx]
            if _as_state(track.state) == TrackState.REMOVED:
                continue
            self._apply_negative_evidence(track)

        events.extend(self._resolve_low_candidate_inheritance(tracks, frame_idx))

        for idx, track in enumerate(tracks):
            matched_group = next((group_idx for track_idx, group_idx in matches if track_idx == idx), None)
            reacquire_score = reacquire_score_by_track.get(idx)
            if reacquire_score is None:
                reacquire_score = self._evidence_increment(groups[matched_group]) if matched_group is not None else 0.0
            matched_real = matched_group is not None and self._group_has_real_detection(groups[matched_group])
            events.extend(self._transition_track(track, frame_idx, matched_real, reacquire_score))

        spawn_group_indices = self._filter_spawn_groups_near_lost(tracks, groups, unmatched_group_indices)
        spawn_group_indices = self._filter_spawn_groups_near_active(tracks, groups, spawn_group_indices)
        new_tracks, new_events = self.spawn_candidates([groups[idx] for idx in spawn_group_indices], frame_idx, tracks)
        tracks.extend(new_tracks)
        events.extend(new_events)
        tracks = self._prune_stale_removed_tracks(tracks, frame_idx)
        tracks = self._enforce_track_caps(tracks, frame_idx)
        return tracks, events

    def merge_observations(self, observations: list[Observation]) -> list[_ObservationGroup]:
        groups: list[_ObservationGroup] = []
        supported_sources = self._real_detection_sources()
        supported_observations = [
            obs
            for obs in observations
            if str(obs.source) in supported_sources
        ]
        sorted_observations = sorted(
            supported_observations,
            key=lambda obs: (_SOURCE_PRIORITY.get(str(obs.source), 99), int(obs.frame_idx)),
        )

        for obs in sorted_observations:
            obs_box = _box_array(obs.box)
            group_idx = self._find_observation_group(groups, obs_box)
            if group_idx is None:
                groups.append(_ObservationGroup())
                group_idx = len(groups) - 1
            self._add_observation_to_group(groups[group_idx], obs, obs_box)
        return groups

    def associate_tracks_to_observations(
        self,
        tracks: list[EvidenceTrack],
        observation_groups: list[_ObservationGroup],
        track_indices: list[int] | None = None,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if track_indices is None:
            track_indices = [
                idx
                for idx, track in enumerate(tracks)
                if _as_state(track.state) in {TrackState.CANDIDATE, TrackState.ACTIVE}
            ]
        if not track_indices or not observation_groups:
            return [], track_indices, list(range(len(observation_groups)))

        track_boxes = np.vstack([self._predict_box(tracks[idx]) for idx in track_indices])
        obs_boxes = np.vstack([group.box for group in observation_groups])
        iou = _iou_batch(track_boxes, obs_boxes)
        distances = _center_distance_batch(track_boxes, obs_boxes)

        center_thresh = float(self._cfg("MSDC_ASSOC_CENTER_DIST", 80.0))
        iou_thresh = float(self._cfg("MSDC_ASSOC_IOU_THRESH", 0.2))
        valid_pairs = np.logical_or(iou >= iou_thresh, distances <= center_thresh)
        local_track_rows, group_cols = np.where(valid_pairs)
        if local_track_rows.size:
            center_bonus = np.maximum(0.0, 1.0 - distances[local_track_rows, group_cols] / max(center_thresh, 1e-6))
            scores = iou[local_track_rows, group_cols] + 0.5 * center_bonus
            candidates = [
                (float(score), int(track_indices[int(local_track_idx)]), int(group_idx))
                for score, local_track_idx, group_idx in zip(scores, local_track_rows, group_cols)
            ]
        else:
            candidates = []

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_groups: set[int] = set()
        for _, track_idx, group_idx in sorted(candidates, reverse=True):
            if track_idx in used_tracks or group_idx in used_groups:
                continue
            matches.append((track_idx, group_idx))
            used_tracks.add(track_idx)
            used_groups.add(group_idx)

        unmatched_tracks = [idx for idx in track_indices if idx not in used_tracks]
        unmatched_groups = [idx for idx in range(len(observation_groups)) if idx not in used_groups]
        return matches, unmatched_tracks, unmatched_groups

    def spawn_candidates(
        self,
        observation_groups: list[_ObservationGroup],
        frame_idx: int,
        existing_tracks: list[EvidenceTrack] | None = None,
    ) -> tuple[list[EvidenceTrack], list[LifecycleEvent]]:
        existing_tracks = existing_tracks or []
        active_candidates = sum(
            1
            for track in existing_tracks
            if _as_state(track.state) in {TrackState.LOW_CANDIDATE, TrackState.CANDIDATE}
        )
        max_candidates = int(self._cfg("MSDC_MAX_CANDIDATES", 64))

        tracks: list[EvidenceTrack] = []
        events: list[LifecycleEvent] = []
        for group in observation_groups:
            if not self._group_can_spawn_candidate(group):
                continue
            if active_candidates + len(tracks) >= max_candidates:
                break
            evidence_score = self._rounded_score(self._evidence_increment(group))
            gid = self.next_gid
            self.next_gid += 1
            real_det_hits = 1 if self._group_has_real_detection(group) else 0
            state = TrackState.LOW_CANDIDATE if self._group_should_spawn_low_candidate(group) else TrackState.CANDIDATE
            track = EvidenceTrack(
                gid=gid,
                state=state,
                box=group.box,
                velocity=[0.0, 0.0],
                evidence_score=evidence_score,
                hits=1,
                misses=0,
                age=1,
                last_seen=int(frame_idx),
                last_real_det_frame=int(frame_idx) if real_det_hits else -1,
                real_det_hits=real_det_hits,
                last_real_det_box=group.box.copy() if real_det_hits else None,
                low_det_history=self._low_history_from_group(group, frame_idx),
                source_history=list(group.source_history),
                class_id=int(group.class_id),
                class_name=str(group.class_name),
            )
            removed_conflict = self._removed_guard_conflict(group, existing_tracks, frame_idx)
            tracks.append(track)
            if removed_conflict is not None:
                events.append(self._make_removed_guard_event(
                    frame_idx=frame_idx,
                    new_track=track,
                    conflict=removed_conflict,
                ))
                events.append(self._make_event(
                    frame_idx=frame_idx,
                    track=track,
                    event_type="NEW_ID_CREATED",
                    from_state=None,
                    to_state=state,
                    reason="removed_guard_conflict_new_gid",
                    extra={
                        "blocked_gid": int(removed_conflict["removed_gid"]),
                        "new_gid": int(track.gid),
                        "guard_iou": float(removed_conflict["iou"]),
                        "guard_center_distance": float(removed_conflict["center_distance"]),
                    },
                ))
            events.append(self._make_event(
                frame_idx=frame_idx,
                track=track,
                event_type="NEW_LOW_CANDIDATE" if state == TrackState.LOW_CANDIDATE else "NEW_CANDIDATE",
                from_state=None,
                to_state=state,
                reason="spawn_from_observation",
            ))
        return tracks, events

    def active_tracks_to_project_boxes(self, tracks: list[EvidenceTrack]) -> list[dict]:
        return [
            track_to_output_box(track)
            for track in tracks
            if _as_state(track.state) == TrackState.ACTIVE
        ]

    def should_reacquire_frame(self, frame_idx: int) -> bool:
        if not bool(self._cfg("MSDC_USE_REACQUIRE", True)):
            return False
        interval = int(self._cfg("MSDC_REACQUIRE_INTERVAL", 0))
        return bool(interval > 0 and int(frame_idx) % interval == 0)

    def reacquire_lost_tracks(
        self,
        lost_tracks: list[EvidenceTrack],
        observations: list[Observation],
        frame_idx: int,
    ) -> list[dict]:
        groups = self.merge_observations(observations)
        matches = self._reacquire_lost_track_indices(
            tracks=list(lost_tracks),
            groups=groups,
            available_group_indices=list(range(len(groups))),
            frame_idx=frame_idx,
        )
        return [match.debug for match in matches]

    def retired_guard_allows(
        self,
        candidate: EvidenceTrack | _ObservationGroup | Observation,
        retired_tracks: list[EvidenceTrack],
        frame_idx: int,
    ) -> bool:
        group = self._candidate_to_group(candidate)
        return self._removed_guard_conflict(group, retired_tracks, frame_idx) is None

    def _resolve_low_candidate_inheritance(
        self,
        tracks: list[EvidenceTrack],
        frame_idx: int,
    ) -> list[LifecycleEvent]:
        ready_low_indices = [
            idx
            for idx, track in enumerate(tracks)
            if _as_state(track.state) == TrackState.LOW_CANDIDATE
            and self._low_candidate_can_confirm(track, frame_idx)
        ]
        lost_indices = [
            idx
            for idx, track in enumerate(tracks)
            if _as_state(track.state) == TrackState.LOST
        ]
        active_indices = [
            idx
            for idx, track in enumerate(tracks)
            if _as_state(track.state) == TrackState.ACTIVE
        ]
        enabled = bool(self._cfg("MSDC_LOW_INHERIT_ENABLE", True))
        self.last_low_inherit_debug = {
            "enabled": bool(enabled),
            "frame_idx": int(frame_idx),
            "num_ready_low_candidates": int(len(ready_low_indices)),
            "num_lost": int(len(lost_indices)),
            "num_matches": 0,
            "matches": [],
            "num_active_conflicts": 0,
            "active_conflicts": [],
        }
        if not ready_low_indices:
            return []

        events: list[LifecycleEvent] = []
        if not enabled or not lost_indices:
            events.extend(self._remove_ready_low_candidates_near_active(tracks, ready_low_indices, active_indices, frame_idx))
            return events

        candidates: list[tuple[float, int, int, dict]] = []
        for low_idx in ready_low_indices:
            active_conflict = self._low_candidate_active_conflict(tracks[low_idx], tracks, active_indices)
            if active_conflict is not None:
                active_conflict["low_candidate_gid"] = int(tracks[low_idx].gid)
                self.last_low_inherit_debug["active_conflicts"].append(active_conflict)
                continue
            for lost_idx in lost_indices:
                scored = self._score_low_candidate_inheritance(tracks[lost_idx], tracks[low_idx], frame_idx)
                if scored is None:
                    continue
                score, debug = scored
                if score < float(self._cfg("MSDC_LOW_INHERIT_SCORE", 0.40)):
                    continue
                candidates.append((score, lost_idx, low_idx, debug))

        used_lost: set[int] = set()
        used_low: set[int] = set()
        matches: list[tuple[int, int, float, dict]] = []
        for score, lost_idx, low_idx, debug in sorted(candidates, reverse=True, key=lambda item: item[0]):
            if lost_idx in used_lost or low_idx in used_low:
                continue
            used_lost.add(lost_idx)
            used_low.add(low_idx)
            matches.append((lost_idx, low_idx, score, debug))

        for lost_idx, low_idx, score, debug in matches:
            events.extend(self._merge_low_candidate_into_lost(
                lost=tracks[lost_idx],
                low_candidate=tracks[low_idx],
                frame_idx=frame_idx,
                inherit_score=score,
                debug=debug,
            ))

        unmatched_ready_low = [idx for idx in ready_low_indices if idx not in used_low]
        events.extend(self._remove_ready_low_candidates_near_active(tracks, unmatched_ready_low, active_indices, frame_idx))
        self.last_low_inherit_debug["num_matches"] = int(len(matches))
        self.last_low_inherit_debug["matches"] = [debug for _, _, _, debug in matches]
        self.last_low_inherit_debug["num_active_conflicts"] = int(len(self.last_low_inherit_debug["active_conflicts"]))
        return events

    def _remove_ready_low_candidates_near_active(
        self,
        tracks: list[EvidenceTrack],
        ready_low_indices: list[int],
        active_indices: list[int],
        frame_idx: int,
    ) -> list[LifecycleEvent]:
        events: list[LifecycleEvent] = []
        for low_idx in ready_low_indices:
            low_track = tracks[low_idx]
            if _as_state(low_track.state) != TrackState.LOW_CANDIDATE:
                continue
            conflict = self._low_candidate_active_conflict(low_track, tracks, active_indices)
            if conflict is None:
                continue
            conflict["low_candidate_gid"] = int(low_track.gid)
            self.last_low_inherit_debug["active_conflicts"].append(conflict)
            low_track.retired_signature = self._internal_merged_signature(low_track, frame_idx, merged_into_gid=None)
            low_track.state = TrackState.REMOVED
            events.append(self._make_event(
                frame_idx=frame_idx,
                track=low_track,
                event_type="PRUNED_LOW_CANDIDATE_ACTIVE_CONFLICT",
                from_state=TrackState.LOW_CANDIDATE,
                to_state=TrackState.REMOVED,
                reason="ready_low_candidate_overlaps_active_track",
                extra=conflict,
            ))
        self.last_low_inherit_debug["num_active_conflicts"] = int(len(self.last_low_inherit_debug["active_conflicts"]))
        return events

    def _score_low_candidate_inheritance(
        self,
        lost: EvidenceTrack,
        low_candidate: EvidenceTrack,
        frame_idx: int,
    ) -> tuple[float, dict] | None:
        lost_age = int(frame_idx) - int(lost.last_seen)
        max_lost_age = int(self._cfg("MSDC_LOW_INHERIT_MAX_LOST_AGE", self._cfg("MSDC_LOST_MAX_AGE", 80)))
        if lost_age < 0 or lost_age > max_lost_age:
            return None
        require_class_gate = bool(self._cfg("MSDC_LOW_INHERIT_CLASS_MATCH", True))
        class_compatible = self._classes_compatible(lost, low_candidate)

        pred_box, pred_debug = self._predict_low_inherit_lost_box(lost, frame_idx)
        pred_box = pred_box.reshape(1, 4)
        low_box = _box_array(low_candidate.box).reshape(1, 4)
        iou_score = float(_iou_batch(pred_box, low_box)[0, 0])
        center_dist = float(_center_distance_batch(pred_box, low_box)[0, 0])
        center_thresh = float(self._cfg(
            "MSDC_LOW_INHERIT_CENTER_DIST",
            self._cfg("MSDC_REACQUIRE_CENTER_DIST", self._cfg("MSDC_ASSOC_CENTER_DIST", 80.0) * 2.0),
        ))
        iou_thresh = float(self._cfg("MSDC_LOW_INHERIT_IOU_THRESH", 0.02))
        if iou_score < iou_thresh and center_dist > center_thresh:
            return None
        if require_class_gate and not class_compatible:
            class_mismatch_center = float(self._cfg("MSDC_LOW_INHERIT_CLASS_MISMATCH_CENTER_DIST", 80.0))
            if center_dist > class_mismatch_center:
                return None

        group = self._candidate_to_group(low_candidate)
        velocity_consistency = self._velocity_consistency(lost, group)
        if velocity_consistency < float(self._cfg("MSDC_LOW_INHERIT_VELOCITY_MIN", 0.15)):
            return None

        center_score = max(0.0, 1.0 - center_dist / max(center_thresh, 1e-6))
        low_avg_score = self._low_candidate_avg_score(low_candidate, frame_idx)
        recency_score = max(0.0, 1.0 - lost_age / max(max_lost_age, 1))
        score = (
            float(self._cfg("MSDC_LOW_INHERIT_WEIGHT_IOU", 0.35)) * iou_score
            + float(self._cfg("MSDC_LOW_INHERIT_WEIGHT_CENTER", 0.25)) * center_score
            + float(self._cfg("MSDC_LOW_INHERIT_WEIGHT_VELOCITY", 0.20)) * velocity_consistency
            + float(self._cfg("MSDC_LOW_INHERIT_WEIGHT_LOW_SCORE", 0.15)) * low_avg_score
            + float(self._cfg("MSDC_LOW_INHERIT_WEIGHT_RECENCY", 0.05)) * recency_score
        )
        if require_class_gate and not class_compatible:
            score -= float(self._cfg("MSDC_LOW_INHERIT_CLASS_MISMATCH_PENALTY", 0.0))
        score = self._rounded_score(score)
        debug = {
            "frame_idx": int(frame_idx),
            "lost_gid": int(lost.gid),
            "low_candidate_gid": int(low_candidate.gid),
            "public_id": None if lost.public_id is None else int(lost.public_id),
            "class_compatible": bool(class_compatible),
            "inherit_score": float(score),
            "iou": float(round(iou_score, 4)),
            "center_distance": float(round(center_dist, 4)),
            "center_threshold": float(round(center_thresh, 4)),
            "velocity_consistency": float(round(velocity_consistency, 4)),
            "low_avg_score": float(round(low_avg_score, 4)),
            "lost_age": int(lost_age),
            "low_candidate_box": [float(v) for v in _box_array(low_candidate.box).tolist()],
            "lost_pred_box": [float(v) for v in pred_box.reshape(-1).tolist()],
            "lost_pred_source": str(pred_debug.get("source", "default")),
            "lost_pred_velocity": [float(v) for v in pred_debug.get("velocity", [])],
            "lost_pred_age": int(pred_debug.get("age", 0)),
        }
        return score, debug

    def _predict_low_inherit_lost_box(self, track: EvidenceTrack, frame_idx: int) -> tuple[np.ndarray, dict]:
        fallback = self._predict_box(track)
        if not bool(self._cfg("MSDC_LOW_INHERIT_USE_HISTORY_VELOCITY", True)):
            return fallback, {"source": "track_velocity", "velocity": list(np.asarray(track.velocity).reshape(-1)[:2]), "age": 1}

        velocity = self._low_history_velocity(track)
        latest = self._latest_low_history_box(track)
        if velocity is None or latest is None:
            return fallback, {"source": "track_velocity", "velocity": list(np.asarray(track.velocity).reshape(-1)[:2]), "age": 1}

        base_box, base_frame = latest
        age = max(0, int(frame_idx) - int(base_frame))
        age = min(age, int(self._cfg("MSDC_LOW_INHERIT_MAX_PREDICT_AGE", self._cfg("MSDC_LOW_INHERIT_MAX_LOST_AGE", 120))))
        pred = base_box.copy()
        pred[[0, 2]] += float(velocity[0]) * float(age)
        pred[[1, 3]] += float(velocity[1]) * float(age)
        return pred, {
            "source": "low_history_velocity",
            "velocity": [float(velocity[0]), float(velocity[1])],
            "age": int(age),
        }

    def _low_history_velocity(self, track: EvidenceTrack) -> np.ndarray | None:
        history = sorted(
            list(getattr(track, "low_det_history", []) or []),
            key=lambda item: int(item.get("frame_idx", 0)),
        )
        if len(history) < 2:
            return None
        velocities = []
        for prev, curr in zip(history, history[1:]):
            prev_center = prev.get("center", [])
            curr_center = curr.get("center", [])
            if not isinstance(prev_center, (list, tuple)) or not isinstance(curr_center, (list, tuple)):
                continue
            if len(prev_center) < 2 or len(curr_center) < 2:
                continue
            dt = int(curr.get("frame_idx", 0)) - int(prev.get("frame_idx", 0))
            if dt <= 0:
                continue
            displacement = np.asarray(curr_center[:2], dtype=np.float64) - np.asarray(prev_center[:2], dtype=np.float64)
            velocities.append(displacement / float(dt))
        if not velocities:
            return None
        return np.median(np.vstack(velocities), axis=0)

    def _latest_low_history_box(self, track: EvidenceTrack) -> tuple[np.ndarray, int] | None:
        history = list(getattr(track, "low_det_history", []) or [])
        if not history:
            return None
        latest = max(history, key=lambda item: int(item.get("frame_idx", -1)))
        try:
            return _box_array(latest.get("box", track.box)), int(latest.get("frame_idx", track.last_seen))
        except (TypeError, ValueError):
            return None

    def _merge_low_candidate_into_lost(
        self,
        lost: EvidenceTrack,
        low_candidate: EvidenceTrack,
        frame_idx: int,
        inherit_score: float,
        debug: dict,
    ) -> list[LifecycleEvent]:
        lost_from_state = lost.state
        if lost.public_id is None:
            lost.public_id = self.next_public_id
            self.next_public_id += 1

        lost.box = _box_array(low_candidate.box).copy()
        lost.velocity = list(np.asarray(low_candidate.velocity, dtype=np.float64).reshape(-1)[:2])
        lost.evidence_score = self._rounded_score(max(float(lost.evidence_score), float(low_candidate.evidence_score)))
        lost.hits = max(int(lost.hits) + 1, int(low_candidate.hits))
        lost.misses = 0
        lost.age = max(int(lost.age), int(low_candidate.age))
        lost.last_seen = int(frame_idx)
        lost.last_real_det_frame = int(low_candidate.last_real_det_frame)
        lost.real_det_hits = max(int(lost.real_det_hits) + 1, int(low_candidate.real_det_hits))
        real_det_box = low_candidate.last_real_det_box if low_candidate.last_real_det_box is not None else low_candidate.box
        lost.last_real_det_box = _box_array(real_det_box).copy()
        lost.low_det_history = list(getattr(low_candidate, "low_det_history", []) or [])[-int(self._cfg("MSDC_LOW_CONFIRM_WINDOW", 8)):]
        self._append_source_history(lost, list(getattr(low_candidate, "source_history", []) or []))
        if str(low_candidate.class_name) != "unknown":
            lost.class_id = int(low_candidate.class_id)
            lost.class_name = str(low_candidate.class_name)
        lost.state = TrackState.ACTIVE

        inherited_event = self._make_event(
            frame_idx=frame_idx,
            track=lost,
            event_type="LOW_CANDIDATE_INHERITED_LOST",
            from_state=lost_from_state,
            to_state=TrackState.ACTIVE,
            reason="ready_low_candidate_attached_to_nearby_lost_track",
            extra={
                **debug,
                "inherit_score": float(inherit_score),
                "merged_low_candidate_gid": int(low_candidate.gid),
            },
        )

        low_candidate.retired_signature = self._internal_merged_signature(low_candidate, frame_idx, merged_into_gid=lost.gid)
        low_candidate.state = TrackState.REMOVED
        merged_event = self._make_event(
            frame_idx=frame_idx,
            track=low_candidate,
            event_type="LOW_CANDIDATE_MERGED",
            from_state=TrackState.LOW_CANDIDATE,
            to_state=TrackState.REMOVED,
            reason="merged_into_lost_track_public_id",
            extra={
                "merged_into_gid": int(lost.gid),
                "merged_into_public_id": int(lost.public_id),
                "inherit_score": float(inherit_score),
            },
        )
        return [inherited_event, merged_event]

    def _low_candidate_active_conflict(
        self,
        low_candidate: EvidenceTrack,
        tracks: list[EvidenceTrack],
        active_indices: list[int],
    ) -> dict | None:
        active_tracks = [tracks[idx] for idx in active_indices if int(tracks[idx].gid) != int(low_candidate.gid)]
        if not active_tracks:
            return None
        return self._active_search_conflict(active_tracks, self._candidate_to_group(low_candidate))

    @staticmethod
    def _classes_compatible(left: EvidenceTrack, right: EvidenceTrack) -> bool:
        left_id = int(getattr(left, "class_id", -1))
        right_id = int(getattr(right, "class_id", -1))
        if left_id >= 0 and right_id >= 0:
            return left_id == right_id
        left_name = str(getattr(left, "class_name", "unknown"))
        right_name = str(getattr(right, "class_name", "unknown"))
        return "unknown" in {left_name, right_name} or left_name == right_name

    def _low_candidate_avg_score(self, track: EvidenceTrack, frame_idx: int) -> float:
        window = int(self._cfg("MSDC_LOW_CONFIRM_WINDOW", 8))
        history = [
            item
            for item in list(getattr(track, "low_det_history", []) or [])
            if int(frame_idx) - int(item.get("frame_idx", frame_idx)) < window
        ]
        if not history:
            return 0.0
        return float(sum(float(item.get("score", 0.0)) for item in history) / max(1, len(history)))

    def _internal_merged_signature(
        self,
        track: EvidenceTrack,
        frame_idx: int,
        merged_into_gid: int | None,
    ) -> dict:
        signature = self._make_removed_signature(track, frame_idx)
        signature.update({
            "skip_removed_guard": True,
            "merged_internal": True,
            "merged_into_gid": None if merged_into_gid is None else int(merged_into_gid),
        })
        return signature

    def _reacquire_lost_track_indices(
        self,
        tracks: list[EvidenceTrack],
        groups: list[_ObservationGroup],
        available_group_indices: list[int],
        frame_idx: int,
    ) -> list[_ReacquireMatch]:
        lost_indices = [
            idx
            for idx, track in enumerate(tracks)
            if _as_state(track.state) == TrackState.LOST
        ]
        attempted = self.should_reacquire_frame(frame_idx) and bool(lost_indices) and bool(available_group_indices)
        self.last_reacquire_debug = {
            "enabled": int(self._cfg("MSDC_REACQUIRE_INTERVAL", 0)) > 0,
            "interval": int(self._cfg("MSDC_REACQUIRE_INTERVAL", 0)),
            "attempted": bool(attempted),
            "frame_idx": int(frame_idx),
            "num_lost": int(len(lost_indices)),
            "num_groups": int(len(available_group_indices)),
            "num_matches": 0,
            "matches": [],
        }
        if not attempted:
            return []

        candidates: list[tuple[float, int, int, dict]] = []
        for track_idx in lost_indices:
            track = tracks[track_idx]
            for group_idx in available_group_indices:
                candidate = self._score_reacquire_candidate(track, groups[group_idx], frame_idx)
                if candidate is None:
                    continue
                score, debug = candidate
                if score < float(self._cfg("MSDC_REACQUIRE_SCORE", 1.5)):
                    continue
                candidates.append((score, track_idx, group_idx, debug))

        matches: list[_ReacquireMatch] = []
        used_tracks: set[int] = set()
        used_groups: set[int] = set()
        for score, track_idx, group_idx, debug in sorted(candidates, reverse=True, key=lambda item: item[0]):
            if track_idx in used_tracks or group_idx in used_groups:
                continue
            matches.append(_ReacquireMatch(track_idx=track_idx, group_idx=group_idx, score=score, debug=debug))
            used_tracks.add(track_idx)
            used_groups.add(group_idx)

        self.last_reacquire_debug["num_matches"] = int(len(matches))
        self.last_reacquire_debug["matches"] = [match.debug for match in matches]
        return matches

    def _filter_spawn_groups_near_lost(
        self,
        tracks: list[EvidenceTrack],
        groups: list[_ObservationGroup],
        group_indices: list[int],
    ) -> list[int]:
        lost_tracks = [track for track in tracks if _as_state(track.state) == TrackState.LOST]
        if not bool(self._cfg("MSDC_USE_REACQUIRE", True)) or not lost_tracks or not group_indices:
            return group_indices

        kept = []
        suppressed = []
        for group_idx in group_indices:
            group = groups[group_idx]
            conflict = self._lost_search_conflict(lost_tracks, group)
            if conflict is None:
                kept.append(group_idx)
            else:
                conflict["group_idx"] = int(group_idx)
                suppressed.append(conflict)

        if suppressed:
            self.last_reacquire_debug["num_suppressed_spawn_groups"] = int(len(suppressed))
            self.last_reacquire_debug["suppressed_spawn_groups"] = suppressed
        else:
            self.last_reacquire_debug.setdefault("num_suppressed_spawn_groups", 0)
            self.last_reacquire_debug.setdefault("suppressed_spawn_groups", [])
        return kept

    def _filter_spawn_groups_near_active(
        self,
        tracks: list[EvidenceTrack],
        groups: list[_ObservationGroup],
        group_indices: list[int],
    ) -> list[int]:
        active_tracks = [track for track in tracks if _as_state(track.state) == TrackState.ACTIVE]
        if not bool(self._cfg("MSDC_SPAWN_SUPPRESS_ENABLE", True)) or not active_tracks or not group_indices:
            self.last_spawn_suppression_debug["num_groups"] = int(len(group_indices))
            return group_indices

        kept = []
        suppressed = []
        for group_idx in group_indices:
            group = groups[group_idx]
            conflict = self._active_search_conflict(active_tracks, group)
            if conflict is None:
                kept.append(group_idx)
            else:
                conflict["group_idx"] = int(group_idx)
                suppressed.append(conflict)

        self.last_spawn_suppression_debug.update({
            "enabled": True,
            "num_groups": int(len(group_indices)),
            "num_suppressed_spawn_groups": int(len(suppressed)),
            "suppressed_spawn_groups": suppressed,
        })
        return kept

    def _active_search_conflict(self, active_tracks: list[EvidenceTrack], group: _ObservationGroup) -> dict | None:
        center_thresh = float(self._cfg("MSDC_SPAWN_SUPPRESS_CENTER_DIST", self._cfg("MSDC_ASSOC_CENTER_DIST", 80.0)))
        iou_thresh = float(self._cfg("MSDC_SPAWN_SUPPRESS_IOU", 0.1))
        if self._group_should_spawn_low_candidate(group):
            center_thresh = max(
                center_thresh,
                float(self._cfg("MSDC_LOW_SPAWN_SUPPRESS_CENTER_DIST", center_thresh)),
            )
        group_box = group.box.reshape(1, 4)
        best = None
        for track in active_tracks:
            pred_box = self._predict_box(track).reshape(1, 4)
            iou_score = float(_iou_batch(pred_box, group_box)[0, 0])
            center_dist = float(_center_distance_batch(pred_box, group_box)[0, 0])
            if iou_score < iou_thresh and center_dist > center_thresh:
                continue
            conflict = {
                "gid": int(track.gid),
                "public_id": None if track.public_id is None else int(track.public_id),
                "iou": float(round(iou_score, 4)),
                "center_distance": float(round(center_dist, 4)),
                "sources": list(group.source_history),
            }
            if best is None or iou_score > float(best["iou"]):
                best = conflict
        return best

    @staticmethod
    def _real_detection_sources() -> set[str]:
        return {"high_det", "low_det", "reacquire"}

    def _group_has_real_detection(self, group: _ObservationGroup) -> bool:
        return any(source in group.source_scores for source in self._real_detection_sources())

    def _group_has_high_detection(self, group: _ObservationGroup) -> bool:
        return "high_det" in group.source_scores

    def _group_can_spawn_candidate(self, group: _ObservationGroup) -> bool:
        if self._group_has_high_detection(group):
            return True
        low_score = float(group.source_scores.get("low_det", 0.0))
        if low_score >= float(self._cfg("MSDC_LOW_SPAWN_MIN_CONF", 0.30)):
            return True
        return False

    def _group_should_spawn_low_candidate(self, group: _ObservationGroup) -> bool:
        if not bool(self._cfg("MSDC_LOW_CANDIDATE_ENABLE", True)):
            return False
        if self._group_has_high_detection(group):
            return False
        return float(group.source_scores.get("low_det", 0.0)) >= float(self._cfg("MSDC_LOW_SPAWN_MIN_CONF", 0.30))

    def _low_history_from_group(self, group: _ObservationGroup, frame_idx: int) -> list[dict]:
        score = float(group.source_scores.get("low_det", 0.0))
        if score <= 0.0:
            return []
        box = group.box
        return [{
            "frame_idx": int(frame_idx),
            "score": float(score),
            "box": [float(v) for v in box.tolist()],
            "area": float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])),
            "center": [float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)],
        }]

    def _lost_search_conflict(self, lost_tracks: list[EvidenceTrack], group: _ObservationGroup) -> dict | None:
        iou_thresh = float(self._cfg("MSDC_REACQUIRE_IOU_THRESH", 0.05))
        group_box = group.box.reshape(1, 4)
        best = None
        for track in lost_tracks:
            center_thresh = self._reacquire_center_threshold(track)
            pred_box = self._predict_box(track).reshape(1, 4)
            iou_score = float(_iou_batch(pred_box, group_box)[0, 0])
            center_dist = float(_center_distance_batch(pred_box, group_box)[0, 0])
            if iou_score < iou_thresh and center_dist > center_thresh:
                continue
            conflict = {
                "gid": int(track.gid),
                "iou": float(round(iou_score, 4)),
                "center_distance": float(round(center_dist, 4)),
                "center_threshold": float(round(center_thresh, 4)),
                "sources": list(group.source_history),
            }
            if best is None or iou_score > float(best["iou"]):
                best = conflict
        return best

    def _reacquire_center_threshold(self, track: EvidenceTrack) -> float:
        base = float(self._cfg("MSDC_REACQUIRE_CENTER_DIST", self._cfg("MSDC_ASSOC_CENTER_DIST", 80.0) * 2.0))
        scale_factor = float(self._cfg("MSDC_REACQUIRE_CENTER_SCALE_FACTOR", 4.0))
        max_dist = float(self._cfg("MSDC_REACQUIRE_MAX_CENTER_DIST", max(base, 240.0)))
        box = _box_array(track.box)
        width = max(1.0, float(box[2] - box[0]))
        height = max(1.0, float(box[3] - box[1]))
        scaled = max(base, max(width, height) * scale_factor)
        return min(max_dist, scaled)

    def _score_reacquire_candidate(
        self,
        track: EvidenceTrack,
        group: _ObservationGroup,
        frame_idx: int,
    ) -> tuple[float, dict] | None:
        source_scores = dict(group.source_scores)
        if not any(source in source_scores for source in self._real_detection_sources()):
            return None

        pred_box = self._predict_box(track).reshape(1, 4)
        group_box = group.box.reshape(1, 4)
        iou_score = float(_iou_batch(pred_box, group_box)[0, 0])
        center_dist = float(_center_distance_batch(pred_box, group_box)[0, 0])
        iou_thresh = float(self._cfg("MSDC_REACQUIRE_IOU_THRESH", 0.05))
        center_thresh = self._reacquire_center_threshold(track)
        if iou_score < iou_thresh and center_dist > center_thresh:
            return None

        base_score = self._evidence_increment(group)
        score = self._rounded_score(base_score)
        debug = {
            "frame_idx": int(frame_idx),
            "gid": int(track.gid),
            "reacquire_score": float(score),
            "base_score": float(base_score),
            "low_det_score": float(source_scores.get("low_det", 0.0)),
            "iou": float(round(iou_score, 4)),
            "center_distance": float(round(center_dist, 4)),
            "center_threshold": float(round(center_thresh, 4)),
            "box": [float(v) for v in group.box.tolist()],
            "sources": list(group.source_history),
        }
        return score, debug

    def _find_observation_group(self, groups: list[_ObservationGroup], box: np.ndarray) -> int | None:
        if not groups:
            return None
        group_boxes = np.vstack([group.box for group in groups])
        ious = _iou_batch(box.reshape(1, 4), group_boxes).reshape(-1)
        best_idx = int(np.argmax(ious))
        if float(ious[best_idx]) >= float(self._cfg("MSDC_OBS_MERGE_IOU_THRESH", 0.5)):
            return best_idx
        return None

    def _add_observation_to_group(self, group: _ObservationGroup, obs: Observation, box: np.ndarray) -> None:
        group.observations.append(obs)
        group.boxes.append(box)
        score = max(0.0, float(obs.score) * float(obs.reliability))
        group.weights.append(max(score, 1e-6))
        source = str(obs.source)
        group.source_scores[source] = max(float(group.source_scores.get(source, 0.0)), score)
        group.source_history = sorted(group.source_scores, key=lambda item: _SOURCE_PRIORITY.get(item, 99))
        if group.class_name == "unknown" and str(obs.class_name) != "unknown":
            group.class_id = int(obs.class_id)
            group.class_name = str(obs.class_name)

    def _candidate_to_group(self, candidate: EvidenceTrack | _ObservationGroup | Observation) -> _ObservationGroup:
        if isinstance(candidate, _ObservationGroup):
            return candidate
        group = _ObservationGroup()
        if isinstance(candidate, Observation):
            self._add_observation_to_group(group, candidate, _box_array(candidate.box))
            return group
        if isinstance(candidate, EvidenceTrack):
            group.boxes.append(_box_array(candidate.box))
            group.weights.append(1.0)
            group.class_id = int(candidate.class_id)
            group.class_name = str(candidate.class_name)
            return group
        raise TypeError(f"Unsupported candidate type for removed guard: {type(candidate)!r}")

    def _velocity_consistency(self, track: EvidenceTrack, group: _ObservationGroup) -> float:
        velocity = np.asarray(track.velocity, dtype=np.float64).reshape(-1)
        if velocity.size < 2:
            return 0.0
        vel = velocity[:2]
        vel_norm = float(np.linalg.norm(vel))
        if vel_norm <= 1e-6:
            return 1.0
        old_box = _box_array(track.box)
        new_box = group.box
        old_center = np.array([(old_box[0] + old_box[2]) / 2.0, (old_box[1] + old_box[3]) / 2.0])
        new_center = np.array([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])
        disp = new_center - old_center
        disp_norm = float(np.linalg.norm(disp))
        if disp_norm <= 1e-6:
            return 0.5
        cosine = float(np.dot(vel, disp) / max(vel_norm * disp_norm, 1e-9))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _make_removed_signature(self, track: EvidenceTrack, frame_idx: int) -> dict:
        return {
            "last_box": [float(v) for v in _box_array(track.box).tolist()],
            "last_seen": int(track.last_seen),
            "removed_frame_idx": int(frame_idx),
            "last_velocity": [float(v) for v in np.asarray(track.velocity, dtype=np.float64).reshape(-1)[:2].tolist()],
            "class_id": int(track.class_id),
            "class_name": str(track.class_name),
        }

    def _removed_guard_conflict(
        self,
        group: _ObservationGroup,
        existing_tracks: list[EvidenceTrack],
        frame_idx: int,
    ) -> dict | None:
        if not bool(self._cfg("MSDC_REUSE_GUARD_ENABLE", True)):
            self.last_removed_guard_debug = self._empty_removed_guard_debug(enabled=False)
            return None

        guard_frames = int(self._cfg(
            "MSDC_removed_GUARD_FRAMES",
            self._cfg("MSDC_REMOVED_GUARD_FRAMES", 120),
        ))
        if guard_frames <= 0:
            self.last_removed_guard_debug = self._empty_removed_guard_debug(enabled=False)
            return None

        iou_thresh = float(self._cfg("MSDC_REMOVED_GUARD_IOU_THRESH", 0.3))
        center_thresh = float(self._cfg("MSDC_REMOVED_GUARD_CENTER_DIST", self._cfg("MSDC_ASSOC_CENTER_DIST", 80.0)))
        group_box = group.box
        guard_data = self._removed_guard_data(existing_tracks, frame_idx, guard_frames)
        sig_boxes = guard_data["boxes"]
        if sig_boxes.size == 0:
            self.last_removed_guard_debug = {
                "enabled": True,
                "guard_frames": int(guard_frames),
                "num_recent_signatures": 0,
                "num_vetoes": 0,
                "vetoes": [],
            }
            return None

        ious = _iou_batch(group_box.reshape(1, 4), sig_boxes).reshape(-1)
        distances = _center_distance_batch(group_box.reshape(1, 4), sig_boxes).reshape(-1)
        matched_indices = np.where((ious >= iou_thresh) | (distances <= center_thresh))[0]
        debug_limit = int(self._cfg("MSDC_DEBUG_DETAIL_LIMIT", 8))
        vetoes = [
            self._removed_guard_payload(guard_data, int(idx), group_box, float(ious[idx]), float(distances[idx]))
            for idx in matched_indices[:max(0, debug_limit)]
        ]
        best_conflict = None
        if matched_indices.size > 0:
            best_idx = int(matched_indices[int(np.argmax(ious[matched_indices]))])
            best_conflict = self._removed_guard_payload(
                guard_data,
                best_idx,
                group_box,
                float(ious[best_idx]),
                float(distances[best_idx]),
            )

        self.last_removed_guard_debug = {
            "enabled": True,
            "guard_frames": int(guard_frames),
            "num_recent_signatures": int(len(guard_data["gids"])),
            "num_vetoes": int(matched_indices.size),
            "vetoes": vetoes,
        }
        return best_conflict

    def _removed_guard_data(
        self,
        existing_tracks: list[EvidenceTrack],
        frame_idx: int,
        guard_frames: int,
    ) -> dict:
        cache_key = (id(existing_tracks), len(existing_tracks), int(frame_idx), int(guard_frames))
        if self._removed_guard_cache_key == cache_key and self._removed_guard_cache is not None:
            return self._removed_guard_cache

        boxes = []
        gids = []
        removed_frame_indices = []
        ages = []
        signatures = []
        for track in existing_tracks:
            if _as_state(track.state) != TrackState.REMOVED:
                continue
            signature = track.retired_signature or self._make_removed_signature(track, int(frame_idx))
            if bool(signature.get("skip_removed_guard", False)):
                continue
            removed_frame_idx = int(signature.get("removed_frame_idx", signature.get("last_seen", int(track.last_seen))))
            age = int(frame_idx) - removed_frame_idx
            if age < 0 or age > guard_frames:
                continue
            boxes.append(_box_array(signature.get("last_box", track.box)))
            gids.append(int(track.gid))
            removed_frame_indices.append(int(removed_frame_idx))
            ages.append(int(age))
            signatures.append(signature)

        payload = {
            "boxes": np.vstack(boxes) if boxes else np.zeros((0, 4), dtype=np.float64),
            "gids": gids,
            "removed_frame_indices": removed_frame_indices,
            "ages": ages,
            "signatures": signatures,
        }
        self._removed_guard_cache_key = cache_key
        self._removed_guard_cache = payload
        return payload

    @staticmethod
    def _removed_guard_payload(
        guard_data: dict,
        index: int,
        group_box: np.ndarray,
        iou_score: float,
        center_dist: float,
    ) -> dict:
        return {
            "removed_gid": int(guard_data["gids"][index]),
            "removed_frame_idx": int(guard_data["removed_frame_indices"][index]),
            "signature_age": int(guard_data["ages"][index]),
            "iou": float(round(iou_score, 4)),
            "center_distance": float(round(center_dist, 4)),
            "candidate_box": [float(v) for v in group_box.tolist()],
            "removed_signature": guard_data["signatures"][index],
        }

    def _prune_stale_removed_tracks(self, tracks: list[EvidenceTrack], frame_idx: int) -> list[EvidenceTrack]:
        guard_frames = int(self._cfg(
            "MSDC_removed_GUARD_FRAMES",
            self._cfg("MSDC_REMOVED_GUARD_FRAMES", 120),
        ))
        if guard_frames <= 0:
            return [track for track in tracks if _as_state(track.state) != TrackState.REMOVED]

        kept = []
        for track in tracks:
            if _as_state(track.state) != TrackState.REMOVED:
                kept.append(track)
                continue
            signature = track.retired_signature or self._make_removed_signature(track, int(frame_idx))
            if bool(signature.get("skip_removed_guard", False)):
                continue
            removed_frame_idx = int(signature.get("removed_frame_idx", signature.get("last_seen", int(track.last_seen))))
            if int(frame_idx) - removed_frame_idx <= guard_frames:
                kept.append(track)
        return kept

    def _enforce_track_caps(self, tracks: list[EvidenceTrack], frame_idx: int) -> list[EvidenceTrack]:
        max_active = int(self._cfg("MSDC_MAX_ACTIVE_TRACKS", 64))
        max_lost = int(self._cfg("MSDC_MAX_LOST_TRACKS", 32))
        max_candidates = int(self._cfg("MSDC_MAX_CANDIDATES", 32))
        max_low_candidates = int(self._cfg("MSDC_MAX_LOW_CANDIDATES", 24))

        if max_active > 0:
            self._cap_state_pool(tracks, TrackState.ACTIVE, max_active, frame_idx, overflow_state=TrackState.LOST)
        if max_lost > 0:
            self._cap_state_pool(tracks, TrackState.LOST, max_lost, frame_idx, overflow_state=TrackState.REMOVED)
        if max_candidates > 0:
            self._cap_state_pool(tracks, TrackState.CANDIDATE, max_candidates, frame_idx, overflow_state=TrackState.REMOVED)
        if max_low_candidates > 0:
            self._cap_state_pool(tracks, TrackState.LOW_CANDIDATE, max_low_candidates, frame_idx, overflow_state=TrackState.REMOVED)

        max_total = int(self._cfg("MSDC_MAX_TOTAL_TRACKS", 128))
        if max_total > 0:
            tracks = self._cap_total_track_list(tracks, max_total, frame_idx)
        return tracks

    def _cap_state_pool(
        self,
        tracks: list[EvidenceTrack],
        state: TrackState,
        limit: int,
        frame_idx: int,
        overflow_state: TrackState,
    ) -> None:
        state_tracks = [track for track in tracks if _as_state(track.state) == state]
        if len(state_tracks) <= limit:
            return
        kept = set(id(track) for track in sorted(state_tracks, key=self._track_keep_score, reverse=True)[:limit])
        for track in state_tracks:
            if id(track) in kept:
                continue
            if overflow_state == TrackState.REMOVED:
                track.retired_signature = self._make_removed_signature(track, frame_idx)
            track.state = overflow_state
            if overflow_state == TrackState.LOST:
                track.misses = max(int(track.misses), 1)

    def _cap_total_track_list(
        self,
        tracks: list[EvidenceTrack],
        limit: int,
        frame_idx: int,
    ) -> list[EvidenceTrack]:
        live_tracks = [track for track in tracks if _as_state(track.state) != TrackState.REMOVED]
        if len(live_tracks) > limit:
            kept_live = set(id(track) for track in sorted(live_tracks, key=self._track_keep_score, reverse=True)[:limit])
            for track in live_tracks:
                if id(track) in kept_live:
                    continue
                track.retired_signature = self._make_removed_signature(track, frame_idx)
                track.state = TrackState.REMOVED

        if len(tracks) <= limit:
            return tracks
        ordered = sorted(tracks, key=self._total_keep_key, reverse=True)
        return ordered[:limit]

    @staticmethod
    def _track_keep_score(track: EvidenceTrack) -> tuple[float, int, int, int]:
        state = _as_state(track.state)
        state_priority = {
            TrackState.ACTIVE: 4,
            TrackState.LOST: 3,
            TrackState.CANDIDATE: 2,
            TrackState.LOW_CANDIDATE: 1,
            TrackState.REMOVED: 0,
        }.get(state, 0)
        return (
            float(track.evidence_score),
            int(track.last_seen),
            int(track.hits),
            state_priority,
        )

    def _total_keep_key(self, track: EvidenceTrack) -> tuple[int, float, int, int]:
        state = _as_state(track.state)
        state_priority = {
            TrackState.ACTIVE: 5,
            TrackState.LOST: 4,
            TrackState.CANDIDATE: 3,
            TrackState.LOW_CANDIDATE: 2,
            TrackState.REMOVED: 1,
        }.get(state, 0)
        removed_age = int(track.last_seen)
        if state == TrackState.REMOVED:
            signature = track.retired_signature or {}
            removed_age = int(signature.get("removed_frame_idx", signature.get("last_seen", track.last_seen)))
        return (
            state_priority,
            float(track.evidence_score),
            int(removed_age),
            int(track.gid),
        )

    def _make_removed_guard_event(
        self,
        frame_idx: int,
        new_track: EvidenceTrack,
        conflict: dict,
    ) -> LifecycleEvent:
        return LifecycleEvent(
            frame_idx=int(frame_idx),
            gid=int(conflict["removed_gid"]),
            event_type="PREVENT_removed_ID_REUSE",
            from_state=TrackState.REMOVED,
            to_state=TrackState.REMOVED,
            reason="removed_signature_conflict_new_gid_allocated",
            evidence_score=float(new_track.evidence_score),
            box=new_track.box,
            extra={
                "blocked_gid": int(conflict["removed_gid"]),
                "new_gid": int(new_track.gid),
                "guard_iou": float(conflict["iou"]),
                "guard_center_distance": float(conflict["center_distance"]),
                "signature_age": int(conflict["signature_age"]),
                "removed_signature": conflict["removed_signature"],
            },
        )

    def _apply_matched_observation(
        self,
        track: EvidenceTrack,
        group: _ObservationGroup,
        frame_idx: int,
        positive_score: float,
    ) -> None:
        old_box = _box_array(track.box)
        new_box = group.box
        old_center = np.array([(old_box[0] + old_box[2]) / 2.0, (old_box[1] + old_box[3]) / 2.0])
        new_center = np.array([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])

        refresh_primary_box = self._matched_group_refreshes_primary_box(track, group)
        if refresh_primary_box:
            track.velocity = [float(new_center[0] - old_center[0]), float(new_center[1] - old_center[1])]
            track.box = new_box
        track.evidence_score = self._rounded_score(
            float(self._cfg("MSDC_EVIDENCE_ALPHA", 0.85)) * float(track.evidence_score) + positive_score
        )
        track.hits = int(track.hits) + 1
        track.misses = 0
        track.last_seen = int(frame_idx)
        track.last_real_det_frame = int(frame_idx)
        track.real_det_hits = int(track.real_det_hits) + 1
        track.last_real_det_box = new_box.copy()
        self._append_low_det_history(track, group, frame_idx)
        self._append_source_history(track, group.source_history)
        if group.class_name != "unknown":
            track.class_id = int(group.class_id)
            track.class_name = str(group.class_name)

    def _matched_group_refreshes_primary_box(self, track: EvidenceTrack, group: _ObservationGroup) -> bool:
        if _as_state(track.state) != TrackState.ACTIVE:
            return True
        sources = set(group.source_scores)
        return bool(sources & {"high_det", "reacquire"})

    def _apply_negative_evidence(self, track: EvidenceTrack) -> None:
        track.misses = int(track.misses) + 1
        track.evidence_score = self._rounded_score(
            max(
                0.0,
                float(self._cfg("MSDC_EVIDENCE_ALPHA", 0.85)) * float(track.evidence_score)
                - float(self._cfg("MSDC_NEGATIVE_WEIGHT", 0.5)),
            )
        )

    def _transition_track(
        self,
        track: EvidenceTrack,
        frame_idx: int,
        matched: bool,
        reacquire_score: float,
    ) -> list[LifecycleEvent]:
        state = _as_state(track.state)
        current_dt = int(frame_idx) - int(track.last_seen)
        target_state = state
        event_type = ""
        reason = ""
        extra = None
        if state == TrackState.LOW_CANDIDATE:
            if self._low_candidate_can_confirm(track, frame_idx):
                target_state = TrackState.ACTIVE
                event_type = "CONFIRM_LOW_ACTIVE"
                reason = "low_temporal_gate_confirmed"
            elif int(track.age) > 1 and (
                int(track.misses) > int(self._cfg("MSDC_LOW_CONFIRM_MAX_MISSES", 1))
                or int(track.age) > int(self._cfg("MSDC_LOW_CONFIRM_WINDOW", 8))
                or float(track.evidence_score) < float(self._cfg("MSDC_PRUNE_SCORE", 0.1))
            ):
                target_state = TrackState.REMOVED
                event_type = "PRUNED_LOW_CANDIDATE"
                reason = "low_candidate_pruned"
        elif state == TrackState.CANDIDATE:
            confirm_ready = (
                float(track.evidence_score) >= float(self._cfg("MSDC_CONFIRM_SCORE", 2.5))
                and int(track.hits) >= int(self._cfg("MSDC_CONFIRM_MIN_HITS", 3))
                and self._candidate_has_required_detection(track)
            )
            target_state = assign_state(
                p_t=float(track.evidence_score),
                v_t=1.0 if confirm_ready and matched else 0.0,
                m_t=0.0 if confirm_ready else float(self._cfg("MSDC_STATE_THETA_M", 1.0)),
                dt=current_dt,
                config=self.config,
            )
            if target_state == TrackState.ACTIVE:
                event_type = "CONFIRM_ACTIVE"
                reason = "evidence_confirmed"
            elif int(track.age) > 1 and (
                int(track.age) > int(self._cfg("MSDC_CANDIDATE_MAX_AGE", 5))
                or float(track.evidence_score) < float(self._cfg("MSDC_PRUNE_SCORE", 0.1))
            ):
                target_state = TrackState.REMOVED
                event_type = "PRUNED_CANDIDATE"
                reason = "candidate_pruned"
            else:
                target_state = state
        elif state == TrackState.ACTIVE:
            if int(track.misses) > self._active_missing_patience(track, frame_idx):
                target_state = assign_state(
                    p_t=float(track.evidence_score),
                    v_t=0.0,
                    m_t=0.0,
                    dt=current_dt,
                    config=self.config,
                )
                if target_state == TrackState.LOST:
                    event_type = "ACTIVE_TO_LOST"
                    reason = "missing_patience_exceeded"
                else:
                    target_state = state
        elif state == TrackState.LOST:
            if matched and reacquire_score >= float(self._cfg("MSDC_REACQUIRE_SCORE", 1.0)):
                track.misses = 0
                target_state = assign_state(
                    p_t=max(float(track.evidence_score), float(self._cfg("MSDC_CONFIRM_SCORE", 2.5)) + 1e-6),
                    v_t=1.0,
                    m_t=0.0,
                    dt=current_dt,
                    config=self.config,
                )
                if target_state == TrackState.ACTIVE:
                    event_type = "LOST_REACQUIRED"
                    reason = f"reacquire_score>={float(self._cfg('MSDC_REACQUIRE_SCORE', 1.0))}"
                    extra = {"reacquire_score": float(reacquire_score)}
                else:
                    target_state = state
            else:
                timed_out = current_dt > int(self._cfg("MSDC_LOST_MAX_AGE", 5))
                target_state = assign_state(
                    p_t=0.0 if timed_out else float(track.evidence_score),
                    v_t=0.0,
                    m_t=float(self._cfg("MSDC_STATE_THETA_M", 1.0)),
                    dt=current_dt,
                    config=self.config,
                )
                if target_state == TrackState.REMOVED:
                    event_type = "LOST_TO_removed"
                    reason = "lost_timeout"
                else:
                    target_state = state
        if target_state != state and event_type:
            return [
                self._set_state(
                    track,
                    frame_idx,
                    target_state,
                    event_type,
                    reason,
                    extra=extra,
                )
            ]
        return []

    def _set_state(
        self,
        track: EvidenceTrack,
        frame_idx: int,
        to_state: TrackState,
        event_type: str,
        reason: str,
        extra: dict | None = None,
    ) -> LifecycleEvent:
        from_state = track.state
        if to_state == TrackState.ACTIVE and track.public_id is None:
            track.public_id = self.next_public_id
            self.next_public_id += 1
        if to_state == TrackState.REMOVED:
            track.retired_signature = self._make_removed_signature(track, frame_idx)
        track.state = to_state
        return self._make_event(
            frame_idx=frame_idx,
            track=track,
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            extra=extra,
        )

    def _make_event(
        self,
        frame_idx: int,
        track: EvidenceTrack,
        event_type: str,
        from_state: TrackState | str | None,
        to_state: TrackState | str | None,
        reason: str,
        extra: dict | None = None,
    ) -> LifecycleEvent:
        payload_extra = {
            "public_id": None if track.public_id is None else int(track.public_id),
            "hits": int(track.hits),
            "misses": int(track.misses),
            "age": int(track.age),
            "last_seen": int(track.last_seen),
            "last_real_det_frame": int(track.last_real_det_frame),
            "real_det_hits": int(track.real_det_hits),
            "source_history": list(track.source_history),
        }
        if track.retired_signature is not None:
            payload_extra["removed_signature"] = track.retired_signature
        if extra:
            payload_extra.update(extra)
        return LifecycleEvent(
            frame_idx=int(frame_idx),
            gid=int(track.gid),
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            evidence_score=float(track.evidence_score),
            box=track.box,
            extra=payload_extra,
        )

    def _evidence_increment(self, group: _ObservationGroup) -> float:
        total = 0.0
        for source, score in group.source_scores.items():
            total += self._source_weight(source) * float(score)
        return self._rounded_score(total)

    def _source_weight(self, source: str) -> float:
        mapping = {
            "high_det": "MSDC_WEIGHT_HIGH",
            "low_det": "MSDC_WEIGHT_LOW",
            "reacquire": "MSDC_WEIGHT_REACQUIRE",
        }
        return float(self._cfg(mapping.get(source, "MSDC_WEIGHT_LOW"), 1.0))

    def _predict_box(self, track: EvidenceTrack) -> np.ndarray:
        box = _box_array(track.box).copy()
        velocity = np.asarray(track.velocity, dtype=np.float64).reshape(-1)
        if velocity.size >= 2:
            box[[0, 2]] += float(velocity[0])
            box[[1, 3]] += float(velocity[1])
        return box

    def _append_source_history(self, track: EvidenceTrack, sources: list[str]) -> None:
        history = list(track.source_history)
        history.extend(str(source) for source in sources)
        max_size = int(self._cfg("MSDC_SOURCE_HISTORY_SIZE", 16))
        track.source_history = history[-max_size:]

    def _append_low_det_history(self, track: EvidenceTrack, group: _ObservationGroup, frame_idx: int) -> None:
        low_history = self._low_history_from_group(group, frame_idx)
        if not low_history:
            return
        history = list(getattr(track, "low_det_history", []) or [])
        history.extend(low_history)
        max_size = max(
            int(self._cfg("MSDC_LOW_CONFIRM_WINDOW", 8)),
            int(self._cfg("MSDC_LOW_CONFIRM_MIN_HITS", 5)),
        )
        track.low_det_history = history[-max_size:]

    def _low_candidate_can_confirm(self, track: EvidenceTrack, frame_idx: int) -> bool:
        if not bool(self._cfg("MSDC_LOW_CANDIDATE_ENABLE", True)):
            return False
        if int(track.misses) > int(self._cfg("MSDC_LOW_CONFIRM_MAX_MISSES", 1)):
            return False
        history = [
            item
            for item in list(getattr(track, "low_det_history", []) or [])
            if int(frame_idx) - int(item.get("frame_idx", frame_idx)) < int(self._cfg("MSDC_LOW_CONFIRM_WINDOW", 8))
        ]
        min_hits = int(self._cfg("MSDC_LOW_CONFIRM_MIN_HITS", 5))
        if len(history) < min_hits:
            return False
        avg_score = sum(float(item.get("score", 0.0)) for item in history) / max(1, len(history))
        if avg_score < float(self._cfg("MSDC_LOW_CONFIRM_MIN_AVG_SCORE", 0.22)):
            return False
        if not self._low_history_area_stable(history):
            return False
        if not self._low_history_center_step_stable(history):
            return False
        return True

    def _active_missing_patience(self, track: EvidenceTrack, frame_idx: int) -> int:
        return int(self._cfg("MSDC_ACTIVE_MISSING_PATIENCE", 2))

    @staticmethod
    def _group_has_min_size(group: _ObservationGroup, min_size: float) -> bool:
        box = group.box
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        return bool(width >= float(min_size) and height >= float(min_size))

    def _low_history_area_stable(self, history: list[dict]) -> bool:
        areas = [float(item.get("area", 0.0)) for item in history if float(item.get("area", 0.0)) > 0.0]
        if len(areas) < 2:
            return True
        ratio = max(areas) / max(min(areas), 1e-9)
        return bool(ratio <= float(self._cfg("MSDC_LOW_CONFIRM_MAX_AREA_CHANGE", 1.8)))

    def _low_history_center_step_stable(self, history: list[dict]) -> bool:
        if len(history) < 4:
            return True
        centers = []
        for item in history:
            center = item.get("center", [])
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                return False
            centers.append(np.asarray(center[:2], dtype=np.float64))
        steps = [float(np.linalg.norm(centers[idx] - centers[idx - 1])) for idx in range(1, len(centers))]
        if not steps:
            return True
        median_step = float(np.median(steps))
        max_factor = float(self._cfg("MSDC_LOW_CONFIRM_MAX_CENTER_STEP_FACTOR", 3.0))
        tolerance = float(self._cfg("MSDC_ASSOC_CENTER_DIST", 80.0))
        return bool(max(steps) <= max(tolerance, median_step * max_factor))

    def _candidate_has_required_detection(self, track: EvidenceTrack) -> bool:
        if not bool(self._cfg("MSDC_CONFIRM_REQUIRE_DET", True)):
            return True
        min_hits = int(self._cfg(
            "MSDC_CONFIRM_MIN_REAL_DET_HITS",
            self._cfg("MSDC_CONFIRM_MIN_DET_HITS", 2),
        ))
        if int(track.real_det_hits) < min_hits:
            return False
        if bool(self._cfg("MSDC_CONFIRM_REQUIRE_HIGH_DET", True)):
            return any(str(source) == "high_det" for source in track.source_history)
        return True

    def _ensure_next_gid(self, tracks: list[EvidenceTrack]) -> None:
        if not tracks:
            return
        max_gid = max(int(track.gid) for track in tracks)
        self.next_gid = max(self.next_gid, max_gid + 1)

    def _ensure_next_public_id(self, tracks: list[EvidenceTrack]) -> None:
        public_ids = [int(track.public_id) for track in tracks if track.public_id is not None]
        if not public_ids:
            return
        self.next_public_id = max(self.next_public_id, max(public_ids) + 1)

    def _empty_reacquire_debug(self, frame_idx: int | None = None) -> dict:
        return {
            "enabled": int(self._cfg("MSDC_REACQUIRE_INTERVAL", 0)) > 0,
            "use_reacquire": bool(self._cfg("MSDC_USE_REACQUIRE", True)),
            "interval": int(self._cfg("MSDC_REACQUIRE_INTERVAL", 0)),
            "attempted": False,
            "frame_idx": None if frame_idx is None else int(frame_idx),
            "num_lost": 0,
            "num_groups": 0,
            "num_matches": 0,
            "matches": [],
            "num_suppressed_spawn_groups": 0,
            "suppressed_spawn_groups": [],
        }

    def _empty_low_inherit_debug(self, frame_idx: int | None = None) -> dict:
        return {
            "enabled": bool(self._cfg("MSDC_LOW_INHERIT_ENABLE", True)),
            "frame_idx": None if frame_idx is None else int(frame_idx),
            "num_ready_low_candidates": 0,
            "num_lost": 0,
            "num_matches": 0,
            "matches": [],
            "num_active_conflicts": 0,
            "active_conflicts": [],
        }

    def _empty_removed_guard_debug(self, enabled: bool | None = None) -> dict:
        if enabled is None:
            enabled = bool(self._cfg("MSDC_REUSE_GUARD_ENABLE", True)) and int(self._cfg(
                "MSDC_removed_GUARD_FRAMES",
                self._cfg("MSDC_REMOVED_GUARD_FRAMES", 120),
            )) > 0
        return {
            "enabled": bool(enabled),
            "guard_frames": int(self._cfg(
                "MSDC_removed_GUARD_FRAMES",
                self._cfg("MSDC_REMOVED_GUARD_FRAMES", 120),
            )),
            "num_recent_signatures": 0,
            "num_vetoes": 0,
            "vetoes": [],
        }

    def _empty_spawn_suppression_debug(self, frame_idx: int | None = None) -> dict:
        return {
            "enabled": bool(self._cfg("MSDC_SPAWN_SUPPRESS_ENABLE", True)),
            "frame_idx": None if frame_idx is None else int(frame_idx),
            "num_groups": 0,
            "num_suppressed_spawn_groups": 0,
            "suppressed_spawn_groups": [],
        }

    def _cfg(self, name: str, default):
        return getattr(self.config, name, default)

    @staticmethod
    def _rounded_score(value: float) -> float:
        return round(float(value), 4)
