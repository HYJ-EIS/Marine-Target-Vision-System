"""
MS-DC-ELT-lite tracker coordinator.

This class wires high-threshold detections, low-only detections, optional motion
seeds, active-only template observations, low-frequency lost reacquire,
removed-ID guard diagnostics, and EvidenceStateUpdater together. The formal v3
variant keeps motion seed disabled and relies on the bounded low-det candidate
pool for re-capture candidates.
"""

from __future__ import annotations

import heapq
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.evidence_state import EvidenceStateUpdater
from target_module.image_detect_module.utils.motion_seed import MotionSeedGenerator
from target_module.image_detect_module.utils.msdc_types import (
    EvidenceTrack,
    Observation,
    TrackState,
    event_to_dict,
    observation_from_project_box,
    track_to_dict,
    track_to_output_box,
    xywh_to_xyxy,
    xyxy_to_project_box,
)
from target_module.image_detect_module.utils.roi_redetect import ROIRedetector
from target_module.image_detect_module.utils.template_lock import TemplateLock


class MSDCLifecycleTracker:
    def __init__(self, config=None, file_type: str | None = None, debug_dir: str | Path | None = None, processor=None):
        self.config = config or Config
        self.file_type = file_type
        self.tracks: list[EvidenceTrack] = []
        self.motion_seed = MotionSeedGenerator(self.config)
        self.roi_redetector = ROIRedetector(self.config, processor=processor)
        self.evidence_updater = EvidenceStateUpdater(self.config)
        self.template_lock = TemplateLock(self.config)
        self.output_candidates = bool(getattr(self.config, "MSDC_OUTPUT_CANDIDATES", False))
        self.debug_events = bool(getattr(self.config, "MSDC_DEBUG_EVENTS", False))
        self.debug_dir = self._resolve_debug_dir(debug_dir)
        self.last_events = []
        self.last_debug_info: dict[str, Any] = {}
        self.last_motion_debug: dict[str, Any] = {}
        self.last_template_debug: dict[str, Any] = self._empty_template_debug()
        self.last_reacquire_debug: dict[str, Any] = {}
        self.last_low_inherit_debug: dict[str, Any] = {}
        self.last_removed_guard_debug: dict[str, Any] = {}
        self.last_spawn_suppression_debug: dict[str, Any] = {}
        self.last_output_nms_debug: dict[str, Any] = self._empty_output_nms_debug()
        self.last_roi_redetect_debug: dict[str, Any] = {}
        self.last_low_observation_budget_debug: dict[str, Any] = self._empty_low_observation_budget_debug()
        self.last_timing_debug: dict[str, float] = self._empty_timing_debug()

        if self.debug_events and self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
        file_type: str,
        high_boxes: list[dict] | None = None,
        low_boxes: list[dict] | None = None,
    ) -> list[dict]:
        """
        Update the lifecycle tracker and return project-compatible active boxes.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("MSDCLifecycleTracker.update expects a non-empty numpy frame")

        idx = int(frame_idx)
        modality = str(file_type or self.file_type or "unknown")
        self.file_type = modality
        update_start = time.perf_counter()
        step_start = update_start

        high_boxes = list(high_boxes or [])
        low_boxes = list(low_boxes or [])
        raw_low_only_boxes = self._filter_low_only_boxes(high_boxes, low_boxes)
        low_only_boxes = self._budget_low_only_boxes(raw_low_only_boxes, self.tracks)
        low_filter_s = time.perf_counter() - step_start

        step_start = time.perf_counter()
        roi_low_boxes, roi_debug = self.roi_redetector.update(
            frame=frame,
            tracks=self.tracks,
            frame_idx=idx,
            file_type=modality,
            existing_boxes=high_boxes + low_boxes,
        )
        self.last_roi_redetect_debug = dict(roi_debug or {})
        roi_redetect_s = time.perf_counter() - step_start

        step_start = time.perf_counter()
        if bool(getattr(self.config, "MSDC_USE_MOTION", True)):
            motion_boxes, motion_debug = self.motion_seed.update(frame, frame_idx=idx)
        else:
            motion_boxes, motion_debug = [], {"frame_idx": idx, "num_motion_boxes": 0, "disabled": True}
        self.last_motion_debug = dict(motion_debug or {})
        motion_s = time.perf_counter() - step_start

        step_start = time.perf_counter()
        observations = []
        observations.extend(self._boxes_to_observations(high_boxes, "high_det", idx, modality))
        observations.extend(self._boxes_to_observations(low_only_boxes, "low_det", idx, modality))
        observations.extend(self._boxes_to_observations(roi_low_boxes, "roi_low_det", idx, modality))
        observations.extend(self._boxes_to_observations(motion_boxes, "motion", idx, modality))
        observation_build_s = time.perf_counter() - step_start

        step_start = time.perf_counter()
        template_observations = []
        if self._template_enabled():
            template_observations = self.template_lock.match_active_tracks(self.tracks, frame, idx, modality=modality)
            if self.evidence_updater.should_reacquire_frame(idx):
                template_observations.extend(self.template_lock.match_lost_tracks(self.tracks, frame, idx, modality=modality))
        observations.extend(template_observations)
        template_match_s = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self.tracks, events = self.evidence_updater.update_tracks(self.tracks, observations, idx)
        self.last_events = events
        evidence_update_s = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self.last_template_debug = self._sync_active_templates(frame, template_observations, idx)
        self.last_reacquire_debug = dict(getattr(self.evidence_updater, "last_reacquire_debug", {}) or {})
        self.last_low_inherit_debug = dict(getattr(self.evidence_updater, "last_low_inherit_debug", {}) or {})
        self.last_removed_guard_debug = dict(getattr(self.evidence_updater, "last_removed_guard_debug", {}) or {})
        self.last_spawn_suppression_debug = dict(getattr(
            self.evidence_updater,
            "last_spawn_suppression_debug",
            {},
        ) or {})
        template_sync_s = time.perf_counter() - step_start

        step_start = time.perf_counter()
        output_boxes = self._tracks_to_output_boxes(self.tracks, frame_idx=idx)
        output_s = time.perf_counter() - step_start

        step_start = time.perf_counter()
        if self.debug_events and self.debug_dir is not None:
            self.last_debug_info = self._frame_debug_info(
                frame_idx=idx,
                file_type=modality,
                high_boxes=high_boxes,
                low_boxes=low_boxes,
                low_only_boxes=low_only_boxes,
                roi_low_boxes=roi_low_boxes,
                motion_boxes=motion_boxes,
                template_observations=template_observations,
                observations=observations,
                output_boxes=output_boxes,
            )
            self._write_debug(events, self.last_debug_info)
        else:
            self.last_debug_info = self._minimal_frame_debug_info(
                frame_idx=idx,
                file_type=modality,
                high_boxes=high_boxes,
                low_boxes=low_boxes,
                low_only_boxes=low_only_boxes,
                roi_low_boxes=roi_low_boxes,
                motion_boxes=motion_boxes,
                template_observations=template_observations,
                observations=observations,
                output_boxes=output_boxes,
            )
        debug_s = time.perf_counter() - step_start
        self.last_timing_debug = {
            "low_filter_s": float(low_filter_s),
            "roi_redetect_s": float(roi_redetect_s),
            "motion_s": float(motion_s),
            "observation_build_s": float(observation_build_s),
            "template_match_s": float(template_match_s),
            "evidence_update_s": float(evidence_update_s),
            "template_sync_s": float(template_sync_s),
            "output_s": float(output_s),
            "debug_s": float(debug_s),
            "total_update_s": float(time.perf_counter() - update_start),
        }
        return output_boxes

    def reset(self) -> None:
        self.tracks = []
        self.motion_seed.reset()
        self.roi_redetector.reset()
        self.evidence_updater = EvidenceStateUpdater(self.config)
        self.template_lock.reset()
        self.last_events = []
        self.last_debug_info = {}
        self.last_motion_debug = {}
        self.last_template_debug = self._empty_template_debug()
        self.last_reacquire_debug = {}
        self.last_low_inherit_debug = {}
        self.last_removed_guard_debug = {}
        self.last_spawn_suppression_debug = {}
        self.last_output_nms_debug = self._empty_output_nms_debug()
        self.last_roi_redetect_debug = {}
        self.last_low_observation_budget_debug = self._empty_low_observation_budget_debug()
        self.last_timing_debug = self._empty_timing_debug()

    @staticmethod
    def _empty_timing_debug() -> dict[str, float]:
        return {
            "low_filter_s": 0.0,
            "roi_redetect_s": 0.0,
            "motion_s": 0.0,
            "observation_build_s": 0.0,
            "template_match_s": 0.0,
            "evidence_update_s": 0.0,
            "template_sync_s": 0.0,
            "output_s": 0.0,
            "debug_s": 0.0,
            "total_update_s": 0.0,
        }

    def _tracks_to_output_boxes(self, tracks: list[EvidenceTrack], frame_idx: int | None = None) -> list[dict]:
        active_boxes = []
        candidate_boxes = []
        for track in tracks:
            state = track.state if isinstance(track.state, TrackState) else TrackState(str(track.state))
            if state == TrackState.ACTIVE:
                output_box = self._track_to_output_box(track, frame_idx=frame_idx)
                if output_box is not None:
                    active_boxes.append(output_box)
            elif self.output_candidates and state == TrackState.CANDIDATE:
                candidate_boxes.append(track_to_output_box(track))

        active_boxes = self._nms_output_boxes(active_boxes)
        return active_boxes + candidate_boxes

    def _track_to_output_box(self, track: EvidenceTrack, frame_idx: int | None = None) -> dict | None:
        if float(track.evidence_score) <= 0.0:
            return None
        age_since_real = int(getattr(track, "last_real_det_frame", -1))
        if age_since_real < 0:
            return None
        if frame_idx is None:
            frame_idx = int(getattr(track, "last_seen", -1))
        real_age = max(0, frame_idx - int(track.last_real_det_frame))
        max_real_age = int(getattr(self.config, "MSDC_OUTPUT_MAX_REAL_DET_AGE", 3))
        if real_age > max_real_age:
            return None

        box = self._predict_track_box_for_output(track, real_age)
        if not self._valid_output_xyxy(box):
            return None
        output_id = track.public_id if track.public_id is not None else track.gid
        return xyxy_to_project_box(
            box,
            track_id=output_id,
            confidence=track.evidence_score,
            cls=track.class_name,
            class_confidence=track.evidence_score,
            gid=track.gid,
            public_id=track.public_id,
            lifecycle_state=self._track_state(track).value,
            real_det_age=real_age,
        )

    def _predict_track_box_for_output(self, track: EvidenceTrack, real_age: int) -> np.ndarray:
        box = np.asarray(track.box, dtype=np.float64).reshape(4).copy()
        if real_age <= 0:
            return box
        velocity = np.asarray(track.velocity, dtype=np.float64).reshape(-1)
        if velocity.size >= 2:
            box[[0, 2]] += float(velocity[0]) * float(real_age)
            box[[1, 3]] += float(velocity[1]) * float(real_age)
        return box

    def _valid_output_xyxy(self, box: np.ndarray) -> bool:
        min_size = int(getattr(self.config, "MSDC_OUTPUT_MIN_BOX_SIZE", 12))
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        return bool(width >= min_size and height >= min_size)

    def _nms_output_boxes(self, boxes: list[dict]) -> list[dict]:
        enabled = bool(getattr(self.config, "MSDC_OUTPUT_NMS_ENABLE", True))
        if not enabled or len(boxes) <= 1:
            self.last_output_nms_debug = {
                "enabled": enabled,
                "num_input": int(len(boxes)),
                "num_output": int(len(boxes)),
                "num_suppressed": 0,
                "suppressed": [],
            }
            return boxes

        iou_thresh = float(getattr(self.config, "MSDC_OUTPUT_NMS_IOU", 0.3))
        center_thresh = float(getattr(self.config, "MSDC_OUTPUT_NMS_CENTER_DIST", 30.0))
        fragment_ratio = float(getattr(self.config, "MSDC_OUTPUT_NMS_FRAGMENT_AREA_RATIO", 0.35))
        containment_thresh = float(getattr(self.config, "MSDC_OUTPUT_NMS_CONTAINMENT_RATIO", 0.50))
        order = sorted(range(len(boxes)), key=lambda idx: float(boxes[idx].get("confidence", 0.0)), reverse=True)
        kept_indices = []
        suppressed = []

        for idx in order:
            box = boxes[idx]
            conflict = None
            for kept_idx in kept_indices:
                kept_box = boxes[kept_idx]
                iou_score = self._box_iou(box, kept_box)
                center_dist = self._box_center_distance(box, kept_box)
                containment = self._box_containment(box, kept_box)
                area_ratio = self._box_area(box) / max(self._box_area(kept_box), 1e-10)
                is_fragment = (
                    area_ratio <= fragment_ratio
                    and (containment >= containment_thresh or center_dist <= center_thresh)
                )
                if (
                    iou_score >= iou_thresh
                    or (center_thresh > 0.0 and center_dist <= center_thresh and area_ratio <= 1.2)
                    or is_fragment
                ):
                    conflict = {
                        "track_id": int(box.get("track_id", -1)),
                        "gid": int(box.get("gid", box.get("track_id", -1))),
                        "kept_track_id": int(kept_box.get("track_id", -1)),
                        "kept_gid": int(kept_box.get("gid", kept_box.get("track_id", -1))),
                        "iou": float(round(iou_score, 4)),
                        "center_distance": float(round(center_dist, 4)),
                        "containment": float(round(containment, 4)),
                        "area_ratio": float(round(area_ratio, 4)),
                        "fragment": bool(is_fragment),
                    }
                    break
            if conflict is None:
                kept_indices.append(idx)
            else:
                suppressed.append(conflict)

        kept = [boxes[idx] for idx in kept_indices]
        self.last_output_nms_debug = {
            "enabled": True,
            "iou_thresh": float(iou_thresh),
            "center_dist_thresh": float(center_thresh),
            "fragment_area_ratio": float(fragment_ratio),
            "containment_thresh": float(containment_thresh),
            "num_input": int(len(boxes)),
            "num_output": int(len(kept)),
            "num_suppressed": int(len(suppressed)),
            "suppressed": self._limit_debug_list(suppressed),
        }
        return kept

    @staticmethod
    def _box_iou(a: dict, b: dict) -> float:
        ax1, ay1, ax2, ay2 = float(a.get("x", 0.0)), float(a.get("y", 0.0)), 0.0, 0.0
        bx1, by1, bx2, by2 = float(b.get("x", 0.0)), float(b.get("y", 0.0)), 0.0, 0.0
        ax2 = ax1 + max(0.0, float(a.get("w", 0.0)))
        ay2 = ay1 + max(0.0, float(a.get("h", 0.0)))
        bx2 = bx1 + max(0.0, float(b.get("w", 0.0)))
        by2 = by1 + max(0.0, float(b.get("h", 0.0)))
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return float(inter / max(area_a + area_b - inter, 1e-10))

    @staticmethod
    def _box_area(box: dict) -> float:
        return float(max(0.0, float(box.get("w", 0.0))) * max(0.0, float(box.get("h", 0.0))))

    @staticmethod
    def _box_containment(a: dict, b: dict) -> float:
        ax1 = float(a.get("x", 0.0))
        ay1 = float(a.get("y", 0.0))
        ax2 = ax1 + max(0.0, float(a.get("w", 0.0)))
        ay2 = ay1 + max(0.0, float(a.get("h", 0.0)))
        bx1 = float(b.get("x", 0.0))
        by1 = float(b.get("y", 0.0))
        bx2 = bx1 + max(0.0, float(b.get("w", 0.0)))
        by2 = by1 + max(0.0, float(b.get("h", 0.0)))
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        return float(inter / max((ax2 - ax1) * (ay2 - ay1), 1e-10))

    @staticmethod
    def _box_center_distance(a: dict, b: dict) -> float:
        acx = float(a.get("x", 0.0)) + 0.5 * float(a.get("w", 0.0))
        acy = float(a.get("y", 0.0)) + 0.5 * float(a.get("h", 0.0))
        bcx = float(b.get("x", 0.0)) + 0.5 * float(b.get("w", 0.0))
        bcy = float(b.get("y", 0.0)) + 0.5 * float(b.get("h", 0.0))
        return float(np.hypot(acx - bcx, acy - bcy))

    def _boxes_to_observations(
        self,
        boxes: list[dict],
        source: str,
        frame_idx: int,
        file_type: str,
    ) -> list[Observation]:
        observations = []
        class_to_id = {name: idx for idx, name in enumerate(getattr(self.config, "CLASSES", []))}
        for box in boxes:
            if not self._is_valid_project_box(box):
                continue
            class_name = str(box.get("class", "unknown"))
            class_id = class_to_id.get(class_name, -1)
            observations.append(observation_from_project_box(
                box=box,
                source=source,
                frame_idx=frame_idx,
                modality=file_type,
                reliability=1.0,
                class_id=class_id,
            ))
        return observations

    def _filter_low_only_boxes(self, high_boxes: list[dict], low_boxes: list[dict]) -> list[dict]:
        valid_low = [box for box in low_boxes if self._is_valid_project_box(box)]
        valid_high = [box for box in high_boxes if self._is_valid_project_box(box)]
        if not valid_high:
            return valid_low

        high_xyxy = np.vstack([xywh_to_xyxy(box) for box in valid_high])
        threshold = float(getattr(self.config, "MSDC_LOW_HIGH_IOU_THRESH", 0.5))
        low_only = []
        for box in valid_low:
            low_xyxy = xywh_to_xyxy(box)
            if float(np.max(self._iou_one_to_many(low_xyxy, high_xyxy))) < threshold:
                low_only.append(box)
        return low_only

    def _budget_low_only_boxes(self, low_only_boxes: list[dict], tracks: list[EvidenceTrack]) -> list[dict]:
        topk = max(0, int(getattr(self.config, "MSDC_LOW_OBS_TOPK", 0)))
        global_topk = max(0, int(getattr(self.config, "MSDC_LOW_OBS_GLOBAL_TOPK", topk)))
        per_track_nearest = max(0, int(getattr(self.config, "MSDC_LOW_OBS_PER_TRACK_NEAREST", 1)))
        max_per_frame = max(0, int(getattr(self.config, "MSDC_LOW_OBS_MAX_PER_FRAME", 0)))
        min_conf = float(getattr(self.config, "MSDC_LOW_OBS_MIN_CONF", 0.0))
        valid = [box for box in low_only_boxes if self._is_valid_project_box(box)]
        enabled = bool(topk > 0 or min_conf > 0.0)
        after_min_conf = [
            box for box in valid
            if float(box.get("confidence", box.get("score", 0.0))) >= min_conf
        ] if enabled else list(valid)
        proximity_enabled = bool(getattr(self.config, "MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY", True))
        proximity_tracks = [
            track
            for track in tracks
            if self._track_state(track) in {TrackState.ACTIVE, TrackState.LOST}
        ]
        after_gate = after_min_conf
        gate_mask = np.ones(len(after_min_conf), dtype=bool)
        iou_matrix = np.zeros((len(after_min_conf), len(proximity_tracks)), dtype=np.float64)
        distance_matrix = np.zeros((len(after_min_conf), len(proximity_tracks)), dtype=np.float64)
        if proximity_enabled and proximity_tracks and after_min_conf:
            gate_mask, iou_matrix, distance_matrix = self._low_boxes_motion_gate(after_min_conf, proximity_tracks)
            after_gate = [
                box
                for box, keep in zip(after_min_conf, gate_mask)
                if bool(keep)
            ]

        gated_original_indices = [idx for idx, keep in enumerate(gate_mask.tolist()) if bool(keep)]
        score_key = lambda box: float(box.get("confidence", box.get("score", 0.0)))
        selected_indices: list[int] = []
        kept_global_count = 0
        kept_nearest_count = 0
        if global_topk > 0:
            ranked = heapq.nlargest(global_topk, gated_original_indices, key=lambda idx: score_key(after_min_conf[idx]))
            selected_indices.extend(ranked)
            kept_global_count = len(ranked)
        elif enabled:
            ranked = sorted(gated_original_indices, key=lambda idx: score_key(after_min_conf[idx]), reverse=True)
            selected_indices.extend(ranked)
            kept_global_count = len(ranked)
        elif enabled:
            selected_indices.extend(gated_original_indices)
            kept_global_count = len(gated_original_indices)
        else:
            selected_indices.extend(gated_original_indices)
            kept_global_count = len(gated_original_indices)

        if per_track_nearest > 0 and proximity_enabled and proximity_tracks and gated_original_indices:
            gated_set = set(gated_original_indices)
            for track_col in range(len(proximity_tracks)):
                ordered_by_distance = sorted(
                    gated_original_indices,
                    key=lambda idx: (
                        float(distance_matrix[idx, track_col]),
                        -float(iou_matrix[idx, track_col]),
                        -score_key(after_min_conf[idx]),
                    ),
                )
                for idx in ordered_by_distance[:per_track_nearest]:
                    if idx in gated_set:
                        selected_indices.append(idx)
                        kept_nearest_count += 1

        deduped_indices = list(dict.fromkeys(selected_indices))
        if max_per_frame > 0 and len(deduped_indices) > max_per_frame:
            deduped_indices = deduped_indices[:max_per_frame]
        output = [after_min_conf[idx] for idx in deduped_indices]
        ordered_count = len(after_gate)
        self.last_low_observation_budget_debug = {
            "enabled": enabled,
            "topk": int(topk),
            "global_topk": int(global_topk),
            "per_track_nearest": int(per_track_nearest),
            "max_per_frame": int(max_per_frame),
            "min_conf": float(min_conf),
            "track_proximity_enabled": bool(proximity_enabled),
            "track_proximity_center_dist": float(getattr(self.config, "MSDC_LOW_OBS_MOTION_GATE_CENTER_DIST", getattr(self.config, "MSDC_LOW_OBS_TRACK_PROXIMITY_CENTER_DIST", 240.0))),
            "track_proximity_iou": float(getattr(self.config, "MSDC_LOW_OBS_MOTION_GATE_IOU", getattr(self.config, "MSDC_LOW_OBS_TRACK_PROXIMITY_IOU", 0.01))),
            "track_proximity_track_count": int(len(proximity_tracks)),
            "num_input": int(len(low_only_boxes)),
            "num_valid": int(len(valid)),
            "num_after_min_conf": int(len(after_min_conf)),
            "num_after_track_proximity": int(len(after_gate)),
            "num_after_motion_gate": int(len(after_gate)),
            "num_output": int(len(output)),
            "num_filtered_invalid": int(len(low_only_boxes) - len(valid)),
            "num_filtered_min_conf": int(len(valid) - len(after_min_conf)),
            "num_filtered_track_proximity": int(len(after_min_conf) - len(after_gate)),
            "num_filtered_motion_gate": int(len(after_min_conf) - len(after_gate)),
            "num_filtered_topk": int(max(0, ordered_count - len(output))),
            "num_kept_global_topk": int(kept_global_count),
            "num_kept_per_track_nearest": int(kept_nearest_count),
            "active_track_count": int(sum(1 for track in tracks if self._track_state(track) == TrackState.ACTIVE)),
            "lost_track_count": int(sum(1 for track in tracks if self._track_state(track) == TrackState.LOST)),
        }
        return output

    def _low_boxes_motion_gate(self, boxes: list[dict], tracks: list[EvidenceTrack]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not boxes:
            return np.zeros(0, dtype=bool), np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
        if not tracks:
            return np.ones(len(boxes), dtype=bool), np.zeros((len(boxes), 0), dtype=np.float64), np.zeros((len(boxes), 0), dtype=np.float64)
        box_xyxy = np.vstack([xywh_to_xyxy(box) for box in boxes])
        track_xyxy = np.vstack([self._predict_track_xyxy(track) for track in tracks])
        iou = self._iou_matrix(box_xyxy, track_xyxy)
        distances = self._center_distance_matrix(box_xyxy, track_xyxy)
        iou_thresh = float(getattr(
            self.config,
            "MSDC_LOW_OBS_MOTION_GATE_IOU",
            getattr(self.config, "MSDC_LOW_OBS_TRACK_PROXIMITY_IOU", 0.01),
        ))
        center_thresh = float(getattr(
            self.config,
            "MSDC_LOW_OBS_MOTION_GATE_CENTER_DIST",
            getattr(self.config, "MSDC_LOW_OBS_TRACK_PROXIMITY_CENTER_DIST", 240.0),
        ))
        keep_mask = np.logical_or(
            np.max(iou, axis=1) >= iou_thresh,
            np.min(distances, axis=1) <= center_thresh,
        )
        return keep_mask, iou, distances

    def _low_boxes_near_tracks_mask(self, boxes: list[dict], tracks: list[EvidenceTrack]) -> np.ndarray:
        if not boxes or not tracks:
            return np.ones(len(boxes), dtype=bool)
        box_xyxy = np.vstack([xywh_to_xyxy(box) for box in boxes])
        track_xyxy = np.vstack([self._predict_track_xyxy(track) for track in tracks])
        iou = self._iou_matrix(box_xyxy, track_xyxy)
        distances = self._center_distance_matrix(box_xyxy, track_xyxy)
        iou_thresh = float(getattr(self.config, "MSDC_LOW_OBS_TRACK_PROXIMITY_IOU", 0.01))
        center_thresh = float(getattr(self.config, "MSDC_LOW_OBS_TRACK_PROXIMITY_CENTER_DIST", 240.0))
        return np.logical_or(
            np.max(iou, axis=1) >= iou_thresh,
            np.min(distances, axis=1) <= center_thresh,
        )

    @staticmethod
    def _predict_track_xyxy(track: EvidenceTrack) -> np.ndarray:
        box = np.asarray(track.box, dtype=np.float64).reshape(4).copy()
        velocity = np.asarray(track.velocity, dtype=np.float64).reshape(-1)
        if velocity.size >= 2:
            box[[0, 2]] += float(velocity[0])
            box[[1, 3]] += float(velocity[1])
        return box

    def _template_enabled(self) -> bool:
        return bool(getattr(self.config, "MSDC_USE_TEMPLATE", False)) and bool(
            getattr(self.config, "MSDC_TEMPLATE_ENABLE", False)
        )

    @staticmethod
    def _empty_low_observation_budget_debug() -> dict:
        return {
            "enabled": False,
            "topk": 0,
            "global_topk": 0,
            "per_track_nearest": 0,
            "max_per_frame": 0,
            "min_conf": 0.0,
            "num_input": 0,
            "num_valid": 0,
            "num_after_min_conf": 0,
            "num_after_track_proximity": 0,
            "num_after_motion_gate": 0,
            "num_output": 0,
            "num_filtered_invalid": 0,
            "num_filtered_min_conf": 0,
            "num_filtered_track_proximity": 0,
            "num_filtered_motion_gate": 0,
            "num_filtered_topk": 0,
            "num_kept_global_topk": 0,
            "num_kept_per_track_nearest": 0,
            "track_proximity_enabled": False,
            "track_proximity_center_dist": 0.0,
            "track_proximity_iou": 0.0,
            "track_proximity_track_count": 0,
            "active_track_count": 0,
            "lost_track_count": 0,
        }

    @staticmethod
    def _iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if len(left) == 0 or len(right) == 0:
            return np.zeros((len(left), len(right)), dtype=np.float64)
        xx1 = np.maximum(left[:, 0:1], right[:, 0].reshape(1, -1))
        yy1 = np.maximum(left[:, 1:2], right[:, 1].reshape(1, -1))
        xx2 = np.minimum(left[:, 2:3], right[:, 2].reshape(1, -1))
        yy2 = np.minimum(left[:, 3:4], right[:, 3].reshape(1, -1))
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        left_area = np.maximum(0.0, left[:, 2] - left[:, 0]) * np.maximum(0.0, left[:, 3] - left[:, 1])
        right_area = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(0.0, right[:, 3] - right[:, 1])
        union = left_area[:, None] + right_area[None, :] - inter
        return inter / np.maximum(union, 1e-10)

    @staticmethod
    def _center_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if len(left) == 0 or len(right) == 0:
            return np.zeros((len(left), len(right)), dtype=np.float64)
        left_cx = (left[:, 0] + left[:, 2]) / 2.0
        left_cy = (left[:, 1] + left[:, 3]) / 2.0
        right_cx = (right[:, 0] + right[:, 2]) / 2.0
        right_cy = (right[:, 1] + right[:, 3]) / 2.0
        return np.sqrt((left_cx[:, None] - right_cx[None, :]) ** 2 + (left_cy[:, None] - right_cy[None, :]) ** 2)

    @staticmethod
    def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros(0, dtype=np.float64)
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter
        return inter / np.maximum(union, 1e-10)

    @staticmethod
    def _is_valid_project_box(box: dict) -> bool:
        if not isinstance(box, dict):
            return False
        try:
            xyxy = xywh_to_xyxy(box)
        except (TypeError, ValueError):
            return False
        return bool(xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1])

    def _frame_debug_info(
        self,
        frame_idx: int,
        file_type: str,
        high_boxes: list[dict],
        low_boxes: list[dict],
        low_only_boxes: list[dict],
        roi_low_boxes: list[dict],
        motion_boxes: list[dict],
        template_observations: list[Observation],
        observations: list[Observation],
        output_boxes: list[dict],
    ) -> dict:
        active_count = sum(1 for track in self.tracks if self._track_state(track) == TrackState.ACTIVE)
        candidate_count = sum(1 for track in self.tracks if self._track_state(track) == TrackState.CANDIDATE)
        low_candidate_count = sum(1 for track in self.tracks if self._track_state(track) == TrackState.LOW_CANDIDATE)
        lost_count = sum(1 for track in self.tracks if self._track_state(track) == TrackState.LOST)
        removed_count = sum(1 for track in self.tracks if self._track_state(track) == TrackState.REMOVED)

        return {
            "frame_idx": int(frame_idx),
            "file_type": str(file_type),
            "num_high": int(len(high_boxes)),
            "num_low": int(len(low_boxes)),
            "num_low_only": int(len(low_only_boxes)),
            "num_roi_low": int(len(roi_low_boxes)),
            "num_motion": int(len(motion_boxes)),
            "num_template": int(len(template_observations)),
            "num_observations": int(len(observations)),
            "num_events": int(len(self.last_events)),
            "active_track_count": int(active_count),
            "candidate_track_count": int(candidate_count),
            "low_candidate_track_count": int(low_candidate_count),
            "lost_track_count": int(lost_count),
            "removed_track_count": int(removed_count),
            "output_box_count": int(len(output_boxes)),
            "motion_debug": self.last_motion_debug,
            "roi_redetect_debug": self._compact_roi_redetect_debug(self.last_roi_redetect_debug),
            "low_observation_budget_debug": dict(self.last_low_observation_budget_debug),
            "template_score": self.last_template_debug["template_score"],
            "template_updated": self.last_template_debug["template_updated"],
            "template_match_box": self.last_template_debug["template_match_box"],
            "template_debug": self._compact_template_debug(self.last_template_debug),
            "reacquire_debug": self._compact_reacquire_debug(self.last_reacquire_debug),
            "low_inherit_debug": self._compact_low_inherit_debug(self.last_low_inherit_debug),
            "removed_guard_debug": self._compact_removed_guard_debug(self.last_removed_guard_debug),
            "spawn_suppression_debug": self._compact_spawn_suppression_debug(self.last_spawn_suppression_debug),
            "output_nms_debug": self._compact_output_nms_debug(self.last_output_nms_debug),
            "tracks": self._debug_track_snapshot(),
            "output_boxes": output_boxes,
            "_high_boxes": high_boxes,
            "_low_boxes": low_boxes,
            "_low_only_boxes": low_only_boxes,
            "_roi_low_boxes": roi_low_boxes,
        }

    def _minimal_frame_debug_info(
        self,
        frame_idx: int,
        file_type: str,
        high_boxes: list[dict],
        low_boxes: list[dict],
        low_only_boxes: list[dict],
        roi_low_boxes: list[dict],
        motion_boxes: list[dict],
        template_observations: list[Observation],
        observations: list[Observation],
        output_boxes: list[dict],
    ) -> dict:
        active_count = sum(1 for track in self.tracks if self._track_state(track) == TrackState.ACTIVE)
        candidate_count = sum(1 for track in self.tracks if self._track_state(track) == TrackState.CANDIDATE)
        low_candidate_count = sum(1 for track in self.tracks if self._track_state(track) == TrackState.LOW_CANDIDATE)
        lost_count = sum(1 for track in self.tracks if self._track_state(track) == TrackState.LOST)
        return {
            "debug_events": False,
            "frame_idx": int(frame_idx),
            "file_type": str(file_type),
            "num_high": int(len(high_boxes)),
            "num_low": int(len(low_boxes)),
            "num_low_only": int(len(low_only_boxes)),
            "num_roi_low": int(len(roi_low_boxes)),
            "num_motion": int(len(motion_boxes)),
            "num_template": int(len(template_observations)),
            "num_observations": int(len(observations)),
            "num_events": int(len(self.last_events)),
            "active_track_count": int(active_count),
            "candidate_track_count": int(candidate_count),
            "low_candidate_track_count": int(low_candidate_count),
            "lost_track_count": int(lost_count),
            "output_box_count": int(len(output_boxes)),
            "low_observation_budget_debug": dict(self.last_low_observation_budget_debug),
        }

    @staticmethod
    def _track_state(track: EvidenceTrack) -> TrackState:
        return track.state if isinstance(track.state, TrackState) else TrackState(str(track.state))

    def _write_debug(self, events, frame_info: dict) -> None:
        if not self.debug_events or self.debug_dir is None:
            return
        for event in events:
            payload = event_to_dict(event)
            payload["state"] = payload.get("to_state")
            payload["transition_reason"] = payload.get("reason")
            payload["template_score"] = frame_info.get("template_score")
            payload["template_updated"] = frame_info.get("template_updated", False)
            payload["template_match_box"] = frame_info.get("template_match_box")
            payload["reacquire_debug"] = self._compact_reacquire_debug(frame_info.get("reacquire_debug", {}), include_details=False)
            payload["low_inherit_debug"] = self._compact_low_inherit_debug(frame_info.get("low_inherit_debug", {}), include_details=False)
            payload["removed_guard_debug"] = self._compact_removed_guard_debug(frame_info.get("removed_guard_debug", {}), include_details=False)
            payload["spawn_suppression_debug"] = self._compact_spawn_suppression_debug(frame_info.get("spawn_suppression_debug", {}), include_details=False)
            payload["output_nms_debug"] = self._compact_output_nms_debug(frame_info.get("output_nms_debug", {}), include_details=False)
            self._append_jsonl(self.debug_dir / "lifecycle_events.jsonl", payload)
        self._append_jsonl(self.debug_dir / "msdc_tracks.jsonl", frame_info)
        self._append_jsonl(self.debug_dir / "candidate_pool_stats.jsonl", self._candidate_pool_stats(frame_info))
        self._append_jsonl(self.debug_dir / "stage_observations.jsonl", self._stage_observations(frame_info))

    def _candidate_pool_stats(self, frame_info: dict) -> dict:
        tracks = list(frame_info.get("tracks", []))
        candidates = [track for track in tracks if track.get("state") == TrackState.CANDIDATE.value]
        low_candidates = [track for track in tracks if track.get("state") == TrackState.LOW_CANDIDATE.value]
        candidate_scores = [float(track.get("evidence_score", 0.0)) for track in candidates]
        low_candidate_scores = [float(track.get("evidence_score", 0.0)) for track in low_candidates]
        return {
            "frame_idx": int(frame_info.get("frame_idx", 0)),
            "candidate_track_count": int(len(candidates)),
            "low_candidate_track_count": int(len(low_candidates)),
            "active_track_count": int(frame_info.get("active_track_count", 0)),
            "lost_track_count": int(frame_info.get("lost_track_count", 0)),
            "removed_track_count": int(frame_info.get("removed_track_count", 0)),
            "candidate_gids": [int(track.get("gid", -1)) for track in candidates],
            "candidate_evidence_scores": candidate_scores,
            "low_candidate_gids": [int(track.get("gid", -1)) for track in low_candidates],
            "low_candidate_evidence_scores": low_candidate_scores,
            "candidate_source_history": {
                str(track.get("gid", -1)): list(track.get("source_history", []))
                for track in candidates
            },
            "low_candidate_source_history": {
                str(track.get("gid", -1)): list(track.get("source_history", []))
                for track in low_candidates
            },
            "max_candidate_evidence_score": max(candidate_scores) if candidate_scores else 0.0,
            "max_low_candidate_evidence_score": max(low_candidate_scores) if low_candidate_scores else 0.0,
            "num_observations": int(frame_info.get("num_observations", 0)),
            "num_low_only": int(frame_info.get("num_low_only", 0)),
            "num_roi_low": int(frame_info.get("num_roi_low", 0)),
            "num_motion": int(frame_info.get("num_motion", 0)),
            "num_template": int(frame_info.get("num_template", 0)),
            "reacquire_attempted": bool(frame_info.get("reacquire_debug", {}).get("attempted", False)),
            "low_inherit_matches": int(frame_info.get("low_inherit_debug", {}).get("num_matches", 0)),
            "low_inherit_active_conflicts": int(frame_info.get("low_inherit_debug", {}).get("num_active_conflicts", 0)),
            "removed_guard_vetoes": int(frame_info.get("removed_guard_debug", {}).get("num_vetoes", 0)),
            "suppressed_spawn_groups": int(frame_info.get("spawn_suppression_debug", {}).get("num_suppressed_spawn_groups", 0)),
            "output_nms_suppressed": int(frame_info.get("output_nms_debug", {}).get("num_suppressed", 0)),
        }

    def _debug_track_snapshot(self) -> list[dict]:
        limit = int(getattr(self.config, "MSDC_DEBUG_TRACK_SNAPSHOT_LIMIT", 128))
        if limit <= 0:
            return []

        def sort_key(track: EvidenceTrack) -> tuple[int, float, int]:
            state = self._track_state(track)
            priority = {
                TrackState.ACTIVE: 0,
                TrackState.LOST: 1,
                TrackState.CANDIDATE: 2,
                TrackState.LOW_CANDIDATE: 3,
                TrackState.REMOVED: 4,
            }.get(state, 9)
            return (priority, -float(track.evidence_score), int(track.gid))

        snapshot = []
        for track in sorted(self.tracks, key=sort_key):
            if self._track_state(track) == TrackState.REMOVED:
                continue
            snapshot.append(track_to_dict(track))
            if len(snapshot) >= limit:
                break
        return snapshot

    def _compact_template_debug(self, debug: dict | None) -> dict:
        payload = dict(debug or {})
        payload["matches"] = self._limit_debug_list(payload.get("matches", []))
        return payload

    def _compact_reacquire_debug(self, debug: dict | None, include_details: bool = True) -> dict:
        payload = dict(debug or {})
        if include_details:
            payload["matches"] = self._limit_debug_list(payload.get("matches", []))
            payload["suppressed_spawn_groups"] = self._limit_debug_list(payload.get("suppressed_spawn_groups", []))
        else:
            payload.pop("matches", None)
            payload.pop("suppressed_spawn_groups", None)
        return payload

    def _compact_low_inherit_debug(self, debug: dict | None, include_details: bool = True) -> dict:
        payload = dict(debug or {})
        if include_details:
            payload["matches"] = self._limit_debug_list(payload.get("matches", []))
            payload["active_conflicts"] = self._limit_debug_list(payload.get("active_conflicts", []))
        else:
            payload.pop("matches", None)
            payload.pop("active_conflicts", None)
        return payload

    def _compact_removed_guard_debug(self, debug: dict | None, include_details: bool = True) -> dict:
        payload = dict(debug or {})
        if include_details:
            payload["vetoes"] = self._limit_debug_list(payload.get("vetoes", []))
        else:
            payload.pop("vetoes", None)
        return payload

    def _compact_spawn_suppression_debug(self, debug: dict | None, include_details: bool = True) -> dict:
        payload = dict(debug or {})
        if include_details:
            payload["suppressed_spawn_groups"] = self._limit_debug_list(payload.get("suppressed_spawn_groups", []))
        else:
            payload.pop("suppressed_spawn_groups", None)
        return payload

    def _compact_output_nms_debug(self, debug: dict | None, include_details: bool = True) -> dict:
        payload = dict(debug or {})
        if include_details:
            payload["suppressed"] = self._limit_debug_list(payload.get("suppressed", []))
        else:
            payload.pop("suppressed", None)
        return payload

    def _compact_roi_redetect_debug(self, debug: dict | None, include_details: bool = True) -> dict:
        payload = dict(debug or {})
        if include_details:
            payload["rois"] = self._limit_debug_list(payload.get("rois", []))
        else:
            payload.pop("rois", None)
        return payload

    def _stage_observations(self, frame_info: dict) -> dict:
        boxes = []
        for stage, source_boxes in (
            ("high_det", frame_info.get("_high_boxes", [])),
            ("low_det", frame_info.get("_low_boxes", [])),
            ("low_only", frame_info.get("_low_only_boxes", [])),
            ("roi_low_det", frame_info.get("_roi_low_boxes", [])),
            ("output", frame_info.get("output_boxes", [])),
        ):
            for box in source_boxes:
                boxes.append(self._stage_box(stage, box))
        return {
            "frame_idx": int(frame_info.get("frame_idx", 0)),
            "file_type": str(frame_info.get("file_type", "")),
            "boxes": boxes,
        }

    @staticmethod
    def _stage_box(stage: str, box: dict) -> dict:
        payload = {
            "stage": str(stage),
            "x": float(box.get("x", 0.0)),
            "y": float(box.get("y", 0.0)),
            "w": float(box.get("w", 0.0)),
            "h": float(box.get("h", 0.0)),
            "confidence": float(box.get("confidence", box.get("score", 0.0))),
            "class": str(box.get("class", "unknown")),
        }
        if "track_id" in box:
            payload["track_id"] = int(box["track_id"])
        if "gid" in box:
            payload["gid"] = int(box["gid"])
        if "roi_track_gid" in box:
            payload["roi_track_gid"] = int(box["roi_track_gid"])
        return payload

    def _limit_debug_list(self, items: Any) -> list:
        limit = int(getattr(self.config, "MSDC_DEBUG_DETAIL_LIMIT", 8))
        if limit <= 0:
            return []
        if not isinstance(items, list):
            return []
        return items[:limit]

    @staticmethod
    def _append_jsonl(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _resolve_debug_dir(self, debug_dir: str | Path | None) -> Path | None:
        if debug_dir is not None:
            return Path(debug_dir)
        default = getattr(
            self.config,
            "MSDC_LIFECYCLE_DEBUG_OUTPUT_DIR",
            getattr(self.config, "MSDC_MOTION_DEBUG_OUTPUT_DIR", None),
        )
        return Path(default) if default else None

    def _sync_active_templates(self, frame: np.ndarray, template_observations: list[Observation], frame_idx: int) -> dict:
        if not self._template_enabled():
            return self._empty_template_debug()
        updated = False
        track_by_gid = {int(track.gid): track for track in self.tracks}
        for observation in template_observations:
            gid = observation.raw.get("gid") if isinstance(observation.raw, dict) else None
            if gid is None:
                continue
            track = track_by_gid.get(int(gid))
            if track is None or self._track_state(track) != TrackState.ACTIVE:
                continue
            if self.template_lock.update_template(track, frame, observation):
                updated = True

        for track in self.tracks:
            if self._track_state(track) != TrackState.ACTIVE:
                continue
            if track.template is None:
                if int(getattr(track, "last_real_det_frame", -1)) != int(frame_idx):
                    continue
                if self.template_lock.init_template(track, frame):
                    updated = True

        return self._template_debug_summary(updated)

    def _template_debug_summary(self, updated: bool) -> dict:
        matches = list(self.template_lock.last_debug)
        best = None
        if matches:
            best = max(matches, key=lambda item: float(item.get("template_score", 0.0)))
        return {
            "enabled": bool(getattr(self.config, "MSDC_TEMPLATE_ENABLE", True)),
            "template_count": int(self.template_lock.template_count),
            "num_template_matches": int(len(matches)),
            "template_score": None if best is None else best.get("template_score"),
            "template_updated": bool(updated or any(bool(item.get("template_updated", False)) for item in matches)),
            "template_match_box": None if best is None else best.get("template_match_box"),
            "matches": matches,
        }

    @staticmethod
    def _empty_template_debug() -> dict:
        return {
            "enabled": False,
            "template_count": 0,
            "num_template_matches": 0,
            "template_score": None,
            "template_updated": False,
            "template_match_box": None,
            "matches": [],
        }

    @staticmethod
    def _empty_output_nms_debug() -> dict:
        return {
            "enabled": False,
            "num_input": 0,
            "num_output": 0,
            "num_suppressed": 0,
            "suppressed": [],
        }
