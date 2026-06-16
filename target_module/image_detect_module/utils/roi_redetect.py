"""
Track-driven ROI low-threshold redetection for MS-DC-ELT.

The module only uses current track state and the current frame. It does not read
ground truth and is only enabled by the MS-DC-ELT path.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, TrackState, xywh_to_xyxy


class ROIRedetector:
    def __init__(self, config=Config, processor=None):
        self.config = config or Config
        self.processor = processor
        self._cooldown_until: dict[int, int] = {}
        self.last_debug: dict[str, Any] = self._empty_debug(enabled=self.enabled)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "MSDC_USE_ROI_REDETECT", True))

    def reset(self) -> None:
        self._cooldown_until = {}
        self.last_debug = self._empty_debug(enabled=self.enabled)

    def update(
        self,
        frame: np.ndarray,
        tracks: list[EvidenceTrack],
        frame_idx: int,
        file_type: str,
        existing_boxes: list[dict] | None = None,
    ) -> tuple[list[dict], dict]:
        if not self.enabled:
            self.last_debug = self._empty_debug(enabled=False, disabled_reason="config_disabled")
            return [], self.last_debug
        if self.processor is None or not hasattr(self.processor, "process_frame"):
            self.last_debug = self._empty_debug(enabled=True, disabled_reason="processor_unavailable")
            return [], self.last_debug
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.last_debug = self._empty_debug(enabled=True, disabled_reason="invalid_frame")
            return [], self.last_debug

        self._prune_cooldowns(tracks, int(frame_idx))
        candidates, skipped_tracks = self._select_tracks(tracks, int(frame_idx))
        existing = list(existing_boxes or [])
        boxes: list[dict] = []
        rois = []
        num_filtered_existing = 0
        num_filtered_invalid = 0
        num_capped_by_roi = 0
        for track in candidates:
            roi = self._track_roi(track, frame.shape, int(frame_idx))
            if roi is None:
                skipped_tracks.append(self._skip_debug(track, "invalid_roi", int(frame_idx)))
                continue
            x0, y0, x1, y1 = roi
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                skipped_tracks.append(self._skip_debug(track, "empty_roi", int(frame_idx)))
                continue
            inference_frame = crop
            scale = float(getattr(self.config, "MSDC_ROI_REDETECT_UPSCALE", 2.0))
            if scale > 1.0:
                inference_frame = cv2.resize(
                    crop,
                    (max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale)))),
                    interpolation=cv2.INTER_LINEAR,
                )
            low_conf = float(getattr(self.config, "MSDC_ROI_REDETECT_LOW_CONF", 0.12))
            process_context = getattr(self.processor, "use_stage", None)
            if callable(process_context):
                with process_context("roi_redetect"):
                    roi_start = time.perf_counter()
                    stats = self.processor.process_frame(inference_frame, file_type, conf_override=low_conf)
                    roi_elapsed_ms = (time.perf_counter() - roi_start) * 1000.0
            else:
                roi_start = time.perf_counter()
                stats = self.processor.process_frame(inference_frame, file_type, conf_override=low_conf)
                roi_elapsed_ms = (time.perf_counter() - roi_start) * 1000.0
            local_boxes = list((stats or {}).get("boxes", []))
            roi_boxes: list[dict] = []
            for local_box in local_boxes:
                mapped = self._map_box(local_box, x0=x0, y0=y0, scale=scale, track=track)
                if not self._valid_box(mapped):
                    num_filtered_invalid += 1
                    continue
                if self._overlaps_existing(mapped, existing):
                    num_filtered_existing += 1
                    continue
                mapped["_roi_center_distance"] = self._center_distance_to_track(mapped, track, int(frame_idx))
                roi_boxes.append(mapped)
            roi_boxes.sort(
                key=lambda item: (
                    -float(item.get("confidence", item.get("score", 0.0))),
                    float(item.get("_roi_center_distance", 0.0)),
                )
            )
            max_per_roi = max(1, int(getattr(self.config, "MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI", 2)))
            if len(roi_boxes) > max_per_roi:
                num_capped_by_roi += len(roi_boxes) - max_per_roi
                roi_boxes = roi_boxes[:max_per_roi]
            for mapped in roi_boxes:
                mapped.pop("_roi_center_distance", None)
            boxes.extend(roi_boxes)
            if not roi_boxes:
                self._mark_cooldown(track, int(frame_idx))
            rois.append({
                "gid": int(track.gid),
                "public_id": None if track.public_id is None else int(track.public_id),
                "state": self._state(track).value,
                "roi": [int(x0), int(y0), int(x1), int(y1)],
                "num_local_boxes": int(len(local_boxes)),
                "num_kept_boxes": int(len(roi_boxes)),
                "roi_call_ms": float(round(roi_elapsed_ms, 4)),
            })

        self.last_debug = {
            "enabled": True,
            "frame_idx": int(frame_idx),
            "num_candidate_tracks": int(len(candidates)),
            "num_rois": int(len(rois)),
            "num_roi_boxes": int(len(boxes)),
            "num_filtered_existing": int(num_filtered_existing),
            "num_filtered_invalid": int(num_filtered_invalid),
            "num_capped_by_roi": int(num_capped_by_roi),
            "num_skipped_tracks": int(len(skipped_tracks)),
            "skipped_tracks": skipped_tracks[: int(getattr(self.config, "MSDC_DEBUG_DETAIL_LIMIT", 8))],
            "rois": rois[: int(getattr(self.config, "MSDC_DEBUG_DETAIL_LIMIT", 8))],
        }
        return boxes, self.last_debug

    def _select_tracks(self, tracks: list[EvidenceTrack], frame_idx: int) -> tuple[list[EvidenceTrack], list[dict]]:
        active_interval = max(1, int(getattr(self.config, "MSDC_ROI_REDETECT_ACTIVE_INTERVAL", 1)))
        lost_interval = max(1, int(getattr(self.config, "MSDC_ROI_REDETECT_LOST_INTERVAL", 3)))
        active_enabled = bool(getattr(self.config, "MSDC_ROI_REDETECT_ACTIVE_ENABLE", False))
        lost_max_age = int(getattr(self.config, "MSDC_ROI_REDETECT_LOST_MAX_REAL_AGE", 30))
        selected = []
        skipped = []
        for track in list(tracks or []):
            state = self._state(track)
            cooldown_until = int(self._cooldown_until.get(int(track.gid), -1))
            if cooldown_until > int(frame_idx):
                skipped.append(self._skip_debug(track, "cooldown", frame_idx, cooldown_until=cooldown_until))
                continue
            if state == TrackState.ACTIVE:
                if not active_enabled:
                    skipped.append(self._skip_debug(track, "active_disabled", frame_idx))
                    continue
                if int(frame_idx) % active_interval == 0:
                    selected.append(track)
                else:
                    skipped.append(self._skip_debug(track, "active_interval", frame_idx))
            elif state == TrackState.LOST:
                real_age = self._real_det_age(track, frame_idx)
                if lost_max_age >= 0 and real_age > lost_max_age:
                    skipped.append(self._skip_debug(track, "lost_age_exceeded", frame_idx, real_age=real_age))
                    continue
                if int(frame_idx) % lost_interval == 0:
                    selected.append(track)
                else:
                    skipped.append(self._skip_debug(track, "lost_interval", frame_idx, real_age=real_age))
            else:
                skipped.append(self._skip_debug(track, f"state_{state.value}", frame_idx))
        selected.sort(key=lambda item: (0 if self._state(item) == TrackState.LOST else 1, -float(item.evidence_score), int(item.gid)))
        max_tracks = max(0, int(getattr(self.config, "MSDC_ROI_REDETECT_MAX_TRACKS", 8)))
        capped = selected[:max_tracks]
        for track in selected[max_tracks:]:
            skipped.append(self._skip_debug(track, "max_tracks", frame_idx))
        return capped, skipped

    def _mark_cooldown(self, track: EvidenceTrack, frame_idx: int) -> None:
        cooldown = max(0, int(getattr(self.config, "MSDC_ROI_REDETECT_COOLDOWN_FRAMES", 0)))
        if cooldown <= 0:
            return
        self._cooldown_until[int(track.gid)] = int(frame_idx) + cooldown + 1

    def _prune_cooldowns(self, tracks: list[EvidenceTrack], frame_idx: int) -> None:
        live_gids = {int(track.gid) for track in list(tracks or [])}
        expired = [
            gid for gid, until in self._cooldown_until.items()
            if int(until) <= int(frame_idx) or int(gid) not in live_gids
        ]
        for gid in expired:
            self._cooldown_until.pop(int(gid), None)

    def _skip_debug(self, track: EvidenceTrack, reason: str, frame_idx: int, **extra: Any) -> dict:
        payload = {
            "gid": int(track.gid),
            "public_id": None if track.public_id is None else int(track.public_id),
            "state": self._state(track).value,
            "reason": str(reason),
            "real_det_age": int(self._real_det_age(track, frame_idx)),
        }
        payload.update(extra)
        return payload

    @staticmethod
    def _real_det_age(track: EvidenceTrack, frame_idx: int) -> int:
        last_real = int(getattr(track, "last_real_det_frame", frame_idx))
        return max(0, int(frame_idx) - last_real)

    def _track_roi(self, track: EvidenceTrack, frame_shape: tuple, frame_idx: int) -> tuple[int, int, int, int] | None:
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        if frame_h <= 0 or frame_w <= 0:
            return None
        box = np.asarray(track.last_real_det_box if track.last_real_det_box is not None else track.box, dtype=np.float64).reshape(4).copy()
        age = max(0, int(frame_idx) - int(getattr(track, "last_real_det_frame", frame_idx)))
        velocity = np.asarray(track.velocity, dtype=np.float64).reshape(-1)
        if velocity.size >= 2:
            box[[0, 2]] += float(velocity[0]) * float(age)
            box[[1, 3]] += float(velocity[1]) * float(age)

        width = max(1.0, float(box[2] - box[0]))
        height = max(1.0, float(box[3] - box[1]))
        scale = float(getattr(self.config, "MSDC_ROI_REDETECT_SEARCH_SCALE", 4.0))
        crop_w = max(float(getattr(self.config, "MSDC_ROI_REDETECT_MIN_CROP_SIZE", 64)), width * scale)
        crop_h = max(float(getattr(self.config, "MSDC_ROI_REDETECT_MIN_CROP_SIZE", 64)), height * scale)
        cx = float((box[0] + box[2]) / 2.0)
        cy = float((box[1] + box[3]) / 2.0)
        x0 = max(0, int(round(cx - crop_w / 2.0)))
        y0 = max(0, int(round(cy - crop_h / 2.0)))
        x1 = min(frame_w, int(round(cx + crop_w / 2.0)))
        y1 = min(frame_h, int(round(cy + crop_h / 2.0)))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _map_box(self, local_box: dict, x0: int, y0: int, scale: float, track: EvidenceTrack) -> dict:
        mapped = dict(local_box)
        inv = 1.0 / max(float(scale), 1e-9)
        mapped["x"] = int(round(float(local_box.get("x", 0.0)) * inv + x0))
        mapped["y"] = int(round(float(local_box.get("y", 0.0)) * inv + y0))
        mapped["w"] = max(0, int(round(float(local_box.get("w", 0.0)) * inv)))
        mapped["h"] = max(0, int(round(float(local_box.get("h", 0.0)) * inv)))
        mapped["confidence"] = float(local_box.get("confidence", local_box.get("score", 0.0)))
        mapped["class"] = str(local_box.get("class", getattr(track, "class_name", "unknown")))
        if "class_confidence" in local_box:
            mapped["class_confidence"] = float(local_box["class_confidence"])
        mapped["source"] = "roi_low_det"
        mapped["roi_track_gid"] = int(track.gid)
        mapped["roi_track_state"] = self._state(track).value
        mapped["roi_origin"] = [int(x0), int(y0)]
        mapped["roi_scale"] = float(scale)
        return mapped

    def _overlaps_existing(self, box: dict, existing_boxes: list[dict]) -> bool:
        if not existing_boxes:
            return False
        threshold = float(getattr(self.config, "MSDC_ROI_REDETECT_EXISTING_IOU", 0.5))
        box_xyxy = xywh_to_xyxy(box)
        for existing in existing_boxes:
            if self._iou_xyxy(box_xyxy, xywh_to_xyxy(existing)) >= threshold:
                return True
        return False

    def _valid_box(self, box: dict) -> bool:
        width = float(box.get("w", 0.0))
        height = float(box.get("h", 0.0))
        min_size = float(getattr(self.config, "MSDC_ROI_REDETECT_MIN_BOX_SIZE", 12))
        return bool(width >= min_size and height >= min_size)

    def _center_distance_to_track(self, box: dict, track: EvidenceTrack, frame_idx: int) -> float:
        box_xyxy = xywh_to_xyxy(box)
        track_box = np.asarray(track.last_real_det_box if track.last_real_det_box is not None else track.box, dtype=np.float64).reshape(4).copy()
        age = max(0, int(frame_idx) - int(getattr(track, "last_real_det_frame", frame_idx)))
        velocity = np.asarray(track.velocity, dtype=np.float64).reshape(-1)
        if velocity.size >= 2:
            track_box[[0, 2]] += float(velocity[0]) * float(age)
            track_box[[1, 3]] += float(velocity[1]) * float(age)
        box_center = np.array([(box_xyxy[0] + box_xyxy[2]) / 2.0, (box_xyxy[1] + box_xyxy[3]) / 2.0])
        track_center = np.array([(track_box[0] + track_box[2]) / 2.0, (track_box[1] + track_box[3]) / 2.0])
        return float(np.linalg.norm(box_center - track_center))

    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        ix1 = max(float(a[0]), float(b[0]))
        iy1 = max(float(a[1]), float(b[1]))
        ix2 = min(float(a[2]), float(b[2]))
        iy2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        return float(inter / max(area_a + area_b - inter, 1e-10))

    @staticmethod
    def _state(track: EvidenceTrack) -> TrackState:
        return track.state if isinstance(track.state, TrackState) else TrackState(str(track.state))

    @staticmethod
    def _empty_debug(enabled: bool, disabled_reason: str | None = None) -> dict:
        payload = {
            "enabled": bool(enabled),
            "num_candidate_tracks": 0,
            "num_rois": 0,
            "num_roi_boxes": 0,
            "num_filtered_existing": 0,
            "num_filtered_invalid": 0,
            "num_capped_by_roi": 0,
            "num_skipped_tracks": 0,
            "skipped_tracks": [],
            "rois": [],
        }
        if disabled_reason:
            payload["disabled_reason"] = disabled_reason
        return payload
