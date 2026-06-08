"""
Active-only lightweight template locking for MS-DC-ELT.

The first implementation uses local OpenCV template matching. It does not run
on candidate tracks, does not do global search, and stores only template
metadata on EvidenceTrack so JSONL diagnostics stay compact.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, Observation, TrackState


@dataclass
class _TemplateState:
    gid: int
    image: np.ndarray
    box: list[float]
    updated_frame_idx: int
    last_score: float | None = None
    updated_count: int = 0


class TemplateLock:
    def __init__(self, config=Config):
        self.config = config or Config
        self.enabled = bool(
            getattr(self.config, "MSDC_TEMPLATE_ENABLE", True)
            and getattr(self.config, "MSDC_USE_TEMPLATE", True)
        )
        self.max_templates = int(getattr(self.config, "MSDC_MAX_ACTIVE_TEMPLATES", 8))
        self.update_thresh = float(getattr(self.config, "MSDC_TEMPLATE_UPDATE_THRESH", 0.75))
        self.search_scale = float(getattr(self.config, "MSDC_TEMPLATE_SEARCH_SCALE", 2.5))
        self.min_size = int(getattr(self.config, "MSDC_TEMPLATE_MIN_SIZE", 8))
        self.min_std = float(getattr(self.config, "MSDC_TEMPLATE_MIN_STD", 1.0))
        self.update_require_real_det = bool(getattr(self.config, "MSDC_TEMPLATE_UPDATE_REQUIRE_REAL_DET", True))
        self._templates: OrderedDict[int, _TemplateState] = OrderedDict()
        self.last_debug: list[dict] = []

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def has_template(self, gid: int) -> bool:
        return int(gid) in self._templates

    def reset(self) -> None:
        self._templates.clear()
        self.last_debug = []

    def init_template(self, track: EvidenceTrack, frame: np.ndarray) -> bool:
        if not self._can_use_track(track) or frame is None:
            return False

        crop = self._crop_gray(frame, track.box)
        if not self._valid_template(crop):
            return False

        gid = int(track.gid)
        state = _TemplateState(
            gid=gid,
            image=crop,
            box=self._box_list(track.box),
            updated_frame_idx=int(track.last_seen),
        )
        self._templates[gid] = state
        self._templates.move_to_end(gid)
        self._enforce_budget()
        self._sync_track_template_metadata(track, state)
        return self.has_template(gid)

    def match(
        self,
        track: EvidenceTrack,
        frame: np.ndarray,
        frame_idx: int | None = None,
        modality: str = "visible",
        allow_lost: bool = False,
    ) -> Observation | None:
        if not self._can_use_track(track, allow_lost=allow_lost) or frame is None:
            return None
        gid = int(track.gid)
        state = self._templates.get(gid)
        if state is None or not self._valid_template(state.image):
            return None

        search_gray, offset = self._search_crop_gray(frame, track.box, state.image.shape[:2])
        if search_gray is None:
            return None

        method = cv2.TM_CCOEFF_NORMED if float(np.std(state.image)) >= self.min_std else cv2.TM_CCORR_NORMED
        response = cv2.matchTemplate(search_gray, state.image, method)
        if response.size == 0:
            return None

        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        score = round(float(max_val), 4)
        x1 = float(offset[0] + max_loc[0])
        y1 = float(offset[1] + max_loc[1])
        h, w = state.image.shape[:2]
        box = [x1, y1, x1 + float(w), y1 + float(h)]
        debug = {
            "gid": gid,
            "frame_idx": int(frame_idx) if frame_idx is not None else None,
            "template_score": score,
            "template_match_box": [round(v, 4) for v in box],
            "template_updated": False,
            "track_state": self._state(track).value,
        }
        self.last_debug.append(debug)
        return Observation(
            box=box,
            source="template",
            score=score,
            reliability=1.0,
            modality=modality,
            frame_idx=0 if frame_idx is None else int(frame_idx),
            class_id=int(track.class_id),
            class_name=str(track.class_name),
            raw=dict(debug),
        )

    def should_update(self, track: EvidenceTrack, observation: Observation | None, score: float) -> bool:
        if not self._can_use_track(track) or observation is None:
            return False
        if str(observation.source) != "template":
            return False
        if self.update_require_real_det:
            if int(getattr(track, "last_real_det_frame", -1)) != int(observation.frame_idx):
                return False
            if not self._template_matches_current_track_box(track, observation):
                return False
        return float(score) >= self.update_thresh

    def update_template(self, track: EvidenceTrack, frame: np.ndarray, observation: Observation) -> bool:
        score = float(observation.score)
        if not self.should_update(track, observation, score):
            return False

        crop = self._crop_gray(frame, observation.box)
        if not self._valid_template(crop):
            return False

        gid = int(track.gid)
        old_state = self._templates.get(gid)
        updated_count = 1 if old_state is None else int(old_state.updated_count) + 1
        state = _TemplateState(
            gid=gid,
            image=crop,
            box=self._box_list(observation.box),
            updated_frame_idx=int(observation.frame_idx),
            last_score=round(score, 4),
            updated_count=updated_count,
        )
        self._templates[gid] = state
        self._templates.move_to_end(gid)
        self._enforce_budget()
        self._sync_track_template_metadata(track, state)
        self._mark_last_debug_updated(gid, state.last_score)
        return True

    def match_active_tracks(
        self,
        tracks: list[EvidenceTrack],
        frame: np.ndarray,
        frame_idx: int,
        modality: str = "visible",
    ) -> list[Observation]:
        self.last_debug = []
        observations = []
        if not self.enabled:
            return observations
        for track in tracks:
            if self._state(track) != TrackState.ACTIVE:
                continue
            obs = self.match(track, frame, frame_idx=frame_idx, modality=modality)
            if obs is not None:
                observations.append(obs)
        return observations

    def match_lost_tracks(
        self,
        tracks: list[EvidenceTrack],
        frame: np.ndarray,
        frame_idx: int,
        modality: str = "visible",
    ) -> list[Observation]:
        observations = []
        if not self.enabled:
            return observations
        for track in tracks:
            if self._state(track) != TrackState.LOST:
                continue
            obs = self.match(track, frame, frame_idx=frame_idx, modality=modality, allow_lost=True)
            if obs is not None:
                observations.append(obs)
        return observations

    def _can_use_track(self, track: EvidenceTrack, allow_lost: bool = False) -> bool:
        if not self.enabled or self.max_templates <= 0:
            return False
        state = self._state(track)
        return bool(state == TrackState.ACTIVE or (allow_lost and state == TrackState.LOST))

    @staticmethod
    def _state(track: EvidenceTrack) -> TrackState:
        return track.state if isinstance(track.state, TrackState) else TrackState(str(track.state))

    def _template_matches_current_track_box(self, track: EvidenceTrack, observation: Observation) -> bool:
        track_box = np.asarray(track.box, dtype=np.float64).reshape(-1)
        obs_box = np.asarray(observation.box, dtype=np.float64).reshape(-1)
        if track_box.size != 4 or obs_box.size != 4:
            return False
        iou_score = self._box_iou(track_box, obs_box)
        center_dist = self._center_distance(track_box, obs_box)
        track_w = max(1.0, float(track_box[2] - track_box[0]))
        track_h = max(1.0, float(track_box[3] - track_box[1]))
        center_limit = max(8.0, 0.5 * max(track_w, track_h))
        return bool(iou_score >= 0.2 or center_dist <= center_limit)

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        xx1 = max(float(a[0]), float(b[0]))
        yy1 = max(float(a[1]), float(b[1]))
        xx2 = min(float(a[2]), float(b[2]))
        yy2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        return float(inter / max(area_a + area_b - inter, 1e-10))

    @staticmethod
    def _center_distance(a: np.ndarray, b: np.ndarray) -> float:
        acx = (float(a[0]) + float(a[2])) / 2.0
        acy = (float(a[1]) + float(a[3])) / 2.0
        bcx = (float(b[0]) + float(b[2])) / 2.0
        bcy = (float(b[1]) + float(b[3])) / 2.0
        return float(np.hypot(acx - bcx, acy - bcy))

    def _crop_gray(self, frame: np.ndarray, box: Any) -> np.ndarray | None:
        coords = self._clip_box(box, frame.shape)
        if coords is None:
            return None
        x1, y1, x2, y2 = coords
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def _search_crop_gray(
        self,
        frame: np.ndarray,
        box: Any,
        template_shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        base = self._clip_box(box, frame.shape)
        if base is None:
            return None, (0, 0)
        x1, y1, x2, y2 = base
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        search_w = max(template_shape[1], int(round(box_w * self.search_scale)))
        search_h = max(template_shape[0], int(round(box_h * self.search_scale)))

        sx1 = max(0, int(round(cx - search_w / 2.0)))
        sy1 = max(0, int(round(cy - search_h / 2.0)))
        sx2 = min(frame.shape[1], int(round(cx + search_w / 2.0)))
        sy2 = min(frame.shape[0], int(round(cy + search_h / 2.0)))
        if sx2 - sx1 < template_shape[1] or sy2 - sy1 < template_shape[0]:
            return None, (sx1, sy1)

        search = frame[sy1:sy2, sx1:sx2]
        if search.size == 0:
            return None, (sx1, sy1)
        return cv2.cvtColor(search, cv2.COLOR_BGR2GRAY), (sx1, sy1)

    def _valid_template(self, crop: np.ndarray | None) -> bool:
        if crop is None:
            return False
        if crop.ndim != 2:
            return False
        h, w = crop.shape[:2]
        if h < self.min_size or w < self.min_size:
            return False
        return float(np.std(crop)) >= self.min_std

    @staticmethod
    def _clip_box(box: Any, frame_shape: tuple) -> tuple[int, int, int, int] | None:
        arr = np.asarray(box, dtype=np.float64).reshape(-1)
        if arr.size != 4:
            return None
        h, w = frame_shape[:2]
        x1 = max(0, min(w, int(round(arr[0]))))
        y1 = max(0, min(h, int(round(arr[1]))))
        x2 = max(0, min(w, int(round(arr[2]))))
        y2 = max(0, min(h, int(round(arr[3]))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _box_list(box: Any) -> list[float]:
        arr = np.asarray(box, dtype=np.float64).reshape(-1)
        return [float(v) for v in arr.tolist()]

    def _sync_track_template_metadata(self, track: EvidenceTrack, state: _TemplateState) -> None:
        h, w = state.image.shape[:2]
        track.template = {
            "initialized": True,
            "size": [int(w), int(h)],
            "updated_frame_idx": int(state.updated_frame_idx),
            "last_score": state.last_score,
            "updated_count": int(state.updated_count),
        }

    def _mark_last_debug_updated(self, gid: int, score: float | None) -> None:
        for item in reversed(self.last_debug):
            if int(item.get("gid", -1)) == int(gid):
                item["template_updated"] = True
                item["template_score"] = score
                break

    def _enforce_budget(self) -> None:
        while len(self._templates) > max(0, self.max_templates):
            self._templates.popitem(last=False)
