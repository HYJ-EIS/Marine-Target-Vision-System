"""
Adapter for vendored official OC-SORT and BoT-SORT trackers.

The project keeps its existing tracker implementations as the default baseline.
This module only translates the project detection/result dictionaries to and
from the official tracker APIs so experiments can compare the association layer
on the same detector outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from target_module.image_detect_module.config import Config


@dataclass
class _DetectionMeta:
    bbox: np.ndarray
    confidence: float
    cls_id: int


class OfficialTrackerAdapter:
    """Project-format wrapper around vendored official tracker implementations."""

    _CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(Config.CLASSES)}
    _ID_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(Config.CLASSES)}

    def __init__(self, tracker_type: str, frame_rate: float = 25.0,
                 min_hits: int | None = None):
        self.tracker_type = tracker_type
        self.frame_rate = max(1.0, float(frame_rate))
        self.min_hits = min_hits if min_hits is not None else Config.TRACKER_MIN_HITS

        if tracker_type == "official_ocsort":
            from third_party.official_trackers.oc_sort.ocsort_tracker.ocsort import OCSort

            self._tracker = OCSort(
                det_thresh=Config.TRACKER_ACTIVATION_THRESH,
                max_age=Config.TRACKER_LOST_BUFFER,
                min_hits=self.min_hits,
                iou_threshold=Config.TRACKER_MATCH_THRESH,
                delta_t=3,
                asso_func="iou",
                inertia=0.2,
                use_byte=True,
            )
        elif tracker_type == "official_botsort":
            from third_party.official_trackers.bot_sort.tracker.bot_sort import BoTSORT

            self._tracker = BoTSORT(self._botsort_args(), frame_rate=int(self.frame_rate))
        else:
            raise ValueError(f"不支持的官方追踪器类型: {tracker_type}")

    def update(self, detections: list[dict], frame_shape: tuple,
               frame: np.ndarray | None = None) -> list[dict]:
        det_array, metas = self._to_official_detections(detections, frame_shape)

        if self.tracker_type == "official_ocsort":
            tracked = self._tracker.update(det_array, frame_shape[:2], frame_shape[:2])
            return self._format_array_tracks(tracked, metas)

        if frame is None:
            h, w = frame_shape[:2]
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        tracked = self._tracker.update(det_array, frame)
        return self._format_botsort_tracks(tracked, metas)

    def reset(self) -> None:
        self.__init__(self.tracker_type, self.frame_rate, self.min_hits)

    def get_recent_tracks(self, max_time_since_update: int = 1) -> list[dict]:
        return []

    def _botsort_args(self) -> SimpleNamespace:
        method_map = {
            "sparse_flow": "sparseOptFlow",
            "sparseOptFlow": "sparseOptFlow",
            "orb": "orb",
            "none": "none",
        }
        return SimpleNamespace(
            track_high_thresh=Config.TRACKER_ACTIVATION_THRESH,
            track_low_thresh=0.1,
            new_track_thresh=Config.TRACKER_ACTIVATION_THRESH,
            track_buffer=Config.TRACKER_LOST_BUFFER,
            match_thresh=1.0 - Config.TRACKER_MATCH_THRESH,
            proximity_thresh=1.0 - Config.TRACKER_MATCH_THRESH,
            appearance_thresh=0.25,
            with_reid=False,
            fast_reid_config="",
            fast_reid_weights="",
            device="cpu",
            cmc_method=method_map.get(Config.GMC_METHOD, "sparseOptFlow"),
            name="project",
            ablation=False,
            mot20=False,
        )

    def _to_official_detections(
        self,
        detections: list[dict],
        frame_shape: tuple,
    ) -> tuple[np.ndarray, list[_DetectionMeta]]:
        h, w = frame_shape[:2]
        rows: list[list[float]] = []
        metas: list[_DetectionMeta] = []
        for det in detections:
            x1 = float(det["x"])
            y1 = float(det["y"])
            x2 = x1 + float(det["w"])
            y2 = y1 + float(det["h"])
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 <= x1 or y2 <= y1:
                continue

            conf = float(det.get("confidence", 0.0))
            cls_id = int(self._CLASS_TO_ID.get(det.get("class", Config.CLASSES[0]), 0))
            bbox = np.array([x1, y1, x2, y2], dtype=np.float64)
            rows.append([x1, y1, x2, y2, conf])
            metas.append(_DetectionMeta(bbox=bbox, confidence=conf, cls_id=cls_id))

        if not rows:
            return np.empty((0, 5), dtype=np.float64), []
        return np.array(rows, dtype=np.float64), metas

    def _format_array_tracks(self, tracks: np.ndarray,
                             metas: list[_DetectionMeta]) -> list[dict]:
        if tracks is None or len(tracks) == 0:
            return []
        results = []
        for row in np.asarray(tracks):
            x1, y1, x2, y2 = row[:4]
            tid = int(row[4])
            meta = self._best_meta(np.array([x1, y1, x2, y2], dtype=np.float64), metas)
            results.append(self._result_dict(x1, y1, x2, y2, tid, meta))
        return results

    def _format_botsort_tracks(self, tracks: list,
                               metas: list[_DetectionMeta]) -> list[dict]:
        results = []
        for track in tracks:
            x1, y1, x2, y2 = track.tlbr
            meta = self._best_meta(np.array([x1, y1, x2, y2], dtype=np.float64), metas)
            if meta is None:
                meta = _DetectionMeta(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
                    confidence=float(getattr(track, "score", 0.0)),
                    cls_id=0,
                )
            results.append(self._result_dict(x1, y1, x2, y2, int(track.track_id), meta))
        return results

    def _best_meta(self, bbox: np.ndarray,
                   metas: list[_DetectionMeta]) -> _DetectionMeta | None:
        if not metas:
            return None
        ious = np.array([self._iou(bbox, meta.bbox) for meta in metas])
        best_idx = int(np.argmax(ious))
        if ious[best_idx] <= 0:
            return None
        return metas[best_idx]

    def _result_dict(self, x1: float, y1: float, x2: float, y2: float,
                     tid: int, meta: _DetectionMeta | None) -> dict:
        if meta is None:
            meta = _DetectionMeta(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
                confidence=0.0,
                cls_id=0,
            )
        return {
            "track_id": int(tid),
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "confidence": round(float(meta.confidence), 4),
            "class": self._ID_TO_CLASS.get(int(meta.cls_id), Config.CLASSES[0]),
            "class_confidence": round(float(meta.confidence), 4),
        }

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        xx1 = max(float(a[0]), float(b[0]))
        yy1 = max(float(a[1]), float(b[1]))
        xx2 = min(float(a[2]), float(b[2]))
        yy2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        return inter / max(area_a + area_b - inter, 1e-12)
