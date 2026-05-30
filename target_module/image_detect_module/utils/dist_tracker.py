"""
Dist-Tracker style lightweight tracker.

This module implements the FLIT association idea from Dist-Tracker without
vendoring the full Ultralytics-based project. It keeps this repository's
tracking-by-detection contract: project detection dicts in, project tracking
dicts out.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.gmc import GMC
from target_module.image_detect_module.utils.kalman_bbox import KalmanBoxTracker
from target_module.image_detect_module.utils.tracker import _center_distance_batch, _iou_batch


class DistTracker:
    """Small-object tracker using FLIT L2-IoU matching plus optional GMC."""

    _CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(Config.CLASSES)}
    _ID_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(Config.CLASSES)}

    def __init__(
        self,
        frame_rate: float = 25.0,
        max_age: int | None = None,
        min_hits: int | None = None,
    ):
        self.frame_rate = max(1.0, float(frame_rate))
        self.max_age = max_age if max_age is not None else Config.TRACKER_LOST_BUFFER
        self.min_hits = min_hits if min_hits is not None else Config.TRACKER_MIN_HITS
        self.gamma = float(Config.DIST_TRACKER_GAMMA)
        self.match_thresh = float(Config.DIST_TRACKER_MATCH_THRESH)
        self.fuse_score = bool(Config.DIST_TRACKER_FUSE_SCORE)
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self._gmc = GMC(method=Config.GMC_METHOD, downscale=Config.GMC_DOWNSCALE)

    def update(
        self,
        detections: list[dict],
        frame_shape: tuple,
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        self.frame_count += 1

        if frame is not None:
            matrix = self._gmc.apply(frame)
            self._apply_gmc(matrix)

        predicted = self._predict_existing_tracks()
        dets, confs, cls_ids = self._to_arrays(detections, frame_shape)

        if len(dets) == 0:
            self._drop_expired_tracks()
            return []

        if len(self.trackers) == 0:
            for i in range(len(dets)):
                self.trackers.append(
                    KalmanBoxTracker(dets[i], int(cls_ids[i]), float(confs[i]))
                )
            return self._output()

        cost_matrix = self._flit_cost(dets, predicted, confs, frame_shape)
        matched_d, matched_t, unmatched_dets = self._match(cost_matrix)

        for d_idx, t_idx in zip(matched_d, matched_t):
            self.trackers[int(t_idx)].update(
                dets[int(d_idx)],
                int(cls_ids[int(d_idx)]),
                float(confs[int(d_idx)]),
            )

        for d_idx in unmatched_dets:
            self.trackers.append(
                KalmanBoxTracker(dets[int(d_idx)], int(cls_ids[int(d_idx)]), float(confs[int(d_idx)]))
            )

        self._drop_expired_tracks()
        return self._output()

    def reset(self) -> None:
        self.trackers = []
        self.frame_count = 0
        self._gmc.reset()
        KalmanBoxTracker.reset_count()

    def get_recent_tracks(self, max_time_since_update: int = 1) -> list[dict]:
        if max_time_since_update <= 0:
            return []
        results = []
        for trk in self.trackers:
            if trk.time_since_update <= 0 or trk.time_since_update > max_time_since_update:
                continue
            if trk.hits < self.min_hits:
                continue
            x1, y1, x2, y2 = trk.get_state()
            results.append(self._result_dict(x1, y1, x2, y2, trk.id, trk.cls_id, trk.conf, True))
        return results

    def _predict_existing_tracks(self) -> np.ndarray:
        predicted = np.zeros((len(self.trackers), 4), dtype=np.float64)
        to_delete = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            predicted[i] = pred
            if np.any(np.isnan(pred)):
                to_delete.append(i)

        for i in reversed(to_delete):
            self.trackers.pop(i)
            predicted = np.delete(predicted, i, axis=0)
        return predicted

    def _to_arrays(
        self,
        detections: list[dict],
        frame_shape: tuple,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = frame_shape[:2]
        dets, confs, cls_ids = [], [], []
        for det in detections:
            x1 = float(det["x"])
            y1 = float(det["y"])
            x2 = x1 + float(det["w"])
            y2 = y1 + float(det["h"])
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append([x1, y1, x2, y2])
            confs.append(float(det.get("confidence", 0.0)))
            cls_ids.append(int(self._CLASS_TO_ID.get(det.get("class", Config.CLASSES[0]), 0)))

        if not dets:
            return (
                np.empty((0, 4), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=int),
            )
        return (
            np.array(dets, dtype=np.float64),
            np.array(confs, dtype=np.float64),
            np.array(cls_ids, dtype=int),
        )

    def _flit_cost(
        self,
        dets: np.ndarray,
        tracks: np.ndarray,
        confs: np.ndarray,
        frame_shape: tuple,
    ) -> np.ndarray:
        iou_distance = 1.0 - _iou_batch(dets, tracks)
        diagonal = max(float(np.hypot(frame_shape[1], frame_shape[0])), 1.0)
        l2_distance = np.clip(_center_distance_batch(dets, tracks) / diagonal, 0.0, 1.0)

        gamma = min(max(self.gamma, 0.0), 1.0)
        cost = gamma * l2_distance + (1.0 - gamma) * iou_distance
        cost = np.clip(cost, 0.0, 1.0)

        if self.fuse_score and cost.size:
            similarity = 1.0 - cost
            similarity *= confs.reshape(-1, 1)
            cost = 1.0 - similarity
        return cost

    def _match(self, cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows, cols = cost_matrix.shape
        if rows == 0 or cols == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.arange(rows)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_d, matched_t = [], []
        unmatched_d = set(range(rows))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.match_thresh:
                matched_d.append(r)
                matched_t.append(c)
                unmatched_d.discard(r)

        return (
            np.array(matched_d, dtype=int),
            np.array(matched_t, dtype=int),
            np.array(sorted(unmatched_d), dtype=int),
        )

    def _drop_expired_tracks(self) -> None:
        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]

    def _output(self) -> list[dict]:
        results = []
        for trk in self.trackers:
            if trk.time_since_update <= 0 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                x1, y1, x2, y2 = trk.last_observation if trk.observed else trk.get_state()
                results.append(self._result_dict(x1, y1, x2, y2, trk.id, trk.cls_id, trk.conf))
        return results

    def _result_dict(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        track_id: int,
        cls_id: int,
        conf: float,
        is_prediction: bool = False,
    ) -> dict:
        result = {
            "track_id": int(track_id),
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "confidence": round(float(conf), 4),
            "class": self._ID_TO_CLASS.get(int(cls_id), Config.CLASSES[0]),
            "class_confidence": round(float(conf), 4),
        }
        if is_prediction:
            result["is_tracker_prediction"] = True
        return result

    def _apply_gmc(self, matrix: np.ndarray) -> None:
        for trk in self.trackers:
            trk.apply_affine(matrix)
            obs = trk.last_observation
            cx = (obs[0] + obs[2]) / 2
            cy = (obs[1] + obs[3]) / 2
            w, h = obs[2] - obs[0], obs[3] - obs[1]
            new_pt = matrix @ np.array([cx, cy, 1.0])
            trk.last_observation = np.array(
                [
                    new_pt[0] - w / 2,
                    new_pt[1] - h / 2,
                    new_pt[0] + w / 2,
                    new_pt[1] + h / 2,
                ],
                dtype=np.float64,
            )
