"""
多目标追踪模块 — 支持多种追踪算法

支持的追踪器:
    - "bytetrack" : supervision.ByteTrack（原始实现，纯 IoU 匹配）
    - "ocsort"    : OC-SORT（Observation-Centric，虚拟轨迹重更新 + 方向一致性）
    - "botsort"   : BoT-SORT-Lite（OC-SORT + 全局相机运动补偿 GMC）
    - "dist_tracker" : Dist-Tracker FLIT（L2-IoU 融合匹配 + GMC）
    - "official_ocsort"  : vendored 官方 OC-SORT 适配层
    - "official_botsort" : vendored 官方 BoT-SORT 适配层（默认关闭 ReID）

通过 Config.TRACKER_TYPE 选择算法，或在构造时指定 tracker_type 参数。

输入格式（来自 ImageProcessor/target_detection）：
    [{"x": int, "y": int, "w": int, "h": int, "confidence": float, "class": str}, ...]

输出格式：
    [{"track_id": int, "x": int, "y": int, "w": int, "h": int,
      "confidence": float, "class": str}, ...]
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.kalman_bbox import KalmanBoxTracker
from target_module.image_detect_module.utils.gmc import GMC
from target_module.image_detect_module.utils.official_adapter import OfficialTrackerAdapter


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    计算两组 bbox 的 IoU 矩阵

    Parameters
    ----------
    bb_test : (M, 4) [x1, y1, x2, y2]
    bb_gt   : (N, 4) [x1, y1, x2, y2]

    Returns
    -------
    (M, N) IoU 矩阵
    """
    M, N = len(bb_test), len(bb_gt)
    if M == 0 or N == 0:
        return np.zeros((M, N))

    xx1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0].reshape(1, -1))
    yy1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1].reshape(1, -1))
    xx2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2].reshape(1, -1))
    yy2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3].reshape(1, -1))

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_b = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-10)


def _center_distance_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """计算两组 bbox 中心点的欧式距离矩阵 (M, N)"""
    M, N = len(bb_test), len(bb_gt)
    if M == 0 or N == 0:
        return np.zeros((M, N))

    cx_a = (bb_test[:, 0] + bb_test[:, 2]) / 2
    cy_a = (bb_test[:, 1] + bb_test[:, 3]) / 2
    cx_b = (bb_gt[:, 0] + bb_gt[:, 2]) / 2
    cy_b = (bb_gt[:, 1] + bb_gt[:, 3]) / 2

    dx = cx_a[:, None] - cx_b[None, :]
    dy = cy_a[:, None] - cy_b[None, :]
    return np.sqrt(dx ** 2 + dy ** 2)


def _direction_consistency(trk: KalmanBoxTracker, det_bbox: np.ndarray) -> float:
    """
    OC-SORT OCM: 计算轨迹速度方向与 (last_obs → det) 方向的一致性
    返回 [0, 1]，1 表示完全一致
    """
    if np.linalg.norm(trk.velocity) < 1e-4:
        return 0.5  # 无历史速度信息时中性

    new_center = np.array([
        (det_bbox[0] + det_bbox[2]) / 2,
        (det_bbox[1] + det_bbox[3]) / 2,
    ])
    old_center = np.array([
        (trk.last_observation[0] + trk.last_observation[2]) / 2,
        (trk.last_observation[1] + trk.last_observation[3]) / 2,
    ])
    movement = new_center - old_center
    if np.linalg.norm(movement) < 1e-4:
        return 0.5

    cos_sim = np.dot(trk.velocity, movement) / (
        np.linalg.norm(trk.velocity) * np.linalg.norm(movement) + 1e-10
    )
    return (cos_sim + 1) / 2  # 映射到 [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# OC-SORT 追踪器
# ─────────────────────────────────────────────────────────────────────────────

class OCSortTracker:
    """
    OC-SORT: Observation-Centric SORT

    相比 ByteTrack 的改进:
    1. 虚拟轨迹重更新 (ORU): 丢失后重新关联时用线性插值纠正 Kalman 漂移
    2. 方向一致性 (OCM): 用速度方向提升匹配准确性
    3. 两阶段关联: 先 IoU，后中心距离回捞
    """

    def __init__(
        self,
        max_age: int = 50,
        min_hits: int = 1,
        iou_threshold: float = 0.2,
        dist_threshold: float = 200.0,
        use_oru: bool = True,
        use_ocm: bool = True,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.use_oru = use_oru
        self.use_ocm = use_ocm

        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray, confs: np.ndarray,
               cls_ids: np.ndarray) -> list[tuple]:
        """
        Parameters
        ----------
        dets     : (N, 4) [x1, y1, x2, y2]
        confs    : (N,) confidence
        cls_ids  : (N,) class_id

        Returns
        -------
        list of (x1, y1, x2, y2, track_id, cls_id, conf)
        """
        self.frame_count += 1

        # ── 1. 预测所有现有轨迹的下一帧位置 ──────────────────────────────
        predicted_bboxes = np.zeros((len(self.trackers), 4))
        to_del = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            predicted_bboxes[i] = pred
            if np.any(np.isnan(pred)):
                to_del.append(i)

        for i in reversed(to_del):
            self.trackers.pop(i)
            predicted_bboxes = np.delete(predicted_bboxes, i, axis=0)

        # ── 2. 第一阶段关联：IoU ─────────────────────────────────────────
        if len(dets) == 0:
            # 无检测，仅做预测，淘汰过期轨迹
            self.trackers = [
                t for t in self.trackers if t.time_since_update <= self.max_age
            ]
            return []

        if len(self.trackers) == 0:
            # 无现有轨迹，全部初始化
            results = []
            for i in range(len(dets)):
                trk = KalmanBoxTracker(dets[i], int(cls_ids[i]), float(confs[i]))
                self.trackers.append(trk)
            # 仅返回 min_hits 满足的
            return self._output()

        # 计算 IoU
        iou_matrix = _iou_batch(dets, predicted_bboxes)

        # OCM: 用方向一致性加权 IoU
        if self.use_ocm:
            for d_idx in range(len(dets)):
                for t_idx in range(len(self.trackers)):
                    if iou_matrix[d_idx, t_idx] > 0:
                        dc = _direction_consistency(self.trackers[t_idx], dets[d_idx])
                        iou_matrix[d_idx, t_idx] *= (0.5 + 0.5 * dc)

        matched_d, matched_t, unmatched_dets, unmatched_trks = \
            self._linear_assignment(iou_matrix, self.iou_threshold)

        # ── 3. 第二阶段关联：中心距离（回捞 IoU=0 的目标）──────────────────
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            # 使用 last_observation 而非 Kalman 预测（OC-SORT 核心）
            last_obs = np.array([
                self.trackers[t].last_observation for t in unmatched_trks
            ])
            remaining_dets = dets[unmatched_dets]

            dist_matrix = _center_distance_batch(remaining_dets, last_obs)

            m_d2, m_t2, um_d2, um_t2 = \
                self._linear_assignment_dist(dist_matrix, self.dist_threshold,
                                             unmatched_dets, unmatched_trks)

            matched_d = np.concatenate([matched_d, m_d2])
            matched_t = np.concatenate([matched_t, m_t2])
            unmatched_dets = um_d2
            unmatched_trks = um_t2

        # ── 4. 更新匹配的轨迹 ──────────────────────────────────────────────
        for d_idx, t_idx in zip(matched_d, matched_t):
            d_idx, t_idx = int(d_idx), int(t_idx)
            trk = self.trackers[t_idx]
            if self.use_oru and trk.time_since_update > 1:
                trk.re_update_with_virtual_trajectory(
                    dets[d_idx], int(cls_ids[d_idx]), float(confs[d_idx])
                )
            else:
                trk.update(dets[d_idx], int(cls_ids[d_idx]), float(confs[d_idx]))

        # ── 5. 为未匹配的检测创建新轨迹 ──────────────────────────────────
        for d_idx in unmatched_dets:
            trk = KalmanBoxTracker(dets[d_idx], int(cls_ids[d_idx]), float(confs[d_idx]))
            self.trackers.append(trk)

        # ── 6. 淘汰过期轨迹 ──────────────────────────────────────────────
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        return self._output()

    def apply_gmc(self, M: np.ndarray):
        """将 GMC 仿射矩阵应用到所有轨迹的 Kalman 状态"""
        for trk in self.trackers:
            trk.apply_affine(M)
            # 同时更新 last_observation
            obs = trk.last_observation
            cx = (obs[0] + obs[2]) / 2
            cy = (obs[1] + obs[3]) / 2
            w, h = obs[2] - obs[0], obs[3] - obs[1]
            pt = np.array([cx, cy, 1.0])
            new_pt = M @ pt
            trk.last_observation = np.array([
                new_pt[0] - w / 2, new_pt[1] - h / 2,
                new_pt[0] + w / 2, new_pt[1] + h / 2,
            ])

    def _output(self) -> list[tuple]:
        """输出满足 min_hits 条件的轨迹

        对于刚被观测到的轨迹，输出原始检测坐标（last_observation）
        而非 Kalman 滤波后的状态。Kalman 滤波器仍用于帧间预测和匹配，
        但其状态在低帧率/离散输入场景下会因 dt=1 假设产生位置偏移。
        """
        results = []
        for trk in self.trackers:
            if trk.time_since_update <= 0 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # 使用原始观测坐标，避免 Kalman 滤波引入的位置偏移
                bbox = trk.last_observation if trk.observed else trk.get_state()
                results.append((
                    *bbox, trk.id, trk.cls_id, trk.conf
                ))
        return results

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.reset_count()

    # ── 匈牙利匹配 ────────────────────────────────────────────────────────

    @staticmethod
    def _linear_assignment(cost_matrix: np.ndarray, threshold: float):
        """IoU 匹配（越大越好，转换为代价后用匈牙利）"""
        M, N = cost_matrix.shape
        if M == 0 or N == 0:
            return np.array([]), np.array([]), np.arange(M), np.arange(N)

        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        matched_d, matched_t = [], []
        unmatched_d = set(range(M))
        unmatched_t = set(range(N))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= threshold:
                matched_d.append(r)
                matched_t.append(c)
                unmatched_d.discard(r)
                unmatched_t.discard(c)

        return (
            np.array(matched_d, dtype=int),
            np.array(matched_t, dtype=int),
            np.array(sorted(unmatched_d), dtype=int),
            np.array(sorted(unmatched_t), dtype=int),
        )

    @staticmethod
    def _linear_assignment_dist(dist_matrix: np.ndarray, threshold: float,
                                det_indices: np.ndarray, trk_indices: np.ndarray):
        """距离匹配（越小越好）"""
        M, N = dist_matrix.shape
        if M == 0 or N == 0:
            return np.array([]), np.array([]), det_indices, trk_indices

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matched_d, matched_t = [], []
        um_d = set(range(M))
        um_t = set(range(N))

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= threshold:
                matched_d.append(det_indices[r])
                matched_t.append(trk_indices[c])
                um_d.discard(r)
                um_t.discard(c)

        return (
            np.array(matched_d, dtype=int),
            np.array(matched_t, dtype=int),
            np.array([det_indices[i] for i in sorted(um_d)], dtype=int),
            np.array([trk_indices[i] for i in sorted(um_t)], dtype=int),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 统一接口：MultiObjectTracker
# ─────────────────────────────────────────────────────────────────────────────

class MultiObjectTracker:
    """
    统一追踪器接口

    根据 tracker_type 参数选择底层算法:
        "bytetrack" — supervision ByteTrack
        "ocsort"    — OC-SORT（推荐低帧率场景）
        "botsort"   — OC-SORT + GMC（推荐无人机场景）
        "dist_tracker" — Dist-Tracker FLIT 轻量适配层
        "official_ocsort" / "official_botsort" — 官方源码适配层
    """

    _CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(Config.CLASSES)}
    _ID_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(Config.CLASSES)}

    def __init__(self, frame_rate: float = 25.0,
                 tracker_type: str | None = None,
                 min_hits: int | None = None):
        self.frame_rate = max(1.0, float(frame_rate))
        self.tracker_type = tracker_type or Config.TRACKER_TYPE
        self._min_hits = min_hits if min_hits is not None else Config.TRACKER_MIN_HITS

        # 根据帧率动态调整距离阈值
        # 低帧率 → 帧间位移更大 → 需要更大的距离阈值
        fps_scale = max(1.0, 25.0 / self.frame_rate)
        dist_thresh = Config.TRACKER_DIST_THRESH * fps_scale

        if self.tracker_type == "bytetrack":
            import supervision as sv
            self._sv = sv
            self._bt = sv.ByteTrack(
                track_activation_threshold=Config.TRACKER_ACTIVATION_THRESH,
                lost_track_buffer=Config.TRACKER_LOST_BUFFER,
                minimum_matching_threshold=Config.TRACKER_MATCH_THRESH,
                frame_rate=int(self.frame_rate),
            )
        elif self.tracker_type in ("ocsort", "botsort"):
            self._ocsort = OCSortTracker(
                max_age=Config.TRACKER_LOST_BUFFER,
                min_hits=self._min_hits,
                iou_threshold=Config.TRACKER_MATCH_THRESH,
                dist_threshold=dist_thresh,
                use_oru=True,
                use_ocm=True,
            )
            if self.tracker_type == "botsort":
                self._gmc = GMC(
                    method=Config.GMC_METHOD,
                    downscale=Config.GMC_DOWNSCALE,
                )
            else:
                self._gmc = None
        elif self.tracker_type == "dist_tracker":
            from target_module.image_detect_module.utils.dist_tracker import DistTracker
            self._dist_tracker = DistTracker(
                frame_rate=self.frame_rate,
                max_age=Config.TRACKER_LOST_BUFFER,
                min_hits=self._min_hits,
            )
        elif self.tracker_type in ("official_ocsort", "official_botsort"):
            self._official = OfficialTrackerAdapter(
                tracker_type=self.tracker_type,
                frame_rate=self.frame_rate,
                min_hits=self._min_hits,
            )
        else:
            raise ValueError(f"不支持的追踪器类型: {self.tracker_type}")

        print(f"[Tracker] 已初始化 {self.tracker_type} (fps={self.frame_rate:.1f}, "
              f"dist_thresh={dist_thresh:.0f}px)")

    def update(self, detections: list[dict], frame_shape: tuple,
               frame: np.ndarray | None = None) -> list[dict]:
        """
        对单帧执行追踪更新。

        Parameters
        ----------
        detections : list[dict]
            检测结果列表 (xywh 格式)
        frame_shape : tuple
            帧的 shape (H, W[, C])
        frame : np.ndarray | None
            原始帧图像（botsort / dist_tracker 用于 GMC 计算）
        """
        if self.tracker_type == "bytetrack":
            return self._update_bytetrack(detections, frame_shape)
        elif self.tracker_type == "dist_tracker":
            return self._dist_tracker.update(detections, frame_shape, frame=frame)
        elif self.tracker_type in ("official_ocsort", "official_botsort"):
            return self._official.update(detections, frame_shape, frame=frame)
        else:
            return self._update_ocsort(detections, frame_shape, frame)

    def reset(self) -> None:
        """重置追踪状态"""
        if self.tracker_type == "bytetrack":
            self._bt = self._sv.ByteTrack(
                track_activation_threshold=Config.TRACKER_ACTIVATION_THRESH,
                lost_track_buffer=Config.TRACKER_LOST_BUFFER,
                minimum_matching_threshold=Config.TRACKER_MATCH_THRESH,
                frame_rate=int(self.frame_rate),
            )
        elif self.tracker_type == "dist_tracker":
            self._dist_tracker.reset()
        elif self.tracker_type in ("official_ocsort", "official_botsort"):
            self._official.reset()
        else:
            self._ocsort.reset()
            if self._gmc:
                self._gmc.reset()

    def get_recent_tracks(self, max_time_since_update: int = 1) -> list[dict]:
        """Return short-gap predicted tracks for extraction fallback."""
        if max_time_since_update <= 0:
            return []
        if self.tracker_type == "dist_tracker":
            return self._dist_tracker.get_recent_tracks(max_time_since_update=max_time_since_update)
        if self.tracker_type in ("bytetrack", "official_ocsort", "official_botsort"):
            return []

        results = []
        for trk in self._ocsort.trackers:
            if trk.time_since_update <= 0 or trk.time_since_update > max_time_since_update:
                continue
            if trk.hits < self._min_hits:
                continue
            x1, y1, x2, y2 = trk.get_state()
            results.append({
                "track_id": int(trk.id),
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "confidence": round(float(trk.conf), 4),
                "class": self._ID_TO_CLASS.get(int(trk.cls_id), Config.CLASSES[0]),
                "class_confidence": round(float(trk.conf), 4),
                "is_tracker_prediction": True,
            })
        return results

    # ── ByteTrack (supervision) ──────────────────────────────────────────

    def _update_bytetrack(self, detections: list[dict],
                          frame_shape: tuple) -> list[dict]:
        sv = self._sv
        if not detections:
            empty = sv.Detections.empty()
            self._bt.update_with_detections(empty)
            return []

        h, w = frame_shape[:2]
        xyxy_list, conf_list, cls_list = [], [], []
        for d in detections:
            x1, y1 = float(d["x"]), float(d["y"])
            x2, y2 = x1 + float(d["w"]), y1 + float(d["h"])
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 > x1 and y2 > y1:
                xyxy_list.append([x1, y1, x2, y2])
                conf_list.append(float(d.get("confidence", 0.0)))
                cls_list.append(int(self._CLASS_TO_ID.get(d.get("class", "USV"), 0)))

        if not xyxy_list:
            return []

        sv_dets = sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            confidence=np.array(conf_list, dtype=np.float32),
            class_id=np.array(cls_list, dtype=int),
        )
        tracked = self._bt.update_with_detections(sv_dets)
        if tracked.tracker_id is None or len(tracked) == 0:
            return []

        results = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            results.append({
                "track_id": int(tracked.tracker_id[i]),
                "x": int(x1), "y": int(y1),
                "w": int(x2 - x1), "h": int(y2 - y1),
                "confidence": round(float(tracked.confidence[i]), 4),
                "class": self._ID_TO_CLASS.get(int(tracked.class_id[i]), Config.CLASSES[0]),
                "class_confidence": round(float(tracked.confidence[i]), 4),
            })
        return results

    # ── OC-SORT / BoT-SORT ──────────────────────────────────────────────

    def _update_ocsort(self, detections: list[dict], frame_shape: tuple,
                       frame: np.ndarray | None) -> list[dict]:
        # GMC：在 predict 之前补偿相机运动
        if self._gmc is not None and frame is not None:
            M = self._gmc.apply(frame)
            self._ocsort.apply_gmc(M)

        # 转换输入格式
        if not detections:
            dets = np.empty((0, 4))
            confs = np.empty(0)
            cls_ids = np.empty(0, dtype=int)
        else:
            h, w = frame_shape[:2]
            dets_list, confs_list, cls_list = [], [], []
            for d in detections:
                x1, y1 = float(d["x"]), float(d["y"])
                x2, y2 = x1 + float(d["w"]), y1 + float(d["h"])
                x1, x2 = max(0.0, x1), min(float(w), x2)
                y1, y2 = max(0.0, y1), min(float(h), y2)
                if x2 > x1 and y2 > y1:
                    dets_list.append([x1, y1, x2, y2])
                    confs_list.append(float(d.get("confidence", 0.0)))
                    cls_list.append(int(self._CLASS_TO_ID.get(d.get("class", "USV"), 0)))
            if not dets_list:
                dets = np.empty((0, 4))
                confs = np.empty(0)
                cls_ids = np.empty(0, dtype=int)
            else:
                dets = np.array(dets_list, dtype=np.float64)
                confs = np.array(confs_list, dtype=np.float64)
                cls_ids = np.array(cls_list, dtype=int)

        tracked = self._ocsort.update(dets, confs, cls_ids)

        results = []
        for item in tracked:
            x1, y1, x2, y2, tid, cls_id, conf = item
            results.append({
                "track_id": int(tid),
                "x": int(x1), "y": int(y1),
                "w": int(x2 - x1), "h": int(y2 - y1),
                "confidence": round(float(conf), 4),
                "class": self._ID_TO_CLASS.get(int(cls_id), Config.CLASSES[0]),
                "class_confidence": round(float(conf), 4),
            })
        return results
