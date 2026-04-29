"""
SORT/OC-SORT 使用的 Kalman 滤波器

状态向量 (8 维):
    [cx, cy, s, r, vx, vy, vs, vr]
    cx, cy = 中心坐标
    s      = 面积 (width * height)
    r      = 宽高比 (width / height)
    vx, vy, vs, vr = 对应的速度

观测向量 (4 维):
    [cx, cy, s, r]
"""

import numpy as np


class KalmanBoxTracker:
    """为单个目标维护 Kalman 状态"""

    _count = 0  # 全局 ID 计数器

    def __init__(self, bbox: np.ndarray, cls_id: int = 0, conf: float = 0.0):
        """
        Parameters
        ----------
        bbox : np.ndarray
            [x1, y1, x2, y2] 格式的检测框
        cls_id : int
            类别 ID
        conf : float
            置信度
        """
        # 状态转移矩阵 F (8x8)
        self.F = np.eye(8, dtype=np.float64)
        self.F[:4, 4:] = np.eye(4)  # 位置 += 速度 * dt

        # 观测矩阵 H (4x8)
        self.H = np.eye(4, 8, dtype=np.float64)

        # 过程噪声 Q
        self.Q = np.eye(8, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01  # 速度噪声较小
        self.Q[:4, :4] *= 1.0   # 位置噪声

        # 测量噪声 R
        self.R = np.eye(4, dtype=np.float64) * 1.0
        self.R[2, 2] *= 10.0   # 面积测量噪声较大
        self.R[3, 3] *= 10.0   # 宽高比测量噪声较大

        # 状态协方差 P
        self.P = np.eye(8, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0  # 初始速度高度不确定
        self.P *= 10.0

        # 初始状态
        z = self._bbox_to_z(bbox)
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[:4] = z.reshape(4, 1)

        # 追踪元数据
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.cls_id = cls_id
        self.conf = conf

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

        # OC-SORT: 保存历史观测用于虚拟轨迹重更新
        self.history_observations: list[np.ndarray] = [bbox.copy()]
        self.last_observation = bbox.copy()
        self.observed = True  # 本帧是否被观测到

        # 速度方向（用于 OCM）
        self.velocity = np.zeros(2)  # (vx, vy) in pixel space

    def predict(self) -> np.ndarray:
        """预测下一帧位置，返回 [x1, y1, x2, y2]"""
        # 面积不允许为负
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.observed = False

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def update(self, bbox: np.ndarray, cls_id: int = 0, conf: float = 0.0):
        """用新观测更新 Kalman 状态"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.observed = True
        self.cls_id = cls_id
        self.conf = conf

        # 计算速度方向（像素空间，用于 OCM）
        new_center = np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
        ])
        old_center = np.array([
            (self.last_observation[0] + self.last_observation[2]) / 2,
            (self.last_observation[1] + self.last_observation[3]) / 2,
        ])
        self.velocity = new_center - old_center

        self.last_observation = bbox.copy()
        self.history_observations.append(bbox.copy())

        # Kalman update
        z = self._bbox_to_z(bbox).reshape(4, 1)
        y = z - self.H @ self.x  # 残差
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """返回当前状态的 [x1, y1, x2, y2]"""
        return self._z_to_bbox(self.x[:4].flatten())

    def apply_affine(self, M: np.ndarray):
        """
        GMC：对 Kalman 状态施加仿射变换矩阵 (2x3)
        用于补偿全局相机运动
        """
        # 将预测的中心点做仿射变换
        cx, cy = float(self.x[0]), float(self.x[1])
        pt = np.array([cx, cy, 1.0])
        new_pt = M @ pt
        self.x[0] = new_pt[0]
        self.x[1] = new_pt[1]

    # ── OC-SORT: 虚拟轨迹重更新 ──────────────────────────────────────────

    def re_update_with_virtual_trajectory(self, new_bbox: np.ndarray,
                                          cls_id: int, conf: float):
        """
        OC-SORT ORU: 当轨迹在丢失 N 帧后被重新关联时，
        通过线性插值生成虚拟观测，逐帧重做 Kalman update，
        纠正因丢失期间纯预测导致的状态漂移。
        """
        lost_frames = self.time_since_update
        if lost_frames <= 1:
            self.update(new_bbox, cls_id, conf)
            return

        last_obs = self.last_observation
        # 线性插值 lost_frames 个虚拟观测
        for i in range(1, lost_frames + 1):
            t = i / (lost_frames + 1)
            virtual_bbox = last_obs * (1 - t) + new_bbox * t
            self.predict()
            self.update(virtual_bbox, cls_id, conf)

        # 最后用真实观测再更新一次
        self.update(new_bbox, cls_id, conf)

    # ── 坐标变换 ──────────────────────────────────────────────────────────

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """[x1, y1, x2, y2] -> [cx, cy, s, r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([cx, cy, s, r], dtype=np.float64)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """[cx, cy, s, r] -> [x1, y1, x2, y2]"""
        s = max(z[2], 0)
        r = z[3]
        w = np.sqrt(s * r) if s > 0 and r > 0 else 0
        h = s / max(w, 1e-6)
        x1 = z[0] - w / 2.0
        y1 = z[1] - h / 2.0
        x2 = z[0] + w / 2.0
        y2 = z[1] + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float64)

    @classmethod
    def reset_count(cls):
        """重置全局 ID 计数器"""
        cls._count = 0
