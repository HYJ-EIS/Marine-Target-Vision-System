"""
Global Motion Compensation (GMC) — 相机运动补偿

估计相邻帧之间的全局仿射变换矩阵，用于在追踪前移除相机的自身运动。
在无人机/船载相机场景下，相机运动会导致所有目标同步位移，
GMC 能把这部分全局位移消除，让 Kalman 预测更准确。

支持两种方法：
1. sparse_flow: 稀疏光流（Shi-Tomasi + Lucas-Kanade），速度快
2. orb: ORB 特征匹配，更鲁棒但略慢
"""

import cv2
import numpy as np


class GMC:
    """全局运动补偿器"""

    def __init__(self, method: str = "sparse_flow", downscale: int = 2):
        """
        Parameters
        ----------
        method : str
            "sparse_flow" 或 "orb" 或 "none"
        downscale : int
            处理前下采样倍率，加速计算
        """
        self.method = method
        self.downscale = max(1, downscale)
        self._prev_gray = None

        if method == "orb":
            self._orb = cv2.ORB_create(500)
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        计算当前帧相对上一帧的仿射变换矩阵 M (2x3)

        返回
        ----
        M : np.ndarray (2x3)
            仿射变换矩阵。若无上一帧或估计失败，返回单位矩阵。
        """
        if self.method == "none":
            return np.eye(2, 3, dtype=np.float64)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            h, w = gray.shape
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))

        if self._prev_gray is None:
            self._prev_gray = gray
            return np.eye(2, 3, dtype=np.float64)

        try:
            if self.method == "sparse_flow":
                M = self._sparse_flow(self._prev_gray, gray)
            elif self.method == "orb":
                M = self._orb_match(self._prev_gray, gray)
            else:
                M = np.eye(2, 3, dtype=np.float64)
        except Exception:
            M = np.eye(2, 3, dtype=np.float64)

        # 缩放矩阵回原始分辨率
        if self.downscale > 1:
            M[0, 2] *= self.downscale
            M[1, 2] *= self.downscale

        self._prev_gray = gray
        return M

    def reset(self):
        """重置状态（切换视频时调用）"""
        self._prev_gray = None

    # ── 稀疏光流 ──────────────────────────────────────────────────────────

    def _sparse_flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        # Shi-Tomasi 角点检测
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=3,
        )
        p0 = cv2.goodFeaturesToTrack(prev, mask=None, **feature_params)
        if p0 is None or len(p0) < 4:
            return np.eye(2, 3, dtype=np.float64)

        # Lucas-Kanade 光流
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **lk_params)

        # 筛选匹配成功的点
        good_old = p0[status.flatten() == 1]
        good_new = p1[status.flatten() == 1]

        if len(good_old) < 4:
            return np.eye(2, 3, dtype=np.float64)

        # 估计仿射变换（RANSAC 鲁棒估计）
        M, inliers = cv2.estimateAffinePartial2D(
            good_old, good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        return M if M is not None else np.eye(2, 3, dtype=np.float64)

    # ── ORB 特征匹配 ──────────────────────────────────────────────────────

    def _orb_match(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        kp1, des1 = self._orb.detectAndCompute(prev, None)
        kp2, des2 = self._orb.detectAndCompute(curr, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return np.eye(2, 3, dtype=np.float64)

        matches = self._bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) < 4:
            return np.eye(2, 3, dtype=np.float64)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        return M if M is not None else np.eye(2, 3, dtype=np.float64)
