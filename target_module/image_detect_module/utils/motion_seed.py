"""
MS-DC-ELT motion seed generation.

This module extracts class-agnostic motion boxes from adjacent frames. It is a
debug/observation source only; it does not feed OC-SORT, BoT-SORT, or any other
baseline tracker by itself.
"""

from __future__ import annotations

import cv2
import numpy as np

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.gmc import GMC


class MotionSeedGenerator:
    """Generate motion boxes from frame differences with optional GMC."""

    def __init__(self, config=Config):
        self.config = config
        self.enabled = bool(getattr(config, "MSDC_MOTION_ENABLE", True))
        self.use_gmc = bool(getattr(config, "MSDC_MOTION_USE_GMC", True))
        self.diff_percentile = float(getattr(config, "MSDC_MOTION_DIFF_PERCENTILE", 97))
        self.min_area_ratio = float(getattr(config, "MSDC_MOTION_MIN_AREA_RATIO", 1e-6))
        self.max_area_ratio = float(getattr(config, "MSDC_MOTION_MAX_AREA_RATIO", 0.02))
        self.max_boxes = int(getattr(config, "MSDC_MOTION_MAX_BOXES", 64))
        self.clutter_component_thresh = int(getattr(config, "MSDC_MOTION_CLUTTER_COMPONENT_THRESH", 512))
        self.clutter_max_boxes = int(getattr(config, "MSDC_MOTION_CLUTTER_MAX_BOXES", min(12, self.max_boxes)))
        self.min_box_size = int(getattr(config, "MSDC_MOTION_MIN_BOX_SIZE", 4))
        self.debug = bool(getattr(config, "MSDC_MOTION_DEBUG", False))
        self.morph_kernel = max(1, int(getattr(config, "MSDC_MOTION_MORPH_KERNEL", 3)))
        self.max_aspect_ratio = float(getattr(config, "MSDC_MOTION_MAX_ASPECT_RATIO", 8.0))
        self.gmc_method = getattr(config, "GMC_METHOD", "sparse_flow") if self.use_gmc else "none"
        self.gmc_downscale = int(getattr(config, "GMC_DOWNSCALE", 2))
        self._gmc = GMC(method=self.gmc_method, downscale=self.gmc_downscale)
        self._prev_gray: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray = None
        self.last_mask = None
        self._gmc.reset()

    def update(self, frame: np.ndarray, frame_idx: int | None = None) -> tuple[list[dict], dict]:
        """
        Process one BGR frame and return motion boxes plus debug metadata.

        The first frame initializes internal state and returns no boxes because
        no previous frame is available for differencing.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("MotionSeedGenerator.update expects a non-empty numpy frame")

        idx = 0 if frame_idx is None else int(frame_idx)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if not self.enabled:
            self.last_mask = np.zeros((h, w), dtype=np.uint8)
            return [], self._debug_info(idx, 0, 0.0, False, True, 0, 0, False, self.max_boxes)

        if self._prev_gray is None:
            self._prev_gray = gray
            self.last_mask = np.zeros((h, w), dtype=np.uint8)
            if self.use_gmc:
                self._gmc.apply(frame)
            return [], self._debug_info(idx, 0, 0.0, False, True, 0, 0, False, self.max_boxes)

        used_gmc = self.use_gmc and self.gmc_method != "none"
        prev_gray = self._prev_gray
        if used_gmc:
            matrix = self._gmc.apply(frame)
            prev_gray = cv2.warpAffine(
                self._prev_gray,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        diff = cv2.absdiff(gray, prev_gray)
        threshold = float(np.percentile(diff, min(max(self.diff_percentile, 0.0), 100.0)))
        mask = self._threshold_diff(diff, threshold)
        boxes, raw_components, kept_components, clutter_limited, effective_max_boxes = self._mask_to_boxes(mask, diff, idx)

        self._prev_gray = gray
        self.last_mask = mask
        debug_info = self._debug_info(
            idx,
            len(boxes),
            threshold,
            used_gmc,
            False,
            raw_components,
            kept_components,
            clutter_limited,
            effective_max_boxes,
        )
        return boxes, debug_info

    def _threshold_diff(self, diff: np.ndarray, threshold: float) -> np.ndarray:
        if np.max(diff) <= 0:
            return np.zeros(diff.shape, dtype=np.uint8)

        if threshold <= 0:
            mask = (diff > 0).astype(np.uint8) * 255
        else:
            mask = (diff >= threshold).astype(np.uint8) * 255

        kernel = np.ones((self.morph_kernel, self.morph_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _mask_to_boxes(
        self,
        mask: np.ndarray,
        diff: np.ndarray,
        frame_idx: int,
    ) -> tuple[list[dict], int, int, bool, int]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        frame_area = float(mask.shape[0] * mask.shape[1])
        min_area = max(1.0, self.min_area_ratio * frame_area)
        max_area = max(min_area, self.max_area_ratio * frame_area)
        if num_labels > 1:
            label_sums = np.bincount(
                labels.reshape(-1),
                weights=diff.reshape(-1).astype(np.float64),
                minlength=num_labels,
            )
        else:
            label_sums = np.zeros(num_labels, dtype=np.float64)

        boxes: list[dict] = []
        raw_components = max(0, num_labels - 1)
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            area = float(area)
            if area < min_area or area > max_area:
                continue
            if w <= 0 or h <= 0:
                continue
            if min(int(w), int(h)) < max(1, self.min_box_size):
                continue
            aspect_ratio = max(float(w) / float(h), float(h) / float(w))
            if aspect_ratio > self.max_aspect_ratio:
                continue

            score = float(label_sums[label] / max(area, 1.0) / 255.0)
            boxes.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h),
                "score": round(max(0.0, min(score, 1.0)), 4),
                "source": "motion",
                "frame_idx": int(frame_idx),
            })

        boxes.sort(key=lambda box: (box["score"], box["w"] * box["h"]), reverse=True)
        clutter_limited = bool(
            self.clutter_component_thresh > 0
            and raw_components >= self.clutter_component_thresh
        )
        effective_max_boxes = max(0, self.max_boxes)
        if clutter_limited:
            effective_max_boxes = min(effective_max_boxes, max(0, self.clutter_max_boxes))
        capped_boxes = boxes[:effective_max_boxes]
        return capped_boxes, raw_components, len(boxes), clutter_limited, effective_max_boxes

    @staticmethod
    def _debug_info(
        frame_idx: int,
        num_motion_boxes: int,
        diff_threshold: float,
        used_gmc: bool,
        is_first_frame: bool,
        raw_component_count: int,
        kept_component_count: int,
        clutter_limited: bool,
        effective_max_boxes: int,
    ) -> dict:
        return {
            "frame_idx": int(frame_idx),
            "num_motion_boxes": int(num_motion_boxes),
            "diff_threshold": round(float(diff_threshold), 4),
            "used_gmc": bool(used_gmc),
            "is_first_frame": bool(is_first_frame),
            "raw_component_count": int(raw_component_count),
            "kept_component_count": int(kept_component_count),
            "clutter_limited": bool(clutter_limited),
            "effective_max_boxes": int(effective_max_boxes),
        }
