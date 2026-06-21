"""IR thermal presence support for RGB-track ROI redetection experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from tools.evaluation.proben_fusion import compute_center_distance, compute_iou


@dataclass
class ThermalPresence:
    decision: str
    thermal_score: float
    hot_area_ratio: float
    hot_box_ir: dict | None
    roi_ir: dict | None
    reject_reason: str = ""


def transform_box_affine(box: dict, matrix: np.ndarray, shape: tuple[int, ...]) -> dict | None:
    """Transform an xywh box with a 2x3 affine matrix and clip to image shape."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("matrix must have shape 2x3")
    x, y = float(box["x"]), float(box["y"])
    w, h = float(box["w"]), float(box["h"])
    if w <= 0 or h <= 0:
        return None
    corners = np.array(
        [[x, y, 1.0], [x + w, y, 1.0], [x, y + h, 1.0], [x + w, y + h, 1.0]],
        dtype=np.float64,
    )
    mapped = corners @ matrix.T
    x1, y1 = float(np.min(mapped[:, 0])), float(np.min(mapped[:, 1]))
    x2, y2 = float(np.max(mapped[:, 0])), float(np.max(mapped[:, 1]))
    image_h, image_w = int(shape[0]), int(shape[1])
    x1 = min(max(0.0, x1), float(image_w))
    x2 = min(max(0.0, x2), float(image_w))
    y1 = min(max(0.0, y1), float(image_h))
    y2 = min(max(0.0, y2), float(image_h))
    if x2 <= x1 or y2 <= y1:
        return None
    mapped_box = dict(box)
    mapped_box.update({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    return mapped_box


def inverse_affine(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("matrix must have shape 2x3")
    return cv2.invertAffineTransform(matrix)


def map_rgb_box_to_ir(box: dict, ir_to_rgb_affine: np.ndarray, ir_shape: tuple[int, ...]) -> dict | None:
    """Map an RGB xywh box to IR image coordinates."""
    return transform_box_affine(box, inverse_affine(ir_to_rgb_affine), ir_shape)


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _clip_int_roi(box: dict, shape: tuple[int, ...], pad: float = 0.0) -> tuple[int, int, int, int] | None:
    h, w = int(shape[0]), int(shape[1])
    x, y = float(box["x"]), float(box["y"])
    bw, bh = float(box["w"]), float(box["h"])
    if bw <= 0 or bh <= 0:
        return None
    cx, cy = x + bw / 2.0, y + bh / 2.0
    bw *= 1.0 + 2.0 * float(pad)
    bh *= 1.0 + 2.0 * float(pad)
    x0 = max(0, int(math.floor(cx - bw / 2.0)))
    y0 = max(0, int(math.floor(cy - bh / 2.0)))
    x1 = min(w, int(math.ceil(cx + bw / 2.0)))
    y1 = min(h, int(math.ceil(cy + bh / 2.0)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def score_thermal_presence(ir_frame: np.ndarray, roi_ir: dict, args) -> ThermalPresence:
    """Score whether an IR ROI contains a stable local hot response."""
    roi = _clip_int_roi(roi_ir, ir_frame.shape, pad=0.0)
    if roi is None:
        return ThermalPresence("absent", 0.0, 0.0, None, None, "invalid_roi")
    ring = _clip_int_roi(roi_ir, ir_frame.shape, pad=float(args.ir_presence_bg_pad))
    if ring is None:
        ring = roi

    gray = _gray(ir_frame)
    x0, y0, x1, y1 = roi
    rx0, ry0, rx1, ry1 = ring
    roi_pixels = gray[y0:y1, x0:x1]
    bg_pixels = gray[ry0:ry1, rx0:rx1]
    if roi_pixels.size == 0 or bg_pixels.size == 0:
        return ThermalPresence("absent", 0.0, 0.0, None, None, "empty_roi")

    bg_median = float(np.median(bg_pixels))
    bg_std = float(np.std(bg_pixels))
    bg_std = max(bg_std, float(args.ir_presence_min_bg_std))
    roi_p95 = float(np.percentile(roi_pixels, 95))
    thermal_score = (roi_p95 - bg_median) / bg_std
    threshold = bg_median + float(args.ir_presence_z_thresh) * bg_std
    hot_mask = (roi_pixels >= threshold).astype(np.uint8)
    hot_area_ratio = float(np.count_nonzero(hot_mask)) / float(max(1, hot_mask.size))

    hot_box = None
    if np.any(hot_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hot_mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = int(np.argmax(areas)) + 1
            bx = int(stats[best, cv2.CC_STAT_LEFT])
            by = int(stats[best, cv2.CC_STAT_TOP])
            bw = int(stats[best, cv2.CC_STAT_WIDTH])
            bh = int(stats[best, cv2.CC_STAT_HEIGHT])
            hot_box = {"x": float(x0 + bx), "y": float(y0 + by), "w": float(bw), "h": float(bh)}

    present = (
        thermal_score >= float(args.ir_presence_z_thresh)
        and hot_area_ratio >= float(args.ir_presence_min_hot_area_ratio)
        and hot_box is not None
    )
    decision = "present" if present else "absent"
    reason = "" if present else "weak_thermal_response"
    return ThermalPresence(
        decision=decision,
        thermal_score=float(thermal_score),
        hot_area_ratio=float(hot_area_ratio),
        hot_box_ir=hot_box,
        roi_ir={"x": float(x0), "y": float(y0), "w": float(x1 - x0), "h": float(y1 - y0)},
        reject_reason=reason,
    )


def _diag(box: dict) -> float:
    return math.hypot(float(box["w"]), float(box["h"]))


def _matched_by_rgb(track_box: dict, rgb_detections: list[dict], args) -> bool:
    for det in rgb_detections:
        iou = compute_iou(track_box, det)
        dist = compute_center_distance(track_box, det)
        if (
            iou >= float(args.ir_presence_rgb_match_iou)
            or dist <= float(args.ir_presence_rgb_match_dist_factor) * _diag(track_box)
        ):
            return True
    return False


def _pad_box(box: dict, shape: tuple[int, ...], scale: float, min_size: float) -> tuple[int, int, int, int] | None:
    h, w = int(shape[0]), int(shape[1])
    bw = max(float(min_size), float(box["w"]) * float(scale))
    bh = max(float(min_size), float(box["h"]) * float(scale))
    cx = float(box["x"]) + float(box["w"]) / 2.0
    cy = float(box["y"]) + float(box["h"]) / 2.0
    x0 = max(0, int(round(cx - bw / 2.0)))
    y0 = max(0, int(round(cy - bh / 2.0)))
    x1 = min(w, int(round(cx + bw / 2.0)))
    y1 = min(h, int(round(cy + bh / 2.0)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _map_local_box(local_box: dict, x0: int, y0: int, scale: float) -> dict:
    inv = 1.0 / max(float(scale), 1e-9)
    mapped = dict(local_box)
    mapped["x"] = float(local_box.get("x", 0.0)) * inv + float(x0)
    mapped["y"] = float(local_box.get("y", 0.0)) * inv + float(y0)
    mapped["w"] = float(local_box.get("w", 0.0)) * inv
    mapped["h"] = float(local_box.get("h", 0.0)) * inv
    mapped["confidence"] = float(local_box.get("confidence", local_box.get("score", 0.0)))
    return mapped


class IRPresenceSupport:
    """Use IR presence only to trigger RGB ROI low-threshold redetection."""

    def update(
        self,
        frame_id: int,
        rgb_frame: np.ndarray,
        ir_frame: np.ndarray,
        track_boxes: list[dict],
        current_rgb_detections: list[dict],
        ir_to_rgb_affine: np.ndarray,
        processor,
        args,
    ) -> tuple[list[dict], list[dict]]:
        detections: list[dict] = []
        diagnostics: list[dict] = []
        for track in track_boxes:
            if int(track.get("time_since_update", 0)) > int(args.ir_presence_max_track_age):
                diagnostics.append(self._diag_row(frame_id, track, None, None, "skipped", "track_age_exceeded"))
                continue
            if _matched_by_rgb(track, current_rgb_detections, args):
                diagnostics.append(self._diag_row(frame_id, track, None, None, "skipped", "rgb_matched"))
                continue

            roi_ir = map_rgb_box_to_ir(track, ir_to_rgb_affine, ir_frame.shape)
            if roi_ir is None:
                diagnostics.append(self._diag_row(frame_id, track, None, None, "rejected", "rgb_to_ir_mapping_failed"))
                continue
            presence = score_thermal_presence(ir_frame, roi_ir, args)
            if presence.decision != "present" or presence.hot_box_ir is None:
                diagnostics.append(self._diag_row(
                    frame_id, track, presence, None, "rejected", presence.reject_reason
                ))
                continue

            hot_box_rgb = transform_box_affine(presence.hot_box_ir, ir_to_rgb_affine, rgb_frame.shape)
            if hot_box_rgb is None:
                diagnostics.append(self._diag_row(frame_id, track, presence, None, "rejected", "ir_to_rgb_mapping_failed"))
                continue
            roi_det, roi_debug = self._redetect_rgb_roi(
                rgb_frame=rgb_frame,
                trigger_box_rgb=hot_box_rgb,
                track_box=track,
                processor=processor,
                args=args,
            )
            if roi_det is None:
                diagnostics.append(self._diag_row(
                    frame_id, track, presence, roi_debug, "rejected", roi_debug.get("reject_reason", "no_roi_detection")
                ))
                continue
            roi_det["allow_new_track"] = False
            roi_det["support_type"] = "ir_presence_roi_redetect"
            roi_det["linked_track_id"] = int(track.get("track_id", -1))
            detections.append(roi_det)
            diagnostics.append(self._diag_row(frame_id, track, presence, roi_debug, "emitted", ""))
        return detections, diagnostics

    def _redetect_rgb_roi(self, rgb_frame: np.ndarray, trigger_box_rgb: dict, track_box: dict,
                          processor, args) -> tuple[dict | None, dict[str, Any]]:
        roi = _pad_box(
            trigger_box_rgb,
            rgb_frame.shape,
            scale=float(args.ir_presence_roi_pad),
            min_size=float(args.ir_presence_min_crop_size),
        )
        if roi is None:
            return None, {"reject_reason": "invalid_rgb_roi"}
        x0, y0, x1, y1 = roi
        crop = rgb_frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None, {"roi_rgb": [x0, y0, x1, y1], "reject_reason": "empty_rgb_roi"}
        scale = float(args.ir_presence_roi_upscale)
        inference = crop
        if scale > 1.0:
            inference = cv2.resize(
                crop,
                (max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale)))),
                interpolation=cv2.INTER_LINEAR,
            )
        stats = processor.process_frame(inference, "visible", conf_override=float(args.ir_presence_rgb_roi_low_conf))
        local_boxes = list((stats or {}).get("boxes", []))
        accepted: list[dict] = []
        rejected = 0
        for local in local_boxes:
            mapped = _map_local_box(local, x0, y0, scale)
            iou = compute_iou(mapped, track_box)
            dist = compute_center_distance(mapped, track_box)
            passes = (
                iou >= float(args.ir_presence_redetect_iou)
                or dist <= float(args.ir_presence_redetect_dist_factor) * _diag(track_box)
            )
            if not passes:
                rejected += 1
                continue
            mapped["class"] = track_box.get("class", mapped.get("class", "target"))
            mapped["class_confidence"] = float(mapped.get("class_confidence", mapped.get("confidence", 0.0)))
            mapped["_presence_iou"] = iou
            mapped["_presence_center_distance"] = dist
            accepted.append(mapped)
        accepted.sort(
            key=lambda item: (
                -float(item.get("confidence", 0.0)),
                float(item.get("_presence_center_distance", 0.0)),
            )
        )
        debug = {
            "roi_rgb": [int(x0), int(y0), int(x1), int(y1)],
            "num_local_boxes": int(len(local_boxes)),
            "num_rejected_geometry": int(rejected),
        }
        if not accepted:
            debug["reject_reason"] = "no_geometry_consistent_rgb_low_det"
            return None, debug
        out = accepted[0]
        debug["accepted_box_rgb"] = {k: out[k] for k in ("x", "y", "w", "h", "confidence")}
        debug["accepted_iou"] = float(out.pop("_presence_iou", 0.0))
        debug["accepted_center_distance"] = float(out.pop("_presence_center_distance", 0.0))
        return out, debug

    @staticmethod
    def _diag_row(
        frame_id: int,
        track: dict,
        presence: ThermalPresence | None,
        roi_debug: dict | None,
        decision: str,
        reject_reason: str,
    ) -> dict[str, Any]:
        return {
            "frame_id": int(frame_id),
            "track_id": track.get("track_id"),
            "track_box_rgb": {k: track.get(k) for k in ("x", "y", "w", "h")},
            "track_age": int(track.get("time_since_update", 0)),
            "presence_decision": None if presence is None else presence.decision,
            "thermal_score": 0.0 if presence is None else presence.thermal_score,
            "hot_area_ratio": 0.0 if presence is None else presence.hot_area_ratio,
            "roi_ir": None if presence is None else presence.roi_ir,
            "hot_box_ir": None if presence is None else presence.hot_box_ir,
            "roi_redetect": roi_debug,
            "decision": decision,
            "reject_reason": reject_reason,
        }
