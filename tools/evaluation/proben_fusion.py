"""ProbEn-lite RGB/IR late-fusion helpers.

The functions in this module are intentionally detector-agnostic. Inputs and
outputs use the project xywh detection dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class ProbEnFusionResult:
    detections: list[dict]
    diagnostics: list[dict]
    stats: dict[str, Any]


def _confidence(box: dict) -> float:
    return float(box.get("confidence", 0.0))


def _area(box: dict) -> float:
    return max(0.0, float(box.get("w", 0.0))) * max(0.0, float(box.get("h", 0.0)))


def _center(box: dict) -> tuple[float, float]:
    return (
        float(box.get("x", 0.0)) + float(box.get("w", 0.0)) / 2.0,
        float(box.get("y", 0.0)) + float(box.get("h", 0.0)) / 2.0,
    )


def binary_proben_score(p_rgb: float, p_ir: float, eps: float = 1e-4) -> float:
    """Fuse binary object probabilities using the ProbEn odds product."""
    p_rgb = min(1.0 - eps, max(eps, float(p_rgb)))
    p_ir = min(1.0 - eps, max(eps, float(p_ir)))
    positive = p_rgb * p_ir
    negative = (1.0 - p_rgb) * (1.0 - p_ir)
    return positive / max(positive + negative, eps)


def score_weighted_box_fusion(
    rgb_box: dict,
    ir_box_mapped: dict,
    p_rgb: float,
    p_ir: float,
    ir_box_weight: float = 0.5,
) -> dict:
    """Fuse two xywh boxes with score weights and an explicit IR weight."""
    w_rgb = max(0.0, float(p_rgb))
    w_ir = max(0.0, float(ir_box_weight)) * max(0.0, float(p_ir))
    denom = max(w_rgb + w_ir, 1e-9)

    fused = dict(rgb_box)
    for key in ("x", "y", "w", "h"):
        fused[key] = (
            w_rgb * float(rgb_box[key]) + w_ir * float(ir_box_mapped[key])
        ) / denom
    return fused


def compute_iou(box_a: dict, box_b: dict) -> float:
    ax1, ay1 = float(box_a["x"]), float(box_a["y"])
    ax2, ay2 = ax1 + float(box_a["w"]), ay1 + float(box_a["h"])
    bx1, by1 = float(box_b["x"]), float(box_b["y"])
    bx2, by2 = bx1 + float(box_b["w"]), by1 + float(box_b["h"])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(box_a) + _area(box_b) - inter
    return 0.0 if union <= 0 else inter / union


def compute_center_distance(box_a: dict, box_b: dict) -> float:
    acx, acy = _center(box_a)
    bcx, bcy = _center(box_b)
    return math.hypot(acx - bcx, acy - bcy)


def compute_scale_ratio(box_a: dict, box_b: dict) -> float:
    """Return IR/RGB area ratio for two xywh boxes."""
    return _area(box_b) / max(_area(box_a), 1e-9)


def _empty_stats() -> dict[str, Any]:
    return {
        "rgb_ir_proben_matched_count": 0,
        "rgb_ir_proben_promoted_count": 0,
        "rgb_ir_proben_rejected_by_geometry_count": 0,
        "rgb_ir_proben_rejected_by_score_count": 0,
        "rgb_ir_proben_ir_only_count": 0,
        "rgb_ir_proben_avg_score_gain": 0.0,
        "rgb_ir_proben_avg_box_shift": 0.0,
        "rgb_ir_proben_max_box_shift": 0.0,
    }


def _candidate_matches(rgb_box: dict, mapped_ir_dets: list[dict], args) -> tuple[list[dict], list[dict]]:
    candidates = []
    geometry_rejections = []
    rgb_area_side = math.sqrt(max(_area(rgb_box), 1e-9))
    for ir_idx, ir_box in enumerate(mapped_ir_dets):
        iou = compute_iou(rgb_box, ir_box)
        center_distance = compute_center_distance(rgb_box, ir_box)
        scale_ratio = compute_scale_ratio(rgb_box, ir_box)
        passes_position = (
            iou >= float(args.proben_match_iou)
            or center_distance <= float(args.proben_match_dist_factor) * rgb_area_side
        )
        passes_scale = (
            float(args.proben_scale_ratio_min)
            <= scale_ratio
            <= float(args.proben_scale_ratio_max)
        )
        item = {
            "ir_idx": ir_idx,
            "ir_box": ir_box,
            "match_iou": iou,
            "center_distance": center_distance,
            "scale_ratio": scale_ratio,
        }
        if passes_position and passes_scale:
            candidates.append(item)
        elif passes_position and not passes_scale:
            item["reject_reason"] = "scale_ratio_out_of_range"
            geometry_rejections.append(item)
    candidates.sort(key=lambda item: (item["center_distance"], -item["match_iou"]))
    return candidates, geometry_rejections


def match_rgb_ir_boxes(
    rgb_dets: list[dict],
    mapped_ir_dets: list[dict],
    match_iou: float,
    match_dist_factor: float,
    scale_ratio_min: float,
    scale_ratio_max: float,
) -> tuple[dict[int, dict], list[int], list[dict]]:
    """One-to-one match RGB boxes to mapped IR boxes."""
    class Args:
        pass

    args = Args()
    args.proben_match_iou = match_iou
    args.proben_match_dist_factor = match_dist_factor
    args.proben_scale_ratio_min = scale_ratio_min
    args.proben_scale_ratio_max = scale_ratio_max

    matches: dict[int, dict] = {}
    used_ir: set[int] = set()
    geometry_rejections: list[dict] = []
    rgb_order = sorted(range(len(rgb_dets)), key=lambda idx: _confidence(rgb_dets[idx]), reverse=True)
    for rgb_idx in rgb_order:
        candidates, rejected = _candidate_matches(rgb_dets[rgb_idx], mapped_ir_dets, args)
        geometry_rejections.extend({"rgb_idx": rgb_idx, **item} for item in rejected)
        for candidate in candidates:
            if candidate["ir_idx"] in used_ir:
                continue
            matches[rgb_idx] = candidate
            used_ir.add(candidate["ir_idx"])
            break
    unmatched_ir = [idx for idx in range(len(mapped_ir_dets)) if idx not in used_ir]
    return matches, unmatched_ir, geometry_rejections


def _diagnostic_row(
    frame_id: int | None,
    rgb_box: dict | None,
    ir_box: dict | None,
    fused_box: dict | None,
    decision: str,
    reject_reason: str = "",
    match_iou: float = 0.0,
    center_distance: float = 0.0,
    scale_ratio: float = 0.0,
    p_fused: float = 0.0,
) -> dict:
    p_rgb = _confidence(rgb_box) if rgb_box is not None else 0.0
    p_ir = _confidence(ir_box) if ir_box is not None else 0.0
    box_shift = compute_center_distance(rgb_box, fused_box) if rgb_box is not None and fused_box is not None else 0.0
    return {
        "frame_id": frame_id,
        "rgb_box": rgb_box,
        "ir_box_mapped": ir_box,
        "fused_box": fused_box,
        "p_rgb": p_rgb,
        "p_ir": p_ir,
        "p_fused": p_fused,
        "score_gain": p_fused - p_rgb if rgb_box is not None else 0.0,
        "box_shift": box_shift,
        "match_iou": match_iou,
        "center_distance": center_distance,
        "scale_ratio": scale_ratio,
        "decision": decision,
        "reject_reason": reject_reason,
    }


def _nearest_rgb_metrics(ir_box: dict, rgb_dets: list[dict]) -> tuple[float, float]:
    if not rgb_dets:
        return 0.0, 0.0
    best_iou = 0.0
    best_dist = float("inf")
    for rgb_box in rgb_dets:
        best_iou = max(best_iou, compute_iou(rgb_box, ir_box))
        best_dist = min(best_dist, compute_center_distance(rgb_box, ir_box))
    return best_dist, best_iou


def fuse_rgb_ir_proben(
    rgb_dets: list[dict],
    mapped_ir_dets: list[dict],
    args,
    frame_id: int | None = None,
) -> ProbEnFusionResult:
    """Fuse RGB and mapped IR detections with ProbEn-lite."""
    matches, unmatched_ir, geometry_rejections = match_rgb_ir_boxes(
        rgb_dets,
        mapped_ir_dets,
        float(args.proben_match_iou),
        float(args.proben_match_dist_factor),
        float(args.proben_scale_ratio_min),
        float(args.proben_scale_ratio_max),
    )
    detections: list[dict] = []
    diagnostics: list[dict] = []
    stats = _empty_stats()
    score_gains: list[float] = []
    box_shifts: list[float] = []

    for rejected in geometry_rejections:
        stats["rgb_ir_proben_rejected_by_geometry_count"] += 1
        rgb_box = rgb_dets[rejected["rgb_idx"]]
        diagnostics.append(_diagnostic_row(
            frame_id,
            rgb_box,
            rejected["ir_box"],
            None,
            "rejected",
            reject_reason=rejected["reject_reason"],
            match_iou=rejected["match_iou"],
            center_distance=rejected["center_distance"],
            scale_ratio=rejected["scale_ratio"],
        ))

    for rgb_idx, rgb_box in enumerate(rgb_dets):
        p_rgb = _confidence(rgb_box)
        is_high = p_rgb >= float(args.rgb_high_conf)
        is_low = float(args.rgb_low_conf) <= p_rgb < float(args.rgb_high_conf)
        match = matches.get(rgb_idx)
        if match is None:
            if is_high:
                out = dict(rgb_box)
                out["allow_new_track"] = True
                out["support_type"] = "high"
                detections.append(out)
            continue

        ir_box = match["ir_box"]
        p_ir = _confidence(ir_box)
        p_fused = binary_proben_score(p_rgb, p_ir)
        if str(args.proben_box_mode) == "savg":
            out = score_weighted_box_fusion(
                rgb_box,
                ir_box,
                p_rgb,
                p_ir,
                ir_box_weight=float(args.proben_ir_box_weight),
            )
        else:
            out = dict(rgb_box)
        out["confidence"] = p_fused
        out["class"] = rgb_box.get("class")
        out["class_confidence"] = p_fused
        out["support_type"] = "proben"
        out["allow_new_track"] = True if is_high else bool(args.proben_low_allow_new_track)

        stats["rgb_ir_proben_matched_count"] += 1
        score_gain = p_fused - p_rgb
        box_shift = compute_center_distance(rgb_box, out)
        score_gains.append(score_gain)
        box_shifts.append(box_shift)

        if is_low and p_fused < float(args.proben_keep_conf):
            stats["rgb_ir_proben_rejected_by_score_count"] += 1
            diagnostics.append(_diagnostic_row(
                frame_id,
                rgb_box,
                ir_box,
                out,
                "rejected",
                reject_reason="fused_score_below_keep_conf",
                match_iou=match["match_iou"],
                center_distance=match["center_distance"],
                scale_ratio=match["scale_ratio"],
                p_fused=p_fused,
            ))
            continue
        if is_low:
            stats["rgb_ir_proben_promoted_count"] += 1
        if is_high or is_low:
            detections.append(out)
            diagnostics.append(_diagnostic_row(
                frame_id,
                rgb_box,
                ir_box,
                out,
                "emitted",
                match_iou=match["match_iou"],
                center_distance=match["center_distance"],
                scale_ratio=match["scale_ratio"],
                p_fused=p_fused,
            ))

    for ir_idx in unmatched_ir:
        ir_box = mapped_ir_dets[ir_idx]
        nearest_dist, nearest_iou = _nearest_rgb_metrics(ir_box, rgb_dets)
        stats["rgb_ir_proben_ir_only_count"] += 1
        diagnostics.append({
            **_diagnostic_row(frame_id, None, ir_box, None, "ir_only_not_emitted"),
            "nearest_rgb_dist": nearest_dist,
            "nearest_rgb_iou": nearest_iou,
        })

    if score_gains:
        stats["rgb_ir_proben_avg_score_gain"] = sum(score_gains) / len(score_gains)
    if box_shifts:
        stats["rgb_ir_proben_avg_box_shift"] = sum(box_shifts) / len(box_shifts)
        stats["rgb_ir_proben_max_box_shift"] = max(box_shifts)
    return ProbEnFusionResult(detections=detections, diagnostics=diagnostics, stats=stats)
