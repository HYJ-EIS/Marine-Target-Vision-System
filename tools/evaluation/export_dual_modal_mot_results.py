"""
Export RGB-only and RGB+IR-support BoT-SORT results to MOTChallenge format.

This script only uses IR detections as support evidence for low-confidence RGB
detections. It never emits IR-only boxes and never uses MOT GT during inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from tools.evaluation.export_mot_results import format_mot_result_line
from tools.evaluation.ir_presence_support import (
    IRPresenceSupport,
    map_rgb_box_to_ir,
    score_thermal_presence,
)
from tools.evaluation.ir_reacquire_buffer import IRReacquireBuffer
from tools.evaluation.proben_fusion import fuse_rgb_ir_proben


DEFAULT_ALIGNMENT_CONFIG = "/home/hyj/Anti_Drone_Project/IR_RGB_match/config/alignment_config.yaml"
FUSION_CHOICES = (
    "rgb_only",
    "rgb_ir_support",
    "rgb_ir_proben",
    "rgb_ir_proben_lost_reacquire",
    "rgb_ir_presence_roi_redetect",
)
TRACKER_CHOICES = ("botsort",)
REQUIRED_STAGE_TIMINGS = (
    "video_read_decode",
    "detection",
    "alignment_mapping",
    "fusion",
    "low_threshold_detection_or_roi_redetect",
    "motion_template_lifecycle_tracker",
    "visualization_rendering",
    "result_write_export",
)


def load_alignment_config(config_path: str | Path) -> dict[str, Any]:
    """Read the RGB/IR alignment YAML config."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Alignment config does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read alignment_config.yaml") from exc
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except Exception as exc:
        raise ValueError(f"Failed to parse alignment config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Alignment config must be a mapping: {config_path}")
    return config


def _candidate_affine_matrices(report: dict[str, Any]) -> list[Any]:
    candidates = [
        report.get("affine_matrix"),
        report.get("ir_to_rgb_affine"),
        report.get("affine", {}).get("matrix") if isinstance(report.get("affine"), dict) else None,
        report.get("transforms", {}).get("affine", {}).get("matrix")
        if isinstance(report.get("transforms"), dict)
        and isinstance(report.get("transforms", {}).get("affine"), dict)
        else None,
    ]
    return [candidate for candidate in candidates if candidate is not None]


def _candidate_piecewise_affine(report: dict[str, Any]) -> list[dict[str, Any]]:
    transforms = report.get("transforms")
    if not isinstance(transforms, dict):
        return []
    piecewise = transforms.get("piecewise_affine")
    if not isinstance(piecewise, dict):
        return []
    segments = piecewise.get("segments")
    if not isinstance(segments, list):
        return []
    valid_segments = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        matrix = np.asarray(segment.get("matrix"), dtype=np.float64)
        if matrix.shape != (2, 3):
            continue
        valid_segments.append({
            "name": str(segment.get("name", "")),
            "start_frame": int(segment.get("start_frame", 1)),
            "end_frame": int(segment.get("end_frame", 1)),
            "matrix": matrix,
        })
    return valid_segments


def load_calib(calib_path: str | Path) -> np.ndarray | dict[str, Any]:
    """Load an existing IR->RGB affine or piecewise affine matrix from JSON."""
    calib_path = Path(calib_path)
    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration report does not exist: {calib_path}")
    try:
        with calib_path.open("r", encoding="utf-8") as fh:
            report = json.load(fh)
    except Exception as exc:
        raise ValueError(f"Failed to parse calibration JSON: {calib_path}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"Calibration JSON must be an object: {calib_path}")

    piecewise_segments = _candidate_piecewise_affine(report)
    if piecewise_segments:
        piecewise_segments.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
        return {"type": "piecewise_affine", "segments": piecewise_segments}

    for candidate in _candidate_affine_matrices(report):
        matrix = np.asarray(candidate, dtype=np.float64)
        if matrix.shape == (2, 3):
            return matrix
        if matrix.shape == (3, 3) and np.allclose(matrix[2], np.array([0.0, 0.0, 1.0])):
            return matrix[:2]
    raise ValueError(
        "Calibration report does not contain a valid IR->RGB affine matrix "
        "(expected 2x3, or affine 3x3 with last row [0,0,1])"
    )


def resolve_affine_matrix(calib: np.ndarray | dict[str, Any], frame_id: int) -> np.ndarray:
    """Resolve a calibration spec to the 2x3 IR->RGB affine for a frame."""
    if isinstance(calib, dict) and calib.get("type") == "piecewise_affine":
        segments = list(calib.get("segments", []))
        for segment in segments:
            if int(segment["start_frame"]) <= int(frame_id) <= int(segment["end_frame"]):
                return np.asarray(segment["matrix"], dtype=np.float64)
        if segments:
            if int(frame_id) < int(segments[0]["start_frame"]):
                return np.asarray(segments[0]["matrix"], dtype=np.float64)
            return np.asarray(segments[-1]["matrix"], dtype=np.float64)
    matrix = np.asarray(calib, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("calib must resolve to a 2x3 affine matrix")
    return matrix


def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute IoU for xywh boxes."""
    ax1, ay1 = float(box_a["x"]), float(box_a["y"])
    ax2, ay2 = ax1 + float(box_a["w"]), ay1 + float(box_a["h"])
    bx1, by1 = float(box_b["x"]), float(box_b["y"])
    bx2, by2 = bx1 + float(box_b["w"]), by1 + float(box_b["h"])

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_distance(box_a: dict, box_b: dict) -> float:
    """Compute center distance for xywh boxes."""
    acx = float(box_a["x"]) + float(box_a["w"]) / 2.0
    acy = float(box_a["y"]) + float(box_a["h"]) / 2.0
    bcx = float(box_b["x"]) + float(box_b["w"]) / 2.0
    bcy = float(box_b["y"]) + float(box_b["h"]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _diag(box: dict) -> float:
    return math.hypot(float(box["w"]), float(box["h"]))


def map_ir_box_to_rgb(box: dict, affine_matrix: np.ndarray, rgb_shape: tuple[int, ...]) -> dict | None:
    """Map an IR xywh box to RGB coordinates using an existing affine matrix."""
    matrix = np.asarray(affine_matrix, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("affine_matrix must have shape 2x3")

    x, y = float(box["x"]), float(box["y"])
    w, h = float(box["w"]), float(box["h"])
    if w <= 0 or h <= 0:
        return None
    corners = np.array(
        [
            [x, y, 1.0],
            [x + w, y, 1.0],
            [x, y + h, 1.0],
            [x + w, y + h, 1.0],
        ],
        dtype=np.float64,
    )
    mapped = corners @ matrix.T
    x1, y1 = np.min(mapped[:, 0]), np.min(mapped[:, 1])
    x2, y2 = np.max(mapped[:, 0]), np.max(mapped[:, 1])

    rgb_h, rgb_w = rgb_shape[:2]
    x1 = min(max(0.0, float(x1)), float(rgb_w))
    x2 = min(max(0.0, float(x2)), float(rgb_w))
    y1 = min(max(0.0, float(y1)), float(rgb_h))
    y2 = min(max(0.0, float(y2)), float(rgb_h))
    if x2 <= x1 or y2 <= y1:
        return None

    mapped_box = dict(box)
    mapped_box.update({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    return mapped_box


def has_ir_support(rgb_low_box: dict, mapped_ir_boxes: list[dict], args: argparse.Namespace) -> tuple[bool, dict | None]:
    """Return whether a low-confidence RGB box is supported by mapped IR boxes."""
    matches = []
    for ir_box in mapped_ir_boxes:
        iou = compute_iou(rgb_low_box, ir_box)
        dist = center_distance(rgb_low_box, ir_box)
        dist_limit = float(args.ir_rgb_match_dist_factor) * _diag(rgb_low_box)
        if iou >= float(args.ir_rgb_match_iou) or dist <= dist_limit:
            matches.append((iou, -dist, ir_box))
    if not matches:
        return False, None
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return True, matches[0][2]


def _iter_track_support_boxes(tracker: MultiObjectTracker | None) -> list[dict]:
    if tracker is None or getattr(tracker, "tracker_type", None) not in ("ocsort", "botsort"):
        return []
    ocsort = getattr(tracker, "_ocsort", None)
    if ocsort is None:
        return []

    min_hits = int(getattr(tracker, "_min_hits", 1))
    max_age = int(getattr(ocsort, "max_age", 0))
    boxes = []
    for trk in getattr(ocsort, "trackers", []):
        if int(getattr(trk, "hits", 0)) < min_hits:
            continue
        if int(getattr(trk, "time_since_update", 0)) > max_age:
            continue
        state = getattr(trk, "last_observation", None)
        if state is None or len(state) != 4:
            state = trk.get_state()
        x1, y1, x2, y2 = [float(v) for v in state]
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    return boxes


def has_track_support(rgb_low_box: dict, tracker: MultiObjectTracker | None,
                      args: argparse.Namespace) -> bool:
    """Check whether an existing stable track supports a low-confidence RGB box."""
    for track_box in _iter_track_support_boxes(tracker):
        iou = compute_iou(rgb_low_box, track_box)
        dist = center_distance(rgb_low_box, track_box)
        dist_limit = float(args.track_dist_factor) * _diag(track_box)
        if iou >= float(args.track_iou_thresh) or dist <= dist_limit:
            return True
    return False


def _fused_confidence(conf_rgb: float, conf_ir: float, alpha: float) -> float:
    fused = 1.0 - (1.0 - conf_rgb) * (1.0 - alpha * conf_ir)
    return min(1.0, max(conf_rgb, fused))


def fuse_rgb_ir_detections(rgb_boxes: list[dict], ir_boxes_mapped: list[dict],
                           tracker: MultiObjectTracker | None,
                           args: argparse.Namespace) -> list[dict]:
    """Fuse RGB candidates with IR/track support for tracker input."""
    fused: list[dict] = []
    for box in rgb_boxes:
        conf = float(box.get("confidence", 0.0))
        if conf >= float(args.rgb_high_conf):
            out = dict(box)
            out["allow_new_track"] = True
            out["support_type"] = "high"
            fused.append(out)
            continue
        if conf < float(args.rgb_low_conf):
            continue

        supported, best_ir = has_ir_support(box, ir_boxes_mapped, args)
        if supported and best_ir is not None:
            out = dict(box)
            out["confidence"] = _fused_confidence(
                conf,
                float(best_ir.get("confidence", 0.0)),
                float(args.alpha),
            )
            out["allow_new_track"] = True
            out["support_type"] = "ir_support"
            fused.append(out)
            continue

        if has_track_support(box, tracker, args):
            out = dict(box)
            out["allow_new_track"] = False
            out["support_type"] = "track_support"
            fused.append(out)
    return fused


def _extract_boxes(stats: dict | None) -> list[dict]:
    if not stats:
        return []
    boxes = stats.get("boxes", [])
    if not isinstance(boxes, list):
        return []
    return boxes


def format_detection_result_line(frame_id: int, box: dict) -> str:
    """Format one detector/fusion box as a MOTChallenge det-style row."""
    return (
        f"{int(frame_id)},-1,"
        f"{float(box['x']):.2f},{float(box['y']):.2f},"
        f"{float(box['w']):.2f},{float(box['h']):.2f},"
        f"{float(box.get('confidence', 1.0)):.4f},-1,-1,-1"
    )


def _validate_sync_config(config: dict[str, Any]) -> None:
    timing = config.get("timing")
    if not isinstance(timing, dict) or not timing.get("sync_mode"):
        raise ValueError("Alignment config is missing timing.sync_mode")
    sync_mode = str(timing["sync_mode"])
    if sync_mode not in {"frame_index", "frame_by_frame", "per_frame"}:
        raise ValueError(f"Unsupported RGB/IR sync mode for this exporter: {sync_mode}")


def _open_video(path: str | Path, label: str) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} video does not exist: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {label} video: {path}")
    return cap


def _empty_stats(run_name: str, seq_name: str) -> dict[str, Any]:
    stage_timings = {stage: 0.0 for stage in REQUIRED_STAGE_TIMINGS}
    return {
        "run_name": run_name,
        "seq_name": seq_name,
        "processed_frames": 0,
        "rgb_high_boxes_count": 0,
        "rgb_low_boxes_count": 0,
        "rgb_low_promoted_by_ir_count": 0,
        "rgb_low_kept_by_track_support_count": 0,
        "rgb_low_suppressed_count": 0,
        "ir_boxes_count": 0,
        "mapped_valid_ir_boxes_count": 0,
        "output_mot_txt_path": "",
        "output_detection_txt_path": "det.txt",
        "diagnostics_csv_path": "diagnostics.csv",
        "proben_diagnostics_jsonl_path": "proben_diagnostics.jsonl",
        "ir_reacquire_diagnostics_jsonl_path": "ir_reacquire_diagnostics.jsonl",
        "ir_presence_diagnostics_jsonl_path": "ir_presence_diagnostics.jsonl",
        "stage_timings_json_path": "stage_timings.json",
        "total_runtime_seconds": 0.0,
        "runtime_fps": 0.0,
        "stage_timings_seconds": dict(stage_timings),
        "stage_timings_per_frame_seconds": dict(stage_timings),
        "rgb_ir_proben_matched_count": 0,
        "rgb_ir_proben_promoted_count": 0,
        "rgb_ir_proben_rejected_by_geometry_count": 0,
        "rgb_ir_proben_rejected_by_score_count": 0,
        "rgb_ir_proben_ir_only_count": 0,
        "rgb_ir_proben_avg_score_gain": 0.0,
        "rgb_ir_proben_avg_box_shift": 0.0,
        "rgb_ir_proben_max_box_shift": 0.0,
        "ir_reacquire_candidates_created": 0,
        "ir_reacquire_candidates_confirmed": 0,
        "ir_reacquire_emitted_count": 0,
        "ir_reacquire_rejected_low_conf": 0,
        "ir_reacquire_rejected_no_lost_track": 0,
        "ir_reacquire_rejected_gate": 0,
        "ir_reacquire_rejected_not_confirmed": 0,
        "ir_reacquire_rejected_max_per_frame": 0,
        "ir_presence_rgb_verified_count": 0,
        "ir_presence_low_promoted_count": 0,
        "ir_presence_rgb_rejected_count": 0,
        "ir_presence_tracks_checked_count": 0,
        "ir_presence_roi_redetect_emitted_count": 0,
        "ir_presence_rejected_weak_count": 0,
        "ir_presence_rejected_no_rgb_low_det_count": 0,
        "ir_presence_skipped_rgb_matched_count": 0,
    }


def _print_stats(stats: dict[str, Any]) -> None:
    print("[OK] Dual-modal MOT export completed")
    for key in [
        "processed_frames",
        "rgb_high_boxes_count",
        "rgb_low_boxes_count",
        "rgb_low_promoted_by_ir_count",
        "rgb_low_kept_by_track_support_count",
        "rgb_low_suppressed_count",
        "ir_boxes_count",
        "mapped_valid_ir_boxes_count",
        "output_mot_txt_path",
        "output_detection_txt_path",
        "diagnostics_csv_path",
        "proben_diagnostics_jsonl_path",
        "ir_reacquire_diagnostics_jsonl_path",
        "ir_presence_diagnostics_jsonl_path",
        "stage_timings_json_path",
        "total_runtime_seconds",
        "runtime_fps",
        "stage_timings_seconds",
    ]:
        print(f"{key}: {stats[key]}")


def _finalize_stage_timings(stats: dict[str, Any]) -> None:
    processed = max(1, int(stats.get("processed_frames", 0)))
    timings = stats["stage_timings_seconds"]
    stats["stage_timings_seconds"] = {
        key: round(float(timings.get(key, 0.0)), 6) for key in REQUIRED_STAGE_TIMINGS
    }
    stats["stage_timings_per_frame_seconds"] = {
        key: round(float(stats["stage_timings_seconds"][key]) / processed, 6)
        for key in REQUIRED_STAGE_TIMINGS
    }


def _write_diagnostics_csv(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_name",
        "seq_name",
        "processed_frames",
        "rgb_high_boxes_count",
        "rgb_low_boxes_count",
        "rgb_low_promoted_by_ir_count",
        "rgb_low_kept_by_track_support_count",
        "rgb_low_suppressed_count",
        "ir_boxes_count",
        "mapped_valid_ir_boxes_count",
        "rgb_ir_proben_matched_count",
        "rgb_ir_proben_promoted_count",
        "rgb_ir_proben_rejected_by_geometry_count",
        "rgb_ir_proben_rejected_by_score_count",
        "rgb_ir_proben_ir_only_count",
        "rgb_ir_proben_avg_score_gain",
        "rgb_ir_proben_avg_box_shift",
        "rgb_ir_proben_max_box_shift",
        "ir_reacquire_candidates_created",
        "ir_reacquire_candidates_confirmed",
        "ir_reacquire_emitted_count",
        "ir_reacquire_rejected_low_conf",
        "ir_reacquire_rejected_no_lost_track",
        "ir_reacquire_rejected_gate",
        "ir_reacquire_rejected_not_confirmed",
        "ir_reacquire_rejected_max_per_frame",
        "ir_presence_rgb_verified_count",
        "ir_presence_low_promoted_count",
        "ir_presence_rgb_rejected_count",
        "ir_presence_tracks_checked_count",
        "ir_presence_roi_redetect_emitted_count",
        "ir_presence_rejected_weak_count",
        "ir_presence_rejected_no_rgb_low_det_count",
        "ir_presence_skipped_rgb_matched_count",
        "runtime_fps",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: stats.get(field, "") for field in fields})


def _get_lost_track_boxes(tracker: MultiObjectTracker, max_age: int) -> list[dict]:
    """Read stable recently-lost local OC-SORT/BoT-SORT tracks without mutating state."""
    if getattr(tracker, "tracker_type", None) not in ("ocsort", "botsort"):
        return []
    ocsort = getattr(tracker, "_ocsort", None)
    if ocsort is None:
        return []
    boxes = []
    min_hits = int(getattr(tracker, "_min_hits", 1))
    id_to_class = getattr(tracker, "_ID_TO_CLASS", {})
    for trk in getattr(ocsort, "trackers", []):
        if int(getattr(trk, "hits", 0)) < min_hits:
            continue
        age = int(getattr(trk, "time_since_update", 0))
        if age <= 0 or age > int(max_age):
            continue
        state = trk.get_state()
        x1, y1, x2, y2 = [float(v) for v in state]
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({
            "track_id": int(trk.id),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "class": id_to_class.get(int(trk.cls_id), "target"),
            "confidence": float(getattr(trk, "conf", 0.0)),
            "time_since_update": age,
        })
    return boxes


def _get_presence_track_boxes(tracker: MultiObjectTracker, max_age: int) -> list[dict]:
    """Read stable active/recent tracks for IR presence support without mutating state."""
    if getattr(tracker, "tracker_type", None) not in ("ocsort", "botsort"):
        return []
    ocsort = getattr(tracker, "_ocsort", None)
    if ocsort is None:
        return []
    boxes = []
    min_hits = int(getattr(tracker, "_min_hits", 1))
    id_to_class = getattr(tracker, "_ID_TO_CLASS", {})
    for trk in getattr(ocsort, "trackers", []):
        if int(getattr(trk, "hits", 0)) < min_hits:
            continue
        age = int(getattr(trk, "time_since_update", 0))
        if age > int(max_age):
            continue
        state = trk.get_state() if age > 0 else getattr(trk, "last_observation", None)
        if state is None or len(state) != 4:
            state = trk.get_state()
        x1, y1, x2, y2 = [float(v) for v in state]
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({
            "track_id": int(trk.id),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "class": id_to_class.get(int(trk.cls_id), "target"),
            "confidence": float(getattr(trk, "conf", 0.0)),
            "time_since_update": age,
        })
    return boxes


def _presence_verified_box(box: dict, ir_frame: np.ndarray, affine_matrix: np.ndarray,
                           args: argparse.Namespace) -> tuple[dict | None, dict]:
    roi_ir = map_rgb_box_to_ir(box, affine_matrix, ir_frame.shape)
    if roi_ir is None:
        return None, {
            "rgb_box": box,
            "decision": "rejected",
            "reject_reason": "rgb_to_ir_mapping_failed",
        }
    presence = score_thermal_presence(ir_frame, roi_ir, args)
    row = {
        "rgb_box": box,
        "presence_decision": presence.decision,
        "thermal_score": presence.thermal_score,
        "hot_area_ratio": presence.hot_area_ratio,
        "roi_ir": presence.roi_ir,
        "hot_box_ir": presence.hot_box_ir,
    }
    if presence.decision != "present":
        row.update({"decision": "rejected", "reject_reason": presence.reject_reason})
        return None, row
    out = dict(box)
    boost = float(args.ir_presence_conf_boost)
    conf = float(out.get("confidence", 0.0))
    out["confidence"] = min(0.99, conf + boost * (1.0 - conf))
    out["class_confidence"] = min(0.99, float(out.get("class_confidence", conf)) + boost * (1.0 - conf))
    row.update({"decision": "verified", "reject_reason": ""})
    return out, row


def export_dual_modal_video_to_mot_results(
    visible_input: str | Path,
    infrared_input: str | Path | None,
    output_root: str | Path,
    run_name: str,
    seq_name: str,
    tracker: str = "botsort",
    fusion: str = "rgb_only",
    calib: str | Path | None = None,
    config: str | Path = DEFAULT_ALIGNMENT_CONFIG,
    rgb_high_conf: float = 0.50,
    rgb_low_conf: float = 0.30,
    ir_conf: float = 0.45,
    alpha: float = 0.7,
    max_frames: int = 0,
    progress_interval: int = 100,
    track_iou_thresh: float = 0.20,
    track_dist_factor: float = 1.5,
    ir_rgb_match_iou: float = 0.10,
    ir_rgb_match_dist_factor: float = 2.0,
    proben_match_iou: float = 0.15,
    proben_match_dist_factor: float = 1.5,
    proben_keep_conf: float = 0.50,
    proben_ir_box_weight: float = 0.5,
    proben_scale_ratio_min: float = 0.4,
    proben_scale_ratio_max: float = 2.5,
    proben_low_allow_new_track: bool = False,
    proben_box_mode: str = "score_only",
    save_proben_diagnostics: bool = False,
    ir_reacquire_enable: bool = False,
    ir_reacquire_confirm_frames: int = 2,
    ir_reacquire_max_age: int = 30,
    ir_reacquire_iou: float = 0.05,
    ir_reacquire_dist_factor: float = 2.5,
    ir_reacquire_max_per_frame: int = 2,
    ir_reacquire_min_conf: float = 0.45,
    ir_reacquire_candidate_iou: float = 0.10,
    ir_reacquire_candidate_dist_factor: float = 1.5,
    ir_reacquire_max_gap: int = 1,
    ir_reacquire_diagnostics: bool = False,
    ir_presence_enable: bool = False,
    ir_presence_z_thresh: float = 2.0,
    ir_presence_min_hot_area_ratio: float = 0.01,
    ir_presence_bg_pad: float = 1.0,
    ir_presence_min_bg_std: float = 1.0,
    ir_presence_conf_boost: float = 0.10,
    ir_presence_max_track_age: int = 8,
    ir_presence_rgb_match_iou: float = 0.20,
    ir_presence_rgb_match_dist_factor: float = 1.5,
    ir_presence_roi_pad: float = 1.8,
    ir_presence_min_crop_size: float = 64.0,
    ir_presence_roi_upscale: float = 2.0,
    ir_presence_rgb_roi_low_conf: float = 0.20,
    ir_presence_redetect_iou: float = 0.05,
    ir_presence_redetect_dist_factor: float = 2.5,
    ir_presence_diagnostics: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run dual-modal export and return the MOT txt path plus run stats."""
    if tracker != "botsort":
        raise ValueError("Only tracker=botsort is supported")
    if fusion not in FUSION_CHOICES:
        raise ValueError(f"Unsupported fusion mode: {fusion}")
    if rgb_low_conf > rgb_high_conf:
        raise ValueError("--rgb-low-conf must be <= --rgb-high-conf")

    args = argparse.Namespace(
        rgb_high_conf=rgb_high_conf,
        rgb_low_conf=rgb_low_conf,
        ir_conf=ir_conf,
        alpha=alpha,
        track_iou_thresh=track_iou_thresh,
        track_dist_factor=track_dist_factor,
        ir_rgb_match_iou=ir_rgb_match_iou,
        ir_rgb_match_dist_factor=ir_rgb_match_dist_factor,
        proben_match_iou=proben_match_iou,
        proben_match_dist_factor=proben_match_dist_factor,
        proben_keep_conf=proben_keep_conf,
        proben_ir_box_weight=proben_ir_box_weight,
        proben_scale_ratio_min=proben_scale_ratio_min,
        proben_scale_ratio_max=proben_scale_ratio_max,
        proben_low_allow_new_track=proben_low_allow_new_track,
        proben_box_mode=proben_box_mode,
        ir_reacquire_confirm_frames=ir_reacquire_confirm_frames,
        ir_reacquire_max_age=ir_reacquire_max_age,
        ir_reacquire_iou=ir_reacquire_iou,
        ir_reacquire_dist_factor=ir_reacquire_dist_factor,
        ir_reacquire_max_per_frame=ir_reacquire_max_per_frame,
        ir_reacquire_min_conf=ir_reacquire_min_conf,
        ir_reacquire_candidate_iou=ir_reacquire_candidate_iou,
        ir_reacquire_candidate_dist_factor=ir_reacquire_candidate_dist_factor,
        ir_reacquire_max_gap=ir_reacquire_max_gap,
        ir_presence_z_thresh=ir_presence_z_thresh,
        ir_presence_min_hot_area_ratio=ir_presence_min_hot_area_ratio,
        ir_presence_bg_pad=ir_presence_bg_pad,
        ir_presence_min_bg_std=ir_presence_min_bg_std,
        ir_presence_conf_boost=ir_presence_conf_boost,
        ir_presence_max_track_age=ir_presence_max_track_age,
        ir_presence_rgb_match_iou=ir_presence_rgb_match_iou,
        ir_presence_rgb_match_dist_factor=ir_presence_rgb_match_dist_factor,
        ir_presence_roi_pad=ir_presence_roi_pad,
        ir_presence_min_crop_size=ir_presence_min_crop_size,
        ir_presence_roi_upscale=ir_presence_roi_upscale,
        ir_presence_rgb_roi_low_conf=ir_presence_rgb_roi_low_conf,
        ir_presence_redetect_iou=ir_presence_redetect_iou,
        ir_presence_redetect_dist_factor=ir_presence_redetect_dist_factor,
    )

    output_dir = Path(output_root) / run_name
    output_file = output_dir / "data" / f"{seq_name}.txt"
    detection_file = output_dir / "detections" / f"{seq_name}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    detection_file.parent.mkdir(parents=True, exist_ok=True)
    stats = _empty_stats(run_name, seq_name)
    stats["output_mot_txt_path"] = str(output_file)
    stats["output_detection_txt_path"] = str(detection_file)
    diagnostics_csv = output_dir / "diagnostics.csv"
    proben_diagnostics_jsonl = output_dir / "proben_diagnostics.jsonl"
    ir_reacquire_diagnostics_jsonl = output_dir / "ir_reacquire_diagnostics.jsonl"
    ir_presence_diagnostics_jsonl = output_dir / "ir_presence_diagnostics.jsonl"
    stage_timings_json = output_dir / "stage_timings.json"
    stats["diagnostics_csv_path"] = str(diagnostics_csv)
    stats["proben_diagnostics_jsonl_path"] = str(proben_diagnostics_jsonl)
    stats["ir_reacquire_diagnostics_jsonl_path"] = str(ir_reacquire_diagnostics_jsonl)
    stats["ir_presence_diagnostics_jsonl_path"] = str(ir_presence_diagnostics_jsonl)
    stats["stage_timings_json_path"] = str(stage_timings_json)

    affine_calib = None
    if fusion in ("rgb_ir_support", "rgb_ir_proben", "rgb_ir_proben_lost_reacquire", "rgb_ir_presence_roi_redetect"):
        if not infrared_input:
            raise ValueError(f"--infrared-input is required for {fusion}")
        if not calib:
            raise ValueError(f"--calib is required for {fusion}")
        alignment_config = load_alignment_config(config)
        _validate_sync_config(alignment_config)
        affine_calib = load_calib(calib)

    visible_cap = _open_video(visible_input, "visible")
    infrared_cap = None
    if fusion in ("rgb_ir_support", "rgb_ir_proben", "rgb_ir_proben_lost_reacquire", "rgb_ir_presence_roi_redetect"):
        infrared_cap = _open_video(str(infrared_input), "infrared")

    try:
        visible_fps = visible_cap.get(cv2.CAP_PROP_FPS) or 25.0
        visible_frames = int(visible_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        infrared_frames = int(infrared_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if infrared_cap else 0
        available_frames = visible_frames
        if infrared_cap is not None:
            if visible_frames and infrared_frames and visible_frames != infrared_frames:
                print(
                    f"[WARN] RGB/IR frame count mismatch: visible={visible_frames}, infrared={infrared_frames}; "
                    "using the shorter stream",
                    flush=True,
                )
            available_frames = min(v for v in (visible_frames, infrared_frames) if v > 0) if (visible_frames or infrared_frames) else 0
        target_frames = max_frames if max_frames > 0 else available_frames

        detector = td.get_detector()
        processor = getattr(detector, "processor", None)
        if processor is None:
            raise RuntimeError("ImageProcessor is not available")
        mot_tracker = MultiObjectTracker(frame_rate=visible_fps, tracker_type="botsort")

        start_time = time.time()
        proben_score_gain_sum = 0.0
        proben_box_shift_sum = 0.0
        proben_box_shift_count = 0
        ir_reacquire_buffer = IRReacquireBuffer()
        ir_presence_support = IRPresenceSupport()
        seen_candidate_ids: set[int] = set()
        with (
            output_file.open("w", encoding="utf-8") as fh,
            detection_file.open("w", encoding="utf-8") as det_fh,
            proben_diagnostics_jsonl.open("w", encoding="utf-8") if save_proben_diagnostics else open(os.devnull, "w", encoding="utf-8") as proben_diag_fh,
            ir_reacquire_diagnostics_jsonl.open("w", encoding="utf-8") if ir_reacquire_diagnostics else open(os.devnull, "w", encoding="utf-8") as ir_reacquire_diag_fh,
            ir_presence_diagnostics_jsonl.open("w", encoding="utf-8") if ir_presence_diagnostics else open(os.devnull, "w", encoding="utf-8") as ir_presence_diag_fh,
        ):
            frame_id = 0
            while True:
                if target_frames > 0 and frame_id >= target_frames:
                    break
                read_start = time.perf_counter()
                ret_rgb, rgb_frame = visible_cap.read()
                if not ret_rgb:
                    break

                ir_frame = None
                if infrared_cap is not None:
                    ret_ir, ir_frame = infrared_cap.read()
                    if not ret_ir:
                        break
                stats["stage_timings_seconds"]["video_read_decode"] += time.perf_counter() - read_start

                frame_id += 1
                affine_matrix = resolve_affine_matrix(affine_calib, frame_id) if affine_calib is not None else None
                detection_start = time.perf_counter()
                rgb_boxes = _extract_boxes(
                    processor.process_frame(rgb_frame, "visible", conf_override=rgb_low_conf)
                )
                rgb_high_count = sum(1 for box in rgb_boxes if float(box.get("confidence", 0.0)) >= rgb_high_conf)
                rgb_low_count = sum(
                    1
                    for box in rgb_boxes
                    if rgb_low_conf <= float(box.get("confidence", 0.0)) < rgb_high_conf
                )
                stats["rgb_high_boxes_count"] += rgb_high_count
                stats["rgb_low_boxes_count"] += rgb_low_count

                ir_boxes: list[dict] = []
                mapped_ir_boxes: list[dict] = []
                if fusion in ("rgb_ir_support", "rgb_ir_proben", "rgb_ir_proben_lost_reacquire"):
                    ir_boxes = _extract_boxes(
                        processor.process_frame(ir_frame, "infrared", conf_override=ir_conf)
                    )
                    stats["ir_boxes_count"] += len(ir_boxes)
                stats["stage_timings_seconds"]["detection"] += time.perf_counter() - detection_start

                if fusion in ("rgb_ir_support", "rgb_ir_proben", "rgb_ir_proben_lost_reacquire"):
                    mapping_start = time.perf_counter()
                    for ir_box in ir_boxes:
                        mapped = map_ir_box_to_rgb(ir_box, affine_matrix, rgb_frame.shape)
                        if mapped is not None:
                            mapped_ir_boxes.append(mapped)
                    stats["stage_timings_seconds"]["alignment_mapping"] += time.perf_counter() - mapping_start
                    stats["mapped_valid_ir_boxes_count"] += len(mapped_ir_boxes)

                fusion_start = time.perf_counter()
                if fusion == "rgb_only":
                    tracker_inputs = []
                    for box in rgb_boxes:
                        if float(box.get("confidence", 0.0)) >= rgb_high_conf:
                            out = dict(box)
                            out["allow_new_track"] = True
                            out["support_type"] = "high"
                            tracker_inputs.append(out)
                    stats["rgb_low_suppressed_count"] += rgb_low_count
                elif fusion == "rgb_ir_support":
                    tracker_inputs = fuse_rgb_ir_detections(rgb_boxes, mapped_ir_boxes, mot_tracker, args)
                    stats["rgb_low_promoted_by_ir_count"] += sum(
                        1 for box in tracker_inputs if box.get("support_type") == "ir_support"
                    )
                    stats["rgb_low_kept_by_track_support_count"] += sum(
                        1 for box in tracker_inputs if box.get("support_type") == "track_support"
                    )
                    kept_low = stats["rgb_low_promoted_by_ir_count"] + stats["rgb_low_kept_by_track_support_count"]
                    stats["rgb_low_suppressed_count"] = stats["rgb_low_boxes_count"] - kept_low
                elif fusion == "rgb_ir_presence_roi_redetect":
                    tracker_inputs = []
                    for box in rgb_boxes:
                        conf = float(box.get("confidence", 0.0))
                        if conf < rgb_low_conf:
                            continue
                        presence_start = time.perf_counter()
                        verified, row = _presence_verified_box(box, ir_frame, affine_matrix, args)
                        stats["stage_timings_seconds"]["low_threshold_detection_or_roi_redetect"] += (
                            time.perf_counter() - presence_start
                        )
                        row["frame_id"] = frame_id
                        row["source"] = "rgb_detection_presence_verify"
                        if verified is not None:
                            verified["allow_new_track"] = True
                            if conf >= rgb_high_conf:
                                verified["support_type"] = "ir_presence_verified"
                                stats["ir_presence_rgb_verified_count"] += 1
                            else:
                                verified["support_type"] = "ir_presence_low_verified"
                                stats["ir_presence_low_promoted_count"] += 1
                            tracker_inputs.append(verified)
                        elif conf >= rgb_high_conf:
                            out = dict(box)
                            out["allow_new_track"] = True
                            out["support_type"] = "high"
                            tracker_inputs.append(out)
                            stats["ir_presence_rgb_rejected_count"] += 1
                        else:
                            stats["ir_presence_rgb_rejected_count"] += 1
                        if ir_presence_diagnostics:
                            ir_presence_diag_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

                    kept_low = stats["ir_presence_low_promoted_count"]
                    stats["rgb_low_suppressed_count"] = stats["rgb_low_boxes_count"] - kept_low

                    if ir_presence_enable:
                        presence_start = time.perf_counter()
                        track_boxes = _get_presence_track_boxes(mot_tracker, max_age=ir_presence_max_track_age)
                        stats["ir_presence_tracks_checked_count"] += len(track_boxes)
                        roi_dets, presence_rows = ir_presence_support.update(
                            frame_id=frame_id,
                            rgb_frame=rgb_frame,
                            ir_frame=ir_frame,
                            track_boxes=track_boxes,
                            current_rgb_detections=tracker_inputs,
                            ir_to_rgb_affine=affine_matrix,
                            processor=processor,
                            args=args,
                        )
                        stats["stage_timings_seconds"]["low_threshold_detection_or_roi_redetect"] += (
                            time.perf_counter() - presence_start
                        )
                        tracker_inputs.extend(roi_dets)
                        stats["ir_presence_roi_redetect_emitted_count"] += len(roi_dets)
                        for row in presence_rows:
                            reason = row.get("reject_reason", "")
                            if reason == "rgb_matched":
                                stats["ir_presence_skipped_rgb_matched_count"] += 1
                            elif reason in {"weak_thermal_response", "invalid_roi", "empty_roi"}:
                                stats["ir_presence_rejected_weak_count"] += 1
                            elif reason == "no_geometry_consistent_rgb_low_det":
                                stats["ir_presence_rejected_no_rgb_low_det_count"] += 1
                            if ir_presence_diagnostics:
                                ir_presence_diag_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    proben_result = fuse_rgb_ir_proben(rgb_boxes, mapped_ir_boxes, args, frame_id=frame_id)
                    tracker_inputs = proben_result.detections
                    for key in (
                        "rgb_ir_proben_matched_count",
                        "rgb_ir_proben_promoted_count",
                        "rgb_ir_proben_rejected_by_geometry_count",
                        "rgb_ir_proben_rejected_by_score_count",
                        "rgb_ir_proben_ir_only_count",
                    ):
                        stats[key] += int(proben_result.stats.get(key, 0))
                    matched_count = int(proben_result.stats.get("rgb_ir_proben_matched_count", 0))
                    if matched_count > 0:
                        proben_score_gain_sum += float(proben_result.stats["rgb_ir_proben_avg_score_gain"]) * matched_count
                        proben_box_shift_sum += float(proben_result.stats["rgb_ir_proben_avg_box_shift"]) * matched_count
                        proben_box_shift_count += matched_count
                        stats["rgb_ir_proben_max_box_shift"] = max(
                            stats["rgb_ir_proben_max_box_shift"],
                            float(proben_result.stats["rgb_ir_proben_max_box_shift"]),
                        )
                    stats["rgb_low_promoted_by_ir_count"] += int(proben_result.stats.get("rgb_ir_proben_promoted_count", 0))
                    stats["rgb_low_suppressed_count"] = (
                        stats["rgb_low_boxes_count"]
                        - stats["rgb_ir_proben_promoted_count"]
                    )
                    if save_proben_diagnostics:
                        for row in proben_result.diagnostics:
                            proben_diag_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if fusion == "rgb_ir_proben_lost_reacquire" and ir_reacquire_enable:
                        ir_only_boxes = [
                            row["ir_box_mapped"]
                            for row in proben_result.diagnostics
                            if row.get("decision") == "ir_only_not_emitted" and row.get("ir_box_mapped") is not None
                        ]
                        lost_tracks = _get_lost_track_boxes(mot_tracker, max_age=ir_reacquire_max_age)
                        reacquire_dets, reacquire_rows = ir_reacquire_buffer.update(
                            frame_id=frame_id,
                            ir_only_boxes=ir_only_boxes,
                            lost_track_boxes=lost_tracks,
                            args=args,
                        )
                        tracker_inputs.extend(reacquire_dets)
                        stats["ir_reacquire_emitted_count"] += len(reacquire_dets)
                        for row in reacquire_rows:
                            candidate_id = row.get("candidate_id")
                            if candidate_id is not None and candidate_id not in seen_candidate_ids:
                                seen_candidate_ids.add(candidate_id)
                                stats["ir_reacquire_candidates_created"] += 1
                            if row.get("decision") == "emitted":
                                stats["ir_reacquire_candidates_confirmed"] += 1
                            reason = row.get("reject_reason", "")
                            if reason == "low_conf":
                                stats["ir_reacquire_rejected_low_conf"] += 1
                            elif reason == "no_lost_track":
                                stats["ir_reacquire_rejected_no_lost_track"] += 1
                            elif reason == "gate_failed":
                                stats["ir_reacquire_rejected_gate"] += 1
                            elif reason == "not_confirmed":
                                stats["ir_reacquire_rejected_not_confirmed"] += 1
                            elif reason == "max_per_frame":
                                stats["ir_reacquire_rejected_max_per_frame"] += 1
                            if ir_reacquire_diagnostics:
                                ir_reacquire_diag_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["stage_timings_seconds"]["fusion"] += time.perf_counter() - fusion_start

                write_start = time.perf_counter()
                for box in tracker_inputs:
                    det_fh.write(format_detection_result_line(frame_id, box) + "\n")
                stats["stage_timings_seconds"]["result_write_export"] += time.perf_counter() - write_start

                tracker_start = time.perf_counter()
                tracked_boxes = mot_tracker.update(tracker_inputs, rgb_frame.shape, frame=rgb_frame)
                stats["stage_timings_seconds"]["motion_template_lifecycle_tracker"] += time.perf_counter() - tracker_start

                write_start = time.perf_counter()
                for box in tracked_boxes:
                    fh.write(format_mot_result_line(frame_id, box) + "\n")
                stats["stage_timings_seconds"]["result_write_export"] += time.perf_counter() - write_start

                stats["processed_frames"] = frame_id
                if progress_interval > 0 and frame_id % progress_interval == 0:
                    fh.flush()
                    suffix = f"/{target_frames}" if target_frames > 0 else ""
                    print(f"[PROGRESS] {run_name} {seq_name}: frame {frame_id}{suffix}", flush=True)

        runtime = time.time() - start_time
        stats["total_runtime_seconds"] = round(runtime, 6)
        stats["runtime_fps"] = round(stats["processed_frames"] / runtime, 6) if runtime > 0 else 0.0
        if proben_box_shift_count > 0:
            stats["rgb_ir_proben_avg_score_gain"] = round(proben_score_gain_sum / proben_box_shift_count, 6)
            stats["rgb_ir_proben_avg_box_shift"] = round(proben_box_shift_sum / proben_box_shift_count, 6)
            stats["rgb_ir_proben_max_box_shift"] = round(float(stats["rgb_ir_proben_max_box_shift"]), 6)
        stats_write_start = time.perf_counter()
        stats_file = output_dir / "stats.json"
        _write_diagnostics_csv(stats, diagnostics_csv)
        stats["stage_timings_seconds"]["result_write_export"] = round(
            stats["stage_timings_seconds"]["result_write_export"] + (time.perf_counter() - stats_write_start),
            6,
        )
        _finalize_stage_timings(stats)
        stage_timings_json.write_text(
            json.dumps(
                {
                    "stage_timings_seconds": stats["stage_timings_seconds"],
                    "stage_timings_per_frame_seconds": stats["stage_timings_per_frame_seconds"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_stats(stats)
        print(f"stats_json_path: {stats_file}")
        return output_file, stats
    finally:
        visible_cap.release()
        if infrared_cap is not None:
            infrared_cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RGB/IR support BoT-SORT MOTChallenge results")
    parser.add_argument("--visible-input", required=True)
    parser.add_argument("--infrared-input", default="")
    parser.add_argument("--tracker", required=True, choices=TRACKER_CHOICES)
    parser.add_argument("--fusion", required=True, choices=FUSION_CHOICES)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--calib", default="")
    parser.add_argument("--config", default=DEFAULT_ALIGNMENT_CONFIG)
    parser.add_argument("--rgb-high-conf", type=float, default=0.50)
    parser.add_argument("--rgb-low-conf", type=float, default=0.30)
    parser.add_argument("--ir-conf", type=float, default=0.45)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--track-iou-thresh", type=float, default=0.20)
    parser.add_argument("--track-dist-factor", type=float, default=1.5)
    parser.add_argument("--ir-rgb-match-iou", type=float, default=0.10)
    parser.add_argument("--ir-rgb-match-dist-factor", type=float, default=2.0)
    parser.add_argument("--proben-match-iou", type=float, default=0.15)
    parser.add_argument("--proben-match-dist-factor", type=float, default=1.5)
    parser.add_argument("--proben-keep-conf", type=float, default=0.50)
    parser.add_argument("--proben-ir-box-weight", type=float, default=0.5)
    parser.add_argument("--proben-scale-ratio-min", type=float, default=0.4)
    parser.add_argument("--proben-scale-ratio-max", type=float, default=2.5)
    parser.add_argument("--proben-low-allow-new-track", action="store_true")
    parser.add_argument("--proben-box-mode", choices=["score_only", "savg"], default="score_only")
    parser.add_argument("--save-proben-diagnostics", action="store_true")
    parser.add_argument("--ir-reacquire-enable", action="store_true")
    parser.add_argument("--ir-reacquire-confirm-frames", type=int, default=2)
    parser.add_argument("--ir-reacquire-max-age", type=int, default=30)
    parser.add_argument("--ir-reacquire-iou", type=float, default=0.05)
    parser.add_argument("--ir-reacquire-dist-factor", type=float, default=2.5)
    parser.add_argument("--ir-reacquire-max-per-frame", type=int, default=2)
    parser.add_argument("--ir-reacquire-min-conf", type=float, default=0.45)
    parser.add_argument("--ir-reacquire-candidate-iou", type=float, default=0.10)
    parser.add_argument("--ir-reacquire-candidate-dist-factor", type=float, default=1.5)
    parser.add_argument("--ir-reacquire-max-gap", type=int, default=1)
    parser.add_argument("--ir-reacquire-diagnostics", action="store_true")
    parser.add_argument("--ir-presence-enable", action="store_true")
    parser.add_argument("--ir-presence-z-thresh", type=float, default=2.0)
    parser.add_argument("--ir-presence-min-hot-area-ratio", type=float, default=0.01)
    parser.add_argument("--ir-presence-bg-pad", type=float, default=1.0)
    parser.add_argument("--ir-presence-min-bg-std", type=float, default=1.0)
    parser.add_argument("--ir-presence-conf-boost", type=float, default=0.10)
    parser.add_argument("--ir-presence-max-track-age", type=int, default=8)
    parser.add_argument("--ir-presence-rgb-match-iou", type=float, default=0.20)
    parser.add_argument("--ir-presence-rgb-match-dist-factor", type=float, default=1.5)
    parser.add_argument("--ir-presence-roi-pad", type=float, default=1.8)
    parser.add_argument("--ir-presence-min-crop-size", type=float, default=64.0)
    parser.add_argument("--ir-presence-roi-upscale", type=float, default=2.0)
    parser.add_argument("--ir-presence-rgb-roi-low-conf", type=float, default=0.20)
    parser.add_argument("--ir-presence-redetect-iou", type=float, default=0.05)
    parser.add_argument("--ir-presence-redetect-dist-factor", type=float, default=2.5)
    parser.add_argument("--ir-presence-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dual_modal_video_to_mot_results(
        visible_input=args.visible_input,
        infrared_input=args.infrared_input or None,
        output_root=args.output_root,
        run_name=args.run_name,
        seq_name=args.seq_name,
        tracker=args.tracker,
        fusion=args.fusion,
        calib=args.calib or None,
        config=args.config,
        rgb_high_conf=args.rgb_high_conf,
        rgb_low_conf=args.rgb_low_conf,
        ir_conf=args.ir_conf,
        alpha=args.alpha,
        max_frames=args.max_frames,
        progress_interval=args.progress_interval,
        track_iou_thresh=args.track_iou_thresh,
        track_dist_factor=args.track_dist_factor,
        ir_rgb_match_iou=args.ir_rgb_match_iou,
        ir_rgb_match_dist_factor=args.ir_rgb_match_dist_factor,
        proben_match_iou=args.proben_match_iou,
        proben_match_dist_factor=args.proben_match_dist_factor,
        proben_keep_conf=args.proben_keep_conf,
        proben_ir_box_weight=args.proben_ir_box_weight,
        proben_scale_ratio_min=args.proben_scale_ratio_min,
        proben_scale_ratio_max=args.proben_scale_ratio_max,
        proben_low_allow_new_track=args.proben_low_allow_new_track,
        proben_box_mode=args.proben_box_mode,
        save_proben_diagnostics=args.save_proben_diagnostics,
        ir_reacquire_enable=args.ir_reacquire_enable,
        ir_reacquire_confirm_frames=args.ir_reacquire_confirm_frames,
        ir_reacquire_max_age=args.ir_reacquire_max_age,
        ir_reacquire_iou=args.ir_reacquire_iou,
        ir_reacquire_dist_factor=args.ir_reacquire_dist_factor,
        ir_reacquire_max_per_frame=args.ir_reacquire_max_per_frame,
        ir_reacquire_min_conf=args.ir_reacquire_min_conf,
        ir_reacquire_candidate_iou=args.ir_reacquire_candidate_iou,
        ir_reacquire_candidate_dist_factor=args.ir_reacquire_candidate_dist_factor,
        ir_reacquire_max_gap=args.ir_reacquire_max_gap,
        ir_reacquire_diagnostics=args.ir_reacquire_diagnostics,
        ir_presence_enable=args.ir_presence_enable,
        ir_presence_z_thresh=args.ir_presence_z_thresh,
        ir_presence_min_hot_area_ratio=args.ir_presence_min_hot_area_ratio,
        ir_presence_bg_pad=args.ir_presence_bg_pad,
        ir_presence_min_bg_std=args.ir_presence_min_bg_std,
        ir_presence_conf_boost=args.ir_presence_conf_boost,
        ir_presence_max_track_age=args.ir_presence_max_track_age,
        ir_presence_rgb_match_iou=args.ir_presence_rgb_match_iou,
        ir_presence_rgb_match_dist_factor=args.ir_presence_rgb_match_dist_factor,
        ir_presence_roi_pad=args.ir_presence_roi_pad,
        ir_presence_min_crop_size=args.ir_presence_min_crop_size,
        ir_presence_roi_upscale=args.ir_presence_roi_upscale,
        ir_presence_rgb_roi_low_conf=args.ir_presence_rgb_roi_low_conf,
        ir_presence_redetect_iou=args.ir_presence_redetect_iou,
        ir_presence_redetect_dist_factor=args.ir_presence_redetect_dist_factor,
        ir_presence_diagnostics=args.ir_presence_diagnostics,
    )


if __name__ == "__main__":
    main()
