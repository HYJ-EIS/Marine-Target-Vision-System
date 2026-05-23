#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .extract_tracking_frames import (
        build_video_job,
        discover_video_groups,
        merge_raw_and_tracked_boxes,
    )
except ImportError:
    from extract_tracking_frames import (
        build_video_job,
        discover_video_groups,
        merge_raw_and_tracked_boxes,
    )
from target_module.image_detect_module.config import Config


WINDOW_RESULT_COLUMNS = [
    "unit_id",
    "date_batch",
    "modality",
    "pair_status",
    "window_start",
    "window_end",
    "frame_count",
    "low_fps_window",
    "duration_from_metadata",
    "decode_error",
    "primary_scene",
    "sea_ratio",
    "sky_ratio",
    "shoreline_land_ratio",
    "scene_classification_failed",
    "scene_classification_low_confidence",
    "scene_source",
    "target_presence",
    "target_count",
    "track_count",
    "bbox_area_ratio_stats",
    "scale_bin_counts",
    "small_target_count",
    "detector_not_run",
    "usable",
]

VIDEO_RESULT_COLUMNS = [
    "unit_id",
    "modalities_present",
    "pair_status",
    "primary_scene_by_time",
    "covered_scenes",
    "shoreline_mixed_ratio",
    "target_window_ratio",
    "small_target_window_ratio",
    "usable_window_ratio",
    "max_consecutive_unusable_windows",
    "usable",
    "date_batch",
    "metadata_unreadable",
    "t_scale_calibration",
    "t_scale_factor",
]

PRECHECK_COLUMNS = [
    "date_batch",
    "video_file_count",
    "visible_file_count",
    "infrared_file_count",
    "unit_count",
    "paired_count",
    "v_only_count",
    "t_only_count",
    "paired_rate",
    "metadata_readable_count",
    "metadata_unreadable_count",
    "metadata_readable_rate",
    "duration_min_sec",
    "duration_median_sec",
    "duration_max_sec",
    "metadata_unreadable_files",
]

REPORT_COLUMNS = VIDEO_RESULT_COLUMNS + [
    "scene_missing",
    "scene_x_small_target_sparse",
    "paired_modality_gap",
    "metadata_unreadable",
    "low_usable_ratio",
    "paired_visible_proxy_count",
    "dominant_date_share",
    "date_entropy",
    "top1_date_batch",
]


@dataclass
class SceneFrameObservation:
    timestamp_sec: float
    scene_ratios: Dict[str, float]
    horizon_confidence: float
    frame_intensity_variance: float


@dataclass
class TargetFrameObservation:
    timestamp_sec: float
    target_count: int
    track_count: int
    bbox_area_ratios: List[float]
    detector_not_run: bool


@dataclass
class WindowDefinition:
    start_sec: float
    end_sec: float
    frame_indices: List[int]
    low_fps_window: bool


@dataclass
class VideoAnalysisResult:
    unit_id: str
    date_batch: str
    modality: str
    pair_status: str
    modality_present: str
    duration_from_metadata: bool
    metadata_duration_sec: float
    decoded_duration_sec: float
    windows: List[dict]
    open_failed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline batch video scene and target classification")
    parser.add_argument("--input-root", default=Config.DATASET_INPUT_ROOT)
    parser.add_argument("--output-root", default=Config.DATASET_OUTPUT_ROOT)
    parser.add_argument("--window-seconds", type=float, default=Config.DATASET_WINDOW_SECONDS)
    parser.add_argument("--resume", action="store_true", default=Config.DATASET_ENABLE_RESUME)
    parser.add_argument("--workers", type=int, default=Config.DATASET_MAX_WORKERS)
    return parser.parse_args()


def normalize_worker_count(requested_workers: int) -> int:
    cpu_count = os.cpu_count() or 1
    if requested_workers <= 0:
        return 1
    return max(1, min(requested_workers, cpu_count))


def resolve_run_root(output_root: str, resume: bool) -> str:
    os.makedirs(output_root, exist_ok=True)
    if resume:
        candidates = [
            os.path.join(output_root, name)
            for name in os.listdir(output_root)
            if os.path.isdir(os.path.join(output_root, name))
        ]
        if candidates:
            return max(candidates, key=os.path.getmtime)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_root, run_id)


def json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def round_ratio(value: float) -> float:
    return round(float(value), 6)


def list_to_stats(values: Sequence[float]) -> str:
    if not values:
        return json_dumps({"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0})
    return json_dumps(
        {
            "count": len(values),
            "min": round_ratio(min(values)),
            "max": round_ratio(max(values)),
            "mean": round_ratio(sum(values) / len(values)),
            "median": round_ratio(float(statistics.median(values))),
        }
    )


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_csv_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: List[dict], columns: List[str]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_rows(path: str, rows: List[dict], columns: List[str]) -> None:
    if not rows:
        return
    ensure_parent_dir(path)
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: dict) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def get_pair_status(modalities: Dict[str, object]) -> str:
    has_rgb = "rgb" in modalities
    has_ir = "ir" in modalities
    if has_rgb and has_ir:
        return "paired"
    if has_rgb:
        return "V_only"
    return "T_only"


def build_unit_id(job) -> str:
    return f"{job.session_id}__{job.video_id}"


def modality_label(job) -> str:
    return "V" if job.modality == "rgb" else "T"


def estimate_duration_from_metadata(video_path: str) -> Tuple[bool, float, float, int]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return False, 0.0, 0.0, 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps > 0 and frame_count > 0:
        return True, frame_count / fps, fps, frame_count
    return False, 0.0, fps, frame_count


def compute_horizon_confidence(gray: np.ndarray) -> float:
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.abs(sobel_y)
    horizontal_profile = magnitude.mean(axis=1)
    if horizontal_profile.size == 0:
        return 0.0
    profile_mean = float(horizontal_profile.mean())
    if profile_mean <= 1e-6:
        return 0.0
    return float(horizontal_profile.max()) / profile_mean


def downsample_for_scene(frame: np.ndarray) -> np.ndarray:
    max_dim = int(getattr(Config, "DATASET_SCENE_ANALYSIS_MAX_DIM", 0) or 0)
    if max_dim <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return frame
    scale = max_dim / float(longest)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)


def normalize_scene_ratios(sea: float, sky: float, shoreline_land: float) -> Dict[str, float]:
    total = sea + sky + shoreline_land
    if total <= 0:
        return {"sea": 0.0, "sky": 0.0, "shoreline_land": 0.0}
    return {
        "sea": safe_divide(sea, total),
        "sky": safe_divide(sky, total),
        "shoreline_land": safe_divide(shoreline_land, total),
    }


def extract_visible_scene_features(frame: np.ndarray) -> Tuple[Dict[str, float], float, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0].astype(np.float32) * 2.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    sky_mask = (
        (((h >= 170) & (h <= 260)) & (s <= 0.45) & (v >= 0.55))
        | ((s <= 0.20) & (v >= 0.72))
    )
    sea_mask = (
        (((h >= 160) & (h <= 260)) & (s >= 0.18) & (v >= 0.15) & (v <= 0.72))
        | ((v >= 0.10) & (v <= 0.52) & (s >= 0.12))
    )
    edges = cv2.Canny(gray, 60, 160)
    shoreline_mask = (~(sky_mask | sea_mask)) | (edges > 0)
    ratios = normalize_scene_ratios(
        float(sea_mask.mean()),
        float(sky_mask.mean()),
        float(shoreline_mask.mean()),
    )
    return ratios, compute_horizon_confidence(gray), float(np.var(gray))


def extract_infrared_scene_features(frame: np.ndarray) -> Tuple[Dict[str, float], float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_norm = gray.astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray_norm, (5, 5), 0)
    gradient_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)

    upper = smooth[: smooth.shape[0] // 2, :] if smooth.shape[0] else smooth
    lower = smooth[smooth.shape[0] // 2 :, :] if smooth.shape[0] else smooth
    upper_mean = float(upper.mean()) if upper.size else 0.0
    lower_mean = float(lower.mean()) if lower.size else 0.0

    if upper_mean >= lower_mean:
        sky_mask = smooth >= np.quantile(smooth, 0.65)
        sea_mask = smooth <= np.quantile(smooth, 0.45)
    else:
        sky_mask = smooth <= np.quantile(smooth, 0.35)
        sea_mask = smooth >= np.quantile(smooth, 0.55)

    shoreline_mask = magnitude >= np.quantile(magnitude, 0.72)
    shoreline_mask = shoreline_mask | (~(sky_mask | sea_mask))
    ratios = normalize_scene_ratios(
        float(sea_mask.mean()),
        float(sky_mask.mean()),
        float(shoreline_mask.mean()),
    )
    return ratios, compute_horizon_confidence(gray), float(np.var(gray))


def extract_scene_features(frame: np.ndarray, modality: str) -> Tuple[Dict[str, float], float, float]:
    frame = downsample_for_scene(frame)
    if modality == "T":
        return extract_infrared_scene_features(frame)
    return extract_visible_scene_features(frame)


def classify_scene(
    sea_ratio: float,
    sky_ratio: float,
    shoreline_land_ratio: float,
) -> str:
    if shoreline_land_ratio >= Config.DATASET_SCENE_SHORELINE_HIGH_THRESHOLD:
        return "shoreline_mixed"
    if (
        Config.DATASET_SCENE_SHORELINE_LOW_THRESHOLD
        <= shoreline_land_ratio
        < Config.DATASET_SCENE_SHORELINE_HIGH_THRESHOLD
        and sky_ratio >= Config.DATASET_SCENE_SKY_THRESHOLD
        and sea_ratio >= sky_ratio
    ):
        return "nearshore_sea_sky"
    if (
        Config.DATASET_SCENE_SHORELINE_LOW_THRESHOLD
        <= shoreline_land_ratio
        < Config.DATASET_SCENE_SHORELINE_HIGH_THRESHOLD
        and sky_ratio < Config.DATASET_SCENE_SKY_THRESHOLD
        and sea_ratio >= shoreline_land_ratio
    ):
        return "nearshore_sea"
    if (
        sea_ratio + sky_ratio >= 0.9
        and shoreline_land_ratio < Config.DATASET_SCENE_SHORELINE_LOW_THRESHOLD
        and sea_ratio >= 0.2
        and sky_ratio >= 0.2
    ):
        return "sea_sky"
    if sea_ratio >= 0.9:
        return "pure_sea"
    if sky_ratio >= 0.9:
        return "pure_sky"
    dominant = {
        "shoreline_mixed": shoreline_land_ratio,
        "nearshore_sea_sky": min(sea_ratio, sky_ratio),
        "nearshore_sea": sea_ratio,
        "sea_sky": sea_ratio + sky_ratio,
        "pure_sea": sea_ratio,
        "pure_sky": sky_ratio,
    }
    for label in [
        "shoreline_mixed",
        "nearshore_sea_sky",
        "nearshore_sea",
        "sea_sky",
        "pure_sea",
        "pure_sky",
    ]:
        if dominant[label] == max(dominant.values()):
            return label
    return "shoreline_mixed"


def is_t_scene_low_confidence(
    sea_ratio: float,
    sky_ratio: float,
    shoreline_land_ratio: float,
    frame_intensity_variance: float,
    horizon_confidence: float,
) -> bool:
    if max(sea_ratio, sky_ratio, shoreline_land_ratio) < Config.DATASET_T_SCENE_DOMINANT_RATIO_MIN:
        return True
    if frame_intensity_variance < Config.DATASET_T_SCENE_MIN_INTENSITY_VARIANCE:
        return True
    if (
        horizon_confidence < Config.DATASET_T_SCENE_MIN_HORIZON_CONFIDENCE
        and Config.DATASET_SCENE_SHORELINE_LOW_THRESHOLD
        <= shoreline_land_ratio
        < Config.DATASET_SCENE_SHORELINE_HIGH_THRESHOLD
    ):
        return True
    return False


def get_scale_bin(area_ratio: float, modality: str, t_scale_factor: float = 1.0) -> str:
    small_threshold = Config.DATASET_SCALE_SMALL_THRESHOLD
    medium_threshold = Config.DATASET_SCALE_MEDIUM_THRESHOLD
    if modality == "T":
        small_threshold *= t_scale_factor
        medium_threshold *= t_scale_factor
    if area_ratio < small_threshold:
        return "small"
    if area_ratio < medium_threshold:
        return "medium"
    return "large"


def build_time_windows(
    timestamps_sec: Sequence[float],
    decoded_duration_sec: float,
    base_window_seconds: float,
    min_frames_per_window: int,
    max_window_seconds: float,
) -> List[WindowDefinition]:
    if not timestamps_sec:
        return [
            WindowDefinition(
                start_sec=0.0,
                end_sec=base_window_seconds,
                frame_indices=[],
                low_fps_window=True,
            )
        ]
    windows: List[WindowDefinition] = []
    current_start = 0.0
    final_end = max(decoded_duration_sec, timestamps_sec[-1] + 1e-6)
    total_frames = len(timestamps_sec)
    frame_ptr = 0
    while current_start < final_end or (frame_ptr < total_frames and not windows):
        candidate = base_window_seconds
        low_fps = False
        while True:
            end_sec = current_start + candidate
            scan_ptr = frame_ptr
            while scan_ptr < total_frames and timestamps_sec[scan_ptr] < current_start:
                scan_ptr += 1
            end_ptr = scan_ptr
            while end_ptr < total_frames and timestamps_sec[end_ptr] < end_sec:
                end_ptr += 1
            frame_count = end_ptr - scan_ptr
            if frame_count >= min_frames_per_window or candidate >= max_window_seconds:
                if frame_count < min_frames_per_window:
                    low_fps = True
                break
            candidate *= 2.0
        chosen_indices = list(range(scan_ptr, end_ptr))
        windows.append(
            WindowDefinition(
                start_sec=round_ratio(current_start),
                end_sec=round_ratio(min(current_start + candidate, final_end if final_end > current_start else current_start + candidate)),
                frame_indices=chosen_indices,
                low_fps_window=low_fps,
            )
        )
        frame_ptr = end_ptr
        current_start += candidate
        if current_start >= final_end and frame_ptr >= total_frames:
            break
    return windows


def window_overlap_ratio(window_a: dict, window_b: dict) -> float:
    overlap = max(
        0.0,
        min(float(window_a["window_end"]), float(window_b["window_end"]))
        - max(float(window_a["window_start"]), float(window_b["window_start"])),
    )
    shortest = min(
        float(window_a["window_end"]) - float(window_a["window_start"]),
        float(window_b["window_end"]) - float(window_b["window_start"]),
    )
    if shortest <= 0:
        return 0.0
    return overlap / shortest


def find_best_overlap_window(source_window: dict, candidate_windows: Sequence[dict]) -> Optional[dict]:
    eligible = [
        candidate
        for candidate in candidate_windows
        if window_overlap_ratio(source_window, candidate) >= Config.DATASET_ALIGNMENT_MIN_OVERLAP_RATIO
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: window_overlap_ratio(source_window, item))


def evaluate_window_usability(
    frame_count: int,
    scene_classification_failed: bool,
    detector_not_run: bool,
    decode_error: bool,
) -> bool:
    if frame_count < 4:
        return False
    if scene_classification_failed or detector_not_run or decode_error:
        return False
    return True


def max_consecutive_false(flags: Sequence[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if not flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def evaluate_video_usability(usable_window_ratio: float, max_consecutive_unusable_windows: int) -> bool:
    return (
        usable_window_ratio >= Config.DATASET_USABLE_WINDOW_RATIO_THRESHOLD
        and max_consecutive_unusable_windows <= Config.DATASET_MAX_CONSECUTIVE_UNUSABLE_WINDOWS
    )


def build_window_record(
    unit_id: str,
    date_batch: str,
    modality: str,
    pair_status: str,
    duration_from_metadata: bool,
    observations: Sequence[TargetFrameObservation],
    window: WindowDefinition,
    t_scale_factor: float = 1.0,
) -> dict:
    decode_error = False
    scene_failed = not bool(window.frame_indices)
    scene_source = "native"
    if not window.frame_indices:
        sea_ratio = 0.0
        sky_ratio = 0.0
        shoreline_land_ratio = 0.0
        horizon_confidence = 0.0
        intensity_variance = 0.0
        primary_scene = "shoreline_mixed"
        target_count = 0
        track_count = 0
        bbox_area_ratios: List[float] = []
        detector_not_run = True
    else:
        selected = [observations[idx] for idx in window.frame_indices]
        sea_ratio = float(np.mean([item.scene_ratios["sea"] for item in selected]))
        sky_ratio = float(np.mean([item.scene_ratios["sky"] for item in selected]))
        shoreline_land_ratio = float(np.mean([item.scene_ratios["shoreline_land"] for item in selected]))
        horizon_confidence = float(np.mean([item.horizon_confidence for item in selected]))
        intensity_variance = float(np.mean([item.frame_intensity_variance for item in selected]))
        primary_scene = classify_scene(sea_ratio, sky_ratio, shoreline_land_ratio)
        target_count = max((item.target_count for item in selected), default=0)
        track_count = max((item.track_count for item in selected), default=0)
        bbox_area_ratios = [ratio for item in selected for ratio in item.bbox_area_ratios]
        detector_not_run = all(item.detector_not_run for item in selected)

    scene_low_confidence = False
    if modality == "T" and not scene_failed:
        scene_low_confidence = is_t_scene_low_confidence(
            sea_ratio,
            sky_ratio,
            shoreline_land_ratio,
            intensity_variance,
            horizon_confidence,
        )

    scale_bin_counts = {"small": 0, "medium": 0, "large": 0}
    small_target_count = 0
    for area_ratio in bbox_area_ratios:
        scale_bin = get_scale_bin(area_ratio, modality, t_scale_factor=t_scale_factor)
        scale_bin_counts[scale_bin] += 1
        if scale_bin == "small":
            small_target_count += 1

    usable = evaluate_window_usability(
        len(window.frame_indices),
        scene_failed,
        detector_not_run,
        decode_error,
    )
    return {
        "unit_id": unit_id,
        "date_batch": date_batch,
        "modality": modality,
        "pair_status": pair_status,
        "window_start": round_ratio(window.start_sec),
        "window_end": round_ratio(window.end_sec),
        "frame_count": len(window.frame_indices),
        "low_fps_window": window.low_fps_window,
        "duration_from_metadata": duration_from_metadata,
        "decode_error": decode_error,
        "primary_scene": primary_scene,
        "sea_ratio": round_ratio(sea_ratio),
        "sky_ratio": round_ratio(sky_ratio),
        "shoreline_land_ratio": round_ratio(shoreline_land_ratio),
        "scene_classification_failed": scene_failed,
        "scene_classification_low_confidence": scene_low_confidence,
        "scene_source": scene_source,
        "target_presence": bool(target_count > 0),
        "target_count": target_count,
        "track_count": track_count,
        "bbox_area_ratio_stats": list_to_stats(bbox_area_ratios),
        "scale_bin_counts": json_dumps(scale_bin_counts),
        "small_target_count": small_target_count,
        "detector_not_run": detector_not_run,
        "usable": usable,
    }


def build_scene_window_record(
    unit_id: str,
    date_batch: str,
    modality: str,
    pair_status: str,
    duration_from_metadata: bool,
    observations: Sequence[SceneFrameObservation],
    window: WindowDefinition,
) -> dict:
    scene_failed = not bool(window.frame_indices)
    if scene_failed:
        sea_ratio = 0.0
        sky_ratio = 0.0
        shoreline_land_ratio = 0.0
        horizon_confidence = 0.0
        intensity_variance = 0.0
        primary_scene = "shoreline_mixed"
    else:
        selected = [observations[idx] for idx in window.frame_indices]
        sea_ratio = float(np.mean([item.scene_ratios["sea"] for item in selected]))
        sky_ratio = float(np.mean([item.scene_ratios["sky"] for item in selected]))
        shoreline_land_ratio = float(np.mean([item.scene_ratios["shoreline_land"] for item in selected]))
        horizon_confidence = float(np.mean([item.horizon_confidence for item in selected]))
        intensity_variance = float(np.mean([item.frame_intensity_variance for item in selected]))
        primary_scene = classify_scene(sea_ratio, sky_ratio, shoreline_land_ratio)

    scene_low_confidence = False
    if modality == "T" and not scene_failed:
        scene_low_confidence = is_t_scene_low_confidence(
            sea_ratio,
            sky_ratio,
            shoreline_land_ratio,
            intensity_variance,
            horizon_confidence,
        )

    return {
        "unit_id": unit_id,
        "date_batch": date_batch,
        "modality": modality,
        "pair_status": pair_status,
        "window_start": round_ratio(window.start_sec),
        "window_end": round_ratio(window.end_sec),
        "frame_count": len(window.frame_indices),
        "low_fps_window": window.low_fps_window,
        "duration_from_metadata": duration_from_metadata,
        "decode_error": False,
        "primary_scene": primary_scene,
        "sea_ratio": round_ratio(sea_ratio),
        "sky_ratio": round_ratio(sky_ratio),
        "shoreline_land_ratio": round_ratio(shoreline_land_ratio),
        "scene_classification_failed": scene_failed,
        "scene_classification_low_confidence": scene_low_confidence,
        "scene_source": "native",
    }


def build_target_window_record(
    observations: Sequence[TargetFrameObservation],
    window: WindowDefinition,
    modality: str,
    t_scale_factor: float = 1.0,
) -> dict:
    if not window.frame_indices:
        target_count = 0
        track_count = 0
        bbox_area_ratios: List[float] = []
        detector_not_run = True
    else:
        selected = [observations[idx] for idx in window.frame_indices]
        target_count = max((item.target_count for item in selected), default=0)
        track_count = max((item.track_count for item in selected), default=0)
        bbox_area_ratios = [ratio for item in selected for ratio in item.bbox_area_ratios]
        detector_not_run = all(item.detector_not_run for item in selected)

    scale_bin_counts = {"small": 0, "medium": 0, "large": 0}
    small_target_count = 0
    for area_ratio in bbox_area_ratios:
        scale_bin = get_scale_bin(area_ratio, modality, t_scale_factor=t_scale_factor)
        scale_bin_counts[scale_bin] += 1
        if scale_bin == "small":
            small_target_count += 1

    return {
        "target_presence": bool(target_count > 0),
        "target_count": target_count,
        "track_count": track_count,
        "bbox_area_ratio_stats": list_to_stats(bbox_area_ratios),
        "scale_bin_counts": json_dumps(scale_bin_counts),
        "small_target_count": small_target_count,
        "detector_not_run": detector_not_run,
    }


def build_failed_video_result(unit_id: str, date_batch: str, modality: str, pair_status: str) -> VideoAnalysisResult:
    failed_window = {
        "unit_id": unit_id,
        "date_batch": date_batch,
        "modality": modality,
        "pair_status": pair_status,
        "window_start": 0.0,
        "window_end": Config.DATASET_WINDOW_SECONDS,
        "frame_count": 0,
        "low_fps_window": True,
        "duration_from_metadata": False,
        "decode_error": True,
        "primary_scene": "shoreline_mixed",
        "sea_ratio": 0.0,
        "sky_ratio": 0.0,
        "shoreline_land_ratio": 0.0,
        "scene_classification_failed": True,
        "scene_classification_low_confidence": modality == "T",
        "scene_source": "native",
        "target_presence": False,
        "target_count": 0,
        "track_count": 0,
        "bbox_area_ratio_stats": list_to_stats([]),
        "scale_bin_counts": json_dumps({"small": 0, "medium": 0, "large": 0}),
        "small_target_count": 0,
        "detector_not_run": True,
        "usable": False,
    }
    return VideoAnalysisResult(
        unit_id=unit_id,
        date_batch=date_batch,
        modality=modality,
        pair_status=pair_status,
        modality_present=modality,
        duration_from_metadata=False,
        metadata_duration_sec=0.0,
        decoded_duration_sec=0.0,
        windows=[failed_window],
        open_failed=True,
    )


def analyze_scene_video(job, pair_status: str) -> VideoAnalysisResult:
    unit_id = build_unit_id(job)
    metadata_readable, metadata_duration_sec, fps_metadata, frame_count_metadata = estimate_duration_from_metadata(job.video_path)
    capture = cv2.VideoCapture(job.video_path)
    if not capture.isOpened():
        return build_failed_video_result(unit_id, job.session_id.split("__", 1)[0], modality_label(job), pair_status)

    fps = fps_metadata if fps_metadata > 0 else 25.0
    observations: List[SceneFrameObservation] = []
    frame_idx = 0
    last_timestamp_sec = 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        timestamp_sec = pos_msec if pos_msec > 0 else (frame_idx / fps if fps > 0 else frame_idx / 25.0)
        last_timestamp_sec = timestamp_sec
        scene_ratios, horizon_confidence, intensity_variance = extract_scene_features(frame, modality_label(job))
        observations.append(
            SceneFrameObservation(
                timestamp_sec=timestamp_sec,
                scene_ratios=scene_ratios,
                horizon_confidence=horizon_confidence,
                frame_intensity_variance=intensity_variance,
            )
        )
        frame_idx += 1

    capture.release()
    decoded_duration_sec = last_timestamp_sec + safe_divide(1.0, fps) if frame_idx > 0 else 0.0
    timestamps_sec = [item.timestamp_sec for item in observations]
    windows = build_time_windows(
        timestamps_sec=timestamps_sec,
        decoded_duration_sec=decoded_duration_sec,
        base_window_seconds=Config.DATASET_WINDOW_SECONDS,
        min_frames_per_window=Config.DATASET_MIN_FRAMES_PER_WINDOW,
        max_window_seconds=Config.DATASET_LOW_FPS_WINDOW_MAX_SECONDS,
    )
    modality = modality_label(job)
    window_rows = [
        build_scene_window_record(
            unit_id=unit_id,
            date_batch=job.session_id.split("__", 1)[0],
            modality=modality,
            pair_status=pair_status,
            duration_from_metadata=metadata_readable,
            observations=observations,
            window=window,
        )
        for window in windows
    ]
    return VideoAnalysisResult(
        unit_id=unit_id,
        date_batch=job.session_id.split("__", 1)[0],
        modality=modality,
        pair_status=pair_status,
        modality_present=modality,
        duration_from_metadata=metadata_readable,
        metadata_duration_sec=metadata_duration_sec,
        decoded_duration_sec=decoded_duration_sec,
        windows=window_rows,
        open_failed=False,
    )


def analyze_target_video(job, pair_status: str, processor, tracker_cls) -> VideoAnalysisResult:
    unit_id = build_unit_id(job)
    metadata_readable, metadata_duration_sec, fps_metadata, frame_count_metadata = estimate_duration_from_metadata(job.video_path)
    capture = cv2.VideoCapture(job.video_path)
    if not capture.isOpened():
        return build_failed_video_result(unit_id, job.session_id.split("__", 1)[0], modality_label(job), pair_status)

    fps = fps_metadata if fps_metadata > 0 else 25.0
    tracker = tracker_cls(
        frame_rate=fps,
        tracker_type=Config.EXTERNAL_FRAMES_TRACKER,
        min_hits=Config.EXTERNAL_FRAMES_TRACKER_MIN_HITS,
    )

    observations: List[TargetFrameObservation] = []
    frame_idx = 0
    last_timestamp_sec = 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        timestamp_sec = pos_msec if pos_msec > 0 else (frame_idx / fps if fps > 0 else frame_idx / 25.0)
        last_timestamp_sec = timestamp_sec
        stats = processor.process_frame(frame, job.file_type)
        if stats is None:
            raw_boxes = []
            detector_not_run = True
        else:
            raw_boxes = stats.get("boxes", [])
            detector_not_run = False
        tracked_boxes = tracker.update(raw_boxes, frame.shape, frame=frame)
        effective_boxes = merge_raw_and_tracked_boxes(raw_boxes, tracked_boxes)
        frame_area = max(frame.shape[0] * frame.shape[1], 1)
        bbox_area_ratios = [
            round_ratio((box["w"] * box["h"]) / frame_area)
            for box in effective_boxes
            if box.get("w", 0) > 0 and box.get("h", 0) > 0
        ]
        observations.append(
            TargetFrameObservation(
                timestamp_sec=timestamp_sec,
                target_count=len(effective_boxes),
                track_count=len({box["track_id"] for box in effective_boxes if "track_id" in box}),
                bbox_area_ratios=bbox_area_ratios,
                detector_not_run=detector_not_run,
            )
        )
        frame_idx += 1

    capture.release()
    decoded_duration_sec = last_timestamp_sec + safe_divide(1.0, fps) if frame_idx > 0 else 0.0
    timestamps_sec = [item.timestamp_sec for item in observations]
    windows = build_time_windows(
        timestamps_sec=timestamps_sec,
        decoded_duration_sec=decoded_duration_sec,
        base_window_seconds=Config.DATASET_WINDOW_SECONDS,
        min_frames_per_window=Config.DATASET_MIN_FRAMES_PER_WINDOW,
        max_window_seconds=Config.DATASET_LOW_FPS_WINDOW_MAX_SECONDS,
    )
    modality = modality_label(job)
    window_rows = [
        {
            "unit_id": unit_id,
            "date_batch": job.session_id.split("__", 1)[0],
            "modality": modality,
            "pair_status": pair_status,
            "window_start": round_ratio(window.start_sec),
            "window_end": round_ratio(window.end_sec),
            "frame_count": len(window.frame_indices),
            "low_fps_window": window.low_fps_window,
            "duration_from_metadata": metadata_readable,
            "decode_error": False,
            **build_target_window_record(observations, window, modality),
        }
        for window in windows
    ]
    return VideoAnalysisResult(
        unit_id=unit_id,
        date_batch=job.session_id.split("__", 1)[0],
        modality=modality,
        pair_status=pair_status,
        modality_present=modality,
        duration_from_metadata=metadata_readable,
        metadata_duration_sec=metadata_duration_sec,
        decoded_duration_sec=decoded_duration_sec,
        windows=window_rows,
        open_failed=False,
    )


def merge_scene_and_target_results(scene_result: VideoAnalysisResult, target_result: VideoAnalysisResult) -> VideoAnalysisResult:
    target_by_span = {
        (
            row["window_start"],
            row["window_end"],
            row["modality"],
        ): row
        for row in target_result.windows
    }
    merged_windows: List[dict] = []
    for scene_row in scene_result.windows:
        key = (scene_row["window_start"], scene_row["window_end"], scene_row["modality"])
        target_row = target_by_span.get(key)
        merged = dict(scene_row)
        if target_row is None:
            merged.update(
                {
                    "target_presence": False,
                    "target_count": 0,
                    "track_count": 0,
                    "bbox_area_ratio_stats": list_to_stats([]),
                    "scale_bin_counts": json_dumps({"small": 0, "medium": 0, "large": 0}),
                    "small_target_count": 0,
                    "detector_not_run": True,
                }
            )
        else:
            merged.update(
                {
                    "target_presence": target_row["target_presence"],
                    "target_count": target_row["target_count"],
                    "track_count": target_row["track_count"],
                    "bbox_area_ratio_stats": target_row["bbox_area_ratio_stats"],
                    "scale_bin_counts": target_row["scale_bin_counts"],
                    "small_target_count": target_row["small_target_count"],
                    "detector_not_run": target_row["detector_not_run"],
                }
            )
        merged["usable"] = evaluate_window_usability(
            int(merged["frame_count"]),
            bool(merged["scene_classification_failed"]),
            bool(merged["detector_not_run"]),
            bool(merged["decode_error"]),
        )
        merged_windows.append(merged)
    return VideoAnalysisResult(
        unit_id=scene_result.unit_id,
        date_batch=scene_result.date_batch,
        modality=scene_result.modality,
        pair_status=scene_result.pair_status,
        modality_present=scene_result.modality_present,
        duration_from_metadata=scene_result.duration_from_metadata,
        metadata_duration_sec=scene_result.metadata_duration_sec,
        decoded_duration_sec=scene_result.decoded_duration_sec,
        windows=merged_windows,
        open_failed=scene_result.open_failed or target_result.open_failed,
    )


def apply_visible_proxy(ir_result: VideoAnalysisResult, visible_result: VideoAnalysisResult) -> int:
    proxy_count = 0
    visible_windows = visible_result.windows
    for window in ir_result.windows:
        if not window.get("scene_classification_low_confidence"):
            continue
        visible_window = find_best_overlap_window(window, visible_windows)
        if visible_window is None:
            continue
        window["primary_scene"] = visible_window["primary_scene"]
        window["scene_source"] = "paired_visible_proxy"
        proxy_count += 1
    return proxy_count


def parse_area_ratio_stats(stats_json: str) -> dict:
    if not stats_json:
        return {"count": 0, "median": 0.0}
    try:
        return json.loads(stats_json)
    except json.JSONDecodeError:
        return {"count": 0, "median": 0.0}


def calibrate_t_scale(visible_windows: Sequence[dict], infrared_windows: Sequence[dict]) -> Tuple[str, float, int]:
    overlap_ratios: List[float] = []
    for ir_window in infrared_windows:
        if not ir_window.get("target_presence"):
            continue
        visible_window = find_best_overlap_window(ir_window, visible_windows)
        if visible_window is None or not visible_window.get("target_presence"):
            continue
        ir_stats = parse_area_ratio_stats(ir_window["bbox_area_ratio_stats"])
        visible_stats = parse_area_ratio_stats(visible_window["bbox_area_ratio_stats"])
        visible_median = float(visible_stats.get("median", 0.0) or 0.0)
        ir_median = float(ir_stats.get("median", 0.0) or 0.0)
        if visible_median <= 0 or ir_median <= 0:
            continue
        overlap_ratios.append(ir_median / visible_median)
    if len(overlap_ratios) >= Config.DATASET_T_CALIB_MIN_OVERLAP_COUNT:
        return "data_driven", round_ratio(float(statistics.median(overlap_ratios))), len(overlap_ratios)
    return "fallback", Config.DATASET_T_SCALE_DEFAULT, len(overlap_ratios)


def refresh_small_target_counts(windows: List[dict], t_scale_factor: float) -> None:
    for window in windows:
        stats = parse_area_ratio_stats(window["bbox_area_ratio_stats"])
        count = int(stats.get("count", 0) or 0)
        if count == 0:
            window["scale_bin_counts"] = json_dumps({"small": 0, "medium": 0, "large": 0})
            window["small_target_count"] = 0
            continue
        ratio_stats = json.loads(window["bbox_area_ratio_stats"])
        median_ratio = float(ratio_stats.get("median", 0.0) or 0.0)
        if median_ratio <= 0:
            window["small_target_count"] = 0
            continue
        scale_bin = get_scale_bin(median_ratio, window["modality"], t_scale_factor=t_scale_factor)
        counts = {"small": 0, "medium": 0, "large": 0}
        counts[scale_bin] = count
        window["scale_bin_counts"] = json_dumps(counts)
        window["small_target_count"] = count if scale_bin == "small" else 0


def build_video_summary(
    unit_id: str,
    date_batch: str,
    pair_status: str,
    windows: List[dict],
    modalities_present: List[str],
    metadata_unreadable: bool,
    t_scale_calibration: str,
    t_scale_factor: float,
) -> dict:
    primary_scene_counts: Dict[str, int] = {}
    covered_scenes = set()
    usable_flags = []
    target_windows = 0
    small_target_windows = 0
    shoreline_mixed_windows = 0
    for window in windows:
        scene = window["primary_scene"]
        primary_scene_counts[scene] = primary_scene_counts.get(scene, 0) + 1
        covered_scenes.add(scene)
        usable_flags.append(bool(window["usable"]))
        target_windows += 1 if window["target_presence"] else 0
        small_target_windows += 1 if int(window["small_target_count"]) > 0 else 0
        shoreline_mixed_windows += 1 if scene == "shoreline_mixed" else 0

    total_windows = len(windows)
    primary_scene_by_time = "shoreline_mixed"
    if primary_scene_counts:
        primary_scene_by_time = max(
            primary_scene_counts,
            key=lambda key: (primary_scene_counts[key], key),
        )
    usable_window_ratio = round_ratio(safe_divide(sum(1 for flag in usable_flags if flag), total_windows))
    max_consecutive_unusable_windows = max_consecutive_false(usable_flags)
    return {
        "unit_id": unit_id,
        "modalities_present": "|".join(sorted(modalities_present)),
        "pair_status": pair_status,
        "primary_scene_by_time": primary_scene_by_time,
        "covered_scenes": "|".join(sorted(covered_scenes)),
        "shoreline_mixed_ratio": round_ratio(safe_divide(shoreline_mixed_windows, total_windows)),
        "target_window_ratio": round_ratio(safe_divide(target_windows, total_windows)),
        "small_target_window_ratio": round_ratio(safe_divide(small_target_windows, total_windows)),
        "usable_window_ratio": usable_window_ratio,
        "max_consecutive_unusable_windows": max_consecutive_unusable_windows,
        "usable": evaluate_video_usability(usable_window_ratio, max_consecutive_unusable_windows),
        "date_batch": date_batch,
        "metadata_unreadable": metadata_unreadable,
        "t_scale_calibration": t_scale_calibration,
        "t_scale_factor": round_ratio(t_scale_factor),
    }


def build_precheck_rows(input_root: str, groups: Dict[str, Dict[str, object]]) -> Tuple[List[dict], dict]:
    date_stats: Dict[str, dict] = {}
    metadata_unreadable_count = 0
    total_video_count = 0
    for _, modalities in groups.items():
        pair_status = get_pair_status(modalities)
        date_batch = next(iter(modalities.values())).session_id.split("__", 1)[0]
        stats = date_stats.setdefault(
            date_batch,
            {
                "date_batch": date_batch,
                "video_file_count": 0,
                "visible_file_count": 0,
                "infrared_file_count": 0,
                "unit_count": 0,
                "paired_count": 0,
                "v_only_count": 0,
                "t_only_count": 0,
                "metadata_readable_count": 0,
                "metadata_unreadable_count": 0,
                "durations": [],
                "metadata_unreadable_files": [],
            },
        )
        stats["unit_count"] += 1
        stats[f"{pair_status.lower()}_count"] += 1
        for modality_key, job in modalities.items():
            del modality_key
            total_video_count += 1
            stats["video_file_count"] += 1
            if job.modality == "rgb":
                stats["visible_file_count"] += 1
            else:
                stats["infrared_file_count"] += 1
            metadata_readable, duration_sec, _, _ = estimate_duration_from_metadata(job.video_path)
            if metadata_readable:
                stats["metadata_readable_count"] += 1
                stats["durations"].append(duration_sec)
            else:
                stats["metadata_unreadable_count"] += 1
                stats["metadata_unreadable_files"].append(job.video_path)
                metadata_unreadable_count += 1

    rows: List[dict] = []
    for date_batch in sorted(date_stats):
        stats = date_stats[date_batch]
        durations = stats.pop("durations")
        unreadable_files = stats.pop("metadata_unreadable_files")
        rows.append(
            {
                "date_batch": date_batch,
                "video_file_count": stats["video_file_count"],
                "visible_file_count": stats["visible_file_count"],
                "infrared_file_count": stats["infrared_file_count"],
                "unit_count": stats["unit_count"],
                "paired_count": stats["paired_count"],
                "v_only_count": stats["v_only_count"],
                "t_only_count": stats["t_only_count"],
                "paired_rate": round_ratio(safe_divide(stats["paired_count"], stats["unit_count"])),
                "metadata_readable_count": stats["metadata_readable_count"],
                "metadata_unreadable_count": stats["metadata_unreadable_count"],
                "metadata_readable_rate": round_ratio(
                    safe_divide(stats["metadata_readable_count"], stats["video_file_count"])
                ),
                "duration_min_sec": round_ratio(min(durations)) if durations else 0.0,
                "duration_median_sec": round_ratio(float(statistics.median(durations))) if durations else 0.0,
                "duration_max_sec": round_ratio(max(durations)) if durations else 0.0,
                "metadata_unreadable_files": json_dumps(unreadable_files),
            }
        )
    summary = {
        "total_video_count": total_video_count,
        "total_unit_count": len(groups),
        "metadata_unreadable_count": metadata_unreadable_count,
    }
    return rows, summary


def compute_date_concentration(video_rows: Sequence[dict]) -> Tuple[float, float, str]:
    by_date: Dict[str, int] = {}
    for row in video_rows:
        by_date[row["date_batch"]] = by_date.get(row["date_batch"], 0) + 1
    if not by_date:
        return 0.0, 0.0, ""
    total = sum(by_date.values())
    top1_date_batch = max(by_date, key=by_date.get)
    dominant_share = safe_divide(by_date[top1_date_batch], total)
    entropy = 0.0
    for count in by_date.values():
        p = safe_divide(count, total)
        if p > 0:
            entropy -= p * math.log(p, 2)
    return round_ratio(dominant_share), round_ratio(entropy), top1_date_batch


def enrich_report_rows(video_rows: List[dict], window_rows: List[dict]) -> List[dict]:
    dominant_share, date_entropy, top1_date_batch = compute_date_concentration(video_rows)
    windows_by_unit: Dict[str, List[dict]] = {}
    for row in window_rows:
        windows_by_unit.setdefault(row["unit_id"], []).append(row)
    report_rows = []
    for row in video_rows:
        unit_windows = windows_by_unit.get(row["unit_id"], [])
        scene_missing = len(set(window["primary_scene"] for window in unit_windows if window["primary_scene"])) < 1
        scene_small_sparse = all(int(window["small_target_count"]) == 0 for window in unit_windows) if unit_windows else True
        paired_modality_gap = row["pair_status"] != "paired"
        metadata_unreadable = parse_bool(row["metadata_unreadable"])
        low_usable_ratio = float(row["usable_window_ratio"]) < Config.DATASET_USABLE_WINDOW_RATIO_THRESHOLD
        paired_visible_proxy_count = sum(
            1 for window in unit_windows if window.get("scene_source") == "paired_visible_proxy"
        )
        report_row = dict(row)
        report_row.update(
            {
                "scene_missing": scene_missing,
                "scene_x_small_target_sparse": scene_small_sparse,
                "paired_modality_gap": paired_modality_gap,
                "metadata_unreadable": metadata_unreadable,
                "low_usable_ratio": low_usable_ratio,
                "paired_visible_proxy_count": paired_visible_proxy_count,
                "dominant_date_share": dominant_share,
                "date_entropy": date_entropy,
                "top1_date_batch": top1_date_batch,
            }
        )
        report_rows.append(report_row)
    return report_rows


def build_gap_rows(report_rows: Sequence[dict]) -> List[dict]:
    return [
        {
            "unit_id": row["unit_id"],
            "scene_missing": row["scene_missing"],
            "scene_x_small_target_sparse": row["scene_x_small_target_sparse"],
            "paired_modality_gap": row["paired_modality_gap"],
            "metadata_unreadable": row["metadata_unreadable"],
            "low_usable_ratio": row["low_usable_ratio"],
            "paired_visible_proxy_count": row["paired_visible_proxy_count"],
            "dominant_date_share": row["dominant_date_share"],
            "date_entropy": row["date_entropy"],
            "top1_date_batch": row["top1_date_batch"],
        }
        for row in report_rows
    ]


def init_worker() -> None:
    cv2.setNumThreads(1)


def process_group_worker(group_key: str, modalities: Dict[str, object]) -> Tuple[List[dict], dict, int]:
    del group_key
    pair_status = get_pair_status(modalities)
    results: Dict[str, VideoAnalysisResult] = {}
    for modality in ["rgb", "ir"]:
        job = modalities.get(modality)
        if job is None:
            continue
        results[modality] = analyze_scene_video(job, pair_status)
    return results


def process_group(group_key: str, modalities: Dict[str, object], processor, tracker_cls, precomputed_scene_results: Optional[Dict[str, VideoAnalysisResult]] = None) -> Tuple[List[dict], dict, int]:
    del group_key
    pair_status = get_pair_status(modalities)
    scene_results = precomputed_scene_results or {}
    results: Dict[str, VideoAnalysisResult] = {}
    for modality in ["rgb", "ir"]:
        job = modalities.get(modality)
        if job is None:
            continue
        scene_result = scene_results.get(modality)
        if scene_result is None:
            scene_result = analyze_scene_video(job, pair_status)
        target_result = analyze_target_video(job, pair_status, processor, tracker_cls)
        results[modality] = merge_scene_and_target_results(scene_result, target_result)

    t_scale_calibration = "fallback"
    t_scale_factor = Config.DATASET_T_SCALE_DEFAULT
    if "rgb" in results and "ir" in results:
        proxy_count = apply_visible_proxy(results["ir"], results["rgb"])
        t_scale_calibration, t_scale_factor, _ = calibrate_t_scale(
            results["rgb"].windows,
            results["ir"].windows,
        )
        refresh_small_target_counts(results["ir"].windows, t_scale_factor)
    else:
        proxy_count = 0

    window_rows: List[dict] = []
    modalities_present: List[str] = []
    metadata_unreadable = False
    unit_id = ""
    date_batch = ""
    for modality in ["rgb", "ir"]:
        result = results.get(modality)
        if result is None:
            continue
        unit_id = result.unit_id
        date_batch = result.date_batch
        modalities_present.append(result.modality)
        metadata_unreadable = metadata_unreadable or (not result.duration_from_metadata)
        window_rows.extend(result.windows)

    video_row = build_video_summary(
        unit_id=unit_id,
        date_batch=date_batch,
        pair_status=pair_status,
        windows=window_rows,
        modalities_present=modalities_present,
        metadata_unreadable=metadata_unreadable,
        t_scale_calibration=t_scale_calibration,
        t_scale_factor=t_scale_factor,
    )
    video_row["paired_visible_proxy_count"] = proxy_count
    return window_rows, video_row, proxy_count


def main() -> None:
    args = parse_args()
    from target_module.image_detect_module.image_processor import ImageProcessor
    from target_module.image_detect_module.utils.tracker import MultiObjectTracker

    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)
    run_root = resolve_run_root(output_root, args.resume)
    run_id = os.path.basename(run_root)
    worker_count = normalize_worker_count(args.workers)

    precheck_path = os.path.join(run_root, "precheck_by_date.csv")
    window_results_path = os.path.join(run_root, "window_results.csv")
    video_results_path = os.path.join(run_root, "video_results.csv")
    report_all_units_path = os.path.join(run_root, "report_all_units.csv")
    report_paired_only_path = os.path.join(run_root, "report_paired_only.csv")
    gap_report_path = os.path.join(run_root, "gap_report.csv")
    run_summary_path = os.path.join(run_root, "run_summary.json")

    groups = discover_video_groups(input_root)
    if not groups:
        print(f"[WARN] No target videos found under: {input_root}")
        return

    if args.resume and os.path.exists(precheck_path):
        precheck_rows = load_csv_rows(precheck_path)
        precheck_summary = {
            "total_video_count": sum(int(float(row["video_file_count"])) for row in precheck_rows),
            "total_unit_count": sum(int(float(row["unit_count"])) for row in precheck_rows),
            "metadata_unreadable_count": sum(int(float(row["metadata_unreadable_count"])) for row in precheck_rows),
        }
    else:
        precheck_rows, precheck_summary = build_precheck_rows(input_root, groups)
        write_csv(precheck_path, precheck_rows, PRECHECK_COLUMNS)

    existing_window_rows = load_csv_rows(window_results_path) if args.resume else []
    existing_video_rows = load_csv_rows(video_results_path) if args.resume else []
    completed_units = {row["unit_id"] for row in existing_video_rows}
    all_window_rows = list(existing_window_rows)
    all_video_rows = list(existing_video_rows)
    paired_visible_proxy_total = sum(int(row.get("paired_visible_proxy_count", 0) or 0) for row in all_video_rows)
    processor = ImageProcessor()

    if not args.resume:
        write_csv(window_results_path, [], WINDOW_RESULT_COLUMNS)
        write_csv(video_results_path, [], VIDEO_RESULT_COLUMNS + ["paired_visible_proxy_count"])

    pending_groups = []
    for group_key, modalities in groups.items():
        sample_job = next(iter(modalities.values()))
        unit_id = build_unit_id(sample_job)
        if args.resume and unit_id in completed_units:
            print(f"[RESUME] Skip completed unit: {unit_id}")
            continue
        pending_groups.append((group_key, modalities))

    print(f"[INFO] Pending units: {len(pending_groups)} | scene_workers={worker_count} | detect_workers=1")

    precomputed_scene_by_group: Dict[str, Dict[str, VideoAnalysisResult]] = {}
    if not pending_groups:
        pass
    elif worker_count == 1:
        for group_key, modalities in pending_groups:
            precomputed_scene_by_group[group_key] = process_group_worker(group_key, modalities)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_worker,
        ) as executor:
            future_to_group = {
                executor.submit(process_group_worker, group_key, modalities): group_key
                for group_key, modalities in pending_groups
            }
            scene_completed = 0
            for future in concurrent.futures.as_completed(future_to_group):
                group_key = future_to_group[future]
                precomputed_scene_by_group[group_key] = future.result()
                scene_completed += 1
                print(f"[SCENE] {scene_completed}/{len(pending_groups)} {group_key}")

    detection_completed = 0
    for group_key, modalities in pending_groups:
        print(f"[GROUP] {group_key}")
        window_rows, video_row, proxy_count = process_group(
            group_key,
            modalities,
            processor,
            MultiObjectTracker,
            precomputed_scene_results=precomputed_scene_by_group.get(group_key),
        )
        detection_completed += 1
        print(f"[DETECT] {detection_completed}/{len(pending_groups)} {group_key}")
        all_window_rows.extend(window_rows)
        all_video_rows.append(video_row)
        paired_visible_proxy_total += proxy_count
        append_csv_rows(window_results_path, window_rows, WINDOW_RESULT_COLUMNS)
        append_csv_rows(
            video_results_path,
            [video_row],
            VIDEO_RESULT_COLUMNS + ["paired_visible_proxy_count"],
        )

    report_rows = enrich_report_rows(all_video_rows, all_window_rows)
    paired_rows = [row for row in report_rows if row["pair_status"] == "paired"]
    gap_rows = build_gap_rows(report_rows)
    write_csv(report_all_units_path, report_rows, REPORT_COLUMNS)
    write_csv(report_paired_only_path, paired_rows, REPORT_COLUMNS)
    write_csv(
        gap_report_path,
        gap_rows,
        [
            "unit_id",
            "scene_missing",
            "scene_x_small_target_sparse",
            "paired_modality_gap",
            "metadata_unreadable",
            "low_usable_ratio",
            "paired_visible_proxy_count",
            "dominant_date_share",
            "date_entropy",
            "top1_date_batch",
        ],
    )

    summary = dict(precheck_summary)
    summary.update(
        {
            "run_id": run_id,
            "input_root": input_root,
            "output_root": run_root,
            "total_window_count": len(all_window_rows),
            "paired_unit_count": sum(1 for row in all_video_rows if row["pair_status"] == "paired"),
            "v_only_count": sum(1 for row in all_video_rows if row["pair_status"] == "V_only"),
            "t_only_count": sum(1 for row in all_video_rows if row["pair_status"] == "T_only"),
            "paired_visible_proxy_count": paired_visible_proxy_total,
        }
    )
    write_json(run_summary_path, summary)

    print(f"[DONE] Precheck: {precheck_path}")
    print(f"[DONE] Windows: {window_results_path}")
    print(f"[DONE] Videos: {video_results_path}")
    print(f"[DONE] Summary: {run_summary_path}")


if __name__ == "__main__":
    main()
