import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv_utils import imread_unicode, imwrite_unicode
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.constants import BASELINE_TRACKER_CHOICES
from target_module.image_detect_module.utils.file_utils import get_file_type


MANIFEST_FIELDS = [
    "frame_id",
    "frame_path",
    "session_id",
    "modality",
    "source_video",
    "timestamp_sec",
    "label_path",
    "split",
    "is_enriched",
    "is_augmented",
    "source_type",
    "qc_status",
    "annotation_status",
    "keep_reason",
    "class_usv",
    "class_fishship",
    "class_uav",
]

KEEP_REASON_ORDER = [
    "first_detection",
    "new_track",
    "motion_change",
    "pose_angle_change",
    "scale_change",
    "tracker_fill",
    "detection_lost",
    "detection_recovered",
    "empty_scene_distinct",
    "paired_alignment",
]

PHASH_SIZE = 32
PHASH_LOW_FREQ = 8
PHASH_EXEMPT_REASONS = {
    "first_detection",
    "new_track",
    "detection_lost",
    "detection_recovered",
    "paired_alignment",
}
CLASS_TO_ID = {name: idx for idx, name in enumerate(Config.CLASSES)}
MODALITY_TO_FOLDER = {"visible": "rgb", "infrared": "ir"}


@dataclass
class VideoJob:
    video_path: str
    session_id: str
    video_id: str
    modality: str
    file_type: str
    group_key: str


@dataclass
class TrackSnapshot:
    center: Tuple[float, float]
    size: Tuple[float, float]
    area: float
    pose_angle_deg: Optional[float]


@dataclass
class ExportRecord:
    frame_id: str
    frame_idx: int
    timestamp_sec: float
    frame: Optional[bytes]
    boxes: List[dict]
    session_id: str
    modality: str
    source_video: str
    frame_path: str
    label_path: str
    keep_reasons: set[str] = field(default_factory=set)
    is_written: bool = False

    def add_reasons(self, reasons: Iterable[str]) -> None:
        self.keep_reasons.update(reasons)


@dataclass
class VideoExtractionResult:
    job: VideoJob
    fps: float
    frame_count: int
    records: Dict[str, ExportRecord]
    anchor_timestamps: List[Tuple[float, str]]


@dataclass
class RepresentativeMetrics:
    record: ExportRecord
    class_signature: Tuple[Tuple[str, int], ...]
    area_ratio: float
    center: Tuple[float, float]
    scale_basis: float
    sharpness: float
    frame_hash: np.ndarray
    roi_hash: Optional[np.ndarray]


def encode_frame_payload(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("Failed to encode frame payload.")
    return buf.tobytes()


def decode_frame_payload(payload: Optional[bytes]) -> Optional[np.ndarray]:
    if payload is None:
        return None
    arr = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracking-aware RGB/IR frame extraction")
    parser.add_argument("--input-root", default=Config.EXTERNAL_FRAMES_INPUT_ROOT)
    parser.add_argument("--output-root", default=Config.EXTERNAL_FRAMES_OUTPUT_ROOT)
    parser.add_argument(
        "--tracker",
        default=Config.EXTERNAL_FRAMES_TRACKER,
        choices=BASELINE_TRACKER_CHOICES,
    )
    parser.add_argument("--motion-threshold", type=float, default=Config.EXTERNAL_FRAMES_MOTION_THRESHOLD)
    parser.add_argument(
        "--pose-angle-threshold-deg",
        type=float,
        default=Config.EXTERNAL_FRAMES_POSE_ANGLE_THRESHOLD_DEG,
    )
    parser.add_argument("--area-threshold", type=float, default=Config.EXTERNAL_FRAMES_AREA_THRESHOLD)
    parser.add_argument("--pair-tolerance-sec", type=float, default=Config.EXTERNAL_FRAMES_PAIR_TOLERANCE_SEC)
    parser.add_argument(
        "--phash-hamming-threshold",
        type=int,
        default=Config.EXTERNAL_FRAMES_PHASH_HAMMING_THRESHOLD,
    )
    parser.add_argument(
        "--include-empty-frames",
        action="store_true",
        default=Config.EXTERNAL_FRAMES_INCLUDE_EMPTY_FRAMES,
    )
    parser.add_argument(
        "--allow-tracker-fill-labels",
        action="store_true",
        default=Config.EXTERNAL_FRAMES_ALLOW_TRACKER_FILL_LABELS,
    )
    parser.add_argument(
        "--disable-representative-pruning",
        action="store_false",
        dest="enable_representative_pruning",
        default=Config.EXTERNAL_FRAMES_ENABLE_REPRESENTATIVE_PRUNING,
    )
    parser.add_argument(
        "--disable-paired-rgb-reference",
        action="store_false",
        dest="paired_rgb_reference",
        default=Config.EXTERNAL_FRAMES_PAIRED_RGB_REFERENCE,
    )
    parser.add_argument(
        "--disable-paired-reference-include-empty-target",
        action="store_false",
        dest="paired_reference_include_empty_target",
        default=Config.EXTERNAL_FRAMES_PAIRED_REFERENCE_INCLUDE_EMPTY_TARGET,
    )
    parser.add_argument(
        "--representative-global-phash-threshold",
        type=int,
        default=Config.EXTERNAL_FRAMES_REPRESENTATIVE_GLOBAL_HASH_THRESHOLD,
    )
    parser.add_argument(
        "--representative-target-phash-threshold",
        type=int,
        default=Config.EXTERNAL_FRAMES_REPRESENTATIVE_TARGET_HASH_THRESHOLD,
    )
    parser.add_argument(
        "--representative-motion-threshold",
        type=float,
        default=Config.EXTERNAL_FRAMES_REPRESENTATIVE_MOTION_THRESHOLD,
    )
    parser.add_argument(
        "--representative-area-threshold",
        type=float,
        default=Config.EXTERNAL_FRAMES_REPRESENTATIVE_AREA_THRESHOLD,
    )
    parser.add_argument(
        "--representative-max-cluster-span-sec",
        type=float,
        default=Config.EXTERNAL_FRAMES_REPRESENTATIVE_MAX_CLUSTER_SPAN_SEC,
    )
    parser.add_argument(
        "--representative-min-time-gap-sec",
        type=float,
        default=Config.EXTERNAL_FRAMES_REPRESENTATIVE_MIN_TIME_GAP_SEC,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    return parser.parse_args()


def iter_target_videos(input_root: str) -> Iterable[str]:
    for root, _, files in os.walk(input_root):
        for filename in sorted(files):
            basename, ext = os.path.splitext(filename)
            if ext.lower() not in {".mp4", ".avi", ".mov"}:
                continue
            if not (basename.endswith("_V") or basename.endswith("_T")):
                continue
            yield os.path.join(root, filename)


def build_video_job(input_root: str, video_path: str) -> VideoJob:
    rel_path = os.path.relpath(video_path, input_root)
    rel_parts = rel_path.split(os.sep)
    parent_dir = os.path.basename(os.path.dirname(video_path))
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
    if len(rel_parts) > 1:
        date_folder = rel_parts[0]
    else:
        date_folder = grandparent_dir if grandparent_dir else parent_dir
    session_id = f"{date_folder}__{parent_dir}"
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_id = strip_modality_suffix(video_stem)
    file_type = get_file_type(video_stem)
    modality = MODALITY_TO_FOLDER.get(file_type, "rgb")
    group_key = f"{session_id}__{video_id}"
    return VideoJob(
        video_path=os.path.abspath(video_path),
        session_id=session_id,
        video_id=video_id,
        modality=modality,
        file_type=file_type if file_type != "unknown" else "visible",
        group_key=group_key,
    )


def discover_video_groups(input_root: str) -> Dict[str, Dict[str, VideoJob]]:
    groups: Dict[str, Dict[str, VideoJob]] = {}
    for video_path in iter_target_videos(input_root):
        job = build_video_job(input_root, video_path)
        groups.setdefault(job.group_key, {})[job.modality] = job
    return dict(sorted(groups.items()))


def strip_modality_suffix(stem: str) -> str:
    if stem.endswith("_V") or stem.endswith("_T"):
        return stem[:-2]
    return stem


def timestamp_to_token(timestamp_sec: float) -> str:
    return f"{int(round(timestamp_sec * 1000.0)):06d}"


def build_frame_id(session_id: str, video_id: str, timestamp_sec: float, modality: str) -> str:
    token = timestamp_to_token(timestamp_sec)
    frame_id = f"{session_id}__{video_id}__{token}"
    if modality == "ir":
        frame_id += "_ir"
    return frame_id


def build_output_paths(output_root: str, job: VideoJob, frame_id: str) -> Tuple[str, str]:
    image_dir = os.path.join(output_root, "images", job.modality, job.session_id)
    label_dir = os.path.join(output_root, "labels", job.modality, job.session_id)
    image_path = os.path.abspath(os.path.join(image_dir, f"{frame_id}.jpg"))
    label_path = os.path.abspath(os.path.join(label_dir, f"{frame_id}.txt"))
    return image_path, label_path


def build_classes_path(output_root: str, job: VideoJob) -> str:
    return os.path.abspath(
        os.path.join(output_root, "labels", job.modality, job.session_id, "classes.txt")
    )


def normalize_keep_reasons(reasons: Iterable[str]) -> str:
    reason_set = set(reasons)
    return "|".join([reason for reason in KEEP_REASON_ORDER if reason in reason_set])


def compute_frame_phash(frame: np.ndarray) -> np.ndarray:
    gray = to_gray(frame)
    resized = cv2.resize(gray, (PHASH_SIZE, PHASH_SIZE), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32)
    dct = cv2.dct(resized)
    low_freq = dct[:PHASH_LOW_FREQ, :PHASH_LOW_FREQ]
    flat = low_freq.flatten()
    median = np.median(flat[1:]) if flat.size > 1 else flat[0]
    return (low_freq > median).astype(np.uint8)


def compute_boxes_roi_phash(
    frame: np.ndarray,
    boxes: Sequence[dict],
    padding_ratio: float = 0.1,
) -> Optional[np.ndarray]:
    if not boxes:
        return None

    x1 = min(int(box["x"]) for box in boxes)
    y1 = min(int(box["y"]) for box in boxes)
    x2 = max(int(box["x"] + box["w"]) for box in boxes)
    y2 = max(int(box["y"] + box["h"]) for box in boxes)

    pad_x = int(max(2, (x2 - x1) * padding_ratio))
    pad_y = int(max(2, (y2 - y1) * padding_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame.shape[1], x2 + pad_x)
    y2 = min(frame.shape[0], y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return compute_frame_phash(roi)


def phash_hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def angular_difference_deg(angle_a: Optional[float], angle_b: Optional[float]) -> Optional[float]:
    if angle_a is None or angle_b is None:
        return None
    diff = abs(angle_a - angle_b) % 180.0
    return min(diff, 180.0 - diff)


def estimate_pose_angle(frame: np.ndarray, box: dict) -> Optional[float]:
    x1 = max(0, int(box["x"]))
    y1 = max(0, int(box["y"]))
    x2 = min(frame.shape[1], int(box["x"] + box["w"]))
    y2 = min(frame.shape[0], int(box["y"] + box["h"]))
    if x2 - x1 < Config.EXTERNAL_FRAMES_POSE_MIN_PIXELS:
        return None
    if y2 - y1 < Config.EXTERNAL_FRAMES_POSE_MIN_PIXELS:
        return None

    roi = frame[y1:y2, x1:x2]
    gray = to_gray(roi)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    roi_area = float(gray.shape[0] * gray.shape[1])
    if roi_area <= 0:
        return None

    points = extract_pose_points(gray, roi_area)
    if points is None or len(points) < Config.EXTERNAL_FRAMES_POSE_MIN_POINTS:
        return None

    rect = cv2.minAreaRect(points.reshape(-1, 1, 2))
    (_, _), (rect_w, rect_h), rect_angle = rect
    major = max(float(rect_w), float(rect_h))
    minor = min(float(rect_w), float(rect_h))
    if minor > 1e-6 and major / minor >= Config.EXTERNAL_FRAMES_POSE_MIN_ASPECT_RATIO:
        if rect_w < rect_h:
            rect_angle += 90.0
        return float(rect_angle % 180.0)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
    del mean
    major = float(eigenvalues[0][0]) if len(eigenvalues) > 0 else 0.0
    minor = float(eigenvalues[1][0]) if len(eigenvalues) > 1 else 0.0
    if minor <= 1e-6:
        return None
    if major / minor < Config.EXTERNAL_FRAMES_POSE_MIN_ASPECT_RATIO:
        return None

    direction = eigenvectors[0]
    angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
    return angle_deg


def extract_pose_points(gray: np.ndarray, roi_area: float) -> Optional[np.ndarray]:
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_candidates = [otsu_mask, cv2.bitwise_not(otsu_mask)]

    best_points = None
    best_area = 0.0
    for mask in mask_candidates:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            area_ratio = contour_area / roi_area
            if area_ratio < Config.EXTERNAL_FRAMES_POSE_MIN_AREA_RATIO:
                continue
            if area_ratio > 0.9:
                continue
            pts = contour.reshape(-1, 2).astype(np.float32)
            if contour_area > best_area:
                best_points = pts
                best_area = contour_area

    if best_points is not None:
        return best_points

    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < Config.EXTERNAL_FRAMES_POSE_MIN_POINTS:
        return None
    if float(len(xs)) / roi_area < Config.EXTERNAL_FRAMES_POSE_MIN_AREA_RATIO:
        return None
    return np.column_stack((xs, ys)).astype(np.float32)


def snapshot_from_box(frame: np.ndarray, box: dict) -> TrackSnapshot:
    center = (box["x"] + box["w"] / 2.0, box["y"] + box["h"] / 2.0)
    size = (float(box["w"]), float(box["h"]))
    area = float(box["w"] * box["h"])
    pose_angle_deg = estimate_pose_angle(frame, box)
    return TrackSnapshot(center=center, size=size, area=area, pose_angle_deg=pose_angle_deg)


def bbox_iou_xywh(box_a: dict, box_b: dict) -> float:
    ax1, ay1 = float(box_a["x"]), float(box_a["y"])
    ax2, ay2 = ax1 + float(box_a["w"]), ay1 + float(box_a["h"])
    bx1, by1 = float(box_b["x"]), float(box_b["y"])
    bx2, by2 = bx1 + float(box_b["w"]), by1 + float(box_b["h"])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def merge_raw_and_tracked_boxes(raw_boxes: Sequence[dict], tracked_boxes: Sequence[dict]) -> List[dict]:
    if not raw_boxes:
        return [dict(box) for box in tracked_boxes]
    if not tracked_boxes:
        return [dict(box) for box in raw_boxes]

    merged = []
    used_track_indices: set[int] = set()
    for raw in raw_boxes:
        best_idx = -1
        best_iou = 0.0
        for idx, tracked in enumerate(tracked_boxes):
            if idx in used_track_indices:
                continue
            score = bbox_iou_xywh(raw, tracked)
            if score > best_iou:
                best_iou = score
                best_idx = idx

        merged_box = dict(raw)
        if best_idx >= 0 and best_iou >= 0.2:
            tracked = tracked_boxes[best_idx]
            merged_box["track_id"] = tracked["track_id"]
            merged_box["confidence"] = raw.get("confidence", tracked.get("confidence", 0.0))
            merged_box["class_confidence"] = raw.get(
                "class_confidence",
                tracked.get("class_confidence", merged_box.get("confidence", 0.0)),
            )
            used_track_indices.add(best_idx)
        merged.append(merged_box)

    return merged


def extraction_confidence_override(file_type: str) -> float:
    if file_type == "infrared":
        return Config.EXTERNAL_FRAMES_INFRARED_CONF_THRESH
    return Config.EXTERNAL_FRAMES_VISIBLE_CONF_THRESH


def evaluate_detection_keep_reasons(
    tracked_boxes: Sequence[dict],
    frame: np.ndarray,
    previous_states: Dict[int, TrackSnapshot],
    motion_threshold: float,
    pose_angle_threshold_deg: float,
    area_threshold: float,
    seen_detection_frames: bool,
) -> Tuple[set[str], Dict[int, TrackSnapshot]]:
    reasons: set[str] = set()
    current_states: Dict[int, TrackSnapshot] = {}

    for box in tracked_boxes:
        tid = int(box["track_id"])
        current_states[tid] = snapshot_from_box(frame, box)
        previous = previous_states.get(tid)
        if previous is None:
            if seen_detection_frames:
                reasons.add("new_track")
            else:
                reasons.add("first_detection")
            continue

        current = current_states[tid]
        scale_basis = max(previous.size[0], previous.size[1], 1.0)
        center_disp = np.hypot(
            current.center[0] - previous.center[0],
            current.center[1] - previous.center[1],
        )
        if center_disp >= motion_threshold * scale_basis:
            reasons.add("motion_change")

        if previous.area > 0:
            area_change = abs(current.area - previous.area) / previous.area
            if area_change >= area_threshold:
                reasons.add("scale_change")

        pose_diff = angular_difference_deg(current.pose_angle_deg, previous.pose_angle_deg)
        if pose_diff is not None and pose_diff >= pose_angle_threshold_deg:
            reasons.add("pose_angle_change")

    return reasons, current_states


def has_strong_detection_novelty(
    current_states: Dict[int, TrackSnapshot],
    previous_states: Dict[int, TrackSnapshot],
    reasons: Iterable[str],
    motion_threshold: float,
    area_threshold: float,
    pose_threshold_deg: float,
) -> bool:
    reason_set = set(reasons)
    if "first_detection" in reason_set or "new_track" in reason_set:
        return True

    for tid, current in current_states.items():
        previous = previous_states.get(tid)
        if previous is None:
            return True

        scale_basis = max(previous.size[0], previous.size[1], 1.0)
        center_disp = np.hypot(
            current.center[0] - previous.center[0],
            current.center[1] - previous.center[1],
        )
        normalized_motion = center_disp / scale_basis
        if normalized_motion >= motion_threshold:
            return True

        if previous.area > 0:
            area_change = abs(current.area - previous.area) / previous.area
            if area_change >= area_threshold:
                return True

        pose_diff = angular_difference_deg(current.pose_angle_deg, previous.pose_angle_deg)
        if pose_diff is not None and pose_diff >= pose_threshold_deg:
            return True

    return False


def should_keep_by_phash(
    frame_hash: np.ndarray,
    last_kept_hash: Optional[np.ndarray],
    reasons: Iterable[str],
    threshold: int,
) -> bool:
    reason_set = set(reasons)
    if not last_kept_hash is None:
        if reason_set & PHASH_EXEMPT_REASONS:
            return True
        if phash_hamming_distance(frame_hash, last_kept_hash) <= threshold:
            return False
    return True


def should_keep_detection_frame(
    frame_hash: np.ndarray,
    last_frame_hash: Optional[np.ndarray],
    roi_hash: Optional[np.ndarray],
    last_roi_hash: Optional[np.ndarray],
    reasons: Iterable[str],
    global_threshold: int,
    target_threshold: int,
) -> bool:
    reason_set = set(reasons)
    if reason_set & PHASH_EXEMPT_REASONS:
        return True
    if last_frame_hash is None:
        return True

    global_distance = phash_hamming_distance(frame_hash, last_frame_hash)
    if global_distance > global_threshold:
        return True

    if roi_hash is None or last_roi_hash is None:
        return False

    target_distance = phash_hamming_distance(roi_hash, last_roi_hash)
    return target_distance > target_threshold


def get_record_hashes(
    record: ExportRecord,
    cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    cached = cache.get(record.frame_id)
    if cached is not None:
        return cached

    frame = decode_frame_payload(record.frame)
    if frame is None:
        frame = imread_unicode(record.frame_path)
    if frame is None:
        raise RuntimeError(f"Unable to load frame for hash comparison: {record.frame_path}")

    frame_hash = compute_frame_phash(frame)
    roi_hash = compute_boxes_roi_phash(frame, record.boxes)
    cache[record.frame_id] = (frame_hash, roi_hash)
    return frame_hash, roi_hash


def build_class_flags(boxes: Sequence[dict]) -> Tuple[int, int, int]:
    flags = [0, 0, 0]
    for box in boxes:
        cls_id = CLASS_TO_ID.get(box.get("class", ""), None)
        if cls_id is not None and 0 <= cls_id < len(flags):
            flags[cls_id] = 1
    return tuple(flags)


def sanitize_export_boxes(boxes: Sequence[dict], allow_tracker_fill_labels: bool) -> List[dict]:
    sanitized = []
    for box in boxes:
        if not allow_tracker_fill_labels and box.get("is_tracker_prediction"):
            continue
        sanitized.append(dict(box))
    return sanitized


def build_class_signature(boxes: Sequence[dict]) -> Tuple[Tuple[str, int], ...]:
    counts: Dict[str, int] = {}
    for box in boxes:
        cls_name = box.get("class", "")
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return tuple(sorted(counts.items()))


def compute_record_sharpness(frame: np.ndarray, boxes: Sequence[dict]) -> float:
    if boxes:
        x1 = max(0, min(int(box["x"]) for box in boxes))
        y1 = max(0, min(int(box["y"]) for box in boxes))
        x2 = min(frame.shape[1], max(int(box["x"] + box["w"]) for box in boxes))
        y2 = min(frame.shape[0], max(int(box["y"] + box["h"]) for box in boxes))
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            gray = to_gray(roi)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    gray = to_gray(frame)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_representative_metrics(record: ExportRecord) -> RepresentativeMetrics:
    if record.frame is None:
        raise RuntimeError(f"Representative pruning requires in-memory frames: {record.frame_id}")

    frame = decode_frame_payload(record.frame)
    if frame is None:
        raise RuntimeError(f"Unable to decode representative frame: {record.frame_id}")
    h, w = frame.shape[:2]
    area_sum = sum(float(box["w"]) * float(box["h"]) for box in record.boxes)
    area_ratio = area_sum / max(float(h * w), 1.0)

    if record.boxes:
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        scale_basis = 1.0
        for box in record.boxes:
            area = max(1.0, float(box["w"]) * float(box["h"]))
            cx = float(box["x"]) + float(box["w"]) / 2.0
            cy = float(box["y"]) + float(box["h"]) / 2.0
            weighted_x += cx * area
            weighted_y += cy * area
            total_weight += area
            scale_basis = max(scale_basis, float(box["w"]), float(box["h"]))
        center = (weighted_x / total_weight, weighted_y / total_weight)
    else:
        center = (w / 2.0, h / 2.0)
        scale_basis = max(1.0, float(min(w, h)))

    return RepresentativeMetrics(
        record=record,
        class_signature=build_class_signature(record.boxes),
        area_ratio=area_ratio,
        center=center,
        scale_basis=scale_basis,
        sharpness=compute_record_sharpness(frame, record.boxes),
        frame_hash=compute_frame_phash(frame),
        roi_hash=compute_boxes_roi_phash(frame, record.boxes),
    )


def is_similar_representative(
    anchor: RepresentativeMetrics,
    candidate: RepresentativeMetrics,
    args: argparse.Namespace,
) -> bool:
    if candidate.class_signature != anchor.class_signature:
        return False

    if candidate.record.timestamp_sec - anchor.record.timestamp_sec > args.representative_max_cluster_span_sec:
        return False

    if phash_hamming_distance(anchor.frame_hash, candidate.frame_hash) > args.representative_global_phash_threshold:
        return False

    if anchor.roi_hash is not None and candidate.roi_hash is not None:
        if (
            phash_hamming_distance(anchor.roi_hash, candidate.roi_hash)
            > args.representative_target_phash_threshold
        ):
            return False

    scale_basis = max(anchor.scale_basis, candidate.scale_basis, 1.0)
    center_disp = np.hypot(
        candidate.center[0] - anchor.center[0],
        candidate.center[1] - anchor.center[1],
    )
    if center_disp / scale_basis > args.representative_motion_threshold:
        return False

    baseline_area = max(anchor.area_ratio, 1e-6)
    area_change = abs(candidate.area_ratio - anchor.area_ratio) / baseline_area
    if area_change > args.representative_area_threshold:
        return False

    return True


def representative_reason_priority(reasons: Iterable[str]) -> float:
    weights = {
        "new_track": 3.0,
        "pose_angle_change": 2.5,
        "motion_change": 2.0,
        "scale_change": 1.4,
        "first_detection": 1.2,
        "paired_alignment": 0.3,
    }
    return sum(weights.get(reason, 0.0) for reason in set(reasons))


def representative_score(metrics: RepresentativeMetrics) -> float:
    return (
        representative_reason_priority(metrics.record.keep_reasons)
        + metrics.area_ratio * 180.0
        + min(metrics.sharpness, 400.0) / 40.0
        + len(metrics.record.boxes) * 0.4
    )


def prune_representative_records(
    records: Dict[str, ExportRecord],
    args: argparse.Namespace,
) -> Tuple[Dict[str, ExportRecord], List[Tuple[float, str]]]:
    if len(records) <= 1:
        sorted_records = sorted(records.values(), key=lambda record: record.timestamp_sec)
        return (
            {record.frame_id: record for record in sorted_records},
            [(record.timestamp_sec, record.frame_id) for record in sorted_records],
        )

    sorted_metrics = [
        build_representative_metrics(record)
        for record in sorted(records.values(), key=lambda record: record.timestamp_sec)
    ]

    kept_metrics: List[RepresentativeMetrics] = []
    cluster: List[RepresentativeMetrics] = [sorted_metrics[0]]
    anchor = sorted_metrics[0]

    for metrics in sorted_metrics[1:]:
        if is_similar_representative(anchor, metrics, args):
            cluster.append(metrics)
            continue

        kept_metrics.append(max(cluster, key=representative_score))
        cluster = [metrics]
        anchor = metrics

    if cluster:
        kept_metrics.append(max(cluster, key=representative_score))

    kept_metrics.sort(key=lambda item: item.record.timestamp_sec)
    if args.representative_min_time_gap_sec > 0 and len(kept_metrics) > 1:
        gap_filtered: List[RepresentativeMetrics] = [kept_metrics[0]]
        for metrics in kept_metrics[1:]:
            prev = gap_filtered[-1]
            if metrics.record.timestamp_sec - prev.record.timestamp_sec < args.representative_min_time_gap_sec:
                if representative_score(metrics) > representative_score(prev):
                    gap_filtered[-1] = metrics
                continue
            gap_filtered.append(metrics)
        kept_metrics = gap_filtered

    pruned_records = {item.record.frame_id: item.record for item in kept_metrics}
    anchor_timestamps = [(item.record.timestamp_sec, item.record.frame_id) for item in kept_metrics]
    return pruned_records, anchor_timestamps


def select_records_by_reference(
    reference_result: VideoExtractionResult,
    target_result: VideoExtractionResult,
    pair_tolerance_sec: float,
    processor,
    output_root: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, ExportRecord], List[Tuple[float, str]]]:
    if not reference_result.anchor_timestamps or not target_result.records:
        if not reference_result.anchor_timestamps:
            return {}, []

    sorted_target = sorted(target_result.records.values(), key=lambda record: record.timestamp_sec)
    used_ids: set[str] = set()
    selected: Dict[str, ExportRecord] = {}
    anchors: List[Tuple[float, str]] = []
    capture: Optional[cv2.VideoCapture] = None

    try:
        for ref_ts, _ in sorted(reference_result.anchor_timestamps, key=lambda item: item[0]):
            best_record: Optional[ExportRecord] = None
            best_delta: Optional[float] = None

            for record in sorted_target:
                if record.frame_id in used_ids:
                    continue
                delta = abs(record.timestamp_sec - ref_ts)
                if delta > pair_tolerance_sec:
                    continue
                if best_delta is None or delta < best_delta:
                    best_record = record
                    best_delta = delta

            if best_record is None:
                if processor is None:
                    continue
                if capture is None:
                    capture = cv2.VideoCapture(target_result.job.video_path)
                    if not capture.isOpened():
                        raise RuntimeError(f"Unable to open paired reference video: {target_result.job.video_path}")
                frame_idx = nearest_frame_index(ref_ts, target_result.fps, target_result.frame_count)
                actual_ts = frame_idx / target_result.fps if target_result.fps > 0 else ref_ts
                if abs(actual_ts - ref_ts) <= pair_tolerance_sec:
                    frame = read_frame_at_index(capture, frame_idx)
                    if frame is not None:
                        stats = processor.process_frame(
                            frame,
                            target_result.job.file_type,
                            conf_override=extraction_confidence_override(target_result.job.file_type),
                        )
                        boxes = stats.get("boxes", []) if stats else []
                        materialized = make_export_record(
                            output_root,
                            target_result.job,
                            frame_idx,
                            actual_ts,
                            frame,
                            boxes,
                            {"paired_alignment"},
                            allow_tracker_fill_labels=args.allow_tracker_fill_labels,
                            include_empty_frames=(
                                args.include_empty_frames
                                or getattr(args, "paired_reference_include_empty_target", False)
                            ),
                        )
                        if materialized is not None and materialized.frame_id not in used_ids:
                            target_result.records[materialized.frame_id] = materialized
                            sorted_target.append(materialized)
                            sorted_target.sort(key=lambda record: record.timestamp_sec)
                            best_record = materialized

            if best_record is None:
                continue

            best_record.add_reasons({"paired_alignment"})
            selected[best_record.frame_id] = best_record
            used_ids.add(best_record.frame_id)
            anchors.append((best_record.timestamp_sec, best_record.frame_id))
    finally:
        if capture is not None:
            capture.release()

    return selected, anchors


def make_export_record(
    output_root: str,
    job: VideoJob,
    frame_idx: int,
    timestamp_sec: float,
    frame: np.ndarray,
    boxes: Sequence[dict],
    keep_reasons: Iterable[str],
    allow_tracker_fill_labels: bool,
    include_empty_frames: bool,
) -> Optional[ExportRecord]:
    export_boxes = sanitize_export_boxes(boxes, allow_tracker_fill_labels)
    if not export_boxes and not include_empty_frames:
        return None
    frame_id = build_frame_id(job.session_id, job.video_id, timestamp_sec, job.modality)
    frame_path, label_path = build_output_paths(output_root, job, frame_id)
    return ExportRecord(
        frame_id=frame_id,
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        frame=encode_frame_payload(frame),
        boxes=export_boxes,
        session_id=job.session_id,
        modality=job.modality,
        source_video=job.video_path,
        frame_path=frame_path,
        label_path=label_path,
        keep_reasons=set(keep_reasons),
    )


def merge_record(records: Dict[str, ExportRecord], record: ExportRecord) -> None:
    existing = records.get(record.frame_id)
    if existing is None:
        records[record.frame_id] = record
        return
    existing.add_reasons(record.keep_reasons)
    if not existing.boxes and record.boxes:
        existing.boxes = record.boxes
    if record.frame is not None and existing.frame is None:
        existing.frame = record.frame
    if "paired_alignment" in record.keep_reasons and record.frame is not None:
        existing.frame = record.frame


def read_existing_manifest(manifest_path: str) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    if not os.path.isfile(manifest_path):
        return rows
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_id = row.get("frame_id")
            if frame_id:
                rows[frame_id] = row
    return rows


def record_to_manifest_row(record: ExportRecord) -> dict:
    class_usv, class_fishship, class_uav = build_class_flags(record.boxes)
    return {
        "frame_id": record.frame_id,
        "frame_path": record.frame_path,
        "session_id": record.session_id,
        "modality": record.modality,
        "source_video": record.source_video,
        "timestamp_sec": f"{record.timestamp_sec:.3f}",
        "label_path": record.label_path,
        "split": "",
        "is_enriched": False,
        "is_augmented": False,
        "source_type": "raw",
        "qc_status": "pending",
        "annotation_status": "labeled",
        "keep_reason": normalize_keep_reasons(record.keep_reasons),
        "class_usv": class_usv,
        "class_fishship": class_fishship,
        "class_uav": class_uav,
    }


def write_record_files(record: ExportRecord) -> None:
    if record.frame is None:
        raise RuntimeError(f"Frame data is missing for record: {record.frame_id}")
    frame = decode_frame_payload(record.frame)
    if frame is None:
        raise RuntimeError(f"Failed to decode frame payload: {record.frame_id}")
    os.makedirs(os.path.dirname(record.frame_path), exist_ok=True)
    os.makedirs(os.path.dirname(record.label_path), exist_ok=True)

    if not imwrite_unicode(record.frame_path, frame):
        raise RuntimeError(f"Failed to write image: {record.frame_path}")

    with open(record.label_path, "w", encoding="utf-8") as handle:
        for line in build_yolo_lines(frame.shape, record.boxes):
            handle.write(line + "\n")
    record.is_written = True
    record.frame = None


def write_classes_file(classes_path: str) -> None:
    os.makedirs(os.path.dirname(classes_path), exist_ok=True)
    with open(classes_path, "w", encoding="utf-8") as handle:
        for class_name in Config.CLASSES:
            handle.write(class_name + "\n")


def flush_records(records: Dict[str, ExportRecord], existing_ids: set[str]) -> int:
    written = 0
    for record in records.values():
        if record.frame_id in existing_ids or record.is_written:
            continue
        write_record_files(record)
        written += 1
    return written


def build_yolo_lines(frame_shape: Tuple[int, ...], boxes: Sequence[dict]) -> List[str]:
    h, w = frame_shape[:2]
    lines = []
    for box in boxes:
        cls_id = CLASS_TO_ID.get(box.get("class", ""))
        if cls_id is None:
            continue
        x_center = (box["x"] + box["w"] / 2.0) / max(w, 1)
        y_center = (box["y"] + box["h"] / 2.0) / max(h, 1)
        width = box["w"] / max(w, 1)
        height = box["h"] / max(h, 1)
        lines.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return lines


def read_frame_at_index(capture: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        return None
    return frame


def process_video(
    job: VideoJob,
    processor,
    output_root: str,
    args: argparse.Namespace,
    existing_ids: set[str],
) -> VideoExtractionResult:
    from target_module.image_detect_module.utils.tracker import MultiObjectTracker

    capture = cv2.VideoCapture(job.video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {job.video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tracker = MultiObjectTracker(
        frame_rate=fps,
        tracker_type=args.tracker,
        min_hits=Config.EXTERNAL_FRAMES_TRACKER_MIN_HITS,
    )

    records: Dict[str, ExportRecord] = {}
    anchor_timestamps: List[Tuple[float, str]] = []
    previous_states: Dict[int, TrackSnapshot] = {}
    seen_detection_frames = False

    last_kept_hash: Optional[np.ndarray] = None
    last_kept_detection_roi_hash: Optional[np.ndarray] = None
    last_kept_empty_hash: Optional[np.ndarray] = None
    last_kept_empty_idx = -10**9
    min_empty_gap_frames = max(1, int(round(fps * Config.EXTERNAL_FRAMES_EMPTY_SCENE_MIN_GAP_SEC)))

    prev_had_tracks = False
    last_empty_candidate: Optional[Tuple[int, float, np.ndarray]] = None

    max_frames = args.max_frames_per_video if args.max_frames_per_video > 0 else None
    frame_idx = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, frame = capture.read()
        if not ok:
            break

        timestamp_sec = frame_idx / fps if fps > 0 else 0.0
        stats = processor.process_frame(
            frame,
            job.file_type,
            conf_override=extraction_confidence_override(job.file_type),
        )
        raw_boxes = stats.get("boxes", []) if stats else []
        tracked_boxes = tracker.update(raw_boxes, frame.shape, frame=frame)
        effective_boxes = merge_raw_and_tracked_boxes(raw_boxes, tracked_boxes)
        is_tracker_fill = False
        if not effective_boxes:
            recent_tracks = tracker.get_recent_tracks(
                max_time_since_update=Config.EXTERNAL_FRAMES_TRACKER_MAX_MISSED
            )
            if recent_tracks:
                effective_boxes = recent_tracks
                is_tracker_fill = True

        frame_hash = compute_frame_phash(frame)
        detection_roi_hash = compute_boxes_roi_phash(frame, effective_boxes)

        if effective_boxes:
            if last_empty_candidate is not None:
                if not is_tracker_fill:
                    recovery_record = make_export_record(
                        output_root,
                        job,
                        last_empty_candidate[0],
                        last_empty_candidate[1],
                        last_empty_candidate[2],
                        [],
                        {"detection_recovered"},
                        allow_tracker_fill_labels=args.allow_tracker_fill_labels,
                        include_empty_frames=args.include_empty_frames,
                    )
                    if recovery_record is not None:
                        merge_record(records, recovery_record)
                last_empty_candidate = None

            reasons, current_states = evaluate_detection_keep_reasons(
                effective_boxes,
                frame,
                previous_states,
                motion_threshold=args.motion_threshold,
                pose_angle_threshold_deg=args.pose_angle_threshold_deg,
                area_threshold=args.area_threshold,
                seen_detection_frames=seen_detection_frames,
            )
            if is_tracker_fill:
                reasons.add("tracker_fill")

            is_novel_detection = has_strong_detection_novelty(
                current_states,
                previous_states,
                reasons,
                motion_threshold=Config.EXTERNAL_FRAMES_DETECTION_NOVELTY_MOTION_THRESHOLD,
                area_threshold=Config.EXTERNAL_FRAMES_DETECTION_NOVELTY_AREA_THRESHOLD,
                pose_threshold_deg=Config.EXTERNAL_FRAMES_DETECTION_NOVELTY_POSE_THRESHOLD_DEG,
            )

            if reasons and is_novel_detection and should_keep_detection_frame(
                frame_hash,
                last_kept_hash,
                detection_roi_hash,
                last_kept_detection_roi_hash,
                reasons,
                global_threshold=Config.EXTERNAL_FRAMES_ACTIVE_SCENE_GLOBAL_HASH_THRESHOLD,
                target_threshold=Config.EXTERNAL_FRAMES_ACTIVE_SCENE_TARGET_HASH_THRESHOLD,
            ):
                record = make_export_record(
                    output_root,
                    job,
                    frame_idx,
                    timestamp_sec,
                    frame,
                    effective_boxes,
                    reasons,
                    allow_tracker_fill_labels=args.allow_tracker_fill_labels,
                    include_empty_frames=args.include_empty_frames,
                )
                if record is not None:
                    if record.frame_id not in existing_ids:
                        merge_record(records, record)
                    anchor_timestamps.append((timestamp_sec, record.frame_id))
                    last_kept_hash = frame_hash
                    last_kept_detection_roi_hash = detection_roi_hash
                previous_states = current_states
                seen_detection_frames = True
            prev_had_tracks = True
        else:
            if prev_had_tracks:
                lost_record = make_export_record(
                    output_root,
                    job,
                    frame_idx,
                    timestamp_sec,
                    frame,
                    [],
                    {"detection_lost"},
                    allow_tracker_fill_labels=args.allow_tracker_fill_labels,
                    include_empty_frames=args.include_empty_frames,
                )
                if lost_record is not None:
                    if lost_record.frame_id not in existing_ids:
                        merge_record(records, lost_record)
                    anchor_timestamps.append((timestamp_sec, lost_record.frame_id))
                    last_kept_hash = frame_hash
                    last_kept_empty_hash = frame_hash
                    last_kept_empty_idx = frame_idx
            elif (
                last_kept_empty_hash is None
                or frame_idx - last_kept_empty_idx >= min_empty_gap_frames
            ):
                if should_keep_by_phash(
                    frame_hash,
                    last_kept_empty_hash,
                    {"empty_scene_distinct"},
                    args.phash_hamming_threshold,
                ):
                    empty_record = make_export_record(
                        output_root,
                        job,
                        frame_idx,
                        timestamp_sec,
                        frame,
                        [],
                        {"empty_scene_distinct"},
                        allow_tracker_fill_labels=args.allow_tracker_fill_labels,
                        include_empty_frames=args.include_empty_frames,
                    )
                    if empty_record is not None:
                        if empty_record.frame_id not in existing_ids:
                            merge_record(records, empty_record)
                        anchor_timestamps.append((timestamp_sec, empty_record.frame_id))
                        last_kept_hash = frame_hash
                        last_kept_empty_hash = frame_hash
                        last_kept_empty_idx = frame_idx

            last_empty_candidate = (frame_idx, timestamp_sec, frame.copy())
            prev_had_tracks = False

        frame_idx += 1

    capture.release()
    return VideoExtractionResult(
        job=job,
        fps=fps,
        frame_count=frame_count if frame_count > 0 else frame_idx,
        records=records,
        anchor_timestamps=anchor_timestamps,
    )


def align_results(
    source: VideoExtractionResult,
    target: VideoExtractionResult,
    processor,
    output_root: str,
    args: argparse.Namespace,
    existing_ids: set[str],
) -> None:
    if not source.anchor_timestamps:
        return

    capture = cv2.VideoCapture(target.job.video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open paired video: {target.job.video_path}")

    sorted_target_records = sorted(target.records.values(), key=lambda record: record.timestamp_sec)
    target_hash_cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    target_ptr = 0
    last_target_record: Optional[ExportRecord] = None

    try:
        for timestamp_sec, _ in sorted(source.anchor_timestamps, key=lambda item: item[0]):
            frame_idx = nearest_frame_index(timestamp_sec, target.fps, target.frame_count)
            actual_ts = frame_idx / target.fps if target.fps > 0 else timestamp_sec
            if abs(actual_ts - timestamp_sec) > args.pair_tolerance_sec:
                continue

            while target_ptr < len(sorted_target_records) and sorted_target_records[target_ptr].timestamp_sec <= actual_ts:
                last_target_record = sorted_target_records[target_ptr]
                target_ptr += 1

            frame_id = build_frame_id(target.job.session_id, target.job.video_id, actual_ts, target.job.modality)
            existing = target.records.get(frame_id)
            if existing is not None:
                existing.add_reasons({"paired_alignment"})
                last_target_record = existing
                continue
            if frame_id in existing_ids:
                continue

            frame = read_frame_at_index(capture, frame_idx)
            if frame is None:
                continue

            stats = processor.process_frame(
                frame,
                target.job.file_type,
                conf_override=extraction_confidence_override(target.job.file_type),
            )
            boxes = stats.get("boxes", []) if stats else []
            frame_hash = compute_frame_phash(frame)
            roi_hash = compute_boxes_roi_phash(frame, boxes)

            if last_target_record is not None:
                prev_frame_hash, prev_roi_hash = get_record_hashes(last_target_record, target_hash_cache)
                if not should_keep_detection_frame(
                    frame_hash,
                    prev_frame_hash,
                    roi_hash,
                    prev_roi_hash,
                    set(),
                    global_threshold=Config.EXTERNAL_FRAMES_ACTIVE_SCENE_GLOBAL_HASH_THRESHOLD,
                    target_threshold=Config.EXTERNAL_FRAMES_ACTIVE_SCENE_TARGET_HASH_THRESHOLD,
                ):
                    continue

            record = make_export_record(
                output_root,
                target.job,
                frame_idx,
                actual_ts,
                frame,
                boxes,
                {"paired_alignment"},
                allow_tracker_fill_labels=args.allow_tracker_fill_labels,
                include_empty_frames=args.include_empty_frames,
            )
            if record is None:
                continue
            merge_record(target.records, record)
            target_hash_cache[record.frame_id] = (frame_hash, roi_hash)
            last_target_record = record
    finally:
        capture.release()


def nearest_frame_index(timestamp_sec: float, fps: float, frame_count: int) -> int:
    if fps <= 0:
        return 0
    frame_idx = int(round(timestamp_sec * fps))
    if frame_count <= 0:
        return max(0, frame_idx)
    return min(max(frame_idx, 0), frame_count - 1)


def write_manifest(manifest_path: str, rows: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    ordered_rows = [rows[frame_id] for frame_id in sorted(rows)]
    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_rows)


def main() -> None:
    args = parse_args()
    from target_module.image_detect_module.image_processor import ImageProcessor

    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)
    manifest_path = os.path.join(output_root, "manifests", "frames_manifest.csv")

    groups = discover_video_groups(input_root)
    if args.max_videos > 0:
        groups = dict(list(groups.items())[: args.max_videos])

    if not groups:
        print(f"[WARN] No target videos found under: {input_root}")
        return

    existing_rows = read_existing_manifest(manifest_path) if args.resume else {}
    existing_ids = set(existing_rows)
    manifest_rows = dict(existing_rows)
    processor = ImageProcessor()

    processed_groups = 0
    exported_frames = 0

    for group_key, modalities in groups.items():
        print(f"[GROUP] {group_key}")
        results: Dict[str, VideoExtractionResult] = {}
        for modality in ["rgb", "ir"]:
            job = modalities.get(modality)
            if job is None:
                continue
            write_classes_file(build_classes_path(output_root, job))
            print(f"  [VIDEO] {job.video_path}")
            results[modality] = process_video(job, processor, output_root, args, existing_ids)

        if "rgb" in results and "ir" in results:
            align_results(results["rgb"], results["ir"], processor, output_root, args, existing_ids)
            align_results(results["ir"], results["rgb"], processor, output_root, args, existing_ids)

        if "rgb" in results and args.enable_representative_pruning:
            results["rgb"].records, results["rgb"].anchor_timestamps = prune_representative_records(
                results["rgb"].records,
                args,
            )

        if "rgb" in results and "ir" in results and args.paired_rgb_reference:
            results["ir"].records, results["ir"].anchor_timestamps = select_records_by_reference(
                results["rgb"],
                results["ir"],
                args.pair_tolerance_sec,
                processor,
                output_root,
                args,
            )
        elif "ir" in results and args.enable_representative_pruning:
            results["ir"].records, results["ir"].anchor_timestamps = prune_representative_records(
                results["ir"].records,
                args,
            )

        for modality, result in results.items():
            if modality not in {"rgb", "ir"} and args.enable_representative_pruning:
                result.records, result.anchor_timestamps = prune_representative_records(result.records, args)
            exported_frames += flush_records(result.records, existing_ids)
            for record in result.records.values():
                if record.frame_id in existing_ids:
                    continue
                manifest_rows[record.frame_id] = record_to_manifest_row(record)

        processed_groups += 1

    write_manifest(manifest_path, manifest_rows)
    print(f"[DONE] Processed groups: {processed_groups}")
    print(f"[DONE] Exported new frames: {exported_frames}")
    print(f"[DONE] Manifest: {os.path.abspath(manifest_path)}")


if __name__ == "__main__":
    main()
