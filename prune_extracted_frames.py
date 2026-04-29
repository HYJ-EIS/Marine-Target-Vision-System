import argparse
import csv
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cv_utils import imread_unicode
from extract_tracking_frames import (
    compute_boxes_roi_phash,
    compute_frame_phash,
    phash_hamming_distance,
)
from target_module.image_detect_module.config import Config


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

CLASS_ID_TO_NAME = {idx: name for idx, name in enumerate(Config.CLASSES)}


@dataclass
class FrameCandidate:
    row: dict
    image_path: Path
    label_path: Path
    timestamp_sec: float
    keep_reasons: set[str]
    boxes: list[dict]
    frame_shape: tuple[int, int]
    class_signature: tuple[tuple[str, int], ...]
    area_ratio: float
    center: tuple[float, float]
    scale_basis: float
    sharpness: float
    frame_hash: np.ndarray
    roi_hash: Optional[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune extracted frames into a more compact training subset.")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing images/ labels/ manifests/")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root for the compact subset. Defaults to <dataset-root>_compact",
    )
    parser.add_argument("--global-phash-threshold", type=int, default=14)
    parser.add_argument("--roi-phash-threshold", type=int, default=14)
    parser.add_argument("--motion-threshold", type=float, default=0.55)
    parser.add_argument("--area-threshold", type=float, default=0.35)
    parser.add_argument("--max-cluster-span-sec", type=float, default=6.0)
    return parser.parse_args()


def parse_yolo_boxes(label_path: Path, width: int, height: int) -> list[dict]:
    boxes: list[dict] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            box_w = float(parts[3]) * width
            box_h = float(parts[4]) * height
            boxes.append(
                {
                    "class": CLASS_ID_TO_NAME.get(class_id, f"class_{class_id}"),
                    "x": max(0.0, cx - box_w / 2.0),
                    "y": max(0.0, cy - box_h / 2.0),
                    "w": max(1.0, box_w),
                    "h": max(1.0, box_h),
                }
            )
    return boxes


def compute_sharpness(frame: np.ndarray, boxes: list[dict]) -> float:
    if boxes:
        x1 = max(0, min(int(box["x"]) for box in boxes))
        y1 = max(0, min(int(box["y"]) for box in boxes))
        x2 = min(frame.shape[1], max(int(box["x"] + box["w"]) for box in boxes))
        y2 = min(frame.shape[0], max(int(box["y"] + box["h"]) for box in boxes))
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_candidate(row: dict) -> FrameCandidate:
    image_path = Path(row["frame_path"])
    label_path = Path(row["label_path"])
    frame = imread_unicode(str(image_path))
    if frame is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    height, width = frame.shape[:2]
    boxes = parse_yolo_boxes(label_path, width, height)
    area_sum = sum(float(box["w"]) * float(box["h"]) for box in boxes)
    area_ratio = area_sum / max(float(width * height), 1.0)

    if boxes:
        weighted_centers = []
        total_weight = 0.0
        scale_basis = 1.0
        for box in boxes:
            area = max(1.0, float(box["w"]) * float(box["h"]))
            cx = float(box["x"]) + float(box["w"]) / 2.0
            cy = float(box["y"]) + float(box["h"]) / 2.0
            weighted_centers.append((cx, cy, area))
            total_weight += area
            scale_basis = max(scale_basis, float(box["w"]), float(box["h"]))
        center_x = sum(cx * area for cx, _, area in weighted_centers) / total_weight
        center_y = sum(cy * area for _, cy, area in weighted_centers) / total_weight
    else:
        center_x = width / 2.0
        center_y = height / 2.0
        scale_basis = max(1.0, float(min(width, height)))

    class_counter = Counter(box["class"] for box in boxes)
    class_signature = tuple(sorted(class_counter.items()))
    frame_hash = compute_frame_phash(frame)
    roi_hash = compute_boxes_roi_phash(frame, boxes)
    sharpness = compute_sharpness(frame, boxes)

    return FrameCandidate(
        row=row,
        image_path=image_path,
        label_path=label_path,
        timestamp_sec=float(row["timestamp_sec"]),
        keep_reasons=set(filter(None, row["keep_reason"].split("|"))),
        boxes=boxes,
        frame_shape=(height, width),
        class_signature=class_signature,
        area_ratio=area_ratio,
        center=(center_x, center_y),
        scale_basis=scale_basis,
        sharpness=sharpness,
        frame_hash=frame_hash,
        roi_hash=roi_hash,
    )


def is_similar(anchor: FrameCandidate, candidate: FrameCandidate, args: argparse.Namespace) -> bool:
    if candidate.class_signature != anchor.class_signature:
        return False

    time_delta = candidate.timestamp_sec - anchor.timestamp_sec
    if time_delta > args.max_cluster_span_sec:
        return False

    global_distance = phash_hamming_distance(anchor.frame_hash, candidate.frame_hash)
    if global_distance > args.global_phash_threshold:
        return False

    if anchor.roi_hash is not None and candidate.roi_hash is not None:
        roi_distance = phash_hamming_distance(anchor.roi_hash, candidate.roi_hash)
        if roi_distance > args.roi_phash_threshold:
            return False

    anchor_scale = max(anchor.scale_basis, 1.0)
    candidate_scale = max(candidate.scale_basis, 1.0)
    scale_basis = max(anchor_scale, candidate_scale, 1.0)
    center_disp = math.hypot(candidate.center[0] - anchor.center[0], candidate.center[1] - anchor.center[1])
    if center_disp / scale_basis > args.motion_threshold:
        return False

    baseline_area = max(anchor.area_ratio, 1e-6)
    area_change = abs(candidate.area_ratio - anchor.area_ratio) / baseline_area
    if area_change > args.area_threshold:
        return False

    return True


def reason_priority(keep_reasons: set[str]) -> float:
    weights = {
        "new_track": 3.0,
        "pose_angle_change": 2.5,
        "motion_change": 2.0,
        "scale_change": 1.4,
        "first_detection": 1.2,
        "paired_alignment": 0.3,
    }
    return sum(weights.get(reason, 0.0) for reason in keep_reasons)


def candidate_score(candidate: FrameCandidate) -> float:
    return (
        reason_priority(candidate.keep_reasons)
        + candidate.area_ratio * 180.0
        + min(candidate.sharpness, 400.0) / 40.0
        + len(candidate.boxes) * 0.4
    )


def choose_representative(cluster: list[FrameCandidate]) -> FrameCandidate:
    return max(cluster, key=candidate_score)


def load_manifest(dataset_root: Path) -> list[dict]:
    manifest_path = dataset_root / "manifests" / "frames_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["source_video"], row["modality"])
        grouped.setdefault(key, []).append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: float(row["timestamp_sec"]))
    return grouped


def select_subset(rows: list[dict], args: argparse.Namespace) -> list[FrameCandidate]:
    grouped = group_rows(rows)
    selected: list[FrameCandidate] = []

    for _, group_rows_list in grouped.items():
        candidates = [build_candidate(row) for row in group_rows_list]
        if not candidates:
            continue

        cluster: list[FrameCandidate] = [candidates[0]]
        anchor = candidates[0]
        for candidate in candidates[1:]:
            if is_similar(anchor, candidate, args):
                cluster.append(candidate)
                continue

            selected.append(choose_representative(cluster))
            cluster = [candidate]
            anchor = candidate

        if cluster:
            selected.append(choose_representative(cluster))

    selected.sort(key=lambda item: (item.row["modality"], item.row["source_video"], item.timestamp_sec))
    return selected


def copy_selected(selected: list[FrameCandidate], dataset_root: Path, output_root: Path) -> tuple[list[dict], list[dict]]:
    manifest_rows: list[dict] = []
    summary_rows: list[dict] = []
    grouped_counts: Counter = Counter()
    grouped_kept: Counter = Counter()

    all_rows = load_manifest(dataset_root)
    for row in all_rows:
        grouped_counts[(row["source_video"], row["modality"])] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    selected_ids = {candidate.row["frame_id"] for candidate in selected}
    for candidate in selected:
        grouped_kept[(candidate.row["source_video"], candidate.row["modality"])] += 1
        rel_image = candidate.image_path.relative_to(dataset_root)
        rel_label = candidate.label_path.relative_to(dataset_root)
        dst_image = output_root / rel_image
        dst_label = output_root / rel_label
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.image_path, dst_image)
        shutil.copy2(candidate.label_path, dst_label)

        row = dict(candidate.row)
        row["frame_path"] = str(dst_image.resolve())
        row["label_path"] = str(dst_label.resolve())
        manifest_rows.append(row)

    for (source_video, modality), total in sorted(grouped_counts.items()):
        summary_rows.append(
            {
                "source_video": source_video,
                "modality": modality,
                "before_count": total,
                "after_count": grouped_kept[(source_video, modality)],
                "reduction_ratio": f"{(1.0 - grouped_kept[(source_video, modality)] / max(total, 1)):.3f}",
            }
        )

    selected_manifest_ids = {row["frame_id"] for row in manifest_rows}
    if selected_ids != selected_manifest_ids:
        raise RuntimeError("Selected frame set mismatch while writing compact dataset.")

    return manifest_rows, summary_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else dataset_root.with_name(f"{dataset_root.name}_compact")

    rows = load_manifest(dataset_root)
    selected = select_subset(rows, args)
    manifest_rows, summary_rows = copy_selected(selected, dataset_root, output_root)
    manifest_rows.sort(key=lambda row: row["frame_id"])

    write_csv(output_root / "manifests" / "frames_manifest.csv", manifest_rows, MANIFEST_FIELDS)
    write_csv(
        output_root / "manifests" / "prune_summary.csv",
        summary_rows,
        ["source_video", "modality", "before_count", "after_count", "reduction_ratio"],
    )

    print(f"dataset_root={dataset_root}")
    print(f"output_root={output_root}")
    print(f"before_frames={len(rows)}")
    print(f"after_frames={len(selected)}")
    print(f"reduction_ratio={(1.0 - len(selected) / max(len(rows), 1)):.3f}")


if __name__ == "__main__":
    main()
