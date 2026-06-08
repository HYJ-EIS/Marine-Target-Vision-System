"""
MS-DC-ELT Task 1 low-threshold detection debug utility.

This script does not initialize or modify any tracker. It runs the current
detector twice per frame/image: once with the configured high threshold and
once with an MS-DC-ELT low threshold, then writes per-frame low-only statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cv_utils import imread_unicode
import target_module.image_detect_module.target_detection as td
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.file_utils import get_file_type


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def box_to_xyxy(box: dict) -> tuple[float, float, float, float]:
    x1 = float(box.get("x", 0.0))
    y1 = float(box.get("y", 0.0))
    x2 = x1 + float(box.get("w", 0.0))
    y2 = y1 + float(box.get("h", 0.0))
    return x1, y1, x2, y2


def box_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = box_to_xyxy(a)
    bx1, by1, bx2, by2 = box_to_xyxy(b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def filter_low_only_boxes(
    high_boxes: list[dict],
    low_boxes: list[dict],
    iou_threshold: float = Config.MSDC_LOW_HIGH_IOU_THRESH,
) -> tuple[list[dict], int]:
    """Remove low-threshold boxes that overlap any high-threshold box."""
    low_only: list[dict] = []
    overlap_count = 0
    for low_box in low_boxes:
        max_iou = max((box_iou(low_box, high_box) for high_box in high_boxes), default=0.0)
        if max_iou > iou_threshold:
            overlap_count += 1
        else:
            low_only.append(low_box)
    return low_only, overlap_count


def resolve_low_conf_threshold(file_type: str, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    if file_type == "infrared":
        return float(Config.MSDC_LOW_CONF_INFRARED)
    return float(Config.MSDC_LOW_CONF_VISIBLE)


def build_low_det_record(
    frame_idx: int,
    file_type: str,
    high_boxes: list[dict],
    low_boxes: list[dict],
    low_only_boxes: list[dict],
    low_conf_thresh: float,
    low_high_overlap_count: int,
    source_path: str = "",
    timestamp_sec: float | None = None,
) -> dict:
    low_count = len(low_boxes)
    overlap_ratio = round(low_high_overlap_count / low_count, 4) if low_count else 0.0
    record = {
        "frame_idx": int(frame_idx),
        "frame_index": int(frame_idx),
        "file_type": file_type,
        "num_high": len(high_boxes),
        "num_low": low_count,
        "num_low_only": len(low_only_boxes),
        "high_count": len(high_boxes),
        "low_count": low_count,
        "low_only_count": len(low_only_boxes),
        "low_high_overlap_count": int(low_high_overlap_count),
        "low_high_overlap_ratio": overlap_ratio,
        "low_conf_thresh": round(float(low_conf_thresh), 4),
    }
    if source_path:
        record["source_path"] = source_path
    if timestamp_sec is not None:
        record["timestamp_sec"] = round(float(timestamp_sec), 4)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS-DC-ELT low-threshold detection debug")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", default="", help="Input video path")
    source.add_argument("--image-dir", default="", help="Directory containing images")
    parser.add_argument(
        "--file-type",
        default="auto",
        choices=["auto", "visible", "infrared"],
        help="Detector type; auto falls back to visible when the filename is ambiguous",
    )
    parser.add_argument("--low-conf-thresh", type=float, default=None, help="Override low threshold")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=Config.MSDC_LOW_HIGH_IOU_THRESH,
        help="IoU threshold used to remove low boxes overlapping high boxes",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process all frames/images")
    parser.add_argument(
        "--output-dir",
        default=Config.MSDC_DEBUG_LOW_DET_OUTPUT_DIR,
        help="Output root; a unique run directory is created below it",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    return parser.parse_args()


def resolve_file_type(source_name: str, requested_file_type: str) -> str:
    if requested_file_type != "auto":
        return requested_file_type
    file_type = get_file_type(Path(source_name).stem)
    return "visible" if file_type == "unknown" else file_type


def list_images(image_dir: Path, limit: int = 0) -> list[Path]:
    images = [
        path for path in sorted(image_dir.iterdir(), key=lambda p: p.name.lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit > 0:
        return images[:limit]
    return images


def create_run_dir(output_root: Path, source_label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source_label)
    base = output_root / f"{safe_label}_{timestamp}"
    run_dir = base
    counter = 1
    while run_dir.exists():
        run_dir = output_root / f"{safe_label}_{timestamp}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def process_frame_stats(
    processor,
    frame,
    file_type: str,
    low_conf_thresh: float,
    iou_threshold: float,
    frame_idx: int,
    source_path: str = "",
    timestamp_sec: float | None = None,
) -> dict:
    high_stats = processor.process_frame(frame, file_type)
    low_stats = processor.process_frame(frame, file_type, conf_override=low_conf_thresh)
    high_boxes = high_stats.get("boxes", []) if high_stats else []
    low_boxes = low_stats.get("boxes", []) if low_stats else []
    low_only_boxes, overlap_count = filter_low_only_boxes(
        high_boxes=high_boxes,
        low_boxes=low_boxes,
        iou_threshold=iou_threshold,
    )
    return build_low_det_record(
        frame_idx=frame_idx,
        file_type=file_type,
        high_boxes=high_boxes,
        low_boxes=low_boxes,
        low_only_boxes=low_only_boxes,
        low_conf_thresh=low_conf_thresh,
        low_high_overlap_count=overlap_count,
        source_path=source_path,
        timestamp_sec=timestamp_sec,
    )


def update_totals(totals: dict, record: dict) -> None:
    totals["frames"] += 1
    totals["high"] += int(record["num_high"])
    totals["low"] += int(record["num_low"])
    totals["low_only"] += int(record["num_low_only"])
    totals["overlap"] += int(record["low_high_overlap_count"])


def write_summary(
    run_dir: Path,
    source: str,
    source_type: str,
    file_type_arg: str,
    low_conf_thresh: float | None,
    iou_threshold: float,
    totals: dict,
    elapsed_sec: float,
) -> Path:
    frames = int(totals["frames"])
    summary = {
        "source": source,
        "source_type": source_type,
        "file_type_arg": file_type_arg,
        "low_conf_thresh_arg": low_conf_thresh,
        "iou_threshold": round(float(iou_threshold), 4),
        "processed_frames": frames,
        "total_high": int(totals["high"]),
        "total_low": int(totals["low"]),
        "total_low_only": int(totals["low_only"]),
        "total_low_high_overlap": int(totals["overlap"]),
        "avg_high_per_frame": round(totals["high"] / frames, 4) if frames else 0.0,
        "avg_low_per_frame": round(totals["low"] / frames, 4) if frames else 0.0,
        "avg_low_only_per_frame": round(totals["low_only"] / frames, 4) if frames else 0.0,
        "elapsed_sec": round(float(elapsed_sec), 4),
        "processing_fps": round(frames / elapsed_sec, 4) if elapsed_sec > 0 else 0.0,
        "low_det_stats": str(run_dir / "low_det_stats.jsonl"),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def process_video(args: argparse.Namespace, processor, run_dir: Path, jsonl_handle) -> dict:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    totals = {"frames": 0, "high": 0, "low": 0, "low_only": 0, "overlap": 0}
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            file_type = resolve_file_type(str(input_path), args.file_type)
            low_thresh = resolve_low_conf_threshold(file_type, args.low_conf_thresh)
            timestamp_sec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            record = process_frame_stats(
                processor=processor,
                frame=frame,
                file_type=file_type,
                low_conf_thresh=low_thresh,
                iou_threshold=args.iou_threshold,
                frame_idx=frame_idx,
                source_path=str(input_path),
                timestamp_sec=timestamp_sec,
            )
            jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            update_totals(totals, record)
            frame_idx += 1
            if not args.quiet and frame_idx % 25 == 0:
                print(
                    f"[INFO] frame={frame_idx} high={record['num_high']} "
                    f"low={record['num_low']} low_only={record['num_low_only']}",
                    flush=True,
                )
    finally:
        capture.release()
    return totals


def process_image_dir(args: argparse.Namespace, processor, run_dir: Path, jsonl_handle) -> dict:
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")
    images = list_images(image_dir, args.max_frames)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")

    totals = {"frames": 0, "high": 0, "low": 0, "low_only": 0, "overlap": 0}
    for frame_idx, image_path in enumerate(images):
        frame = imread_unicode(str(image_path))
        if frame is None:
            if not args.quiet:
                print(f"[WARN] Failed to read image: {image_path}", flush=True)
            continue
        file_type = resolve_file_type(str(image_path), args.file_type)
        low_thresh = resolve_low_conf_threshold(file_type, args.low_conf_thresh)
        record = process_frame_stats(
            processor=processor,
            frame=frame,
            file_type=file_type,
            low_conf_thresh=low_thresh,
            iou_threshold=args.iou_threshold,
            frame_idx=frame_idx,
            source_path=str(image_path),
        )
        jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        update_totals(totals, record)
        if not args.quiet and (frame_idx + 1) % 25 == 0:
            print(
                f"[INFO] image={frame_idx + 1}/{len(images)} high={record['num_high']} "
                f"low={record['num_low']} low_only={record['num_low_only']}",
                flush=True,
            )
    return totals


def main() -> None:
    args = parse_args()
    source_path = Path(args.input or args.image_dir)
    run_dir = create_run_dir(Path(args.output_dir), source_path.stem or source_path.name)
    stats_path = run_dir / "low_det_stats.jsonl"

    print("[INFO] Initializing detector...", flush=True)
    detector = td.get_detector()
    processor = detector.processor
    if processor is None:
        raise RuntimeError("TargetDetector processor is not available")

    start_time = time.time()
    with stats_path.open("w", encoding="utf-8") as jsonl_handle:
        if args.input:
            totals = process_video(args, processor, run_dir, jsonl_handle)
            source_type = "video"
        else:
            totals = process_image_dir(args, processor, run_dir, jsonl_handle)
            source_type = "image_dir"
    elapsed_sec = time.time() - start_time

    summary_path = write_summary(
        run_dir=run_dir,
        source=str(source_path),
        source_type=source_type,
        file_type_arg=args.file_type,
        low_conf_thresh=args.low_conf_thresh,
        iou_threshold=args.iou_threshold,
        totals=totals,
        elapsed_sec=elapsed_sec,
    )
    print(f"[OK] low-det stats: {stats_path}", flush=True)
    print(f"[OK] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
