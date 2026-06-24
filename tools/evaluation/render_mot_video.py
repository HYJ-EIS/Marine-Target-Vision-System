"""
Render MOTChallenge tracker results back onto a source video.

The MOT result format does not contain class labels, so this renderer can
optionally run the existing detector once per frame and match detections back
to tracker boxes to recover class text for visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
import visualization as vis
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.file_utils import get_file_type


def load_mot_results(path: str | Path) -> dict[int, list[dict]]:
    """Load MOTChallenge rows grouped by 1-based frame id."""
    path = Path(path)
    rows_by_frame: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 7:
                continue
            frame_id = int(float(row[0]))
            box = {
                "track_id": int(float(row[1])),
                "x": float(row[2]),
                "y": float(row[3]),
                "w": float(row[4]),
                "h": float(row[5]),
                "confidence": float(row[6]),
            }
            rows_by_frame.setdefault(frame_id, []).append(box)
    return rows_by_frame


def load_detection_class_cache(path: str | Path, max_frames: int = 0) -> dict[int, list[dict]]:
    """Load replay detector boxes grouped by 1-based frame id for class labels."""
    path = Path(path)
    rows_by_frame: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame_id = int(row.get("frame_id", 0))
            if frame_id <= 0:
                continue
            if max_frames > 0 and frame_id > max_frames:
                continue
            boxes = []
            for key in ("high_boxes", "low_boxes", "boxes"):
                for box in row.get(key, []) or []:
                    item = dict(box)
                    item.setdefault("class_confidence", item.get("confidence", 0.0))
                    boxes.append(item)
                if boxes and key == "boxes":
                    break
            rows_by_frame[frame_id] = boxes
    return rows_by_frame


def _xywh_to_xyxy(box: dict) -> np.ndarray:
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", 0.0))
    h = float(box.get("h", 0.0))
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def _iou_xywh(a: dict, b: dict) -> float:
    box_a = _xywh_to_xyxy(a)
    box_b = _xywh_to_xyxy(b)
    xx1 = max(float(box_a[0]), float(box_b[0]))
    yy1 = max(float(box_a[1]), float(box_b[1]))
    xx2 = min(float(box_a[2]), float(box_b[2]))
    yy2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return inter / max(area_a + area_b - inter, 1e-9)


def _low_conf_for_file_type(file_type: str) -> float:
    if file_type == "infrared":
        return float(Config.MSDC_LOW_CONF_INFRARED)
    return float(Config.MSDC_LOW_CONF_VISIBLE)


def _detect_frame_classes(detector, frame: np.ndarray, file_type: str, conf_override: float | None) -> list[dict]:
    processor = getattr(detector, "processor", None)
    if processor is None or not hasattr(processor, "process_frame"):
        raise RuntimeError("detector.processor.process_frame(frame, file_type, conf_override=...) is unavailable")
    stats = processor.process_frame(frame, file_type, conf_override=conf_override)
    if stats is None:
        return []
    return list(stats.get("boxes", []))


def _assign_classes(
    track_boxes: list[dict],
    detections: list[dict],
    class_cache: dict[int, tuple[str, float]],
    iou_threshold: float,
) -> list[dict]:
    annotated = []
    for box in track_boxes:
        track_id = int(box["track_id"])
        best_det = None
        best_iou = 0.0
        for det in detections:
            iou = _iou_xywh(box, det)
            if iou > best_iou:
                best_iou = iou
                best_det = det

        out = dict(box)
        if best_det is not None and best_iou >= iou_threshold:
            class_name = str(best_det.get("class", "unknown"))
            class_conf = float(best_det.get("class_confidence", best_det.get("confidence", 0.0)))
            class_cache[track_id] = (class_name, class_conf)
        elif track_id in class_cache:
            class_name, class_conf = class_cache[track_id]
        else:
            class_name = "target"
            class_conf = float(out.get("confidence", 0.0))

        out["id"] = track_id
        out["class"] = class_name
        out["class_confidence"] = class_conf
        annotated.append(out)
    return annotated


def _scale_boxes(boxes: Iterable[dict], scale: float) -> list[dict]:
    if abs(scale - 1.0) < 1e-9:
        return [dict(box) for box in boxes]
    scaled = []
    for box in boxes:
        item = dict(box)
        for key in ("x", "y", "w", "h"):
            item[key] = float(item.get(key, 0.0)) * scale
        scaled.append(item)
    return scaled


def render_mot_video(
    input_video: str | Path,
    mot_results: str | Path,
    output_video: str | Path,
    file_type: str | None = None,
    max_frames: int = 0,
    scale: float = 1.0,
    class_source: str = "detector",
    detections_cache: str | Path | None = None,
    class_iou_threshold: float = 0.1,
    progress_interval: int = 100,
) -> Path:
    input_video = Path(input_video)
    mot_results = Path(mot_results)
    output_video = Path(output_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if not mot_results.is_file():
        raise FileNotFoundError(f"MOT result file does not exist: {mot_results}")
    if scale <= 0:
        raise ValueError("--scale must be positive")

    results_by_frame = load_mot_results(mot_results)
    resolved_type = file_type or get_file_type(input_video.stem)
    if resolved_type == "unknown":
        resolved_type = "visible"

    detector = None
    conf_override = None
    cached_detections: dict[int, list[dict]] = {}
    if class_source == "detector":
        detector = td.get_detector()
        conf_override = _low_conf_for_file_type(resolved_type)
    elif class_source == "cache":
        if detections_cache is None:
            raise ValueError("--detections-cache is required when --class-source=cache")
        cached_detections = load_detection_class_cache(detections_cache, max_frames=max_frames)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = max(1, int(round(frame_w * scale)))
    out_h = max(1, int(round(frame_h * scale)))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {output_video}")

    class_cache: dict[int, tuple[str, float]] = {}
    track_history: dict[int, list[tuple[int, int]]] = {}
    trail_len = 30
    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if max_frames > 0 and frame_id > max_frames:
                break

            track_boxes = results_by_frame.get(frame_id, [])
            detections = []
            if detector is not None and track_boxes:
                detections = _detect_frame_classes(detector, frame, resolved_type, conf_override)
            elif cached_detections and track_boxes:
                detections = cached_detections.get(frame_id, [])
            annotated_boxes = _assign_classes(track_boxes, detections, class_cache, class_iou_threshold)

            if abs(scale - 1.0) >= 1e-9:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                annotated_boxes = _scale_boxes(annotated_boxes, scale)

            for box in annotated_boxes:
                tid = int(box["track_id"])
                cx = int(float(box["x"]) + float(box["w"]) / 2.0)
                cy = int(float(box["y"]) + float(box["h"]) / 2.0)
                pts = track_history.setdefault(tid, [])
                pts.append((cx, cy))
                if len(pts) > trail_len:
                    del pts[:-trail_len]

            vis_frame = vis.visualize_detections(
                frame,
                "video",
                {"data": {"boxes": annotated_boxes}},
                frame_num=frame_id,
                track_history=track_history,
            )
            writer.write(vis_frame)

            if progress_interval > 0 and frame_id % progress_interval == 0:
                suffix = f"/{max_frames}" if max_frames > 0 else ""
                print(f"[PROGRESS] rendered frame {frame_id}{suffix}", flush=True)
    finally:
        cap.release()
        writer.release()

    print(f"[OK] annotated video: {output_video.resolve()}", flush=True)
    return output_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MOT tracker results as an annotated video")
    parser.add_argument("--input", required=True, help="Source video path")
    parser.add_argument("--mot-results", required=True, help="MOTChallenge result txt")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--file-type", default="", choices=["", "visible", "infrared"])
    parser.add_argument("--max-frames", type=int, default=0, help="0 means render until video/result end")
    parser.add_argument("--scale", type=float, default=1.0, help="Output scale, e.g. 0.5 for half-size")
    parser.add_argument("--class-source", choices=["detector", "cache", "none"], default="detector")
    parser.add_argument("--detections-cache", default="", help="Replay detection JSONL for --class-source=cache")
    parser.add_argument("--class-iou-threshold", type=float, default=0.1)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mot_video(
        input_video=args.input,
        mot_results=args.mot_results,
        output_video=args.output,
        file_type=args.file_type or None,
        max_frames=args.max_frames,
        scale=args.scale,
        class_source=args.class_source,
        detections_cache=args.detections_cache or None,
        class_iou_threshold=args.class_iou_threshold,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
