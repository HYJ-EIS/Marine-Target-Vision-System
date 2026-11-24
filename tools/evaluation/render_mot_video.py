"""
Render MOTChallenge tracker results back onto a source video.

Project MOT rows preserve the evaluated one-based class in column 8.
Those labels are displayed directly. Detector/cache recovery remains available
for historical classless MOT files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
from tools.evaluation.stage_timing import StageTiming
from tools.evaluation.video_encoding import H264VideoWriter
from tools.evaluation.class_mapping import class_name_from_gt_id, normalize_prediction_class


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
            if len(row) >= 8 and row[7].strip() not in {"", "-1", "-1.0"}:
                class_name = class_name_from_gt_id(row[7])
                mot_class_id = int(float(row[7]))
                box.update({"class": class_name, "class_id": mot_class_id - 1,
                            "mot_class_id": mot_class_id})
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
        if "class" in box or "mot_class_id" in box or "class_id" in box:
            class_id = normalize_prediction_class(box)
            out = dict(box)
            class_name = class_name_from_gt_id(class_id)
            class_conf = float(out.get("class_confidence", out.get("confidence", 0.0)))
            class_cache[track_id] = (class_name, class_conf)
            out.update({"id": track_id, "class": class_name, "class_id": class_id - 1,
                        "mot_class_id": class_id, "class_confidence": class_conf})
            annotated.append(out)
            continue
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
    start_frame: int = 1,
    end_frame: int = 0,
    max_output_width: int = 0,
    video_codec: str = "mp4v",
    ffmpeg_executable: str | None = None,
) -> Path:
    timing = StageTiming()
    input_video = Path(input_video)
    mot_results = Path(mot_results)
    output_video = Path(output_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if not mot_results.is_file():
        raise FileNotFoundError(f"MOT result file does not exist: {mot_results}")
    if scale <= 0:
        raise ValueError("--scale must be positive")
    if start_frame < 1 or (end_frame > 0 and end_frame < start_frame):
        raise ValueError("Frame interval must be 1-based with end >= start")
    if max_output_width < 0:
        raise ValueError("max_output_width must be nonnegative")
    if video_codec not in {"mp4v", "h264"}:
        raise ValueError(f"Unsupported video codec: {video_codec}")

    results_by_frame = load_mot_results(mot_results)
    resolved_type = file_type or get_file_type(input_video.stem)
    if resolved_type == "unknown":
        resolved_type = "visible"

    detector = None
    conf_override = None
    cached_detections: dict[int, list[dict]] = {}
    if class_source not in {"detector", "cache", "none", "mot"}:
        raise ValueError(f"Unsupported class source: {class_source}")
    needs_class_recovery = any("class" not in box for boxes in results_by_frame.values() for box in boxes)
    if class_source == "mot" and needs_class_recovery:
        raise ValueError("MOT file has classless rows; re-export predictions with class IDs in column 8")
    if needs_class_recovery and class_source == "detector":
        detector = td.get_detector()
        conf_override = _low_conf_for_file_type(resolved_type)
    elif needs_class_recovery and class_source == "cache":
        if detections_cache is None:
            raise ValueError("--detections-cache is required when --class-source=cache")
        cached_detections = load_detection_class_cache(detections_cache, max_frames=max_frames)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_output_width > 0:
        scale = min(scale, max_output_width / max(1, frame_w))
    out_w = max(1, int(round(frame_w * scale)))
    out_h = max(1, int(round(frame_h * scale)))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if video_codec == "h264":
            writer = H264VideoWriter(output_video, fps, (out_w, out_h), executable=ffmpeg_executable)
        else:
            writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    except BaseException:
        cap.release()
        raise
    timing.add("write_export", started)
    if not writer.isOpened():
        cap.release()
        if video_codec == "h264":
            writer.abort()
        raise RuntimeError(f"Failed to open VideoWriter: {output_video}")

    class_cache: dict[int, tuple[str, float]] = {}
    track_history: dict[int, list[tuple[int, int]]] = {}
    trail_len = 30
    frame_id = 0
    try:
        while True:
            if max_frames > 0 and frame_id >= max_frames:
                break
            if end_frame > 0 and frame_id >= end_frame:
                break
            started = time.perf_counter()
            ret, frame = cap.read()
            timing.add("read_decode", started)
            if not ret:
                break
            frame_id += 1
            if max_frames > 0 and frame_id > max_frames:
                break

            started = time.perf_counter()
            track_boxes = results_by_frame.get(frame_id, [])
            detections = []
            frame_needs_classes = any("class" not in box for box in track_boxes)
            if detector is not None and frame_needs_classes:
                detections = _detect_frame_classes(detector, frame, resolved_type, conf_override)
            elif cached_detections and frame_needs_classes:
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

            if frame_id < start_frame:
                timing.add("render", started)
                continue

            vis_frame = vis.visualize_detections(
                frame,
                "video",
                {"data": {"boxes": annotated_boxes}},
                frame_num=frame_id,
                track_history=track_history,
            )
            timing.add("render", started)
            started = time.perf_counter()
            writer.write(vis_frame)
            timing.add("write_export", started)
            timing.frames += 1

            if progress_interval > 0 and frame_id % progress_interval == 0:
                suffix = f"/{max_frames}" if max_frames > 0 else ""
                print(f"[PROGRESS] rendered frame {frame_id}{suffix}", flush=True)
    except BaseException:
        if video_codec == "h264":
            writer.abort()
        raise
    finally:
        cap.release()
        started = time.perf_counter()
        writer.release()
        timing.add("write_export", started)

    timing.write(output_video.with_suffix(".timing.json"), phase="render",
                 class_source=class_source if needs_class_recovery else "mot",
                 source_start_frame=start_frame, source_last_frame=frame_id,
                 render_scale=scale, source_width=frame_w, source_height=frame_h,
                 output_width=out_w, output_height=out_h,
                 **({**writer.metadata, "encoder_frames_written": writer.frames_written,
                     "encoder_time_accounting": "write_export includes stdin blocking and final encoder drain"}
                    if video_codec == "h264" else {"video_codec": "mp4v"}))
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
    parser.add_argument("--class-source", choices=["detector", "cache", "none", "mot"], default="detector")
    parser.add_argument("--detections-cache", default="", help="Replay detection JSONL for --class-source=cache")
    parser.add_argument("--class-iou-threshold", type=float, default=0.1)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--start-frame", type=int, default=1, help="First original 1-based frame to export")
    parser.add_argument("--end-frame", type=int, default=0, help="Last original frame, 0 means no extra limit")
    parser.add_argument("--max-output-width", type=int, default=0, help="Cap visualization width; 0 keeps scale")
    parser.add_argument("--video-codec", choices=["mp4v", "h264"], default="mp4v")
    parser.add_argument("--ffmpeg-executable", default="", help="Frozen FFmpeg binary for H.264; otherwise PAPER_FFMPEG_EXECUTABLE")
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
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_output_width=args.max_output_width,
        video_codec=args.video_codec,
        ffmpeg_executable=args.ffmpeg_executable or None,
    )


if __name__ == "__main__":
    main()
