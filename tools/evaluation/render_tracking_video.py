"""
Render detector + tracker output directly to an annotated video.

This is a one-pass visualization helper: it runs the selected tracker and draws
the project box schema, including class and track_id, without first exporting
MOT txt and then running a second detector pass for rendering.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
import visualization as vis
from cv_utils import imwrite_unicode
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.constants import TRACKER_CHOICES
from target_module.image_detect_module.utils.file_utils import get_file_type
from target_module.image_detect_module.utils.msdc_detection import resolve_msdc_high_low_boxes
from target_module.image_detect_module.utils.tracking_update import (
    is_msdc_tracker,
    update_tracking_for_frame,
)
from target_module.image_detect_module.utils.tracker import MultiObjectTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tracker output directly as an annotated video")
    parser.add_argument("--input", required=True, help="Source video path")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--tracker", default="msdc_elt", choices=TRACKER_CHOICES)
    parser.add_argument("--file-type", default="", choices=["", "visible", "infrared"])
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process until the video ends")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--scale", type=float, default=1.0, help="Output scale, e.g. 0.5 for half-size")
    parser.add_argument("--debug-dir", default="", help="Optional MS-DC-ELT debug directory")
    return parser.parse_args()


def _resolve_file_type(input_video: Path, requested: str) -> str:
    if requested:
        return requested
    resolved = get_file_type(input_video.stem)
    return "visible" if resolved == "unknown" else resolved


def _writer_size(width: int, height: int, scale: float) -> tuple[int, int]:
    if scale <= 0:
        raise ValueError("--scale must be positive")
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def render_tracking_video(
    input_video: str | Path,
    output_video: str | Path,
    tracker_type: str = "msdc_elt",
    file_type: str = "",
    max_frames: int = 0,
    progress_interval: int = 100,
    scale: float = 1.0,
    debug_dir: str | Path | None = None,
) -> Path:
    input_video = Path(input_video)
    output_video = Path(output_video)
    if tracker_type not in TRACKER_CHOICES:
        raise ValueError(f"Unsupported tracker: {tracker_type}")
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_w, out_h = _writer_size(width, height, scale)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer: {output_video}")

    file_type = _resolve_file_type(input_video, file_type)
    detector = td.get_detector()
    if is_msdc_tracker(tracker_type):
        from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker

        if not debug_dir:
            Config.MSDC_DEBUG_EVENTS = False
        tracker = None
        lifecycle_tracker = MSDCLifecycleTracker(
            config=Config,
            file_type=file_type,
            debug_dir=Path(debug_dir) if debug_dir else None,
            processor=getattr(detector, "processor", None),
        )
    else:
        tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_type)
        lifecycle_tracker = None

    track_history: dict[int, list[tuple[int, int]]] = {}
    frame_id = 0
    start = time.time()
    tmp_path = output_video.parent / f".{output_video.stem}_frame.jpg"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if max_frames > 0 and frame_id > max_frames:
                break

            low_boxes = None
            if is_msdc_tracker(tracker_type):
                boxes, low_boxes = resolve_msdc_high_low_boxes(detector, frame, file_type)
            else:
                if not imwrite_unicode(str(tmp_path), frame):
                    raise RuntimeError(f"Failed to write temporary frame: {tmp_path}")
                result = detector.detect_from_image_file(str(tmp_path), file_type=file_type)
                boxes = result.get("data", {}).get("boxes", []) if result and result.get("success") else []

            tracked_boxes = update_tracking_for_frame(
                tracker_type=tracker_type,
                tracker=tracker,
                lifecycle_tracker=lifecycle_tracker,
                detector=detector,
                frame=frame,
                frame_idx=frame_id - 1,
                file_type=file_type,
                boxes=boxes,
                low_boxes=low_boxes,
            )

            for box in tracked_boxes:
                track_id = int(box.get("track_id", box.get("id", 0)))
                cx = int(box.get("x", 0)) + int(box.get("w", 0)) // 2
                cy = int(box.get("y", 0)) + int(box.get("h", 0)) // 2
                track_history.setdefault(track_id, []).append((cx, cy))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

            result = {"success": True, "data": {"boxes": tracked_boxes}}
            rendered = vis.visualize_detections(frame, "video", result, frame_num=frame_id, track_history=track_history)
            if scale != 1.0:
                rendered = cv2.resize(rendered, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(rendered)

            if progress_interval > 0 and frame_id % progress_interval == 0:
                elapsed = time.time() - start
                fps_now = frame_id / elapsed if elapsed > 0 else 0.0
                total = total_frames if total_frames > 0 else "?"
                print(f"[PROGRESS] {input_video.stem}: frame {frame_id}/{total}, {fps_now:.2f} fps", flush=True)
    finally:
        cap.release()
        writer.release()
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    print(f"[OK] Annotated video: {output_video}", flush=True)
    return output_video


def main() -> None:
    args = parse_args()
    render_tracking_video(
        input_video=args.input,
        output_video=args.output,
        tracker_type=args.tracker,
        file_type=args.file_type,
        max_frames=args.max_frames,
        progress_interval=args.progress_interval,
        scale=args.scale,
        debug_dir=args.debug_dir or None,
    )


if __name__ == "__main__":
    main()
