"""
Render an MS-DC-ELT diagnostics JSONL file back onto the source video.

Unlike MOT rendering, this uses the stored per-frame ``output_boxes`` from
``msdc_tracks.jsonl``.  It therefore visualizes the exact tracker output from a
previous run without rerunning detection or tracking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import visualization as vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render stored MS-DC-ELT diagnostics as an annotated video")
    parser.add_argument("--input", required=True, help="Source video path")
    parser.add_argument("--diagnostics-jsonl", required=True, help="Path to msdc_tracks.jsonl")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means render all diagnostics frames")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def _read_frame_boxes(path: str | Path, max_frames: int = 0) -> dict[int, list[dict]]:
    boxes_by_frame: dict[int, list[dict]] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            frame_idx = int(payload.get("frame_idx", len(boxes_by_frame)))
            if max_frames > 0 and frame_idx >= max_frames:
                break
            boxes_by_frame[frame_idx] = list(payload.get("output_boxes", []) or [])
    return boxes_by_frame


def render_msdc_diagnostics_video(
    input_video: str | Path,
    diagnostics_jsonl: str | Path,
    output_video: str | Path,
    max_frames: int = 0,
    progress_interval: int = 100,
) -> Path:
    input_video = Path(input_video)
    diagnostics_jsonl = Path(diagnostics_jsonl)
    output_video = Path(output_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if not diagnostics_jsonl.is_file():
        raise FileNotFoundError(f"Diagnostics JSONL does not exist: {diagnostics_jsonl}")

    boxes_by_frame = _read_frame_boxes(diagnostics_jsonl, max_frames=max_frames)
    if not boxes_by_frame:
        raise RuntimeError(f"No output_boxes found in diagnostics: {diagnostics_jsonl}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer: {output_video}")

    track_history: dict[int, list[tuple[int, int]]] = {}
    frame_id = 0
    target_frames = max(boxes_by_frame) + 1
    try:
        while frame_id < target_frames:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = boxes_by_frame.get(frame_id, [])
            for box in boxes:
                track_id = int(box.get("track_id", box.get("id", 0)))
                cx = int(box.get("x", 0)) + int(box.get("w", 0)) // 2
                cy = int(box.get("y", 0)) + int(box.get("h", 0)) // 2
                track_history.setdefault(track_id, []).append((cx, cy))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
            result = {"success": True, "data": {"boxes": boxes}}
            rendered = vis.visualize_detections(frame, "video", result, frame_num=frame_id + 1, track_history=track_history)
            writer.write(rendered)
            frame_id += 1
            if progress_interval > 0 and frame_id % progress_interval == 0:
                print(f"[PROGRESS] rendered {frame_id}/{target_frames}", flush=True)
    finally:
        cap.release()
        writer.release()

    print(f"[OK] Annotated diagnostics video: {output_video}", flush=True)
    return output_video


def main() -> None:
    args = parse_args()
    render_msdc_diagnostics_video(
        input_video=args.input,
        diagnostics_jsonl=args.diagnostics_jsonl,
        output_video=args.output,
        max_frames=args.max_frames,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
