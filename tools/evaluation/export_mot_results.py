"""
Export project tracker output to MOTChallenge result format.

Output layout:
    <output-root>/<tracker>/data/<seq-name>.txt

Each result row follows MOTChallenge tracker result columns:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
from cv_utils import imwrite_unicode
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.file_utils import get_file_type
from target_module.image_detect_module.utils.tracker import MultiObjectTracker


TRACKER_CHOICES = ["bytetrack", "ocsort", "botsort", "official_ocsort", "official_botsort"]


def format_mot_result_line(frame_id: int, box: dict) -> str:
    """Format one tracked box as a MOTChallenge tracker-result row."""
    return (
        f"{int(frame_id)},{int(box['track_id'])},"
        f"{float(box['x']):.2f},{float(box['y']):.2f},"
        f"{float(box['w']):.2f},{float(box['h']):.2f},"
        f"{float(box.get('confidence', 1.0)):.4f},-1,-1,-1"
    )


def export_video_to_mot_results(
    input_video: str | Path,
    output_root: str | Path,
    tracker_type: str,
    seq_name: str | None = None,
    file_type: str | None = None,
    max_frames: int = 0,
) -> Path:
    """Run detector+tracker on every video frame and write MOTChallenge result txt."""
    input_video = Path(input_video)
    output_root = Path(output_root)
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if tracker_type not in TRACKER_CHOICES:
        raise ValueError(f"Unsupported tracker: {tracker_type}")

    seq_name = seq_name or input_video.stem
    file_type = file_type or get_file_type(input_video.stem)
    if file_type == "unknown":
        file_type = "visible"

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector = td.get_detector()
    tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_type)

    output_file = output_root / tracker_type / "data" / f"{seq_name}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="mot_export_", dir=str(output_root))
    tmp_frame_path = os.path.join(tmp_dir, "frame.jpg")
    frame_id = 0
    rows: list[str] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if max_frames > 0 and frame_id > max_frames:
                break

            if not imwrite_unicode(tmp_frame_path, frame):
                raise RuntimeError(f"Failed to write temporary frame: {tmp_frame_path}")

            result = detector.detect_from_image_file(tmp_frame_path, file_type=file_type)
            boxes = []
            if result and result.get("success"):
                boxes = result.get("data", {}).get("boxes", [])

            tracked_boxes = tracker.update(boxes, frame.shape, frame=frame)
            rows.extend(format_mot_result_line(frame_id, box) for box in tracked_boxes)
    finally:
        cap.release()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    output_file.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tracker output as MOTChallenge result txt")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-root", default="results/motchallenge_trackers")
    parser.add_argument("--tracker", required=True, choices=TRACKER_CHOICES)
    parser.add_argument("--seq-name", default="", help="Sequence name; default is input video stem")
    parser.add_argument("--file-type", default="", choices=["", "visible", "infrared"])
    parser.add_argument("--max-frames", type=int, default=0, help="Debug only; 0 means full video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = export_video_to_mot_results(
        input_video=args.input,
        output_root=args.output_root,
        tracker_type=args.tracker,
        seq_name=args.seq_name or None,
        file_type=args.file_type or None,
        max_frames=args.max_frames,
    )
    print(f"[OK] MOTChallenge tracker result: {output_file.resolve()}")
    if args.max_frames > 0:
        print("[WARN] --max-frames is for debugging; formal evaluation should match the GT frame range.")


if __name__ == "__main__":
    main()
