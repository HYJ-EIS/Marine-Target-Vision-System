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
from target_module.image_detect_module.constants import TRACKER_CHOICES
from target_module.image_detect_module.utils.file_utils import get_file_type
from target_module.image_detect_module.utils.msdc_detection import (
    resolve_msdc_high_low_boxes,
    split_msdc_high_from_low_boxes as _split_msdc_high_from_low_boxes,
)
from target_module.image_detect_module.utils.tracking_update import (
    is_msdc_tracker,
    update_tracking_for_frame,
)
from target_module.image_detect_module.utils.tracker import MultiObjectTracker


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
    progress_interval: int = 0,
    tracker_output_name: str | None = None,
) -> Path:
    """Run detector+tracker on every video frame and write MOTChallenge result txt."""
    input_video = Path(input_video)
    output_root = Path(output_root)
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if tracker_type not in TRACKER_CHOICES:
        raise ValueError(f"Unsupported tracker: {tracker_type}")

    seq_name = seq_name or input_video.stem
    tracker_output_name = tracker_output_name or tracker_type
    file_type = file_type or get_file_type(input_video.stem)
    if file_type == "unknown":
        file_type = "visible"

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector = td.get_detector()
    diagnostics_dir = None
    if is_msdc_tracker(tracker_type):
        from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker
        tracker = None
        diagnostics_dir = output_root / tracker_output_name / "diagnostics" / seq_name
        lifecycle_tracker = MSDCLifecycleTracker(
            config=Config,
            file_type=file_type,
            debug_dir=diagnostics_dir,
            processor=getattr(detector, "processor", None),
        )
    else:
        tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_type)
        lifecycle_tracker = None

    output_file = output_root / tracker_output_name / "data" / f"{seq_name}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="mot_export_", dir=str(output_root))
    tmp_frame_path = os.path.join(tmp_dir, "frame.jpg")
    frame_id = 0
    try:
        with output_file.open("w", encoding="utf-8") as fh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                if max_frames > 0 and frame_id > max_frames:
                    break

                low_boxes = None
                if is_msdc_tracker(tracker_type):
                    boxes, low_boxes = resolve_msdc_high_low_boxes(detector, frame, file_type)
                else:
                    if not imwrite_unicode(tmp_frame_path, frame):
                        raise RuntimeError(f"Failed to write temporary frame: {tmp_frame_path}")

                    result = detector.detect_from_image_file(tmp_frame_path, file_type=file_type)
                    boxes = []
                    if result and result.get("success"):
                        boxes = result.get("data", {}).get("boxes", [])

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
                    fh.write(format_mot_result_line(frame_id, box) + "\n")
                if progress_interval > 0 and frame_id % progress_interval == 0:
                    fh.flush()
                    print(
                        f"[PROGRESS] {tracker_type} {seq_name}: frame {frame_id}"
                        + (f"/{max_frames}" if max_frames > 0 else ""),
                        flush=True,
                    )
    finally:
        cap.release()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if diagnostics_dir is not None:
        print(f"[OK] MS-DC-ELT diagnostics: {diagnostics_dir.resolve()}")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tracker output as MOTChallenge result txt")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-root", default="results/motchallenge_trackers")
    parser.add_argument("--tracker", required=True, choices=TRACKER_CHOICES)
    parser.add_argument("--tracker-name", default="", help="Output tracker folder name; default is --tracker")
    parser.add_argument("--seq-name", default="", help="Sequence name; default is input video stem")
    parser.add_argument("--file-type", default="", choices=["", "visible", "infrared"])
    parser.add_argument("--max-frames", type=int, default=0, help="Debug only; 0 means full video")
    parser.add_argument("--progress-interval", type=int, default=0, help="Print and flush progress every N frames; 0 disables")
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
        progress_interval=args.progress_interval,
        tracker_output_name=args.tracker_name or None,
    )
    print(f"[OK] MOTChallenge tracker result: {output_file.resolve()}")
    if args.max_frames > 0:
        print("[WARN] --max-frames is for debugging; formal evaluation should match the GT frame range.")


if __name__ == "__main__":
    main()
