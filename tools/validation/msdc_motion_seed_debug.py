"""
MS-DC-ELT Task 2 motion seed debug utility.

This script runs MotionSeedGenerator on a video and writes per-frame motion
statistics plus optional binary masks. It does not initialize or update any
baseline tracker.
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

from cv_utils import imwrite_unicode
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.motion_seed import MotionSeedGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS-DC-ELT motion seed debug")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument(
        "--output-dir",
        default=Config.MSDC_MOTION_DEBUG_OUTPUT_DIR,
        help="Output root; a unique run directory is created below it",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    parser.add_argument("--save-masks", action="store_true", help="Write motion mask PNG files")
    parser.add_argument("--progress-interval", type=int, default=50, help="Progress print interval")
    return parser.parse_args()


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


def write_motion_debug_outputs(
    stats_path: Path,
    mask_dir: Path,
    frame_idx: int,
    debug_info: dict,
    mask,
    save_mask: bool = False,
) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(debug_info, ensure_ascii=False) + "\n")

    if save_mask and mask is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"frame_{int(frame_idx):06d}.png"
        if not imwrite_unicode(str(mask_path), mask):
            raise OSError(f"Failed to write motion mask: {mask_path}")


def write_summary(
    run_dir: Path,
    input_path: Path,
    stats_path: Path,
    mask_dir: Path,
    processed_frames: int,
    total_motion_boxes: int,
    elapsed_sec: float,
    save_masks: bool,
) -> Path:
    summary = {
        "source_video": str(input_path),
        "processed_frames": int(processed_frames),
        "total_motion_boxes": int(total_motion_boxes),
        "avg_motion_boxes_per_frame": round(total_motion_boxes / processed_frames, 4) if processed_frames else 0.0,
        "elapsed_sec": round(float(elapsed_sec), 4),
        "processing_fps": round(processed_frames / elapsed_sec, 4) if elapsed_sec > 0 else 0.0,
        "motion_seed_stats": str(stats_path),
        "motion_masks_dir": str(mask_dir) if save_masks else "",
        "used_gmc_config": bool(getattr(Config, "MSDC_MOTION_USE_GMC", True)),
        "max_boxes": int(getattr(Config, "MSDC_MOTION_MAX_BOXES", 64)),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def process_video(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    run_dir = create_run_dir(Path(args.output_dir), input_path.stem)
    stats_path = run_dir / "motion_seed_stats.jsonl"
    mask_dir = run_dir / "motion_masks"
    generator = MotionSeedGenerator(Config)

    processed_frames = 0
    total_motion_boxes = 0
    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            boxes, debug_info = generator.update(frame, frame_idx=processed_frames)
            total_motion_boxes += len(boxes)
            write_motion_debug_outputs(
                stats_path=stats_path,
                mask_dir=mask_dir,
                frame_idx=processed_frames,
                debug_info=debug_info,
                mask=generator.last_mask,
                save_mask=args.save_masks,
            )

            processed_frames += 1
            if args.progress_interval > 0 and processed_frames % args.progress_interval == 0:
                print(
                    f"[INFO] frame={processed_frames} motion_boxes={debug_info['num_motion_boxes']} "
                    f"threshold={debug_info['diff_threshold']}",
                    flush=True,
                )
    finally:
        cap.release()

    elapsed_sec = time.time() - start_time
    summary_path = write_summary(
        run_dir=run_dir,
        input_path=input_path,
        stats_path=stats_path,
        mask_dir=mask_dir,
        processed_frames=processed_frames,
        total_motion_boxes=total_motion_boxes,
        elapsed_sec=elapsed_sec,
        save_masks=args.save_masks,
    )
    return stats_path, summary_path


def main() -> None:
    args = parse_args()
    stats_path, summary_path = process_video(args)
    print(f"[OK] motion seed stats: {stats_path}", flush=True)
    print(f"[OK] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
