"""
Run the same RGB/IR video detections through multiple trackers.

The script writes annotated MP4 output plus per-frame tracking data for every
tracker, so tracker behavior can be compared without re-running detection for
each tracker.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
import visualization as vis
from target_module.image_detect_module.utils.tracker import MultiObjectTracker


DEFAULT_IR_VIDEO = Path(
    "/mnt/d/Desktop/UAV_USV标注数据集/multi_target_source_videos/USV/IR/"
    "DJI_20250916100639_0001_T.MP4"
)
DEFAULT_RGB_VIDEO = Path(
    "/mnt/d/Desktop/UAV_USV标注数据集/multi_target_source_videos/USV/RGB/"
    "DJI_20250916100639_0001_V.MP4"
)
DEFAULT_OUTPUT_ROOT = Path("results/tracker_effect_test/DJI_20250916100639_0001")
TRACKER_CHOICES = [
    "bytetrack",
    "ocsort",
    "botsort",
    "dist_tracker",
    "official_ocsort",
    "official_botsort",
]
TRACK_CSV_FIELDS = ["frame", "id", "x", "y", "w", "h", "confidence", "class"]


@dataclass
class VideoJob:
    modality: str
    file_type: str
    video_path: Path


@dataclass
class TrackerRun:
    name: str
    output_dir: Path
    resume_frame: int = 0
    output_video_path: Path | None = None
    tracker: MultiObjectTracker | None = None
    writer: cv2.VideoWriter | None = None
    csv_handle: object | None = None
    csv_writer: csv.DictWriter | None = None
    jsonl_handle: object | None = None
    track_history: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    total_track_boxes: int = 0
    active: bool = True
    error: str = ""

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        for handle_name in ("csv_handle", "jsonl_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)


def normalize_box(box: dict) -> dict:
    """Return a stable JSON-friendly project tracking box."""
    confidence = float(box.get("confidence", 0.0))
    return {
        "id": int(box.get("track_id", box.get("id", 0))),
        "x": int(box.get("x", 0)),
        "y": int(box.get("y", 0)),
        "w": int(box.get("w", 0)),
        "h": int(box.get("h", 0)),
        "confidence": round(confidence, 4),
        "class": str(box.get("class", "unknown")),
    }


def build_frame_record(frame_index: int, timestamp_sec: float, boxes: list[dict]) -> dict:
    return {
        "frame_index": int(frame_index),
        "timestamp_sec": round(float(timestamp_sec), 4),
        "boxes": boxes,
    }


def write_tracks_csv_row(writer: csv.DictWriter, frame_index: int, box: dict) -> None:
    writer.writerow({
        "frame": int(frame_index),
        "id": int(box.get("track_id", box.get("id", 0))),
        "x": int(box.get("x", 0)),
        "y": int(box.get("y", 0)),
        "w": int(box.get("w", 0)),
        "h": int(box.get("h", 0)),
        "confidence": f"{float(box.get('confidence', 0.0)):.4f}",
        "class": str(box.get("class", "unknown")),
    })


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def resolve_tracker_resume_frame(output_root: Path, modality: str, tracker_name: str) -> int:
    return count_jsonl_rows(output_root / modality / tracker_name / "frames.jsonl")


def annotated_video_filename(resume_frame: int) -> str:
    if resume_frame == 0:
        return "annotated.mp4"
    return f"annotated_part_{resume_frame + 1:06d}.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-tracker effect test on RGB/IR videos")
    parser.add_argument("--ir-video", default=str(DEFAULT_IR_VIDEO), help="IR input video path")
    parser.add_argument("--rgb-video", default=str(DEFAULT_RGB_VIDEO), help="RGB input video path")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory")
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=["IR", "RGB"],
        choices=["IR", "RGB"],
        help="Modalities to process; omit to run both IR and RGB",
    )
    parser.add_argument(
        "--trackers",
        nargs="*",
        default=None,
        choices=TRACKER_CHOICES,
        help="Tracker names; omit to run all trackers",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    parser.add_argument("--progress-interval", type=int, default=50, help="Progress print interval in frames")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append tracks/frames after existing frames.jsonl rows and write a new "
            "annotated_part_*.mp4 segment. Previous frames are replayed to restore "
            "tracker state."
        ),
    )
    return parser.parse_args()


def resolve_trackers(trackers: Iterable[str] | None) -> list[str]:
    selected = list(trackers or TRACKER_CHOICES)
    return selected or TRACKER_CHOICES.copy()


def build_video_jobs(ir_video: str | Path, rgb_video: str | Path,
                     modalities: Iterable[str] | None = None) -> list[VideoJob]:
    selected = set(modalities or ["IR", "RGB"])
    jobs: list[VideoJob] = []
    if "IR" in selected:
        jobs.append(VideoJob("IR", "infrared", Path(ir_video)))
    if "RGB" in selected:
        jobs.append(VideoJob("RGB", "visible", Path(rgb_video)))
    return jobs


def make_detection_result(file_type: str, boxes: list[dict]) -> dict:
    return {
        "success": True,
        "message": "Success",
        "type": file_type,
        "data": {
            "boxes": boxes,
            "count": len(boxes),
            "result_image_path": "",
        },
    }


def initialize_tracker_run(
    tracker_name: str,
    output_dir: Path,
    fps: float,
    frame_size: tuple[int, int],
    resume_frame: int = 0,
) -> TrackerRun:
    output_dir.mkdir(parents=True, exist_ok=True)
    run = TrackerRun(name=tracker_name, output_dir=output_dir, resume_frame=resume_frame)
    try:
        run.tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        run.output_video_path = output_dir / annotated_video_filename(resume_frame)
        run.writer = cv2.VideoWriter(str(run.output_video_path), fourcc, fps, frame_size)
        if not run.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {run.output_video_path}")

        tracks_path = output_dir / "tracks.csv"
        append_outputs = resume_frame > 0
        write_header = not tracks_path.is_file() or tracks_path.stat().st_size == 0 or not append_outputs
        run.csv_handle = tracks_path.open(
            "a" if append_outputs else "w",
            newline="",
            encoding="utf-8",
        )
        run.csv_writer = csv.DictWriter(run.csv_handle, fieldnames=TRACK_CSV_FIELDS)
        if write_header:
            run.csv_writer.writeheader()

        run.jsonl_handle = (output_dir / "frames.jsonl").open(
            "a" if append_outputs else "w",
            encoding="utf-8",
        )
    except Exception as exc:
        run.active = False
        run.error = f"{type(exc).__name__}: {exc}"
        (output_dir / "error.txt").write_text(run.error + "\n", encoding="utf-8")
        run.close()
    return run


def write_summary(
    run: TrackerRun,
    job: VideoJob,
    frame_width: int,
    frame_height: int,
    fps: float,
    total_source_frames: int,
    processed_frames: int,
    total_detection_boxes: int,
    elapsed_sec: float,
    resume_enabled: bool,
) -> None:
    summary = {
        "source_video": str(job.video_path),
        "modality": job.modality,
        "file_type": job.file_type,
        "tracker": run.name,
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "source_fps": round(float(fps), 4),
        "source_total_frames": int(total_source_frames),
        "processed_frames": int(processed_frames),
        "total_detection_boxes": int(total_detection_boxes),
        "total_track_boxes": int(run.total_track_boxes),
        "elapsed_sec": round(float(elapsed_sec), 4),
        "processing_fps": round(processed_frames / elapsed_sec, 4) if elapsed_sec > 0 else 0.0,
        "resume_enabled": bool(resume_enabled),
        "resume_from_frame": int(run.resume_frame),
        "annotated_video": str(run.output_video_path.name) if run.output_video_path else "",
        "error": run.error,
    }
    (run.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_track_history(run: TrackerRun, boxes: list[dict], trail_len: int = 30) -> None:
    for box in boxes:
        track_id = int(box["track_id"])
        cx = int(box["x"]) + int(box["w"]) // 2
        cy = int(box["y"]) + int(box["h"]) // 2
        run.track_history.setdefault(track_id, []).append((cx, cy))
        if len(run.track_history[track_id]) > trail_len:
            run.track_history[track_id].pop(0)


def process_video(
    job: VideoJob,
    output_root: Path,
    tracker_names: list[str],
    processor,
    max_frames: int = 0,
    progress_interval: int = 50,
    resume: bool = False,
) -> None:
    if not job.video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {job.video_path}")

    capture = cv2.VideoCapture(str(job.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {job.video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (frame_width, frame_height)

    runs: list[TrackerRun] = []
    for tracker_name in tracker_names:
        resume_frame = resolve_tracker_resume_frame(output_root, job.modality, tracker_name) if resume else 0
        output_dir = output_root / job.modality / tracker_name
        run = initialize_tracker_run(tracker_name, output_dir, fps, frame_size, resume_frame)
        runs.append(run)
        print(
            f"[{job.modality}] {tracker_name}: resume from frame {resume_frame}, "
            f"video={run.output_video_path if run.output_video_path else 'n/a'}",
            flush=True,
        )

    total_detection_boxes = 0
    processed_frames = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            processed_frames += 1
            if max_frames > 0 and processed_frames > max_frames:
                processed_frames -= 1
                break

            timestamp_sec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            stats = processor.process_frame(frame, job.file_type)
            raw_boxes = stats.get("boxes", []) if stats else []
            total_detection_boxes += len(raw_boxes)

            active_outputs = 0
            for run in runs:
                if not run.active or run.tracker is None:
                    continue
                should_write_outputs = processed_frames > run.resume_frame
                try:
                    tracked_boxes = run.tracker.update(raw_boxes, frame.shape, frame=frame)
                    run.total_track_boxes += len(tracked_boxes) if should_write_outputs else 0
                    update_track_history(run, tracked_boxes)

                    if should_write_outputs and run.csv_writer is not None and run.jsonl_handle is not None:
                        for box in tracked_boxes:
                            write_tracks_csv_row(run.csv_writer, processed_frames, box)
                        record = build_frame_record(
                            frame_index=processed_frames,
                            timestamp_sec=timestamp_sec,
                            boxes=[normalize_box(box) for box in tracked_boxes],
                        )
                        run.jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        active_outputs += len(tracked_boxes)

                    if should_write_outputs and run.writer is not None:
                        result = make_detection_result(job.file_type, tracked_boxes)
                        annotated = vis.visualize_detections(
                            frame,
                            "video",
                            result,
                            frame_num=processed_frames,
                            track_history=run.track_history,
                        )
                        run.writer.write(annotated)
                except Exception as exc:
                    run.active = False
                    run.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                    (run.output_dir / "error.txt").write_text(run.error + "\n", encoding="utf-8")
                    run.close()

            if progress_interval > 0 and processed_frames % progress_interval == 0:
                elapsed = time.time() - start_time
                speed = processed_frames / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{job.modality}] frame {processed_frames}/{total_source_frames} "
                    f"detections={len(raw_boxes)} active_outputs={active_outputs} speed={speed:.2f} fps",
                    flush=True,
                )
    finally:
        elapsed_sec = time.time() - start_time
        capture.release()
        for run in runs:
            run.close()
            write_summary(
                run=run,
                job=job,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=fps,
                total_source_frames=total_source_frames,
                processed_frames=processed_frames,
                total_detection_boxes=total_detection_boxes,
                elapsed_sec=elapsed_sec,
                resume_enabled=resume,
            )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    tracker_names = resolve_trackers(args.trackers)
    jobs = build_video_jobs(args.ir_video, args.rgb_video, args.modalities)

    print("[INFO] Initializing detector...", flush=True)
    detector = td.get_detector()
    processor = detector.processor
    if processor is None:
        raise RuntimeError("TargetDetector processor is not available")

    for job in jobs:
        print(f"[INFO] Processing {job.modality}: {job.video_path}", flush=True)
        process_video(
            job=job,
            output_root=output_root,
            tracker_names=tracker_names,
            processor=processor,
            max_frames=args.max_frames,
            progress_interval=args.progress_interval,
            resume=args.resume,
        )

    print(f"[OK] Outputs written to: {output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
