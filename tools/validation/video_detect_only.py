"""
Run ONNX detection on local videos and save annotated MP4 outputs.

This script is intentionally detection-only: it does not initialize a tracker
and does not write track IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
import visualization as vis


DETECTION_CSV_FIELDS = [
    "frame_index",
    "timestamp_sec",
    "detection_id",
    "x",
    "y",
    "w",
    "h",
    "confidence",
    "class",
    "class_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection-only annotation on local videos")
    parser.add_argument("--input", action="append", required=True, help="Input video path; repeat for multiple videos")
    parser.add_argument("--output-dir", default="results/detection_only", help="Directory for annotated videos")
    parser.add_argument(
        "--file-type",
        default="visible",
        choices=["visible", "infrared"],
        help="Detector type to use for every input video",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    parser.add_argument("--progress-interval", type=int, default=100, help="Progress print interval")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Override detector confidence threshold; omit to use Config threshold",
    )
    return parser.parse_args()


def make_detection_result(file_type: str, stats: dict | None) -> dict:
    boxes = stats.get("boxes", []) if stats else []
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


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_detected.mp4"


def normalize_box(box: dict) -> dict:
    confidence = float(box.get("confidence", 0.0))
    return {
        "id": int(box.get("id", 0)),
        "x": int(box.get("x", 0)),
        "y": int(box.get("y", 0)),
        "w": int(box.get("w", 0)),
        "h": int(box.get("h", 0)),
        "confidence": round(confidence, 4),
        "class": str(box.get("class", "unknown")),
        "class_confidence": round(float(box.get("class_confidence", confidence)), 4),
    }


def contiguous_ranges(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def write_video_analysis(output_dir: Path, input_path: Path, summary: dict,
                         frame_detection_counts: dict[int, int],
                         class_counts: Counter,
                         confidences: list[float],
                         areas: list[int]) -> Path:
    processed_frames = int(summary["processed_frames"])
    detected_frames = sorted(k for k, v in frame_detection_counts.items() if v > 0)
    spans = contiguous_ranges(detected_frames)
    empty_frames = processed_frames - len(detected_frames)

    def mean_or_zero(values: list[float | int]) -> float:
        return round(float(statistics.mean(values)), 4) if values else 0.0

    def median_or_zero(values: list[float | int]) -> float:
        return round(float(statistics.median(values)), 4) if values else 0.0

    report_path = output_dir / f"{input_path.stem}_analysis.md"
    class_lines = [
        f"- `{name}`: {count}"
        for name, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
    ] or ["- none"]
    span_lines = [
        f"- frames {start}-{end} ({end - start + 1} frames)"
        for start, end in spans[:20]
    ] or ["- none"]
    if len(spans) > 20:
        span_lines.append(f"- ... {len(spans) - 20} more spans")

    lines = [
        f"# Detection Analysis: {input_path.name}",
        "",
        "## Inputs",
        "",
        f"- Source video: `{summary['source_video']}`",
        f"- Output video: `{summary['output_video']}`",
        f"- Detector type: `{summary['file_type']}`",
        f"- Resolution: {summary['frame_width']}x{summary['frame_height']}",
        f"- Source FPS: {summary['source_fps']}",
        "",
        "## Frame Coverage",
        "",
        f"- Processed frames: {processed_frames}",
        f"- Frames with detections: {len(detected_frames)}",
        f"- Empty frames: {empty_frames}",
        f"- Detection coverage: {round(len(detected_frames) / processed_frames * 100, 2) if processed_frames else 0.0}%",
        f"- Total detection boxes: {summary['total_detection_boxes']}",
        f"- Average boxes/frame: {round(summary['total_detection_boxes'] / processed_frames, 4) if processed_frames else 0.0}",
        "",
        "## Class Counts",
        "",
        *class_lines,
        "",
        "## Confidence",
        "",
        f"- Min: {round(min(confidences), 4) if confidences else 0.0}",
        f"- Mean: {mean_or_zero(confidences)}",
        f"- Median: {median_or_zero(confidences)}",
        f"- Max: {round(max(confidences), 4) if confidences else 0.0}",
        "",
        "## Box Area",
        "",
        f"- Min pixels: {min(areas) if areas else 0}",
        f"- Mean pixels: {mean_or_zero(areas)}",
        f"- Median pixels: {median_or_zero(areas)}",
        f"- Max pixels: {max(areas) if areas else 0}",
        "",
        "## Detection Frame Spans",
        "",
        *span_lines,
        "",
        "## Interpretation",
        "",
        "- These are detector outputs only; no tracker or temporal smoothing was applied.",
        "- Low frame coverage means the current RGB model only finds the target intermittently in this video.",
        "- Small box areas indicate far/small targets, where confidence and localization usually fluctuate more.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_combined_analysis(output_dir: Path, summaries: list[dict]) -> Path:
    report_path = output_dir / "detection_analysis.md"
    lines = [
        "# Detection-Only Batch Analysis",
        "",
        "| Video | Frames | Detection boxes | Boxes/frame | Processing FPS |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        frames = int(summary["processed_frames"])
        boxes = int(summary["total_detection_boxes"])
        lines.append(
            f"| {Path(summary['source_video']).name} | {frames} | {boxes} | "
            f"{round(boxes / frames, 4) if frames else 0.0} | {summary['processing_fps']} |"
        )
    lines.extend([
        "",
        "Per-frame outputs:",
        "",
    ])
    for summary in summaries:
        stem = Path(summary["source_video"]).stem
        lines.extend([
            f"- `{stem}_detections.csv`",
            f"- `{stem}_frames.jsonl`",
            f"- `{stem}_analysis.md`",
        ])
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def process_video(
    input_path: Path,
    output_dir: Path,
    file_type: str,
    processor,
    max_frames: int = 0,
    progress_interval: int = 100,
    conf_threshold: float | None = None,
) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {input_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_path_for(input_path, output_dir)
    detections_csv_path = output_dir / f"{input_path.stem}_detections.csv"
    frames_jsonl_path = output_dir / f"{input_path.stem}_frames.jsonl"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video writer: {output_video}")

    processed_frames = 0
    total_detection_boxes = 0
    frame_detection_counts: dict[int, int] = {}
    class_counts: Counter = Counter()
    confidences: list[float] = []
    areas: list[int] = []
    start_time = time.time()
    try:
        with detections_csv_path.open("w", newline="", encoding="utf-8") as csv_handle, \
                frames_jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
            csv_writer = csv.DictWriter(csv_handle, fieldnames=DETECTION_CSV_FIELDS)
            csv_writer.writeheader()

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                processed_frames += 1
                if max_frames > 0 and processed_frames > max_frames:
                    processed_frames -= 1
                    break

                timestamp_sec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                if timestamp_sec <= 0:
                    timestamp_sec = (processed_frames - 1) / fps if fps > 0 else 0.0

                stats = processor.process_frame(frame, file_type, conf_override=conf_threshold)
                boxes = [normalize_box(box) for box in (stats.get("boxes", []) if stats else [])]
                total_detection_boxes += len(boxes)
                frame_detection_counts[processed_frames] = len(boxes)

                for box in boxes:
                    confidence = float(box["confidence"])
                    class_counts[str(box["class"])] += 1
                    confidences.append(confidence)
                    areas.append(int(box["w"]) * int(box["h"]))
                    csv_writer.writerow({
                        "frame_index": processed_frames,
                        "timestamp_sec": f"{timestamp_sec:.4f}",
                        "detection_id": int(box["id"]),
                        "x": int(box["x"]),
                        "y": int(box["y"]),
                        "w": int(box["w"]),
                        "h": int(box["h"]),
                        "confidence": f"{confidence:.4f}",
                        "class": str(box["class"]),
                        "class_confidence": f"{float(box['class_confidence']):.4f}",
                    })

                jsonl_handle.write(json.dumps({
                    "frame_index": processed_frames,
                    "timestamp_sec": round(timestamp_sec, 4),
                    "boxes": boxes,
                }, ensure_ascii=False) + "\n")

                annotated = vis.visualize_detections(
                    frame,
                    "image",
                    make_detection_result(file_type, {"boxes": boxes}),
                )
                writer.write(annotated)

                if progress_interval > 0 and processed_frames % progress_interval == 0:
                    elapsed = time.time() - start_time
                    speed = processed_frames / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{input_path.name}] frame {processed_frames}/{total_source_frames} "
                        f"detections={len(boxes)} speed={speed:.2f} fps",
                        flush=True,
                    )
    finally:
        capture.release()
        writer.release()

    elapsed_sec = time.time() - start_time
    summary = {
        "source_video": str(input_path),
        "output_video": str(output_video),
        "detections_csv": str(detections_csv_path),
        "frames_jsonl": str(frames_jsonl_path),
        "file_type": file_type,
        "confidence_threshold": conf_threshold,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "source_fps": round(fps, 4),
        "source_total_frames": total_source_frames,
        "processed_frames": processed_frames,
        "total_detection_boxes": total_detection_boxes,
        "elapsed_sec": round(elapsed_sec, 4),
        "processing_fps": round(processed_frames / elapsed_sec, 4) if elapsed_sec > 0 else 0.0,
    }
    analysis_path = write_video_analysis(
        output_dir=output_dir,
        input_path=input_path,
        summary=summary,
        frame_detection_counts=frame_detection_counts,
        class_counts=class_counts,
        confidences=confidences,
        areas=areas,
    )
    summary["analysis_report"] = str(analysis_path)
    summary_path = output_dir / f"{input_path.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("[INFO] Initializing detector...", flush=True)
    detector = td.get_detector()
    if detector.processor is None:
        raise RuntimeError("TargetDetector processor is not available")

    summaries = []
    for raw_input in args.input:
        input_path = Path(raw_input)
        print(f"[INFO] Processing {input_path}", flush=True)
        summary = process_video(
            input_path=input_path,
            output_dir=output_dir,
            file_type=args.file_type,
            processor=detector.processor,
            max_frames=args.max_frames,
            progress_interval=args.progress_interval,
            conf_threshold=args.conf_threshold,
        )
        summaries.append(summary)
        print(f"[OK] {summary['output_video']}", flush=True)

    report_path = write_combined_analysis(output_dir, summaries)
    print(f"[OK] {report_path}", flush=True)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
