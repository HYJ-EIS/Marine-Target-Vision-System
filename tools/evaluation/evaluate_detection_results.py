"""Evaluate MOT-style detection result rows against MOT GT boxes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def _read_rows(path: str | Path) -> dict[int, list[dict]]:
    rows_by_frame: dict[int, list[dict]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 7:
                continue
            frame_id = int(float(row[0]))
            box = {
                "x": float(row[2]),
                "y": float(row[3]),
                "w": float(row[4]),
                "h": float(row[5]),
                "confidence": float(row[6]),
            }
            rows_by_frame.setdefault(frame_id, []).append(box)
    return rows_by_frame


def _area(box: dict) -> float:
    return max(0.0, float(box["w"])) * max(0.0, float(box["h"]))


def _iou(box_a: dict, box_b: dict) -> float:
    ax1, ay1 = float(box_a["x"]), float(box_a["y"])
    ax2, ay2 = ax1 + float(box_a["w"]), ay1 + float(box_a["h"])
    bx1, by1 = float(box_b["x"]), float(box_b["y"])
    bx2, by2 = bx1 + float(box_b["w"]), by1 + float(box_b["h"])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(box_a) + _area(box_b) - inter
    return 0.0 if union <= 0 else inter / union


def _round(value: float) -> float:
    return round(float(value), 6)


def evaluate_detection_results(
    gt_file: str | Path,
    det_file: str | Path,
    iou_thresh: float = 0.5,
    small_area_thresh: float = 1024.0,
) -> dict:
    """Greedily match detections to GT per frame and return detection metrics."""
    gt_by_frame = _read_rows(gt_file)
    det_by_frame = _read_rows(det_file)
    frames = sorted(set(gt_by_frame) | set(det_by_frame))
    tp = fp = fn = 0
    small_gt = small_tp = 0
    conf_sum = 0.0
    det_count = 0

    for frame_id in frames:
        gt_boxes = gt_by_frame.get(frame_id, [])
        det_boxes = sorted(det_by_frame.get(frame_id, []), key=lambda box: box["confidence"], reverse=True)
        matched_gt: set[int] = set()
        small_gt_indices = {idx for idx, gt in enumerate(gt_boxes) if _area(gt) <= small_area_thresh}
        small_gt += len(small_gt_indices)

        for det in det_boxes:
            det_count += 1
            conf_sum += float(det["confidence"])
            best_idx = None
            best_iou = 0.0
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = _iou(det, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_idx)
                if best_idx in small_gt_indices:
                    small_tp += 1
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "frames": len(frames),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "fp_per_frame": _round(fp / max(len(frames), 1)),
        "small_gt": small_gt,
        "small_tp": small_tp,
        "small_recall": _round(small_tp / max(small_gt, 1)),
        "mean_confidence": _round(conf_sum / max(det_count, 1)),
        "detections": det_count,
        "gt_detections": sum(len(v) for v in gt_by_frame.values()),
        "iou_thresh": float(iou_thresh),
        "small_area_thresh": float(small_area_thresh),
    }


def write_summary_csv(summaries: Iterable[dict], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(summaries)
    fields = [
        "name",
        "frames",
        "detections",
        "gt_detections",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "fp_per_frame",
        "small_gt",
        "small_tp",
        "small_recall",
        "mean_confidence",
        "iou_thresh",
        "small_area_thresh",
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MOT-style detection result rows")
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--det-file", action="append", required=True,
                        help="Name=path or plain path. Can be passed multiple times.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--small-area-thresh", type=float, default=1024.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for item in args.det_file:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem
        summary = evaluate_detection_results(
            args.gt_file,
            path,
            iou_thresh=args.iou_thresh,
            small_area_thresh=args.small_area_thresh,
        )
        summary["name"] = name
        summaries.append(summary)
        print(
            f"{name}: precision={summary['precision']}, recall={summary['recall']}, "
            f"fp={summary['fp']}, fn={summary['fn']}, fp_per_frame={summary['fp_per_frame']}, "
            f"small_recall={summary['small_recall']}"
        )
    csv_path = write_summary_csv(summaries, args.output_csv)
    print(f"[OK] detection summary: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
