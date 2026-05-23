import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv_utils import imread_unicode
from tools.dataset.extract_tracking_frames import (
    ExportRecord,
    encode_frame_payload,
    prune_representative_records,
)
from target_module.image_detect_module.config import Config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FRAME_NAME_RE = re.compile(r"^(?P<video>.+?)_f(?P<frame>\d+)_t(?P<timestamp>-?\d+(?:\.\d+)?)s$")
SUMMARY_FIELDS = ["group_id", "before_count", "after_count", "reduction_ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter already extracted LabelMe frames with current representative-frame logic.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def pruning_args() -> SimpleNamespace:
    return SimpleNamespace(
        representative_global_phash_threshold=Config.EXTERNAL_FRAMES_REPRESENTATIVE_GLOBAL_HASH_THRESHOLD,
        representative_target_phash_threshold=Config.EXTERNAL_FRAMES_REPRESENTATIVE_TARGET_HASH_THRESHOLD,
        representative_motion_threshold=Config.EXTERNAL_FRAMES_REPRESENTATIVE_MOTION_THRESHOLD,
        representative_area_threshold=Config.EXTERNAL_FRAMES_REPRESENTATIVE_AREA_THRESHOLD,
        representative_max_cluster_span_sec=Config.EXTERNAL_FRAMES_REPRESENTATIVE_MAX_CLUSTER_SPAN_SEC,
        representative_min_time_gap_sec=Config.EXTERNAL_FRAMES_REPRESENTATIVE_MIN_TIME_GAP_SEC,
    )


def frame_key(path: Path) -> tuple[str, int, float]:
    match = FRAME_NAME_RE.match(path.stem)
    if not match:
        return path.stem, 0, 0.0
    return (
        match.group("video"),
        int(match.group("frame")),
        float(match.group("timestamp")),
    )


def json_path_for(image_path: Path, input_dir: Path) -> Path | None:
    root_json = image_path.with_suffix(".json")
    if root_json.exists():
        return root_json
    nested_json = input_dir / "json" / f"{image_path.stem}.json"
    if nested_json.exists():
        return nested_json
    return None


def parse_labelme_boxes(json_path: Path | None) -> list[dict]:
    if json_path is None:
        return []

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    boxes: list[dict] = []
    for shape in payload.get("shapes", []):
        points = shape.get("points") or []
        if len(points) < 2:
            continue

        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append(
            {
                "class": str(shape.get("label") or "unknown"),
                "x": x1,
                "y": y1,
                "w": max(1.0, x2 - x1),
                "h": max(1.0, y2 - y1),
            }
        )
    return boxes


def build_records(input_dir: Path) -> tuple[dict[str, dict[str, ExportRecord]], dict[str, Path | None]]:
    grouped: dict[str, dict[str, ExportRecord]] = {}
    json_paths: dict[str, Path | None] = {}

    image_paths = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=frame_key,
    )
    for image_path in image_paths:
        group_id, frame_idx, timestamp_sec = frame_key(image_path)
        frame = imread_unicode(str(image_path))
        if frame is None:
            raise RuntimeError(f"Unable to read image: {image_path}")

        label_path = json_path_for(image_path, input_dir)
        record = ExportRecord(
            frame_id=image_path.stem,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            frame=encode_frame_payload(frame),
            boxes=parse_labelme_boxes(label_path),
            session_id=group_id,
            modality="rgb",
            source_video=group_id,
            frame_path=str(image_path),
            label_path=str(label_path) if label_path is not None else "",
            keep_reasons=set(),
        )
        grouped.setdefault(group_id, {})[record.frame_id] = record
        json_paths[record.frame_id] = label_path

    return grouped, json_paths


def copy_selected(
    selected_records: list[ExportRecord],
    json_paths: dict[str, Path | None],
    input_dir: Path,
    output_dir: Path,
) -> None:
    output_json_dir = output_dir / "json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    for record in selected_records:
        src_image = Path(record.frame_path)
        dst_image = output_dir / src_image.name
        shutil.copy2(src_image, dst_image)

        src_json = json_paths.get(record.frame_id)
        if src_json is not None and src_json.exists():
            shutil.copy2(src_json, output_dir / src_json.name)

        nested_json = input_dir / "json" / f"{record.frame_id}.json"
        if nested_json.exists():
            shutil.copy2(nested_json, output_json_dir / nested_json.name)


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    path = output_dir / "filter_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    grouped, json_paths = build_records(input_dir)
    selected_records: list[ExportRecord] = []
    summary_rows: list[dict] = []
    args_for_pruning = pruning_args()

    for group_id in sorted(grouped):
        records = grouped[group_id]
        pruned, _ = prune_representative_records(records, args_for_pruning)
        selected = [pruned[frame_id] for frame_id in sorted(pruned, key=lambda item: pruned[item].timestamp_sec)]
        selected_records.extend(selected)
        summary_rows.append(
            {
                "group_id": group_id,
                "before_count": len(records),
                "after_count": len(selected),
                "reduction_ratio": f"{(1.0 - len(selected) / max(len(records), 1)):.3f}",
            }
        )

    copy_selected(selected_records, json_paths, input_dir, output_dir)
    write_summary(output_dir, summary_rows)

    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"groups={len(grouped)}")
    print(f"before_frames={sum(len(records) for records in grouped.values())}")
    print(f"after_frames={len(selected_records)}")
    print(f"reduction_ratio={(1.0 - len(selected_records) / max(sum(len(records) for records in grouped.values()), 1)):.3f}")


if __name__ == "__main__":
    main()
