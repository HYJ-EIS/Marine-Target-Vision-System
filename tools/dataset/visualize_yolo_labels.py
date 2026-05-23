import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv_utils import imread_unicode, imwrite_unicode
from target_module.image_detect_module.config import Config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO labels for an extracted frame dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing images/ and labels/ folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for visualization images. Defaults to <dataset-root>_visualized.",
    )
    parser.add_argument(
        "--draw-empty-note",
        action="store_true",
        help="Draw a small 'NO_LABEL' note on images whose label file is empty.",
    )
    return parser.parse_args()


def yolo_to_xyxy(parts: list[str], width: int, height: int) -> tuple[int, int, int, int, int] | None:
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        cx = float(parts[1]) * width
        cy = float(parts[2]) * height
        box_w = float(parts[3]) * width
        box_h = float(parts[4]) * height
    except ValueError:
        return None

    x1 = max(0, min(width - 1, int(round(cx - box_w / 2))))
    y1 = max(0, min(height - 1, int(round(cy - box_h / 2))))
    x2 = max(0, min(width - 1, int(round(cx + box_w / 2))))
    y2 = max(0, min(height - 1, int(round(cy + box_h / 2))))
    return class_id, x1, y1, x2, y2


def load_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    if not label_path.exists():
        return []

    boxes: list[tuple[int, int, int, int, int]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            box = yolo_to_xyxy(line.split(), width, height)
            if box is not None:
                boxes.append(box)
    return boxes


def draw_boxes(image, boxes: list[tuple[int, int, int, int, int]], draw_empty_note: bool):
    vis = image.copy()
    height, width = vis.shape[:2]
    font_scale = max(0.5, min(width, height) / 900.0)
    thickness = max(2, int(round(min(width, height) / 320.0)))

    if not boxes and draw_empty_note:
        cv2.putText(
            vis,
            "NO_LABEL",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    for class_id, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        class_name = Config.CLASSES[class_id] if 0 <= class_id < len(Config.CLASSES) else f"class_{class_id}"
        label = f"{class_id}:{class_name}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_y1 = max(0, y1 - text_h - baseline - 6)
        text_y2 = text_y1 + text_h + baseline + 6
        text_x2 = min(width - 1, x1 + text_w + 10)
        cv2.rectangle(vis, (x1, text_y1), (text_x2, text_y2), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 5, text_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    return vis


def resolve_label_path(dataset_root: Path, image_path: Path) -> Path:
    relative_path = image_path.relative_to(dataset_root / "images")
    return dataset_root / "labels" / relative_path.with_suffix(".txt")


def visualize_dataset(dataset_root: Path, output_root: Path, draw_empty_note: bool) -> tuple[int, int]:
    image_root = dataset_root / "images"
    output_image_root = output_root / "images"
    output_image_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    processed = 0
    labeled = 0
    for image_path in image_paths:
        image = imread_unicode(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = resolve_label_path(dataset_root, image_path)
        boxes = load_yolo_boxes(label_path, width, height)
        vis = draw_boxes(image, boxes, draw_empty_note)

        relative_path = image_path.relative_to(image_root)
        output_path = output_image_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(str(output_path), vis)

        processed += 1
        if boxes:
            labeled += 1

    return processed, labeled


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else dataset_root.with_name(f"{dataset_root.name}_visualized")

    processed, labeled = visualize_dataset(dataset_root, output_root, args.draw_empty_note)
    print(f"dataset_root={dataset_root}")
    print(f"output_root={output_root}")
    print(f"processed_images={processed}")
    print(f"images_with_boxes={labeled}")
    print(f"images_without_boxes={processed - labeled}")


if __name__ == "__main__":
    main()
