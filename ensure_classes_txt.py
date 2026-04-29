import argparse
from pathlib import Path

from target_module.image_detect_module.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure classes.txt exists in every label directory.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing labels/")
    return parser.parse_args()


def write_classes_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for class_name in Config.CLASSES:
            handle.write(class_name + "\n")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    label_root = dataset_root / "labels"
    written = 0

    for session_dir in sorted(path for path in label_root.rglob("*") if path.is_dir()):
        txt_files = [
            child
            for child in session_dir.iterdir()
            if child.is_file() and child.suffix.lower() == ".txt" and child.name != "classes.txt"
        ]
        if not txt_files:
            continue
        classes_path = session_dir / "classes.txt"
        write_classes_file(classes_path)
        written += 1

    print(f"dataset_root={dataset_root}")
    print(f"classes_files_written={written}")


if __name__ == "__main__":
    main()
