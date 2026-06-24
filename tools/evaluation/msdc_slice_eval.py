"""Build MS-DC short-miss slice manifests from per-GT diagnostics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SLICE_FIELDS = ["seq_name", "tracker", "gt_id", "start_frame", "end_frame", "length", "slice_type"]


def _parse_segments(text: str | None) -> list[tuple[int, int]]:
    """Parse semicolon-separated frame segments like ``10-12;30``."""
    segments: list[tuple[int, int]] = []
    for raw_part in str(text or "").split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
        else:
            start_text = end_text = part
        try:
            start_frame = int(start_text.strip())
            end_frame = int(end_text.strip())
        except ValueError:
            continue
        if end_frame < start_frame:
            continue
        segments.append((start_frame, end_frame))
    return segments


def find_short_missing_segments(
    rows: list[dict[str, str]] | list[dict],
    min_len: int = 1,
    max_len: int = 30,
) -> list[dict[str, int | str]]:
    """Return short missing-frame slices from per-GT diagnostic rows."""
    slices: list[dict[str, int | str]] = []
    for row in rows:
        gt_id = row.get("gt_id", "")
        for start_frame, end_frame in _parse_segments(row.get("missed_segments", "")):
            length = end_frame - start_frame + 1
            if int(min_len) <= length <= int(max_len):
                slices.append({
                    "gt_id": str(gt_id),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "length": int(length),
                    "slice_type": "short_miss",
                })
    return slices


def read_per_gt_diagnostics(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_slice_manifest(output_path: str | Path, rows: list[dict]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SLICE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def build_slice_manifest_from_diagnostics(
    diagnostics_root: str | Path,
    output_path: str | Path,
    max_len: int = 30,
) -> Path:
    diagnostics_root = Path(diagnostics_root)
    if not diagnostics_root.is_dir():
        raise FileNotFoundError(f"Diagnostics root does not exist or is not a directory: {diagnostics_root}")
    per_gt_paths = sorted(diagnostics_root.glob("*/*_per_gt_diagnostics.csv"))
    if not per_gt_paths:
        raise FileNotFoundError(f"No per-GT diagnostics CSV files found under: {diagnostics_root}")

    manifest_rows: list[dict[str, int | str]] = []
    for csv_path in per_gt_paths:
        seq_name = csv_path.parent.name
        suffix = "_per_gt_diagnostics.csv"
        tracker = csv_path.name[: -len(suffix)] if csv_path.name.endswith(suffix) else csv_path.stem
        for row in find_short_missing_segments(read_per_gt_diagnostics(csv_path), max_len=max_len):
            manifest_rows.append({
                "seq_name": seq_name,
                "tracker": tracker,
                **row,
            })
    return write_slice_manifest(output_path, manifest_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build short-miss slice manifest from per-GT diagnostics")
    parser.add_argument("--diagnostics-root", required=True, help="Root containing seq/*_per_gt_diagnostics.csv files")
    parser.add_argument("--output", required=True, help="Output slice manifest CSV")
    parser.add_argument("--max-len", type=int, default=30, help="Maximum inclusive short miss length")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = build_slice_manifest_from_diagnostics(
        diagnostics_root=args.diagnostics_root,
        output_path=args.output,
        max_len=int(args.max_len),
    )
    print(f"[OK] slice manifest: {output_path.resolve()}")


if __name__ == "__main__":
    main()
