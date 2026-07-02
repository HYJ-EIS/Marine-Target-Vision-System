"""Evaluate MOT metrics on concatenated short-miss slices."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import METRIC_FIELDS  # noqa: E402
from tools.evaluation.motchallenge_eval import run_motchallenge_eval  # noqa: E402

SLICE_RESULT_FIELDS = ["seq_name", "tracker", "num_slices", "total_slice_frames", *METRIC_FIELDS]
EVAL_SUMMARY_FIELDS = ["seq_name", "tracker", *METRIC_FIELDS]


def _frame_bounds(slice_row: dict[str, str]) -> tuple[int, int]:
    start_frame = int(slice_row["start_frame"])
    end_frame = int(slice_row["end_frame"])
    if end_frame < start_frame:
        raise ValueError(f"Invalid slice frame range: {start_frame}-{end_frame}")
    return start_frame, end_frame


def _mot_frame(line: str) -> int:
    first_col = line.split(",", 1)[0].strip()
    return int(first_col)


def frame_in_any_slice(frame_id: int, slices: list[dict[str, str]]) -> bool:
    """Return whether a frame lies in any inclusive slice interval."""
    for slice_row in slices:
        start_frame, end_frame = _frame_bounds(slice_row)
        if start_frame <= int(frame_id) <= end_frame:
            return True
    return False


def reindex_mot_line(line: str, slice_start: int) -> str:
    """Reindex a MOT line so ``slice_start`` becomes synthetic frame 1."""
    stripped = line.strip()
    if not stripped:
        return ""
    columns = stripped.split(",")
    columns[0] = str(int(columns[0]) - int(slice_start) + 1)
    return ",".join(columns)


def _reindex_mot_line_with_offset(line: str, slice_start: int, timeline_offset: int) -> str:
    columns = reindex_mot_line(line, slice_start).split(",")
    columns[0] = str(int(columns[0]) + int(timeline_offset))
    return ",".join(columns)


def read_csv(path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_slice_metric_rows(output_csv, rows) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SLICE_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def _write_eval_summary_rows(output_csv: str | Path, rows: list[dict]) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def _read_mot_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing MOT file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _clip_and_reindex_lines(source_lines: Iterable[str], slices: list[dict[str, str]]) -> list[str]:
    clipped: list[str] = []
    timeline_offset = 0
    for slice_row in slices:
        start_frame, end_frame = _frame_bounds(slice_row)
        for line in source_lines:
            frame_id = _mot_frame(line)
            if start_frame <= frame_id <= end_frame:
                clipped.append(_reindex_mot_line_with_offset(line, start_frame, timeline_offset))
        timeline_offset += end_frame - start_frame + 1
    return clipped


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text((text + "\n") if text else "", encoding="utf-8")


def _write_seqinfo(path: Path, seq_name: str, seq_length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "[Sequence]",
            f"name={seq_name}",
            "imDir=img1",
            "frameRate=25",
            f"seqLength={int(seq_length)}",
            "imWidth=1920",
            "imHeight=1080",
            "imExt=.jpg",
            "",
        ]),
        encoding="utf-8",
    )


def _group_manifest_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        seq_name = str(row.get("seq_name", "")).strip()
        tracker = str(row.get("tracker", "")).strip()
        if not seq_name or not tracker:
            raise ValueError(f"Slice manifest row must include seq_name and tracker: {row}")
        _frame_bounds(row)
        grouped[(seq_name, tracker)].append(row)
    return dict(grouped)


def _total_slice_frames(slices: list[dict[str, str]]) -> int:
    total = 0
    for slice_row in slices:
        start_frame, end_frame = _frame_bounds(slice_row)
        total += end_frame - start_frame + 1
    return total


def build_clipped_mot_datasets(
    slice_manifest: str | Path,
    ablation_root: str | Path,
    output_root: str | Path,
) -> list[dict[str, str | int]]:
    """Build synthetic MOT datasets for all ``(seq_name, tracker)`` slice groups."""
    ablation_root = Path(ablation_root)
    output_root = Path(output_root)
    gt_root = output_root / "mot_gt"
    trackers_root = output_root / "trackers"
    groups = _group_manifest_rows(read_csv(slice_manifest))

    dataset_rows: list[dict[str, str | int]] = []
    for (seq_name, tracker), slices in groups.items():
        synthetic_seq = f"{seq_name}__{tracker}__short_miss"
        total_frames = _total_slice_frames(slices)
        source_gt = ablation_root / "mot_gt" / seq_name / "gt" / "gt.txt"
        source_pred = ablation_root / "trackers" / tracker / "data" / f"{seq_name}.txt"

        gt_lines = _clip_and_reindex_lines(_read_mot_lines(source_gt), slices)
        pred_lines = _clip_and_reindex_lines(_read_mot_lines(source_pred), slices)
        _write_lines(gt_root / synthetic_seq / "gt" / "gt.txt", gt_lines)
        _write_lines(trackers_root / tracker / "data" / f"{synthetic_seq}.txt", pred_lines)
        _write_seqinfo(gt_root / synthetic_seq / "seqinfo.ini", synthetic_seq, total_frames)
        dataset_rows.append({
            "seq_name": seq_name,
            "tracker": tracker,
            "synthetic_seq": synthetic_seq,
            "num_slices": len(slices),
            "total_slice_frames": total_frames,
        })
    return dataset_rows


def build_slice_metrics(
    slice_manifest: str | Path,
    ablation_root: str | Path,
    output_root: str | Path,
) -> Path:
    output_root = Path(output_root)
    dataset_rows = build_clipped_mot_datasets(
        slice_manifest=slice_manifest,
        ablation_root=ablation_root,
        output_root=output_root,
    )
    if not dataset_rows:
        _write_eval_summary_rows(output_root / "eval" / "motchallenge_summary.csv", [])
        return write_slice_metric_rows(output_root / "slice_metrics.csv", [])

    metric_rows: list[dict[str, str | int | float]] = []
    for dataset_row in dataset_rows:
        tracker = str(dataset_row["tracker"])
        synthetic_seq = str(dataset_row["synthetic_seq"])
        summary = run_motchallenge_eval(
            gt_root=output_root / "mot_gt",
            trackers_root=output_root / "trackers",
            output_root=output_root / "eval" / synthetic_seq,
            trackers=[tracker],
            sequences=[synthetic_seq],
            print_results=False,
        )
        metrics = summary.get(tracker, {})
        metric_rows.append({
            "seq_name": dataset_row["seq_name"],
            "tracker": tracker,
            "num_slices": dataset_row["num_slices"],
            "total_slice_frames": dataset_row["total_slice_frames"],
            **{field: metrics.get(field, "") for field in METRIC_FIELDS},
        })
    _write_eval_summary_rows(output_root / "eval" / "motchallenge_summary.csv", metric_rows)
    return write_slice_metric_rows(output_root / "slice_metrics.csv", metric_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MOT metrics on short-miss slice windows")
    parser.add_argument("--slice-manifest", required=True, help="CSV produced by msdc_slice_eval.py")
    parser.add_argument("--ablation-root", required=True, help="Ablation run root containing mot_gt/ and trackers/")
    parser.add_argument("--output-root", required=True, help="Output root for clipped MOT data and slice metrics")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_csv = build_slice_metrics(
        slice_manifest=args.slice_manifest,
        ablation_root=args.ablation_root,
        output_root=args.output_root,
    )
    print(f"[OK] slice metrics: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
