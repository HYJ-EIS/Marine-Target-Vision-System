"""
Run formal MOTChallenge-style tracking evaluation with vendored TrackEval.

Expected input layout:

GT root:
    <gt-root>/<seq>/seqinfo.ini
    <gt-root>/<seq>/gt/gt.txt

Tracker root:
    <trackers-root>/<tracker-name>/data/<seq>.txt

The GT must contain real cross-frame identity IDs. Tracker-only output cannot
produce formal IDF1/HOTA/MOTA scores.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

# TrackEval still uses removed NumPy aliases in some code paths.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

import third_party.trackeval as trackeval


def _discover_sequences(gt_root: Path) -> list[str]:
    seqs = []
    for child in sorted(gt_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "seqinfo.ini").is_file() and (child / "gt" / "gt.txt").is_file():
            seqs.append(child.name)
    return seqs


def _discover_trackers(trackers_root: Path) -> list[str]:
    trackers = []
    for child in sorted(trackers_root.iterdir()):
        if child.is_dir() and (child / "data").is_dir():
            trackers.append(child.name)
    return trackers


def _validate_inputs(gt_root: Path, trackers_root: Path,
                     trackers: list[str], sequences: list[str]) -> None:
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT root does not exist: {gt_root}")
    if not trackers_root.is_dir():
        raise FileNotFoundError(f"Tracker root does not exist: {trackers_root}")
    if not sequences:
        raise ValueError("No MOTChallenge sequences found. Need <seq>/seqinfo.ini and <seq>/gt/gt.txt.")
    if not trackers:
        raise ValueError("No tracker result folders found. Need <tracker>/data/<seq>.txt.")

    for seq in sequences:
        gt_file = gt_root / seq / "gt" / "gt.txt"
        seqinfo = gt_root / seq / "seqinfo.ini"
        if not seqinfo.is_file():
            raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo}")
        if not gt_file.is_file():
            raise FileNotFoundError(f"Missing gt.txt: {gt_file}")
    for tracker in trackers:
        for seq in sequences:
            tracker_file = trackers_root / tracker / "data" / f"{seq}.txt"
            if not tracker_file.is_file():
                raise FileNotFoundError(f"Missing tracker result file: {tracker_file}")


def _summary_from_trackeval(output_res: dict, trackers: list[str]) -> dict[str, dict[str, float]]:
    dataset_res = output_res["MotChallenge2DBox"]
    summary: dict[str, dict[str, float]] = {}
    for tracker in trackers:
        combined = dataset_res[tracker]["COMBINED_SEQ"]["pedestrian"]
        hota = float(np.mean(combined["HOTA"]["HOTA"])) * 100.0
        mota = float(combined["CLEAR"]["MOTA"]) * 100.0
        idf1 = float(combined["Identity"]["IDF1"]) * 100.0
        summary[tracker] = {
            "HOTA": round(hota, 5),
            "MOTA": round(mota, 5),
            "IDF1": round(idf1, 5),
            "IDSW": int(combined["CLEAR"]["IDSW"]),
            "FP": int(combined["CLEAR"]["CLR_FP"]),
            "FN": int(combined["CLEAR"]["CLR_FN"]),
            "IDTP": int(combined["Identity"]["IDTP"]),
            "IDFP": int(combined["Identity"]["IDFP"]),
            "IDFN": int(combined["Identity"]["IDFN"]),
        }
    return summary


def _write_summary_csv(summary: dict[str, dict[str, float]], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "motchallenge_summary.csv"
    fields = ["tracker", "HOTA", "MOTA", "IDF1", "IDSW", "FP", "FN", "IDTP", "IDFP", "IDFN"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tracker, values in summary.items():
            writer.writerow({"tracker": tracker, **values})
    return csv_path


def run_motchallenge_eval(
    gt_root: str | Path,
    trackers_root: str | Path,
    output_root: str | Path,
    trackers: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    print_results: bool = True,
) -> dict[str, dict[str, float]]:
    """Run official TrackEval HOTA/CLEAR/Identity metrics and return key scores."""
    gt_root = Path(gt_root)
    trackers_root = Path(trackers_root)
    output_root = Path(output_root)

    seq_list = list(sequences) if sequences else _discover_sequences(gt_root)
    tracker_list = list(trackers) if trackers else _discover_trackers(trackers_root)
    _validate_inputs(gt_root, trackers_root, tracker_list, seq_list)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "BREAK_ON_ERROR": True,
        "PRINT_RESULTS": print_results,
        "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "LOG_ON_ERROR": None,
    })

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_root),
        "TRACKERS_FOLDER": str(trackers_root),
        "OUTPUT_FOLDER": str(output_root),
        "TRACKERS_TO_EVAL": tracker_list,
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SEQ_INFO": {seq: None for seq in seq_list},
        "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "PRINT_CONFIG": False,
    })

    metric_config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(metric_config),
        trackeval.metrics.CLEAR(metric_config),
        trackeval.metrics.Identity(metric_config),
    ]
    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
    summary = _summary_from_trackeval(output_res, tracker_list)
    _write_summary_csv(summary, output_root)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal MOTChallenge IDF1/HOTA/MOTA evaluation")
    parser.add_argument("--gt-root", required=True, help="Root containing <seq>/gt/gt.txt and <seq>/seqinfo.ini")
    parser.add_argument("--trackers-root", required=True, help="Root containing <tracker>/data/<seq>.txt")
    parser.add_argument("--output-root", default="results/motchallenge_eval")
    parser.add_argument("--trackers", nargs="*", default=None, help="Tracker folder names to evaluate")
    parser.add_argument("--sequences", nargs="*", default=None, help="Sequence names to evaluate")
    parser.add_argument("--quiet", action="store_true", help="Suppress TrackEval metric tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_motchallenge_eval(
        gt_root=args.gt_root,
        trackers_root=args.trackers_root,
        output_root=args.output_root,
        trackers=args.trackers,
        sequences=args.sequences,
        print_results=not args.quiet,
    )
    print("\nFormal MOTChallenge metrics (percent, TrackEval):")
    for tracker, values in summary.items():
        print(
            f"{tracker}: HOTA={values['HOTA']:.5f}, "
            f"MOTA={values['MOTA']:.5f}, IDF1={values['IDF1']:.5f}, "
            f"IDSW={values['IDSW']}, FP={values['FP']}, FN={values['FN']}"
        )


if __name__ == "__main__":
    main()
