from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_METRICS = ["HOTA", "MOTA", "IDF1", "IDSW", "FP", "FN", "IDTP", "IDFP", "IDFN"]
REQUIRED_SPEED = [
    "processed_frames",
    "total_time_s",
    "mean_latency_ms",
    "mean_fps",
    "detector_calls_total",
    "detector_calls_high_det",
    "detector_calls_low_det",
    "detector_calls_tracker_update",
    "detector_calls_roi_redetect",
    "mean_read_ms",
    "mean_high_det_ms",
    "mean_low_det_ms",
    "mean_roi_redetect_ms",
    "mean_tracker_ms",
    "mean_render_ms",
    "mean_write_ms",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _has_nonempty_glob(root: Path, pattern: str) -> bool:
    return any(path.is_file() and path.stat().st_size > 0 for path in root.glob(pattern))


def _csv_has_fields(path: Path, fields: list[str]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    rows = _read_csv(path)
    if not rows:
        return False
    return all(field in rows[0] and str(rows[0].get(field, "")).strip() for field in fields)


def validate_run(run_root: str | Path) -> dict[str, object]:
    root = Path(run_root)
    missing: list[str] = []
    if not root.is_dir():
        return {"ok": False, "missing": [f"run_root:{root}"], "run_root": str(root)}

    glob_checks = [
        ("mot_result", "**/trackers/*/data/*.txt"),
        ("trackeval_summary", "**/eval/motchallenge_summary.csv"),
        ("stage_observations", "**/trackers/*/diagnostics/**/stage_observations.jsonl"),
        ("candidate_pool_stats", "**/trackers/*/diagnostics/**/candidate_pool_stats.jsonl"),
        ("diagnostic_csv", "**/diagnostics/**/*.csv"),
        ("visualization", "**/visualizations/**/*.mp4"),
        ("path_manifest", "summary/path_manifest.csv"),
    ]
    for label, pattern in glob_checks:
        if not _has_nonempty_glob(root, pattern):
            missing.append(f"{label}:{pattern}")

    main_results = root / "summary" / "main_results.csv"
    if not _csv_has_fields(main_results, REQUIRED_METRICS):
        missing.append(f"metrics:{main_results}")

    speed_results = root / "summary" / "speed_results.csv"
    if not _csv_has_fields(speed_results, REQUIRED_SPEED):
        missing.append(f"speed:{speed_results}")

    return {"ok": not missing, "missing": missing, "run_root": str(root)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MS-DC-ELT formal run artifacts")
    parser.add_argument("--run-root", required=True, help="Formal run root directory")
    parser.add_argument("--json-output", default="", help="Optional path to write validation JSON")
    args = parser.parse_args()

    result = validate_run(args.run_root)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
