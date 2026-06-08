import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.msdc_experiment_summary import (
    METHOD_LABELS,
    build_analysis_text,
    load_metric_rows,
    write_ablation_results,
    write_main_results,
    write_path_manifest,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_main_results_maps_trackers_to_method_labels(tmp_path):
    summary = tmp_path / "eval" / "motchallenge_summary.csv"
    _write_csv(summary, [
        {"tracker": "ocsort", "HOTA": "10", "DetA": "20", "AssA": "30", "MOTA": "40", "IDF1": "50", "IDSW": "6", "FP": "7", "FN": "8", "IDTP": "9", "IDFP": "10", "IDFN": "11"},
        {"tracker": "botsort", "HOTA": "11", "DetA": "21", "AssA": "31", "MOTA": "41", "IDF1": "51", "IDSW": "5", "FP": "6", "FN": "7", "IDTP": "8", "IDFP": "9", "IDFN": "10"},
        {"tracker": "Ours-full", "HOTA": "12", "DetA": "22", "AssA": "32", "MOTA": "42", "IDF1": "52", "IDSW": "4", "FP": "5", "FN": "6", "IDTP": "7", "IDFP": "8", "IDFN": "9"},
    ])

    out = tmp_path / "main_results.csv"
    rows = write_main_results(
        metric_rows=load_metric_rows(summary),
        output_csv=out,
        run_name="main_full",
        benchmark_root=tmp_path,
    )

    assert [row["method"] for row in rows] == [
        METHOD_LABELS["ocsort"],
        METHOD_LABELS["botsort"],
        METHOD_LABELS["Ours-full"],
    ]
    assert out.read_text(encoding="utf-8-sig").splitlines()[0].startswith("run_name,method,tracker")


def test_ablation_results_keeps_variant_names(tmp_path):
    summary = tmp_path / "eval" / "motchallenge_summary.csv"
    _write_csv(summary, [
        {"tracker": "Ours-full", "HOTA": "60", "DetA": "61", "AssA": "62", "MOTA": "63", "IDF1": "64", "IDSW": "1", "FP": "2", "FN": "3", "IDTP": "4", "IDFP": "5", "IDFN": "6"},
        {"tracker": "Ours-no-template", "HOTA": "50", "DetA": "51", "AssA": "52", "MOTA": "53", "IDF1": "54", "IDSW": "9", "FP": "8", "FN": "7", "IDTP": "6", "IDFP": "5", "IDFN": "4"},
    ])

    rows = write_ablation_results(
        metric_rows=load_metric_rows(summary),
        output_csv=tmp_path / "ablation_results.csv",
        run_name="ablation_full",
        benchmark_root=tmp_path,
    )

    assert rows[0]["variant"] == "Ours-full"
    assert rows[1]["variant"] == "Ours-no-template"


def test_analysis_text_reports_direction_without_inventing_values():
    rows = [
        {"method": "FFCA-YOLO + OC-SORT", "IDF1": "50", "IDSW": "10", "HOTA": "20", "AssA": "30", "FP": "5", "FN": "100"},
        {"method": "FFCA-YOLO + MS-DC-ELT", "IDF1": "55", "IDSW": "8", "HOTA": "25", "AssA": "35", "FP": "7", "FN": "90"},
    ]
    text = build_analysis_text(rows, focus_name="FFCA-YOLO + MS-DC-ELT", baseline_names=["FFCA-YOLO + OC-SORT"])
    assert "IDF1" in text
    assert "55" in text
    assert "10" in text
    assert "8" in text


def test_path_manifest_lists_existing_outputs(tmp_path):
    mot_file = tmp_path / "trackers" / "ocsort" / "data" / "seq.txt"
    mot_file.parent.mkdir(parents=True)
    mot_file.write_text("", encoding="utf-8")
    rows = write_path_manifest(tmp_path, tmp_path / "manifest.csv")
    assert any(row["kind"] == "mot_result" and row["path"].endswith("seq.txt") for row in rows)
