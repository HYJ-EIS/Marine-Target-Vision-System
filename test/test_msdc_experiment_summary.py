import csv
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.msdc_experiment_summary import (
    METHOD_LABELS,
    SPEED_FIELDS,
    build_analysis_text,
    load_metric_rows,
    main as summary_main,
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


def _write_summary(path: Path) -> None:
    _write_csv(path, [
        {"tracker": "ocsort", "HOTA": "10", "DetA": "20", "AssA": "30", "MOTA": "40", "IDF1": "50", "IDSW": "6", "FP": "7", "FN": "8", "IDTP": "9", "IDFP": "10", "IDFN": "11"},
        {"tracker": "Ours-full", "HOTA": "12", "DetA": "22", "AssA": "32", "MOTA": "42", "IDF1": "52", "IDSW": "4", "FP": "5", "FN": "6", "IDTP": "7", "IDFP": "8", "IDFN": "9"},
    ])


def _run_summary_cli(monkeypatch, tmp_path: Path, speed_csv: Path | None = None) -> tuple[Path, Path, Path]:
    main_root = tmp_path / "main"
    ablation_root = tmp_path / "ablation"
    output_root = tmp_path / "summary"
    report_output = tmp_path / "MSDC_EXPERIMENT_REPORT.md"
    docs_output = tmp_path / "docs" / "MSDC_EXPERIMENT_RESULT.md"
    _write_summary(main_root / "eval" / "motchallenge_summary.csv")
    _write_summary(ablation_root / "eval" / "motchallenge_summary.csv")

    argv = [
        "msdc_experiment_summary.py",
        "--main-root",
        str(main_root),
        "--ablation-root",
        str(ablation_root),
        "--output-root",
        str(output_root),
        "--report-output",
        str(report_output),
        "--docs-output",
        str(docs_output),
    ]
    if speed_csv is not None:
        argv.extend(["--speed-csv", str(speed_csv)])
    monkeypatch.setattr(sys, "argv", argv)
    summary_main()
    return output_root, report_output, docs_output


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


def test_ablation_results_reports_roi_redetect_switch_for_known_variants(tmp_path):
    variants = [
        "Ours-full",
        "Ours-lite-no-motion",
        "Ours-lite-no-low-det",
        "Ours-no-template",
        "Ours-no-reacquire",
        "Ours-no-removed-guard",
    ]
    summary = tmp_path / "eval" / "motchallenge_summary.csv"
    _write_csv(summary, [
        {"tracker": variant, "HOTA": "60", "DetA": "61", "AssA": "62", "MOTA": "63", "IDF1": "64", "IDSW": "1", "FP": "2", "FN": "3", "IDTP": "4", "IDFP": "5", "IDFN": "6"}
        for variant in variants
    ])

    rows = write_ablation_results(
        metric_rows=load_metric_rows(summary),
        output_csv=tmp_path / "ablation_results.csv",
        run_name="ablation_full",
        benchmark_root=tmp_path,
    )

    assert "MSDC_USE_ROI_REDETECT" in rows[0]
    assert {row["variant"]: row["MSDC_USE_ROI_REDETECT"] for row in rows} == {
        variant: "True" for variant in variants
    }


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


def test_path_manifest_lists_commands_and_failures_outputs(tmp_path):
    commands = tmp_path / "metadata" / "commands.jsonl"
    failures = tmp_path / "metadata" / "failures.csv"
    commands.parent.mkdir(parents=True)
    commands.write_text("{}", encoding="utf-8")
    failures.write_text("stage,error\nexport,failed\n", encoding="utf-8")

    rows = write_path_manifest(tmp_path, tmp_path / "manifest.csv")

    assert any(row["kind"] == "commands_jsonl" and row["path"].endswith("commands.jsonl") for row in rows)
    assert any(row["kind"] == "failure_csv" and row["path"].endswith("failures.csv") for row in rows)


def test_cli_preflights_required_summaries_before_creating_outputs(tmp_path, monkeypatch, capsys):
    ablation_root = tmp_path / "ablation"
    output_root = tmp_path / "summary"
    report_output = tmp_path / "MSDC_EXPERIMENT_REPORT.md"
    docs_output = tmp_path / "docs" / "MSDC_EXPERIMENT_RESULT.md"
    _write_summary(ablation_root / "eval" / "motchallenge_summary.csv")

    monkeypatch.setattr(sys, "argv", [
        "msdc_experiment_summary.py",
        "--main-root",
        str(tmp_path / "main"),
        "--ablation-root",
        str(ablation_root),
        "--output-root",
        str(output_root),
        "--report-output",
        str(report_output),
        "--docs-output",
        str(docs_output),
    ])

    with pytest.raises(SystemExit):
        summary_main()

    assert "Missing required summary CSV" in capsys.readouterr().err
    assert not output_root.exists()
    assert not report_output.exists()
    assert not docs_output.exists()


def test_cli_normalizes_speed_csv_missing_columns_to_stable_schema(tmp_path, monkeypatch):
    speed_csv = tmp_path / "speed" / "speed_results.csv"
    _write_csv(speed_csv, [{
        "run_id": "r1",
        "tracker": "ocsort",
        "mean_fps": "12.5",
        "detector_calls_roi_redetect": "3",
        "mean_roi_redetect_ms": "4.5",
        "mean_render_ms": "0",
        "mean_write_ms": "0",
    }])

    output_root, _, _ = _run_summary_cli(monkeypatch, tmp_path, speed_csv)

    with (output_root / "speed_results.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert reader.fieldnames == [*SPEED_FIELDS, "status", "failure"]
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["tracker"] == "ocsort"
    assert rows[0]["mean_fps"] == "12.5"
    assert rows[0]["commit_hash"] == "N/A"
    assert rows[0]["p95_latency_ms"] == "N/A"
    assert rows[0]["detector_calls_roi_redetect"] == "3"
    assert rows[0]["mean_roi_redetect_ms"] == "4.5"
    assert rows[0]["mean_render_ms"] == "0"
    assert rows[0]["mean_write_ms"] == "0"
    assert rows[0]["status"] == "ok"
    assert rows[0]["failure"] == ""


def test_cli_reports_empty_speed_csv_as_malformed(tmp_path, monkeypatch):
    speed_csv = tmp_path / "speed" / "speed_results.csv"
    speed_csv.parent.mkdir(parents=True)
    speed_csv.write_text("", encoding="utf-8")

    output_root, _, _ = _run_summary_cli(monkeypatch, tmp_path, speed_csv)
    rows = load_metric_rows(output_root / "speed_results.csv")

    assert rows[0]["status"] == "malformed"
    assert "Malformed speed CSV" in rows[0]["failure"]
    assert rows[0]["run_id"] == "N/A"
    assert rows[0]["mean_fps"] == "N/A"


def test_cli_manifest_includes_generated_report_artifacts(tmp_path, monkeypatch):
    speed_csv = tmp_path / "speed" / "speed_results.csv"
    _write_csv(speed_csv, [{"run_id": "r1", "tracker": "ocsort", "mean_fps": "12.5"}])

    output_root, report_output, docs_output = _run_summary_cli(monkeypatch, tmp_path, speed_csv)
    manifest_rows = load_metric_rows(output_root / "path_manifest.csv")
    paths = {row["path"] for row in manifest_rows}

    assert str(output_root / "main_results.csv") in paths
    assert str(output_root / "analysis_main.md") in paths
    assert str(report_output) in paths
    assert str(docs_output) in paths


def test_v2_ablation_switches_are_reported():
    from tools.evaluation.msdc_experiment_summary import _ablation_switches

    switches = _ablation_switches("v2_candidate_real2_age8")
    assert switches["MSDC_USE_LOW_DET"] == "True"
    assert switches["MSDC_USE_REACQUIRE"] == "True"
    assert switches["MSDC_USE_ROI_REDETECT"] == "True"
    assert switches["MSDC_REUSE_GUARD_ENABLE"] == "True"
    assert switches["MSDC_USE_TEMPLATE"] == "False"
    assert switches["MSDC_EXPORT_SHARE_LOW_HIGH_DET"] == "True"
    assert switches["MSDC_CONFIRM_MIN_REAL_DET_HITS"] == "2"
    assert switches["MSDC_CANDIDATE_MAX_AGE"] == "8"


def test_v2_ablation_switches_report_all_changed_fields():
    from tools.evaluation.msdc_experiment_summary import ABLATION_FIELDS, _ablation_switches

    for field in [
        "MSDC_ROI_REDETECT_LOST_INTERVAL",
        "MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI",
        "MSDC_REACQUIRE_MAX_CENTER_DIST",
    ]:
        assert field in ABLATION_FIELDS

    roi_switches = _ablation_switches("v2_roi_budget_active8_max2")
    assert roi_switches["MSDC_ROI_REDETECT_LOST_INTERVAL"] == "3"
    assert roi_switches["MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI"] == "1"

    reacquire_switches = _ablation_switches("v2_reacquire_interval5_center240")
    assert reacquire_switches["MSDC_REACQUIRE_CENTER_DIST"] == "240"
    assert reacquire_switches["MSDC_REACQUIRE_MAX_CENTER_DIST"] == "320"
