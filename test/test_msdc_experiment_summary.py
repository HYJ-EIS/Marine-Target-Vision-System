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
    _ablation_switches,
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
        {"tracker": "v3_candidate_topk_no_roi_no_motion", "HOTA": "12", "DetA": "22", "AssA": "32", "MOTA": "42", "IDF1": "52", "IDSW": "4", "FP": "5", "FN": "6", "IDTP": "7", "IDFP": "8", "IDFN": "9"},
    ])


def _run_summary_cli(
    monkeypatch,
    tmp_path: Path,
    speed_csv: Path | None = None,
    with_optional_sources: bool = False,
) -> tuple[Path, Path, Path]:
    main_root = tmp_path / "main"
    ablation_root = tmp_path / "ablation"
    output_root = tmp_path / "summary"
    report_output = tmp_path / "MSDC_EXPERIMENT_REPORT.md"
    docs_output = tmp_path / "docs" / "MSDC_EXPERIMENT_RESULT.md"
    _write_summary(main_root / "eval" / "motchallenge_summary.csv")
    _write_summary(ablation_root / "eval" / "motchallenge_summary.csv")
    if with_optional_sources:
        _write_csv(
            tmp_path / "sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv",
            [{"tracker": "msdc_v3_replay", "MOTA": "42", "IDF1": "52"}],
        )
        _write_csv(
            tmp_path / "slice/slice_metrics/slice_metrics.csv",
            [{"slice_id": "short_gap_1", "tracker": "msdc_v3_replay", "MOTA": "41"}],
        )
        _write_csv(
            main_root / "diagnostics/msdc_diagnostic_summary.csv",
            [
                {
                    "seq_name": "seq",
                    "tracker": "msdc_v3_replay",
                    "low_candidate_opportunities": "10",
                    "reacquire_opportunities": "N/A",
                    "inherit_opportunities": "5",
                }
            ],
        )

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
        {"tracker": "msdc_elt", "HOTA": "12", "DetA": "22", "AssA": "32", "MOTA": "42", "IDF1": "52", "IDSW": "4", "FP": "5", "FN": "6", "IDTP": "7", "IDFP": "8", "IDFN": "9"},
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
        METHOD_LABELS["msdc_elt"],
    ]
    assert out.read_text(encoding="utf-8-sig").splitlines()[0].startswith("run_name,method,tracker")


def test_main_results_include_v3_candidate_topk_no_roi_no_motion_as_msdc_v3(tmp_path):
    summary = tmp_path / "eval" / "motchallenge_summary.csv"
    _write_csv(summary, [
        {"tracker": "ocsort", "HOTA": "10", "DetA": "20", "AssA": "30", "MOTA": "40", "IDF1": "50", "IDSW": "6", "FP": "7", "FN": "8", "IDTP": "9", "IDFP": "10", "IDFN": "11"},
        {"tracker": "botsort", "HOTA": "11", "DetA": "21", "AssA": "31", "MOTA": "41", "IDF1": "51", "IDSW": "5", "FP": "6", "FN": "7", "IDTP": "8", "IDFP": "9", "IDFN": "10"},
        {"tracker": "v3_candidate_topk_no_roi_no_motion", "HOTA": "12", "DetA": "22", "AssA": "32", "MOTA": "42", "IDF1": "52", "IDSW": "4", "FP": "5", "FN": "6", "IDTP": "7", "IDFP": "8", "IDFN": "9"},
    ])

    rows = write_main_results(
        metric_rows=load_metric_rows(summary),
        output_csv=tmp_path / "main_results.csv",
        run_name="main_full",
        benchmark_root=tmp_path,
    )

    assert [row["tracker"] for row in rows] == ["ocsort", "botsort", "v3_candidate_topk_no_roi_no_motion"]
    assert rows[2]["method"] == METHOD_LABELS["v3_candidate_topk_no_roi_no_motion"]


def test_main_results_include_replay_tracker_names(tmp_path):
    summary = tmp_path / "eval" / "motchallenge_summary.csv"
    _write_csv(summary, [
        {"tracker": "bytetrack_replay", "HOTA": "9", "DetA": "19", "AssA": "29", "MOTA": "39", "IDF1": "49", "IDSW": "7", "FP": "8", "FN": "9", "IDTP": "10", "IDFP": "11", "IDFN": "12"},
        {"tracker": "ocsort_replay", "HOTA": "10", "DetA": "20", "AssA": "30", "MOTA": "40", "IDF1": "50", "IDSW": "6", "FP": "7", "FN": "8", "IDTP": "9", "IDFP": "10", "IDFN": "11"},
        {"tracker": "botsort_replay", "HOTA": "11", "DetA": "21", "AssA": "31", "MOTA": "41", "IDF1": "51", "IDSW": "5", "FP": "6", "FN": "7", "IDTP": "8", "IDFP": "9", "IDFN": "10"},
        {"tracker": "msdc_v3_replay", "HOTA": "12", "DetA": "22", "AssA": "32", "MOTA": "42", "IDF1": "52", "IDSW": "4", "FP": "5", "FN": "6", "IDTP": "7", "IDFP": "8", "IDFN": "9"},
    ])

    rows = write_main_results(
        metric_rows=load_metric_rows(summary),
        output_csv=tmp_path / "main_results.csv",
        run_name="main_full",
        benchmark_root=tmp_path,
    )

    assert [row["tracker"] for row in rows] == [
        "bytetrack_replay",
        "ocsort_replay",
        "botsort_replay",
        "msdc_v3_replay",
    ]
    assert [row["method"] for row in rows] == [
        METHOD_LABELS["bytetrack"],
        METHOD_LABELS["ocsort"],
        METHOD_LABELS["botsort"],
        METHOD_LABELS["msdc_v3"],
    ]


def test_ablation_results_keeps_variant_names(tmp_path):
    summary = tmp_path / "eval" / "motchallenge_summary.csv"
    _write_csv(summary, [
        {"tracker": "Ours-no-reacquire", "HOTA": "60", "DetA": "61", "AssA": "62", "MOTA": "63", "IDF1": "64", "IDSW": "1", "FP": "2", "FN": "3", "IDTP": "4", "IDFP": "5", "IDFN": "6"},
        {"tracker": "v2_candidate_topk", "HOTA": "50", "DetA": "51", "AssA": "52", "MOTA": "53", "IDF1": "54", "IDSW": "9", "FP": "8", "FN": "7", "IDTP": "6", "IDFP": "5", "IDFN": "4"},
    ])

    rows = write_ablation_results(
        metric_rows=load_metric_rows(summary),
        output_csv=tmp_path / "ablation_results.csv",
        run_name="ablation_full",
        benchmark_root=tmp_path,
    )

    assert rows[0]["variant"] == "Ours-no-reacquire"
    assert rows[1]["variant"] == "v2_candidate_topk"


def test_ablation_results_reports_retained_switches_for_known_variants(tmp_path):
    variants = [
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

    by_variant = {row["variant"]: row for row in rows}
    assert by_variant["Ours-no-reacquire"]["MSDC_USE_REACQUIRE"] == "False"
    assert by_variant["Ours-no-removed-guard"]["MSDC_REUSE_GUARD_ENABLE"] == "False"


def test_ablation_results_reports_new_v2_speed_ablation_switches(tmp_path):
    variants = [
        "v2_output_age5_size8",
        "v2_candidate_low3_window6",
        "v2_candidate_real2_age8",
        "v2_candidate_topk",
        "v3_candidate_topk_no_roi_no_motion",
        "v2_speed_diag_off",
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
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["v2_output_age5_size8"]["MSDC_OUTPUT_MAX_REAL_DET_AGE"] == "5"
    assert by_variant["v2_candidate_low3_window6"]["MSDC_LOW_CONFIRM_MIN_HITS"] == "3"
    assert by_variant["v2_candidate_real2_age8"]["MSDC_CONFIRM_MIN_REAL_DET_HITS"] == "2"
    assert by_variant["v2_candidate_topk"]["MSDC_LOW_OBS_TOPK"] == "32"
    assert by_variant["v3_candidate_topk_no_roi_no_motion"]["MSDC_LOW_OBS_TOPK"] == "32"
    assert by_variant["v2_speed_diag_off"]["MSDC_DEBUG_EVENTS"] == "False"


def test_ablation_results_reports_formal_v3_variant_switches(tmp_path):
    variants = [
        "msdc_v3",
        "no_low_candidate",
        "hits_only_no_evidence",
        "no_output_nms",
        "low_budget_topk16",
        "pending_recovery_candidate",
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
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["msdc_v3"]["MSDC_LOW_CANDIDATE_ENABLE"] == "1"
    assert by_variant["msdc_v3"]["MSDC_REACQUIRE_INTERVAL"] == "1"
    assert by_variant["msdc_v3"]["MSDC_REACQUIRE_SCORE"] == "0.8"
    assert by_variant["msdc_v3"]["MSDC_PENDING_RECOVERY_ENABLE"] == "0"
    assert by_variant["no_low_candidate"]["MSDC_LOW_CANDIDATE_ENABLE"] == "0"
    assert by_variant["hits_only_no_evidence"]["MSDC_EVIDENCE_MODE"] == "hits_only"
    assert by_variant["no_output_nms"]["MSDC_OUTPUT_NMS_ENABLE"] == "0"
    assert by_variant["low_budget_topk16"]["MSDC_LOW_OBS_TOPK"] == "16"
    assert by_variant["low_budget_topk16"]["MSDC_LOW_OBS_MAX_PER_FRAME"] == "32"
    assert by_variant["pending_recovery_candidate"]["MSDC_PENDING_RECOVERY_ENABLE"] == "1"
    assert by_variant["pending_recovery_candidate"]["MSDC_PENDING_RECOVERY_FRAMES"] == "1"


def test_ablation_switches_normalizes_replay_suffix_for_formal_variants():
    assert _ablation_switches("no_low_candidate_replay")["MSDC_LOW_CANDIDATE_ENABLE"] == "0"
    assert _ablation_switches("no_direct_reacquire_replay")["MSDC_USE_REACQUIRE"] == "0"
    assert _ablation_switches("hits_only_no_evidence_replay")["MSDC_EVIDENCE_MODE"] == "hits_only"


def test_ablation_results_include_variant_key_after_variant_for_replay_names(tmp_path):
    summary = tmp_path / "eval" / "motchallenge_summary.csv"
    _write_csv(summary, [
        {"tracker": "no_low_candidate_replay", "HOTA": "60", "DetA": "61", "AssA": "62", "MOTA": "63", "IDF1": "64", "IDSW": "1", "FP": "2", "FN": "3", "IDTP": "4", "IDFP": "5", "IDFN": "6"},
    ])

    output_csv = tmp_path / "ablation_results.csv"
    rows = write_ablation_results(
        metric_rows=load_metric_rows(summary),
        output_csv=output_csv,
        run_name="ablation_full",
        benchmark_root=tmp_path,
    )

    header = output_csv.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert header[:3] == ["run_name", "variant", "variant_key"]
    assert rows[0]["variant"] == "no_low_candidate_replay"
    assert rows[0]["variant_key"] == "no_low_candidate"
    assert rows[0]["MSDC_LOW_CANDIDATE_ENABLE"] == "0"


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


def test_cli_materializes_optional_sensitivity_slice_and_diagnostic_sources(tmp_path, monkeypatch):
    output_root, report_output, _ = _run_summary_cli(
        monkeypatch,
        tmp_path,
        with_optional_sources=True,
    )

    sensitivity_rows = load_metric_rows(output_root / "sensitivity_results.csv")
    slice_rows = load_metric_rows(output_root / "slice_metrics.csv")
    diagnostic_rows = load_metric_rows(output_root / "diagnostic_results.csv")
    report_text = report_output.read_text(encoding="utf-8")

    assert sensitivity_rows == [{"tracker": "msdc_v3_replay", "MOTA": "42", "IDF1": "52"}]
    assert slice_rows == [{"slice_id": "short_gap_1", "tracker": "msdc_v3_replay", "MOTA": "41"}]
    assert diagnostic_rows[0]["low_candidate_opportunities"] == "10"
    assert diagnostic_rows[0]["reacquire_opportunities"] == "N/A"
    assert "## Sensitivity Results" in report_text
    assert "## Slice Metrics" in report_text
    assert "## Diagnostic Results" in report_text


def test_cli_materializes_missing_rows_for_absent_optional_sources(tmp_path, monkeypatch):
    output_root, _, _ = _run_summary_cli(monkeypatch, tmp_path)

    sensitivity_rows = load_metric_rows(output_root / "sensitivity_results.csv")
    slice_rows = load_metric_rows(output_root / "slice_metrics.csv")
    diagnostic_rows = load_metric_rows(output_root / "diagnostic_results.csv")

    assert sensitivity_rows[0]["status"] == "missing"
    assert "sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv" in sensitivity_rows[0]["failure"]
    assert slice_rows[0]["status"] == "missing"
    assert "slice/slice_metrics/slice_metrics.csv" in slice_rows[0]["failure"]
    assert diagnostic_rows[0]["status"] == "missing"
    assert "diagnostics/msdc_diagnostic_summary.csv" in diagnostic_rows[0]["failure"]


def test_v2_ablation_switches_are_reported():
    from tools.evaluation.msdc_experiment_summary import _ablation_switches

    switches = _ablation_switches("v2_candidate_real2_age8")
    assert switches["MSDC_USE_REACQUIRE"] == "True"
    assert switches["MSDC_REUSE_GUARD_ENABLE"] == "True"
    assert switches["MSDC_CONFIRM_MIN_REAL_DET_HITS"] == "2"
    assert switches["MSDC_CANDIDATE_MAX_AGE"] == "8"


def test_v2_ablation_switches_report_all_changed_fields():
    from tools.evaluation.msdc_experiment_summary import ABLATION_FIELDS, _ablation_switches

    for field in [
        "MSDC_REACQUIRE_MAX_CENTER_DIST",
    ]:
        assert field in ABLATION_FIELDS

    reacquire_switches = _ablation_switches("v2_reacquire_interval5_center240")
    assert reacquire_switches["MSDC_REACQUIRE_CENTER_DIST"] == "240"
    assert reacquire_switches["MSDC_REACQUIRE_MAX_CENTER_DIST"] == "320"
