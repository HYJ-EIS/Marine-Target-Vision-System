import csv
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.validate_msdc_formal_run import REQUIRED_SPEED, validate_run


REQUIRED_DIAGNOSTIC_FIELDS = [
    "low_candidate_created",
    "low_candidate_confirmed",
    "low_candidate_precision",
    "low_candidate_opportunities",
    "low_candidate_recall",
    "reacquire_attempts",
    "reacquire_opportunities",
    "reacquire_success",
    "inherit_opportunities",
    "inherit_correct",
    "inherit_wrong",
    "inherit_ambiguous",
]


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_valid_run(root):
    mot_dir = root / "main/main_full/trackers/msdc_elt/data"
    mot_dir.mkdir(parents=True)
    (mot_dir / "seq.txt").write_text("1,1,10,10,20,20,1,-1,-1,-1\n", encoding="utf-8")

    tracker_diag_dir = root / "main/main_full/trackers/msdc_elt/diagnostics/seq"
    tracker_diag_dir.mkdir(parents=True)
    (tracker_diag_dir / "stage_observations.jsonl").write_text(
        json.dumps({"frame_idx": 0, "boxes": []}) + "\n",
        encoding="utf-8",
    )
    (tracker_diag_dir / "candidate_pool_stats.jsonl").write_text(
        json.dumps({"frame_idx": 0, "candidate_count": 0}) + "\n",
        encoding="utf-8",
    )

    diag_dir = root / "main/main_full/diagnostics/seq"
    diag_dir.mkdir(parents=True)
    (diag_dir / "msdc_elt_per_gt_diagnostics.csv").write_text(
        "gt_id,coverage_ratio\n1,1.0\n",
        encoding="utf-8",
    )
    summary_diag_dir = root / "main/main_full/diagnostics"
    _write_csv(
        summary_diag_dir / "msdc_diagnostic_summary.csv",
        [
            {
                "seq_name": "seq",
                "tracker": "msdc_elt",
                "rows": "1",
                **{field: "N/A" for field in REQUIRED_DIAGNOSTIC_FIELDS},
            }
        ],
        ["seq_name", "tracker", "rows", *REQUIRED_DIAGNOSTIC_FIELDS],
    )
    eval_dir = root / "main/main_full/eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "motchallenge_summary.csv").write_text(
        "tracker,HOTA,MOTA,IDF1,IDSW,FP,FN,IDTP,IDFP,IDFN\nmsdc_elt,67,80,66,64,3000,21000,1,2,3\n",
        encoding="utf-8",
    )

    vis_dir = root / "main/main_full/visualizations/seq"
    vis_dir.mkdir(parents=True)
    (vis_dir / "msdc_elt.mp4").write_bytes(b"mp4")

    _write_csv(
        root / "slice/slice_manifest.csv",
        [{"seq_name": "seq", "video_path": "seq.mp4"}],
        ["seq_name", "video_path"],
    )
    _write_csv(
        root / "slice/slice_metrics/slice_metrics.csv",
        [{"slice_id": "seq_1", "tracker": "msdc_elt", "MOTA": "80"}],
        ["slice_id", "tracker", "MOTA"],
    )
    _write_csv(
        root / "sensitivity/sensitivity_full/sensitivity_matrix.csv",
        [{"variant": "msdc_v3", "MOTA": "80"}],
        ["variant", "MOTA"],
    )
    _write_csv(
        root / "sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv",
        [{"tracker": "msdc_v3", "MOTA": "80"}],
        ["tracker", "MOTA"],
    )
    effective_config_dir = root / "main/main_full/effective_config"
    effective_config_dir.mkdir(parents=True)
    (effective_config_dir / "msdc_elt.json").write_text(
        json.dumps({"tracker": "msdc_elt", "MSDC_LOW_CANDIDATE_ENABLE": "1"}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        root / "summary/main_results.csv",
        [
            {
                "tracker": "msdc_elt",
                "HOTA": "67.0",
                "MOTA": "80.0",
                "IDF1": "66.0",
                "IDSW": "64",
                "FP": "3000",
                "FN": "21000",
                "IDTP": "1",
                "IDFP": "2",
                "IDFN": "3",
            }
        ],
        ["tracker", "HOTA", "MOTA", "IDF1", "IDSW", "FP", "FN", "IDTP", "IDFP", "IDFN"],
    )
    _write_csv(
        root / "summary/speed_results.csv",
        [
            {
                "tracker": "msdc_elt",
                "processed_frames": "1000",
                "total_time_s": "200",
                "mean_latency_ms": "200",
                "mean_fps": "5",
                "detector_calls_total": "1100",
                "detector_calls_high_det": "0",
                "detector_calls_low_det": "1000",
                "detector_calls_tracker_update": "0",
                "mean_read_decode_ms": "1",
                "mean_low_detection_ms": "80",
                "mean_high_split_ms": "1",
                "mean_low_filter_budget_ms": "2",
                "mean_observation_build_ms": "3",
                "mean_evidence_update_ms": "4",
                "mean_output_nms_ms": "5",
                "mean_render_write_ms": "0",
                "mean_read_ms": "1",
                "mean_high_det_ms": "0",
                "mean_low_det_ms": "80",
                "mean_tracker_ms": "30",
                "mean_render_ms": "0",
                "mean_write_ms": "0",
            }
        ],
        [
            "tracker",
            "processed_frames",
            "total_time_s",
            "mean_latency_ms",
            "mean_fps",
            "detector_calls_total",
            "detector_calls_high_det",
            "detector_calls_low_det",
            "detector_calls_tracker_update",
            "mean_read_decode_ms",
            "mean_low_detection_ms",
            "mean_high_split_ms",
            "mean_low_filter_budget_ms",
            "mean_observation_build_ms",
            "mean_evidence_update_ms",
            "mean_output_nms_ms",
            "mean_render_write_ms",
            "mean_read_ms",
            "mean_high_det_ms",
            "mean_low_det_ms",
            "mean_tracker_ms",
            "mean_render_ms",
            "mean_write_ms",
        ],
    )
    _write_csv(
        root / "summary/path_manifest.csv",
        [
            {"kind": "mot_result", "path": str(mot_dir / "seq.txt")},
            {"kind": "trackeval_summary", "path": str(eval_dir / "motchallenge_summary.csv")},
        ],
        ["kind", "path"],
    )


def test_validate_run_accepts_complete_formal_outputs(tmp_path):
    _make_valid_run(tmp_path)

    result = validate_run(tmp_path)

    assert result["ok"] is True
    assert result["missing"] == []
    assert result["run_root"] == str(tmp_path)


def test_validate_run_rejects_missing_visualization(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "main/main_full/visualizations/seq/msdc_elt.mp4").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("visualization" in item for item in result["missing"])


def test_validate_run_rejects_missing_candidate_pool_stats(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "main/main_full/trackers/msdc_elt/diagnostics/seq/candidate_pool_stats.jsonl").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("candidate_pool_stats" in item for item in result["missing"])


def test_validate_run_rejects_missing_trackeval_summary(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "main/main_full/eval/motchallenge_summary.csv").unlink()
    (tmp_path / "sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("trackeval_summary" in item for item in result["missing"])


def test_validate_run_rejects_missing_path_manifest(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "summary/path_manifest.csv").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("path_manifest" in item for item in result["missing"])


def test_validate_run_rejects_missing_diagnostic_summary(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "main/main_full/diagnostics/msdc_diagnostic_summary.csv").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("diagnostic_summary" in item for item in result["missing"])


def test_validate_run_rejects_missing_slice_manifest(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "slice/slice_manifest.csv").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("slice_manifest" in item for item in result["missing"])


def test_validate_run_rejects_missing_sensitivity_matrix(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "sensitivity/sensitivity_full/sensitivity_matrix.csv").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("sensitivity_matrix" in item for item in result["missing"])


def test_validate_run_rejects_header_only_diagnostic_summary(tmp_path):
    _make_valid_run(tmp_path)
    _write_csv(
        tmp_path / "main/main_full/diagnostics/msdc_diagnostic_summary.csv",
        [],
        ["seq_name", "tracker", "rows"],
    )

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("diagnostic_summary" in item for item in result["missing"])


def test_validate_run_rejects_header_only_sensitivity_matrix(tmp_path):
    _make_valid_run(tmp_path)
    _write_csv(
        tmp_path / "sensitivity/sensitivity_full/sensitivity_matrix.csv",
        [],
        ["variant", "MOTA"],
    )

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("sensitivity_matrix" in item for item in result["missing"])


def test_validate_run_rejects_missing_effective_config(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "main/main_full/effective_config/msdc_elt.json").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("effective_config" in item for item in result["missing"])


def test_validate_run_rejects_missing_sensitivity_metrics(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("sensitivity_metrics" in item for item in result["missing"])


def test_validate_run_rejects_missing_slice_metrics(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "slice/slice_metrics/slice_metrics.csv").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("slice_metrics" in item for item in result["missing"])


def test_validate_run_rejects_missing_diagnostic_required_field(tmp_path):
    _make_valid_run(tmp_path)
    fields = ["seq_name", "tracker", "rows", *REQUIRED_DIAGNOSTIC_FIELDS[:-1]]
    _write_csv(
        tmp_path / "main/main_full/diagnostics/msdc_diagnostic_summary.csv",
        [{field: "N/A" for field in fields}],
        fields,
    )

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("diagnostic_summary_fields" in item for item in result["missing"])


def test_validate_run_rejects_one_invalid_diagnostic_summary_even_if_another_is_valid(tmp_path):
    _make_valid_run(tmp_path)
    bad_summary = tmp_path / "ablation/ablation_full/diagnostics/msdc_diagnostic_summary.csv"
    fields = ["seq_name", "tracker", "rows", *REQUIRED_DIAGNOSTIC_FIELDS[:-1]]
    _write_csv(
        bad_summary,
        [{field: "N/A" for field in fields}],
        fields,
    )

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any(
        "diagnostic_summary_fields" in item and str(bad_summary) in item
        for item in result["missing"]
    )


def test_validate_run_rejects_header_only_invalid_diagnostic_summary_even_if_another_is_valid(tmp_path):
    _make_valid_run(tmp_path)
    bad_summary = tmp_path / "ablation/ablation_full/diagnostics/msdc_diagnostic_summary.csv"
    fields = ["seq_name", "tracker", "rows", *REQUIRED_DIAGNOSTIC_FIELDS[:-1]]
    _write_csv(bad_summary, [], fields)

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any(
        "diagnostic_summary_fields" in item and str(bad_summary) in item
        for item in result["missing"]
    )


def test_validate_run_rejects_placeholder_speed_values(tmp_path):
    _make_valid_run(tmp_path)
    _write_csv(
        tmp_path / "summary/speed_results.csv",
        [{
            "tracker": "N/A",
            "processed_frames": "N/A",
            "total_time_s": "N/A",
            "mean_latency_ms": "N/A",
            "mean_fps": "N/A",
            "detector_calls_total": "N/A",
            "detector_calls_high_det": "N/A",
            "detector_calls_low_det": "N/A",
            "detector_calls_tracker_update": "N/A",
            "mean_read_decode_ms": "N/A",
            "mean_low_detection_ms": "N/A",
            "mean_high_split_ms": "N/A",
            "mean_low_filter_budget_ms": "N/A",
            "mean_observation_build_ms": "N/A",
            "mean_evidence_update_ms": "N/A",
            "mean_output_nms_ms": "N/A",
            "mean_render_write_ms": "N/A",
            "mean_read_ms": "N/A",
            "mean_high_det_ms": "N/A",
            "mean_low_det_ms": "N/A",
            "mean_tracker_ms": "N/A",
            "mean_render_ms": "N/A",
            "mean_write_ms": "N/A",
            "status": "missing",
            "failure": "Speed CSV not found",
        }],
        [
            "tracker",
            "processed_frames",
            "total_time_s",
            "mean_latency_ms",
            "mean_fps",
            "detector_calls_total",
            "detector_calls_high_det",
            "detector_calls_low_det",
            "detector_calls_tracker_update",
            "mean_read_decode_ms",
            "mean_low_detection_ms",
            "mean_high_split_ms",
            "mean_low_filter_budget_ms",
            "mean_observation_build_ms",
            "mean_evidence_update_ms",
            "mean_output_nms_ms",
            "mean_render_write_ms",
            "mean_read_ms",
            "mean_high_det_ms",
            "mean_low_det_ms",
            "mean_tracker_ms",
            "mean_render_ms",
            "mean_write_ms",
            "status",
            "failure",
        ],
    )

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("speed" in item for item in result["missing"])


def test_validate_run_rejects_placeholder_values_in_second_speed_row(tmp_path):
    _make_valid_run(tmp_path)
    valid_speed_row = {field: "1" for field in REQUIRED_SPEED}
    valid_speed_row["tracker"] = "msdc_elt"
    invalid_speed_row = dict(valid_speed_row)
    invalid_speed_row["tracker"] = "botsort"
    invalid_speed_row["mean_low_detection_ms"] = "N/A"
    _write_csv(
        tmp_path / "summary/speed_results.csv",
        [valid_speed_row, invalid_speed_row],
        ["tracker", *REQUIRED_SPEED],
    )

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("speed" in item for item in result["missing"])
