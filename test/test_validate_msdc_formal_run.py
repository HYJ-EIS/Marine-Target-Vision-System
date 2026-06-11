import csv
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.validate_msdc_formal_run import validate_run


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

    diag_dir = root / "main/main_full/diagnostics/seq"
    diag_dir.mkdir(parents=True)
    (diag_dir / "msdc_elt_per_gt_diagnostics.csv").write_text(
        "gt_id,coverage_ratio\n1,1.0\n",
        encoding="utf-8",
    )

    vis_dir = root / "main/main_full/visualizations/seq"
    vis_dir.mkdir(parents=True)
    (vis_dir / "msdc_elt.mp4").write_bytes(b"mp4")

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
                "detector_calls_roi_redetect": "100",
                "mean_read_ms": "1",
                "mean_high_det_ms": "0",
                "mean_low_det_ms": "80",
                "mean_roi_redetect_ms": "10",
                "mean_tracker_ms": "30",
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
            "detector_calls_roi_redetect",
            "mean_read_ms",
            "mean_high_det_ms",
            "mean_low_det_ms",
            "mean_roi_redetect_ms",
            "mean_tracker_ms",
        ],
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
