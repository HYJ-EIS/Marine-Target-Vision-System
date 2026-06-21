import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.export_dual_modal_mot_results import (
    REQUIRED_STAGE_TIMINGS,
    _empty_stats,
    format_detection_result_line,
    compute_iou,
    fuse_rgb_ir_detections,
    has_ir_support,
    load_calib,
    map_ir_box_to_rgb,
    resolve_affine_matrix,
)


def _args(**overrides):
    values = {
        "rgb_high_conf": 0.50,
        "rgb_low_conf": 0.30,
        "alpha": 0.7,
        "ir_rgb_match_iou": 0.10,
        "ir_rgb_match_dist_factor": 2.0,
        "track_iou_thresh": 0.20,
        "track_dist_factor": 1.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_calib_requires_existing_affine_matrix(tmp_path):
    report = tmp_path / "alignment_report.json"
    report.write_text(
        json.dumps({"transforms": {"affine": {"matrix": [[1, 0, 5], [0, 1, 7]]}}}),
        encoding="utf-8",
    )

    matrix = load_calib(report)

    assert matrix.shape == (2, 3)
    assert matrix.tolist() == [[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]]


def test_load_calib_supports_piecewise_affine_segments(tmp_path):
    report = tmp_path / "alignment_report.json"
    report.write_text(
        json.dumps({
            "transforms": {
                "piecewise_affine": {
                    "segments": [
                        {
                            "name": "early",
                            "start_frame": 1,
                            "end_frame": 1200,
                            "matrix": [[1, 0, 5], [0, 1, 7]],
                        },
                        {
                            "name": "late",
                            "start_frame": 1201,
                            "end_frame": 1700,
                            "matrix": [[2, 0, 9], [0, 2, 11]],
                        },
                    ]
                }
            }
        }),
        encoding="utf-8",
    )

    calib = load_calib(report)

    assert calib["type"] == "piecewise_affine"
    assert resolve_affine_matrix(calib, 100).tolist() == [[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]]
    assert resolve_affine_matrix(calib, 1500).tolist() == [[2.0, 0.0, 9.0], [0.0, 2.0, 11.0]]


def test_load_calib_rejects_missing_affine_matrix(tmp_path):
    report = tmp_path / "bad_report.json"
    report.write_text(json.dumps({"transforms": {"homography": {"matrix": np.eye(3).tolist()}}}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_calib(report)


def test_map_ir_box_to_rgb_uses_affine_and_clips_to_rgb_shape():
    matrix = np.array([[2.0, 0.0, 5.0], [0.0, 3.0, 7.0]], dtype=np.float64)
    box = {"x": 10, "y": 20, "w": 30, "h": 40, "confidence": 0.6, "class": "UAV"}

    mapped = map_ir_box_to_rgb(box, matrix, rgb_shape=(100, 80, 3))

    assert mapped is not None
    assert mapped["x"] == pytest.approx(25.0)
    assert mapped["y"] == pytest.approx(67.0)
    assert mapped["w"] == pytest.approx(55.0)
    assert mapped["h"] == pytest.approx(33.0)
    assert mapped["confidence"] == 0.6


def test_has_ir_support_selects_highest_iou_then_confidence_fusion():
    rgb_low = {"x": 100, "y": 100, "w": 20, "h": 20, "confidence": 0.4, "class": "UAV"}
    ir_boxes = [
        {"x": 160, "y": 160, "w": 20, "h": 20, "confidence": 0.9, "class": "USV"},
        {"x": 102, "y": 100, "w": 20, "h": 20, "confidence": 0.5, "class": "USV"},
    ]

    supported, best_ir = has_ir_support(rgb_low, ir_boxes, _args())
    fused = fuse_rgb_ir_detections([rgb_low], ir_boxes, tracker=None, args=_args())

    assert supported is True
    assert best_ir is ir_boxes[1]
    assert fused[0]["support_type"] == "ir_support"
    assert fused[0]["allow_new_track"] is True
    assert fused[0]["class"] == "UAV"
    assert fused[0]["confidence"] == pytest.approx(1 - (1 - 0.4) * (1 - 0.7 * 0.5))


def test_fusion_suppresses_low_boxes_without_ir_or_track_support():
    rgb_boxes = [
        {"x": 0, "y": 0, "w": 20, "h": 20, "confidence": 0.6, "class": "UAV"},
        {"x": 100, "y": 100, "w": 20, "h": 20, "confidence": 0.4, "class": "UAV"},
    ]

    fused = fuse_rgb_ir_detections(rgb_boxes, [], tracker=None, args=_args())

    assert len(fused) == 1
    assert fused[0]["support_type"] == "high"
    assert fused[0]["allow_new_track"] is True


def test_compute_iou_uses_xywh_boxes():
    assert compute_iou({"x": 0, "y": 0, "w": 10, "h": 10}, {"x": 5, "y": 0, "w": 10, "h": 10}) == pytest.approx(1 / 3)


def test_empty_stats_include_formal_timing_and_diagnostic_paths():
    stats = _empty_stats("botsort_rgb", "seq")

    assert set(REQUIRED_STAGE_TIMINGS) <= set(stats["stage_timings_seconds"])
    assert set(REQUIRED_STAGE_TIMINGS) <= set(stats["stage_timings_per_frame_seconds"])
    assert stats["stage_timings_seconds"]["visualization_rendering"] == 0.0
    assert stats["stage_timings_seconds"]["low_threshold_detection_or_roi_redetect"] == 0.0
    assert stats["diagnostics_csv_path"].endswith("diagnostics.csv")
    assert stats["stage_timings_json_path"].endswith("stage_timings.json")
    assert stats["output_detection_txt_path"].endswith("det.txt")
    assert stats["proben_diagnostics_jsonl_path"].endswith("proben_diagnostics.jsonl")
    assert stats["ir_presence_diagnostics_jsonl_path"].endswith("ir_presence_diagnostics.jsonl")
    assert stats["rgb_ir_proben_matched_count"] == 0
    assert stats["rgb_ir_proben_promoted_count"] == 0
    assert stats["ir_presence_roi_redetect_emitted_count"] == 0


def test_format_detection_result_line_uses_mot_detection_columns():
    box = {"x": 10, "y": 20, "w": 30, "h": 40, "confidence": 0.87654}

    line = format_detection_result_line(frame_id=3, box=box)

    assert line == "3,-1,10.00,20.00,30.00,40.00,0.8765,-1,-1,-1"
