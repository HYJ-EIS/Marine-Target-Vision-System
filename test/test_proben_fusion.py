import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.proben_fusion import (
    binary_proben_score,
    compute_scale_ratio,
    fuse_rgb_ir_proben,
    score_weighted_box_fusion,
)


def _args(**overrides):
    values = {
        "rgb_high_conf": 0.50,
        "rgb_low_conf": 0.30,
        "proben_match_iou": 0.15,
        "proben_match_dist_factor": 1.5,
        "proben_keep_conf": 0.50,
        "proben_ir_box_weight": 0.5,
        "proben_scale_ratio_min": 0.4,
        "proben_scale_ratio_max": 2.5,
        "proben_low_allow_new_track": False,
        "proben_box_mode": "score_only",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_binary_proben_score_raises_confidence_when_modalities_agree():
    assert binary_proben_score(0.40, 0.70) == pytest.approx(0.60869565)


def test_binary_proben_score_clamps_zero_and_one():
    assert 0.0 < binary_proben_score(0.0, 0.7) < 1.0
    assert 0.0 < binary_proben_score(1.0, 0.7) < 1.0


def test_score_weighted_box_fusion_can_bias_toward_rgb_box():
    rgb = {"x": 100, "y": 100, "w": 20, "h": 10, "confidence": 0.4, "class": "UAV"}
    ir = {"x": 110, "y": 100, "w": 20, "h": 10, "confidence": 0.8, "class": "USV"}

    weak_ir = score_weighted_box_fusion(rgb, ir, p_rgb=0.4, p_ir=0.8, ir_box_weight=0.25)
    strong_ir = score_weighted_box_fusion(rgb, ir, p_rgb=0.4, p_ir=0.8, ir_box_weight=1.0)

    assert weak_ir["x"] < strong_ir["x"]
    assert weak_ir["x"] == pytest.approx(103.3333333)
    assert strong_ir["x"] == pytest.approx(106.6666667)


def test_compute_scale_ratio_uses_ir_area_over_rgb_area():
    rgb = {"x": 0, "y": 0, "w": 10, "h": 10}
    ir = {"x": 0, "y": 0, "w": 20, "h": 5}

    assert compute_scale_ratio(rgb, ir) == pytest.approx(1.0)


def test_proben_promotes_matched_low_rgb_and_suppresses_ir_only():
    rgb_low = {"x": 100, "y": 100, "w": 20, "h": 20, "confidence": 0.4, "class": "UAV"}
    ir_match = {"x": 101, "y": 101, "w": 20, "h": 20, "confidence": 0.7, "class": "USV"}
    ir_only = {"x": 300, "y": 300, "w": 20, "h": 20, "confidence": 0.9, "class": "USV"}

    result = fuse_rgb_ir_proben([rgb_low], [ir_match, ir_only], _args())

    assert len(result.detections) == 1
    fused = result.detections[0]
    assert fused["support_type"] == "proben"
    assert fused["confidence"] == pytest.approx(binary_proben_score(0.4, 0.7))
    assert fused["allow_new_track"] is False
    assert fused["class"] == "UAV"
    assert result.stats["rgb_ir_proben_matched_count"] == 1
    assert result.stats["rgb_ir_proben_promoted_count"] == 1
    assert result.stats["rgb_ir_proben_ir_only_count"] == 1
    assert any(row["decision"] == "ir_only_not_emitted" for row in result.diagnostics)


def test_proben_rejects_scale_mismatch_and_does_not_duplicate_rgb_box():
    rgb = {"x": 100, "y": 100, "w": 20, "h": 20, "confidence": 0.8, "class": "UAV"}
    huge_ir = {"x": 70, "y": 70, "w": 80, "h": 80, "confidence": 0.9, "class": "USV"}

    result = fuse_rgb_ir_proben([rgb], [huge_ir], _args())

    assert len(result.detections) == 1
    assert result.detections[0]["support_type"] == "high"
    assert result.detections[0]["confidence"] == 0.8
    assert result.stats["rgb_ir_proben_rejected_by_geometry_count"] == 1


def test_proben_savg_mode_moves_box_but_score_only_keeps_rgb_box():
    rgb = {"x": 100, "y": 100, "w": 20, "h": 20, "confidence": 0.8, "class": "UAV"}
    ir = {"x": 110, "y": 100, "w": 20, "h": 20, "confidence": 0.8, "class": "USV"}

    score_only = fuse_rgb_ir_proben([rgb], [ir], _args(proben_box_mode="score_only"))
    savg = fuse_rgb_ir_proben([rgb], [ir], _args(proben_box_mode="savg"))

    assert score_only.detections[0]["x"] == 100
    assert savg.detections[0]["x"] > 100
    assert len(score_only.detections) == 1
    assert len(savg.detections) == 1
