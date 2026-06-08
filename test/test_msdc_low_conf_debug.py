import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from tools.validation.msdc_low_conf_debug import (
    build_low_det_record,
    filter_low_only_boxes,
    resolve_low_conf_threshold,
)


def _box(x: int, y: int, w: int = 10, h: int = 10, conf: float = 0.3) -> dict:
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "confidence": conf,
        "class": "UAV",
        "class_confidence": conf,
    }


def test_filter_low_only_boxes_removes_low_boxes_overlapping_high_boxes():
    high_boxes = [_box(10, 10, conf=0.8)]
    low_boxes = [
        _box(10, 10, conf=0.8),
        _box(11, 11, conf=0.35),
        _box(80, 80, conf=0.2),
    ]

    low_only, overlap_count = filter_low_only_boxes(
        high_boxes=high_boxes,
        low_boxes=low_boxes,
        iou_threshold=0.5,
    )

    assert overlap_count == 2
    assert low_only == [low_boxes[2]]


def test_build_low_det_record_contains_required_aliases_and_overlap_ratio():
    high_boxes = [_box(10, 10, conf=0.8)]
    low_boxes = [_box(10, 10, conf=0.8), _box(80, 80, conf=0.2)]
    low_only = [low_boxes[1]]

    record = build_low_det_record(
        frame_idx=0,
        file_type="visible",
        high_boxes=high_boxes,
        low_boxes=low_boxes,
        low_only_boxes=low_only,
        low_conf_thresh=0.18,
        low_high_overlap_count=1,
    )

    assert record["frame_idx"] == 0
    assert record["frame_index"] == 0
    assert record["file_type"] == "visible"
    assert record["num_high"] == 1
    assert record["num_low"] == 2
    assert record["num_low_only"] == 1
    assert record["high_count"] == 1
    assert record["low_count"] == 2
    assert record["low_only_count"] == 1
    assert record["low_conf_thresh"] == 0.18
    assert record["low_high_overlap_ratio"] == 0.5
    assert json.loads(json.dumps(record, ensure_ascii=False)) == record


def test_resolve_low_conf_threshold_uses_modality_specific_config():
    assert resolve_low_conf_threshold("visible") == Config.MSDC_LOW_CONF_VISIBLE
    assert resolve_low_conf_threshold("infrared") == Config.MSDC_LOW_CONF_INFRARED
    assert resolve_low_conf_threshold("visible", override=0.12) == 0.12
