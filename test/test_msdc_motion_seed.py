import json
import sys
from pathlib import Path

import cv2
import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.utils.motion_seed import MotionSeedGenerator
from tools.validation.msdc_motion_seed_debug import write_motion_debug_outputs


class _MotionConfig:
    MSDC_MOTION_ENABLE = True
    MSDC_MOTION_USE_GMC = False
    MSDC_MOTION_DIFF_PERCENTILE = 90
    MSDC_MOTION_MIN_AREA_RATIO = 1e-5
    MSDC_MOTION_MAX_AREA_RATIO = 0.5
    MSDC_MOTION_MAX_BOXES = 64
    MSDC_MOTION_DEBUG = True


def _frame_with_square(x: int | None = None, y: int = 24) -> np.ndarray:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    if x is not None:
        cv2.rectangle(frame, (x, y), (x + 14, y + 14), (255, 255, 255), -1)
    return frame


def _iou_xyxy(box: dict, xyxy: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    gx1, gy1, gx2, gy2 = xyxy
    ix1, iy1 = max(x1, gx1), max(y1, gy1)
    ix2, iy2 = min(x2, gx2), min(y2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, x2 - x1) * max(0, y2 - y1)
    area_b = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def test_first_frame_initializes_without_motion_boxes():
    generator = MotionSeedGenerator(_MotionConfig)

    boxes, debug = generator.update(_frame_with_square(20), frame_idx=0)

    assert boxes == []
    assert debug["frame_idx"] == 0
    assert debug["num_motion_boxes"] == 0
    assert debug["used_gmc"] is False
    assert debug["is_first_frame"] is True


def test_moving_square_produces_motion_box_overlapping_current_object():
    generator = MotionSeedGenerator(_MotionConfig)
    generator.update(_frame_with_square(20), frame_idx=0)

    boxes, debug = generator.update(_frame_with_square(42), frame_idx=1)

    assert debug["num_motion_boxes"] >= 1
    assert any(_iou_xyxy(box, (42, 24, 57, 39)) > 0 for box in boxes)
    for box in boxes:
        assert {"x", "y", "w", "h", "x1", "y1", "x2", "y2", "score", "source", "frame_idx"} <= set(box)
        assert box["source"] == "motion"
        assert box["frame_idx"] == 1
        assert 0.0 <= box["score"] <= 1.0


def test_identical_frames_return_no_motion_boxes():
    generator = MotionSeedGenerator(_MotionConfig)
    frame = _frame_with_square(20)
    generator.update(frame, frame_idx=0)

    boxes, debug = generator.update(frame.copy(), frame_idx=1)

    assert boxes == []
    assert debug["num_motion_boxes"] == 0


def test_motion_boxes_are_capped_by_config_max_boxes():
    class OneBoxConfig(_MotionConfig):
        MSDC_MOTION_MAX_BOXES = 1
        MSDC_MOTION_MIN_AREA_RATIO = 1e-6

    generator = MotionSeedGenerator(OneBoxConfig)
    first = np.zeros((120, 120, 3), dtype=np.uint8)
    second = first.copy()
    for i in range(6):
        x = 8 + i * 18
        cv2.rectangle(second, (x, 20), (x + 6, 26), (255, 255, 255), -1)
    generator.update(first, frame_idx=0)

    boxes, debug = generator.update(second, frame_idx=1)

    assert len(boxes) == 1
    assert debug["num_motion_boxes"] == 1
    assert debug["raw_component_count"] >= 6


def test_motion_clutter_caps_boxes_when_components_explode():
    class ClutterConfig(_MotionConfig):
        MSDC_MOTION_MAX_BOXES = 64
        MSDC_MOTION_MIN_AREA_RATIO = 1e-6
        MSDC_MOTION_CLUTTER_COMPONENT_THRESH = 5
        MSDC_MOTION_CLUTTER_MAX_BOXES = 2

    generator = MotionSeedGenerator(ClutterConfig)
    first = np.zeros((120, 120, 3), dtype=np.uint8)
    second = first.copy()
    for i in range(12):
        x = 5 + (i % 6) * 18
        y = 10 + (i // 6) * 40
        cv2.rectangle(second, (x, y), (x + 5, y + 5), (255, 255, 255), -1)
    generator.update(first, frame_idx=0)

    boxes, debug = generator.update(second, frame_idx=1)

    assert debug["raw_component_count"] >= 12
    assert debug["clutter_limited"] is True
    assert len(boxes) == 2
    assert debug["num_motion_boxes"] == 2


def test_write_motion_debug_outputs_writes_jsonl_and_mask(tmp_path):
    stats_path = tmp_path / "motion_seed_stats.jsonl"
    mask_dir = tmp_path / "motion_masks"
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 255

    write_motion_debug_outputs(
        stats_path=stats_path,
        mask_dir=mask_dir,
        frame_idx=3,
        debug_info={
            "frame_idx": 3,
            "num_motion_boxes": 2,
            "diff_threshold": 12.5,
            "used_gmc": False,
        },
        mask=mask,
        save_mask=True,
    )

    record = json.loads(stats_path.read_text(encoding="utf-8").strip())
    assert record["frame_idx"] == 3
    assert record["num_motion_boxes"] == 2
    assert record["diff_threshold"] == 12.5
    assert record["used_gmc"] is False
    assert (mask_dir / "frame_000003.png").is_file()
