import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.ir_presence_support import (  # noqa: E402
    IRPresenceSupport,
    map_rgb_box_to_ir,
    score_thermal_presence,
    transform_box_affine,
)


def _args(**overrides):
    values = {
        "ir_presence_bg_pad": 1.0,
        "ir_presence_min_bg_std": 1.0,
        "ir_presence_z_thresh": 3.0,
        "ir_presence_min_hot_area_ratio": 0.02,
        "ir_presence_max_track_age": 10,
        "ir_presence_rgb_match_iou": 0.2,
        "ir_presence_rgb_match_dist_factor": 1.5,
        "ir_presence_roi_pad": 2.0,
        "ir_presence_min_crop_size": 16,
        "ir_presence_roi_upscale": 1.0,
        "ir_presence_rgb_roi_low_conf": 0.2,
        "ir_presence_redetect_iou": 0.05,
        "ir_presence_redetect_dist_factor": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class DummyProcessor:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def process_frame(self, frame, file_type, conf_override=None):
        self.calls.append((frame.shape, file_type, conf_override))
        return {"boxes": [dict(box) for box in self.boxes]}


def test_transform_box_affine_clips_to_image_shape():
    matrix = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, 7.0]], dtype=np.float64)
    box = {"x": 10, "y": 20, "w": 30, "h": 40}

    mapped = transform_box_affine(box, matrix, shape=(100, 80, 3))

    assert mapped["x"] == pytest.approx(25.0)
    assert mapped["y"] == pytest.approx(47.0)
    assert mapped["w"] == pytest.approx(55.0)
    assert mapped["h"] == pytest.approx(53.0)


def test_map_rgb_box_to_ir_uses_inverse_affine():
    ir_to_rgb = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 20.0]], dtype=np.float64)
    rgb_box = {"x": 30, "y": 40, "w": 20, "h": 10}

    mapped = map_rgb_box_to_ir(rgb_box, ir_to_rgb, ir_shape=(100, 100, 3))

    assert mapped["x"] == pytest.approx(10.0)
    assert mapped["y"] == pytest.approx(10.0)
    assert mapped["w"] == pytest.approx(10.0)
    assert mapped["h"] == pytest.approx(5.0)


def test_score_thermal_presence_detects_hot_component():
    frame = np.full((80, 80), 30, dtype=np.uint8)
    frame[30:40, 30:40] = 180
    roi = {"x": 25, "y": 25, "w": 25, "h": 25}

    presence = score_thermal_presence(frame, roi, _args())

    assert presence.decision == "present"
    assert presence.thermal_score > 3.0
    assert presence.hot_area_ratio > 0.02
    assert presence.hot_box_ir is not None


def test_score_thermal_presence_rejects_flat_roi():
    frame = np.full((80, 80), 30, dtype=np.uint8)
    roi = {"x": 25, "y": 25, "w": 25, "h": 25}

    presence = score_thermal_presence(frame, roi, _args())

    assert presence.decision == "absent"
    assert presence.hot_box_ir is None


def test_presence_support_emits_only_geometry_consistent_rgb_roi_detection():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    ir = np.full((100, 100), 20, dtype=np.uint8)
    ir[40:50, 40:50] = 200
    track = {"track_id": 7, "x": 38, "y": 38, "w": 16, "h": 16, "class": "UAV", "time_since_update": 0}
    processor = DummyProcessor([
        {"x": 6, "y": 6, "w": 12, "h": 12, "confidence": 0.25, "class": "UAV"},
        {"x": 50, "y": 50, "w": 10, "h": 10, "confidence": 0.9, "class": "UAV"},
    ])

    detections, diagnostics = IRPresenceSupport().update(
        frame_id=2,
        rgb_frame=rgb,
        ir_frame=ir,
        track_boxes=[track],
        current_rgb_detections=[],
        ir_to_rgb_affine=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        processor=processor,
        args=_args(),
    )

    assert len(detections) == 1
    assert detections[0]["support_type"] == "ir_presence_roi_redetect"
    assert detections[0]["allow_new_track"] is False
    assert detections[0]["linked_track_id"] == 7
    assert diagnostics[0]["decision"] == "emitted"
    assert processor.calls[0][1:] == ("visible", 0.2)


def test_presence_support_skips_tracks_already_matched_by_rgb():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    ir = np.full((100, 100), 20, dtype=np.uint8)
    ir[40:50, 40:50] = 200
    track = {"track_id": 7, "x": 38, "y": 38, "w": 16, "h": 16, "class": "UAV", "time_since_update": 0}
    processor = DummyProcessor([{"x": 6, "y": 6, "w": 12, "h": 12, "confidence": 0.25, "class": "UAV"}])

    detections, diagnostics = IRPresenceSupport().update(
        frame_id=2,
        rgb_frame=rgb,
        ir_frame=ir,
        track_boxes=[track],
        current_rgb_detections=[{"x": 39, "y": 39, "w": 14, "h": 14, "confidence": 0.8}],
        ir_to_rgb_affine=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        processor=processor,
        args=_args(),
    )

    assert detections == []
    assert diagnostics[0]["reject_reason"] == "rgb_matched"
    assert processor.calls == []
