import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils import msdc_detection


class DummyProcessor:
    def __init__(self):
        self.calls = []

    def process_frame(self, frame, file_type, conf_override=None):
        self.calls.append((frame.shape, file_type, conf_override))
        if conf_override is None:
            return {"boxes": [{"x": 9, "y": 9, "w": 9, "h": 9, "confidence": 0.91, "class": "UAV"}]}
        return {
            "boxes": [
                {"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.70, "class": "UAV"},
                {"x": 5, "y": 6, "w": 7, "h": 8, "confidence": 0.20, "class": "UAV"},
            ],
        }


class DummyDetector:
    def __init__(self):
        self.processor = DummyProcessor()


def _frame():
    return np.zeros((16, 16, 3), dtype=np.uint8)


def test_resolve_msdc_high_low_boxes_uses_one_low_call_when_shared(monkeypatch):
    detector = DummyDetector()
    monkeypatch.setattr(Config, "MSDC_EXPORT_SHARE_LOW_HIGH_DET", True, raising=False)
    monkeypatch.setattr(Config, "MSDC_USE_LOW_DET", True, raising=False)

    high_boxes, low_boxes = msdc_detection.resolve_msdc_high_low_boxes(detector, _frame(), "visible")

    assert detector.processor.calls == [(_frame().shape, "visible", Config.MSDC_LOW_CONF_VISIBLE)]
    assert high_boxes == [{"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.70, "class": "UAV"}]
    assert low_boxes == [
        {"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.70, "class": "UAV"},
        {"x": 5, "y": 6, "w": 7, "h": 8, "confidence": 0.20, "class": "UAV"},
    ]


def test_resolve_msdc_high_low_boxes_uses_two_calls_when_shared_disabled(monkeypatch):
    detector = DummyDetector()
    monkeypatch.setattr(Config, "MSDC_EXPORT_SHARE_LOW_HIGH_DET", False, raising=False)
    monkeypatch.setattr(Config, "MSDC_USE_LOW_DET", True, raising=False)

    high_boxes, low_boxes = msdc_detection.resolve_msdc_high_low_boxes(detector, _frame(), "visible")

    assert detector.processor.calls == [
        (_frame().shape, "visible", None),
        (_frame().shape, "visible", Config.MSDC_LOW_CONF_VISIBLE),
    ]
    assert high_boxes == [{"x": 9, "y": 9, "w": 9, "h": 9, "confidence": 0.91, "class": "UAV"}]
    assert len(low_boxes) == 2


def test_resolve_msdc_high_low_boxes_falls_back_to_high_only_when_low_disabled(monkeypatch):
    detector = DummyDetector()
    monkeypatch.setattr(Config, "MSDC_EXPORT_SHARE_LOW_HIGH_DET", True, raising=False)
    monkeypatch.setattr(Config, "MSDC_USE_LOW_DET", False, raising=False)

    high_boxes, low_boxes = msdc_detection.resolve_msdc_high_low_boxes(detector, _frame(), "visible")

    assert detector.processor.calls == [(_frame().shape, "visible", None)]
    assert high_boxes == [{"x": 9, "y": 9, "w": 9, "h": 9, "confidence": 0.91, "class": "UAV"}]
    assert low_boxes == []


def test_config_defaults_to_candidate_topk_no_roi_profile():
    assert Config.MSDC_EXPORT_SHARE_LOW_HIGH_DET is True
    assert Config.MSDC_LOW_OBS_TOPK == 32
    assert Config.MSDC_LOW_OBS_MIN_CONF == 0.25
    assert Config.MSDC_USE_ROI_REDETECT is False
    assert Config.MSDC_USE_TEMPLATE is False
    assert Config.MSDC_TEMPLATE_ENABLE is False
