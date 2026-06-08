import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, TrackState
from target_module.image_detect_module.utils.roi_redetect import ROIRedetector


class ROITestConfig(Config):
    MSDC_USE_ROI_REDETECT = True
    MSDC_ROI_REDETECT_LOW_CONF = 0.12
    MSDC_ROI_REDETECT_ACTIVE_INTERVAL = 1
    MSDC_ROI_REDETECT_LOST_INTERVAL = 3
    MSDC_ROI_REDETECT_MAX_TRACKS = 8
    MSDC_ROI_REDETECT_SEARCH_SCALE = 2.0
    MSDC_ROI_REDETECT_UPSCALE = 2.0
    MSDC_ROI_REDETECT_MIN_CROP_SIZE = 32
    MSDC_ROI_REDETECT_EXISTING_IOU = 0.5
    MSDC_ROI_REDETECT_MIN_BOX_SIZE = 8
    MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI = 2


class DummyProcessor:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def process_frame(self, frame, file_type, conf_override=None):
        self.calls.append((frame.shape, file_type, conf_override))
        return {"boxes": list(self.boxes)}


def _frame():
    return np.zeros((100, 120, 3), dtype=np.uint8)


def _track(state=TrackState.ACTIVE, box=None, velocity=None, gid=3):
    return EvidenceTrack(
        gid=gid,
        public_id=gid,
        state=state,
        box=box or [40, 40, 60, 60],
        velocity=velocity or [0.0, 0.0],
        evidence_score=3.0,
        hits=4,
        misses=0,
        age=4,
        last_seen=4,
        last_real_det_frame=4,
        real_det_hits=4,
        class_id=2,
        class_name="UAV",
    )


def test_roi_redetect_maps_resized_crop_boxes_back_to_full_frame():
    processor = DummyProcessor([
        {"x": 30, "y": 30, "w": 20, "h": 20, "confidence": 0.5, "class": "UAV"},
    ])
    redetector = ROIRedetector(ROITestConfig, processor=processor)

    boxes, debug = redetector.update(
        frame=_frame(),
        tracks=[_track()],
        frame_idx=5,
        file_type="visible",
        existing_boxes=[],
    )

    assert processor.calls == [((80, 80, 3), "visible", ROITestConfig.MSDC_ROI_REDETECT_LOW_CONF)]
    assert boxes == [{
        "x": 45,
        "y": 45,
        "w": 10,
        "h": 10,
        "confidence": 0.5,
        "class": "UAV",
        "source": "roi_low_det",
        "roi_track_gid": 3,
        "roi_track_state": "active",
        "roi_origin": [30, 30],
        "roi_scale": 2.0,
    }]
    assert debug["num_roi_boxes"] == 1
    assert debug["num_rois"] == 1


def test_roi_redetect_skips_candidates_and_removed_tracks():
    processor = DummyProcessor([
        {"x": 10, "y": 10, "w": 10, "h": 10, "confidence": 0.5, "class": "UAV"},
    ])
    redetector = ROIRedetector(ROITestConfig, processor=processor)

    boxes, debug = redetector.update(
        frame=_frame(),
        tracks=[_track(state=TrackState.CANDIDATE), _track(state=TrackState.REMOVED, gid=4)],
        frame_idx=5,
        file_type="visible",
        existing_boxes=[],
    )

    assert boxes == []
    assert processor.calls == []
    assert debug["num_rois"] == 0


def test_roi_redetect_filters_boxes_already_covered_by_existing_detections():
    processor = DummyProcessor([
        {"x": 30, "y": 30, "w": 20, "h": 20, "confidence": 0.5, "class": "UAV"},
    ])
    redetector = ROIRedetector(ROITestConfig, processor=processor)

    boxes, debug = redetector.update(
        frame=_frame(),
        tracks=[_track()],
        frame_idx=5,
        file_type="visible",
        existing_boxes=[{"x": 45, "y": 45, "w": 10, "h": 10, "confidence": 0.9, "class": "UAV"}],
    )

    assert boxes == []
    assert debug["num_filtered_existing"] == 1


def test_roi_redetect_filters_tiny_boxes_before_reacquire():
    processor = DummyProcessor([
        {"x": 20, "y": 18, "w": 10, "h": 6, "confidence": 0.9, "class": "UAV"},
    ])
    redetector = ROIRedetector(ROITestConfig, processor=processor)

    boxes, debug = redetector.update(
        frame=_frame(),
        tracks=[_track()],
        frame_idx=5,
        file_type="visible",
        existing_boxes=[],
    )

    assert boxes == []
    assert debug["num_filtered_invalid"] == 1
    assert debug["num_roi_boxes"] == 0


def test_roi_redetect_caps_boxes_per_roi_by_score_and_track_center():
    processor = DummyProcessor([
        {"x": 2, "y": 2, "w": 18, "h": 18, "confidence": 0.4, "class": "UAV"},
        {"x": 30, "y": 30, "w": 18, "h": 18, "confidence": 0.7, "class": "UAV"},
        {"x": 36, "y": 30, "w": 18, "h": 18, "confidence": 0.6, "class": "UAV"},
    ])
    redetector = ROIRedetector(ROITestConfig, processor=processor)

    boxes, debug = redetector.update(
        frame=_frame(),
        tracks=[_track()],
        frame_idx=5,
        file_type="visible",
        existing_boxes=[],
    )

    assert len(boxes) == 2
    assert [box["confidence"] for box in boxes] == [0.7, 0.6]
    assert debug["num_capped_by_roi"] == 1
