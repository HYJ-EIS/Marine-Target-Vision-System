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
    MSDC_ROI_REDETECT_ACTIVE_ENABLE = True
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
    MSDC_ROI_REDETECT_LOST_MAX_REAL_AGE = 30
    MSDC_ROI_REDETECT_COOLDOWN_FRAMES = 0


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


def test_roi_redetect_uses_processor_stage_context_when_available():
    from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, TrackState
    from target_module.image_detect_module.utils.roi_redetect import ROIRedetector

    class StageProcessor:
        def __init__(self):
            self.stage = "unspecified"
            self.calls = []

        def use_stage(self, stage):
            processor = self

            class _Context:
                def __enter__(self):
                    self.previous = processor.stage
                    processor.stage = stage
                    return processor

                def __exit__(self, exc_type, exc, tb):
                    processor.stage = self.previous
                    return False

            return _Context()

        def process_frame(self, frame, file_type, conf_override=None):
            self.calls.append((self.stage, frame.shape, file_type, conf_override))
            return {"boxes": [{"x": 2, "y": 2, "w": 10, "h": 10, "confidence": 0.8, "class": "UAV"}]}

    class ROIStageConfig(Config):
        MSDC_USE_ROI_REDETECT = True
        MSDC_ROI_REDETECT_ACTIVE_ENABLE = True
        MSDC_ROI_REDETECT_ACTIVE_INTERVAL = 1
        MSDC_ROI_REDETECT_MAX_TRACKS = 1
        MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI = 1
        MSDC_ROI_REDETECT_MIN_CROP_SIZE = 8
        MSDC_ROI_REDETECT_EXISTING_IOU = 1.1

    processor = StageProcessor()
    redetector = ROIRedetector(ROIStageConfig, processor=processor)
    track = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.ACTIVE,
        box=[10, 10, 30, 30],
        evidence_score=3.0,
        hits=4,
        misses=0,
        age=4,
        last_seen=0,
        last_real_det_frame=0,
        real_det_hits=4,
        class_id=2,
        class_name="UAV",
    )

    redetector.update(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        tracks=[track],
        frame_idx=1,
        file_type="visible",
        existing_boxes=[],
    )

    assert processor.calls
    assert {call[0] for call in processor.calls} == {"roi_redetect"}
    assert processor.stage == "unspecified"


def test_roi_redetect_defaults_skip_active_tracks_and_limit_stale_lost_tracks():
    class LostOnlyConfig(ROITestConfig):
        MSDC_ROI_REDETECT_ACTIVE_ENABLE = False
        MSDC_ROI_REDETECT_LOST_INTERVAL = 1
        MSDC_ROI_REDETECT_LOST_MAX_REAL_AGE = 30

    processor = DummyProcessor([
        {"x": 30, "y": 30, "w": 20, "h": 20, "confidence": 0.5, "class": "UAV"},
    ])
    redetector = ROIRedetector(LostOnlyConfig, processor=processor)
    active = _track(state=TrackState.ACTIVE, gid=1)
    recent_lost = _track(state=TrackState.LOST, gid=2)
    stale_lost = _track(state=TrackState.LOST, gid=3)
    stale_lost.last_real_det_frame = 0

    boxes, debug = redetector.update(
        frame=_frame(),
        tracks=[active, recent_lost, stale_lost],
        frame_idx=31,
        file_type="visible",
        existing_boxes=[],
    )

    assert len(boxes) == 1
    assert boxes[0]["roi_track_gid"] == 2
    assert [call[1:] for call in processor.calls] == [("visible", LostOnlyConfig.MSDC_ROI_REDETECT_LOW_CONF)]
    assert debug["num_candidate_tracks"] == 1
    assert debug["num_skipped_tracks"] == 2
    assert {item["reason"] for item in debug["skipped_tracks"]} == {"active_disabled", "lost_age_exceeded"}


def test_roi_redetect_applies_cooldown_after_empty_roi_result():
    class CooldownConfig(ROITestConfig):
        MSDC_ROI_REDETECT_ACTIVE_ENABLE = False
        MSDC_ROI_REDETECT_LOST_INTERVAL = 1
        MSDC_ROI_REDETECT_COOLDOWN_FRAMES = 4

    processor = DummyProcessor([])
    redetector = ROIRedetector(CooldownConfig, processor=processor)
    lost = _track(state=TrackState.LOST, gid=7)

    first_boxes, first_debug = redetector.update(_frame(), [lost], 6, "visible", existing_boxes=[])
    second_boxes, second_debug = redetector.update(_frame(), [lost], 7, "visible", existing_boxes=[])

    assert first_boxes == []
    assert second_boxes == []
    assert len(processor.calls) == 1
    assert first_debug["num_rois"] == 1
    assert second_debug["num_rois"] == 0
    assert second_debug["skipped_tracks"][0]["reason"] == "cooldown"


def test_roi_redetect_prunes_expired_and_absent_track_cooldowns():
    class CooldownConfig(ROITestConfig):
        MSDC_ROI_REDETECT_ACTIVE_ENABLE = False
        MSDC_ROI_REDETECT_LOST_INTERVAL = 1
        MSDC_ROI_REDETECT_COOLDOWN_FRAMES = 1

    processor = DummyProcessor([])
    redetector = ROIRedetector(CooldownConfig, processor=processor)
    lost_a = _track(state=TrackState.LOST, gid=7)
    lost_b = _track(state=TrackState.LOST, gid=8)

    redetector.update(_frame(), [lost_a], 6, "visible", existing_boxes=[])
    assert 7 in redetector._cooldown_until

    redetector.update(_frame(), [lost_b], 9, "visible", existing_boxes=[])

    assert 7 not in redetector._cooldown_until
