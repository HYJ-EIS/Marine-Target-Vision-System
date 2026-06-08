import sys
from pathlib import Path

import cv2
import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, TrackState
from target_module.image_detect_module.utils.template_lock import TemplateLock


class TemplateTestConfig(Config):
    MSDC_TEMPLATE_ENABLE = True
    MSDC_MAX_ACTIVE_TEMPLATES = 2
    MSDC_TEMPLATE_UPDATE_THRESH = 0.7
    MSDC_TEMPLATE_SEARCH_SCALE = 3.0
    MSDC_TEMPLATE_MIN_SIZE = 8


class DisabledTemplateConfig(TemplateTestConfig):
    MSDC_TEMPLATE_ENABLE = False


def _pattern_patch(size=12):
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            patch[y, x] = ((x * 17 + y * 3) % 255, (x * 5 + y * 23) % 255, (x * 11 + y * 7) % 255)
    cv2.circle(patch, (size // 2, size // 2), 3, (255, 255, 255), -1)
    return patch


def _frame_with_patch(x=20, y=20, size=12):
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[y:y + size, x:x + size] = _pattern_patch(size)
    return frame


def _track(gid=1, state=TrackState.ACTIVE, box=None):
    return EvidenceTrack(
        gid=gid,
        state=state,
        box=box or [20, 20, 32, 32],
        evidence_score=3.0,
        hits=3,
        misses=0,
        age=3,
        last_seen=0,
        last_real_det_frame=1,
        real_det_hits=4,
        class_id=2,
        class_name="UAV",
    )


def test_candidate_tracks_do_not_initialize_templates():
    lock = TemplateLock(TemplateTestConfig)
    candidate = _track(state=TrackState.CANDIDATE)

    initialized = lock.init_template(candidate, _frame_with_patch())

    assert initialized is False
    assert candidate.template is None
    assert lock.template_count == 0


def test_active_track_initializes_matches_and_updates_template():
    lock = TemplateLock(TemplateTestConfig)
    track = _track()

    assert lock.init_template(track, _frame_with_patch()) is True
    assert track.template["initialized"] is True
    assert lock.template_count == 1

    observation = lock.match(track, _frame_with_patch(x=24, y=22), frame_idx=1, modality="visible")

    assert observation is not None
    assert observation.source == "template"
    assert observation.score >= TemplateTestConfig.MSDC_TEMPLATE_UPDATE_THRESH
    assert observation.raw["template_score"] == observation.score
    assert observation.raw["template_match_box"][0] in range(23, 26)
    assert observation.raw["template_match_box"][1] in range(21, 24)

    assert lock.should_update(track, observation, observation.score) is True
    assert lock.update_template(track, _frame_with_patch(x=24, y=22), observation) is True
    assert track.template["last_score"] == observation.score
    assert track.template["updated_count"] == 1


def test_template_update_requires_same_frame_real_detection():
    lock = TemplateLock(TemplateTestConfig)
    track = _track()
    track.last_real_det_frame = 0

    assert lock.init_template(track, _frame_with_patch()) is True
    observation = lock.match(track, _frame_with_patch(x=24, y=22), frame_idx=1, modality="visible")

    assert observation is not None
    assert lock.should_update(track, observation, observation.score) is False
    assert lock.update_template(track, _frame_with_patch(x=24, y=22), observation) is False


def test_template_count_is_bounded_to_max_active_templates():
    lock = TemplateLock(TemplateTestConfig)
    frame = _frame_with_patch()

    for gid in (1, 2, 3):
        assert lock.init_template(_track(gid=gid), frame) is True

    assert lock.template_count == TemplateTestConfig.MSDC_MAX_ACTIVE_TEMPLATES
    assert lock.has_template(1) is False
    assert lock.has_template(2) is True
    assert lock.has_template(3) is True


def test_disabled_template_lock_is_noop():
    lock = TemplateLock(DisabledTemplateConfig)
    track = _track()

    assert lock.init_template(track, _frame_with_patch()) is False
    assert lock.match(track, _frame_with_patch(), frame_idx=1) is None
    assert lock.template_count == 0
