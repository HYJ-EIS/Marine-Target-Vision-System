import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config


def test_default_onnx_model_paths_exist_on_current_platform():
    assert Path(Config.ONNX_VISIBLE_MODEL_PATH).is_file()
    assert Path(Config.ONNX_INFRARED_MODEL_PATH).is_file()


def test_msdc_v2_formal_defaults_avoid_template_and_duplicate_full_frame_detection():
    assert Config.MSDC_USE_TEMPLATE is False
    assert Config.MSDC_TEMPLATE_ENABLE is False
    assert Config.MSDC_EXPORT_SHARE_LOW_HIGH_DET is True


def test_msdc_v2_roi_budget_defaults_are_low_frequency():
    assert Config.MSDC_ROI_REDETECT_MAX_TRACKS == 2
    assert Config.MSDC_ROI_REDETECT_ACTIVE_INTERVAL == 8
    assert Config.MSDC_ROI_REDETECT_LOST_INTERVAL == 3
    assert Config.MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI == 1


def test_msdc_v2_tuning_knobs_are_environment_backed(monkeypatch):
    import importlib
    import target_module.image_detect_module.config as config_module

    monkeypatch.setenv("MSDC_CONFIRM_MIN_HITS", "3")
    monkeypatch.setenv("MSDC_CONFIRM_SCORE", "2.0")
    monkeypatch.setenv("MSDC_CANDIDATE_MAX_AGE", "8")
    monkeypatch.setenv("MSDC_REACQUIRE_INTERVAL", "1")
    monkeypatch.setenv("MSDC_REACQUIRE_SCORE", "1.2")
    monkeypatch.setenv("MSDC_REACQUIRE_CENTER_DIST", "240")

    reloaded = importlib.reload(config_module)
    try:
        assert reloaded.Config.MSDC_CONFIRM_MIN_HITS == 3
        assert reloaded.Config.MSDC_CONFIRM_SCORE == 2.0
        assert reloaded.Config.MSDC_CANDIDATE_MAX_AGE == 8
        assert reloaded.Config.MSDC_REACQUIRE_INTERVAL == 1
        assert reloaded.Config.MSDC_REACQUIRE_SCORE == 1.2
        assert reloaded.Config.MSDC_REACQUIRE_CENTER_DIST == 240.0
    finally:
        importlib.reload(config_module)
