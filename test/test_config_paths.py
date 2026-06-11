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


def test_msdc_v2_roi_budget_defaults_are_low_frequency(monkeypatch):
    import importlib
    import target_module.image_detect_module.config as config_module

    roi_env_names = [
        "MSDC_ROI_REDETECT_ACTIVE_INTERVAL",
        "MSDC_ROI_REDETECT_LOST_INTERVAL",
        "MSDC_ROI_REDETECT_MAX_TRACKS",
        "MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI",
    ]

    try:
        with monkeypatch.context() as env:
            for name in roi_env_names:
                env.delenv(name, raising=False)

            reloaded = importlib.reload(config_module)
            assert reloaded.Config.MSDC_ROI_REDETECT_MAX_TRACKS == 2
            assert reloaded.Config.MSDC_ROI_REDETECT_ACTIVE_INTERVAL == 8
            assert reloaded.Config.MSDC_ROI_REDETECT_LOST_INTERVAL == 3
            assert reloaded.Config.MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI == 1
    finally:
        importlib.reload(config_module)


def test_msdc_v2_tuning_knobs_are_environment_backed(monkeypatch):
    import importlib
    import target_module.image_detect_module.config as config_module

    env_names = [
        "MSDC_CONFIRM_MIN_HITS",
        "MSDC_CONFIRM_SCORE",
        "MSDC_CANDIDATE_MAX_AGE",
        "MSDC_REACQUIRE_INTERVAL",
        "MSDC_REACQUIRE_SCORE",
        "MSDC_REACQUIRE_IOU_THRESH",
        "MSDC_REACQUIRE_CENTER_DIST",
    ]

    try:
        with monkeypatch.context() as clean_env:
            for name in env_names:
                clean_env.delenv(name, raising=False)

            with monkeypatch.context() as override_env:
                override_env.setenv("MSDC_CONFIRM_MIN_HITS", "3")
                override_env.setenv("MSDC_CONFIRM_SCORE", "2.0")
                override_env.setenv("MSDC_CANDIDATE_MAX_AGE", "8")
                override_env.setenv("MSDC_REACQUIRE_INTERVAL", "1")
                override_env.setenv("MSDC_REACQUIRE_SCORE", "1.2")
                override_env.setenv("MSDC_REACQUIRE_IOU_THRESH", "0.12")
                override_env.setenv("MSDC_REACQUIRE_CENTER_DIST", "240")

                reloaded = importlib.reload(config_module)
                assert reloaded.Config.MSDC_CONFIRM_MIN_HITS == 3
                assert reloaded.Config.MSDC_CONFIRM_SCORE == 2.0
                assert reloaded.Config.MSDC_CANDIDATE_MAX_AGE == 8
                assert reloaded.Config.MSDC_REACQUIRE_INTERVAL == 1
                assert reloaded.Config.MSDC_REACQUIRE_SCORE == 1.2
                assert reloaded.Config.MSDC_REACQUIRE_IOU_THRESH == 0.12
                assert reloaded.Config.MSDC_REACQUIRE_CENTER_DIST == 240.0

            reloaded = importlib.reload(config_module)
            assert reloaded.Config.MSDC_CONFIRM_MIN_HITS == 4
            assert reloaded.Config.MSDC_CONFIRM_SCORE == 2.5
            assert reloaded.Config.MSDC_CANDIDATE_MAX_AGE == 5
            assert reloaded.Config.MSDC_REACQUIRE_INTERVAL == 5
            assert reloaded.Config.MSDC_REACQUIRE_SCORE == 1.5
            assert reloaded.Config.MSDC_REACQUIRE_IOU_THRESH == 0.05
            assert reloaded.Config.MSDC_REACQUIRE_CENTER_DIST == 160.0
    finally:
        importlib.reload(config_module)
