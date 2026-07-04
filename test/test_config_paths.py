import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config


def test_default_onnx_model_paths_exist_on_current_platform():
    assert Path(Config.ONNX_VISIBLE_MODEL_PATH).is_file()
    assert Path(Config.ONNX_INFRARED_MODEL_PATH).is_file()


def test_msdc_v3_formal_defaults_use_candidate_topk_profile():
    assert Config.MSDC_DEBUG_EVENTS is False
    assert Config.MSDC_LOW_OBS_TOPK == 32
    assert Config.MSDC_LOW_OBS_MIN_CONF == 0.25


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
            assert reloaded.Config.MSDC_REACQUIRE_INTERVAL == 1
            assert reloaded.Config.MSDC_REACQUIRE_SCORE == 0.8
            assert reloaded.Config.MSDC_REACQUIRE_IOU_THRESH == 0.05
            assert reloaded.Config.MSDC_REACQUIRE_CENTER_DIST == 160.0
    finally:
        importlib.reload(config_module)


def test_msdc_low_observation_budget_knobs_are_environment_backed(monkeypatch):
    import importlib
    import target_module.image_detect_module.config as config_module

    env_names = [
        "MSDC_LOW_OBS_TOPK",
        "MSDC_LOW_OBS_MIN_CONF",
    ]

    try:
        with monkeypatch.context() as override_env:
            for name in env_names:
                override_env.delenv(name, raising=False)
            override_env.setenv("MSDC_LOW_OBS_TOPK", "16")
            override_env.setenv("MSDC_LOW_OBS_MIN_CONF", "0.28")

            reloaded = importlib.reload(config_module)
            assert reloaded.Config.MSDC_LOW_OBS_TOPK == 16
            assert reloaded.Config.MSDC_LOW_OBS_MIN_CONF == 0.28
    finally:
        importlib.reload(config_module)


def test_msdc_lifecycle_acceleration_knobs_are_environment_backed(monkeypatch):
    import importlib
    import target_module.image_detect_module.config as config_module

    env_names = [
        "MSDC_LOST_MAX_AGE",
        "MSDC_REMOVED_GUARD_FRAMES",
        "MSDC_MAX_ACTIVE_TRACKS",
        "MSDC_MAX_LOST_TRACKS",
        "MSDC_MAX_CANDIDATES",
        "MSDC_MAX_LOW_CANDIDATES",
        "MSDC_MAX_TOTAL_TRACKS",
        "MSDC_LOW_OBS_GLOBAL_TOPK",
        "MSDC_LOW_OBS_PER_TRACK_NEAREST",
        "MSDC_LOW_OBS_MAX_PER_FRAME",
        "MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY",
        "MSDC_LOW_OBS_TRACK_PROXIMITY_CENTER_DIST",
        "MSDC_LOW_OBS_TRACK_PROXIMITY_IOU",
        "MSDC_LOW_OBS_MOTION_GATE_CENTER_DIST",
        "MSDC_LOW_OBS_MOTION_GATE_IOU",
        "MSDC_PENDING_RECOVERY_FRAMES",
    ]

    try:
        with monkeypatch.context() as override_env:
            for name in env_names:
                override_env.delenv(name, raising=False)
            override_env.setenv("MSDC_LOST_MAX_AGE", "33")
            override_env.setenv("MSDC_REMOVED_GUARD_FRAMES", "44")
            override_env.setenv("MSDC_MAX_ACTIVE_TRACKS", "11")
            override_env.setenv("MSDC_MAX_LOST_TRACKS", "12")
            override_env.setenv("MSDC_MAX_CANDIDATES", "13")
            override_env.setenv("MSDC_MAX_LOW_CANDIDATES", "14")
            override_env.setenv("MSDC_MAX_TOTAL_TRACKS", "15")
            override_env.setenv("MSDC_LOW_OBS_GLOBAL_TOPK", "16")
            override_env.setenv("MSDC_LOW_OBS_PER_TRACK_NEAREST", "2")
            override_env.setenv("MSDC_LOW_OBS_MAX_PER_FRAME", "18")
            override_env.setenv("MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY", "0")
            override_env.setenv("MSDC_LOW_OBS_TRACK_PROXIMITY_CENTER_DIST", "88")
            override_env.setenv("MSDC_LOW_OBS_TRACK_PROXIMITY_IOU", "0.07")
            override_env.setenv("MSDC_LOW_OBS_MOTION_GATE_CENTER_DIST", "99")
            override_env.setenv("MSDC_LOW_OBS_MOTION_GATE_IOU", "0.08")
            override_env.setenv("MSDC_PENDING_RECOVERY_FRAMES", "3")

            reloaded = importlib.reload(config_module)
            assert reloaded.Config.MSDC_LOST_MAX_AGE == 33
            assert reloaded.Config.MSDC_REMOVED_GUARD_FRAMES == 44
            assert reloaded.Config.MSDC_removed_GUARD_FRAMES == 44
            assert reloaded.Config.MSDC_MAX_ACTIVE_TRACKS == 11
            assert reloaded.Config.MSDC_MAX_LOST_TRACKS == 12
            assert reloaded.Config.MSDC_MAX_CANDIDATES == 13
            assert reloaded.Config.MSDC_MAX_LOW_CANDIDATES == 14
            assert reloaded.Config.MSDC_MAX_TOTAL_TRACKS == 15
            assert reloaded.Config.MSDC_LOW_OBS_GLOBAL_TOPK == 16
            assert reloaded.Config.MSDC_LOW_OBS_PER_TRACK_NEAREST == 2
            assert reloaded.Config.MSDC_LOW_OBS_MAX_PER_FRAME == 18
            assert reloaded.Config.MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY is False
            assert reloaded.Config.MSDC_LOW_OBS_TRACK_PROXIMITY_CENTER_DIST == 88.0
            assert reloaded.Config.MSDC_LOW_OBS_TRACK_PROXIMITY_IOU == 0.07
            assert reloaded.Config.MSDC_LOW_OBS_MOTION_GATE_CENTER_DIST == 99.0
            assert reloaded.Config.MSDC_LOW_OBS_MOTION_GATE_IOU == 0.08
            assert reloaded.Config.MSDC_PENDING_RECOVERY_FRAMES == 3

            override_env.delenv("MSDC_PENDING_RECOVERY_FRAMES", raising=False)
            reloaded = importlib.reload(config_module)
            assert reloaded.Config.MSDC_PENDING_RECOVERY_FRAMES == 1
    finally:
        importlib.reload(config_module)
