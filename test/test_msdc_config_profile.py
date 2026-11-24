"""Verify that shared edits and v3 overrides reach replay and child processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from target_module.image_detect_module.config import (
    Config, MSDC_V3_OVERRIDES, build_msdc_v3_config, msdc_config_to_env,
)
from tools.evaluation.detection_replay_benchmark import msdc_variant_context
from tools.evaluation.msdc_speed_benchmark import _coerce_formal_v3_value


def test_shared_edit_inherited_and_v3_override_wins(monkeypatch):
    monkeypatch.setattr(Config, 'MSDC_OUTPUT_MIN_BOX_SIZE', 7)
    monkeypatch.setattr(Config, 'MSDC_ASSOC_IOU_THRESH', .31)
    monkeypatch.setattr(Config, 'MSDC_MAX_ACTIVE_TRACKS', 16)
    profile = build_msdc_v3_config()
    assert profile['MSDC_OUTPUT_MIN_BOX_SIZE'] == 7
    assert profile['MSDC_ASSOC_IOU_THRESH'] == .31
    assert profile['MSDC_MAX_ACTIVE_TRACKS'] == 128
    with msdc_variant_context('msdc_v3', env_override=msdc_config_to_env(profile)):
        assert Config.MSDC_OUTPUT_MIN_BOX_SIZE == 7
        assert Config.MSDC_ASSOC_IOU_THRESH == .31
        assert Config.MSDC_MAX_ACTIVE_TRACKS == 128
    assert Config.MSDC_MAX_ACTIVE_TRACKS == 16


def test_v3_override_validated_and_alias_synchronized(monkeypatch):
    monkeypatch.setitem(MSDC_V3_OVERRIDES, 'MSDC_REMOVED_GUARD_FRAMES', 60)
    assert build_msdc_v3_config()['MSDC_removed_GUARD_FRAMES'] == 60
    monkeypatch.setitem(MSDC_V3_OVERRIDES, 'MSDC_TYPO', 1)
    with pytest.raises(ValueError, match='MSDC_TYPO'):
        build_msdc_v3_config()


def test_complete_profile_reaches_subprocess(monkeypatch, tmp_path):
    # Include a shared field that used to ignore environment exports.
    monkeypatch.setattr(Config, 'MSDC_ASSOC_IOU_THRESH', .31)
    monkeypatch.setattr(Config, 'MSDC_EVIDENCE_ALPHA', .7)
    monkeypatch.setitem(MSDC_V3_OVERRIDES, 'MSDC_MAX_ACTIVE_TRACKS', 96)
    profile = build_msdc_v3_config()
    profile['MSDC_LIFECYCLE_DEBUG_OUTPUT_DIR'] = str(tmp_path / 'Mixed Case')
    env = os.environ.copy()
    env.update(msdc_config_to_env(profile))
    code = (
        'import json; from target_module.image_detect_module.config import Config; '
        'print(json.dumps({k: getattr(Config, k) for k in dir(Config) '
        'if k.startswith("MSDC_") and not callable(getattr(Config,k))}))'
    )
    child = subprocess.run([sys.executable, '-c', code], env=env,
                           cwd=Path(__file__).resolve().parents[1],
                           text=True, capture_output=True, check=True)
    assert json.loads(child.stdout) == profile
    assert _coerce_formal_v3_value(
        'MSDC_LIFECYCLE_DEBUG_OUTPUT_DIR', profile['MSDC_LIFECYCLE_DEBUG_OUTPUT_DIR']
    ) == profile['MSDC_LIFECYCLE_DEBUG_OUTPUT_DIR']


def test_replay_restores_config_and_environment_after_exception(monkeypatch):
    monkeypatch.setenv('MSDC_EVIDENCE_ALPHA', '0.123')
    monkeypatch.setattr(Config, 'MSDC_EVIDENCE_ALPHA', .123)
    with pytest.raises(RuntimeError, match='stop'):
        with msdc_variant_context('msdc_v3'):
            assert Config.MSDC_EVIDENCE_ALPHA == .85
            raise RuntimeError('stop')
    assert Config.MSDC_EVIDENCE_ALPHA == .123
    assert os.environ['MSDC_EVIDENCE_ALPHA'] == '0.123'
