import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_module.image_detect_module.paper_config import apply_frozen_paper_config


def _inputs(tmp_path):
    spec = {"class_names": ["USV", "fishship", "UAV"],
            "frozen_parameters": {
                "full_profile": {"MSDC_ASSOC_USE_CENTER_GATE": True,
                                 "MSDC_CANDIDATE_CONFIRM_SCORE": 2.5},
                "baseline_and_detector_config": {"CLASSES": ["USV", "fishship", "UAV"],
                                                 "BYTETRACK_LOST_BUFFER": 40}},
            "variants": [{"name": "paper_no_a", "overrides": {"MSDC_ASSOC_USE_CENTER_GATE": False}}]}
    config = SimpleNamespace(MSDC_ASSOC_USE_CENTER_GATE=False, MSDC_CANDIDATE_CONFIRM_SCORE=9.0,
                             CLASSES=["old"], BYTETRACK_LOST_BUFFER=99)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return spec, config, path


def test_applies_full_and_baseline_before_registry_without_replacing_dicts(tmp_path):
    spec, config, path = _inputs(tmp_path)
    profile, env = {"stale": 4}, {"stale": "4"}
    profile_alias, env_alias = profile, env
    apply_frozen_paper_config(config, profile, env, path)
    assert config.BYTETRACK_LOST_BUFFER == 40
    assert config.MSDC_CANDIDATE_CONFIRM_SCORE == 2.5
    assert profile_alias == spec["frozen_parameters"]["full_profile"]
    assert env_alias == {"MSDC_ASSOC_USE_CENTER_GATE": "1", "MSDC_CANDIDATE_CONFIRM_SCORE": "2.5"}
    config.CLASSES.append("other")
    assert spec["class_names"] == ["USV", "fishship", "UAV"]


@pytest.mark.parametrize("bad", ["unknown", "type", "class_order", "variant_key"])
def test_rejects_incompatible_specs_before_mutating_config(tmp_path, bad):
    spec, config, path = _inputs(tmp_path)
    before = copy.deepcopy(vars(config))
    if bad == "unknown":
        spec["frozen_parameters"]["full_profile"]["MSDC_MISSING"] = True
    elif bad == "type":
        spec["frozen_parameters"]["full_profile"]["MSDC_ASSOC_USE_CENTER_GATE"] = "false"
    elif bad == "class_order":
        spec["class_names"].reverse()
    else:
        spec["variants"][0]["overrides"] = {"MSDC_NOT_FROZEN": 2}
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError):
        apply_frozen_paper_config(config, {}, {}, path)
    assert vars(config) == before
