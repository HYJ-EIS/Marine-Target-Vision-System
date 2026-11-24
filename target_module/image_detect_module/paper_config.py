"""Apply the paper's fixed experiment values before trackers are imported."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path


def _check_type(key, actual, expected):
    if isinstance(expected, bool):
        valid = type(actual) is bool
    elif isinstance(expected, int):
        valid = type(actual) is int
    elif isinstance(expected, float):
        valid = isinstance(actual, (float, int)) and not isinstance(actual, bool)
    else:
        valid = isinstance(actual, type(expected))
    if not valid:
        raise ValueError(f"Fixed parameter {key} has incompatible type: {type(actual).__name__}")


def apply_frozen_paper_config(config, full_profile, formal_env, spec_path):
    """Apply one spec in place; imported profile dictionaries keep their identity."""
    spec_path = Path(spec_path).expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    fixed = spec["frozen_parameters"]
    profile = fixed["full_profile"]
    baseline = fixed["baseline_and_detector_config"]
    changes = {**baseline, **profile}
    for key, value in changes.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown fixed parameter {key}; implement the paper configuration first")
        _check_type(key, value, getattr(config, key))
    if tuple(baseline["CLASSES"]) != tuple(spec["class_names"]):
        raise ValueError("Fixed detector classes and paper class_names differ")
    if len(spec["variants"]) != len({row["name"] for row in spec["variants"]}):
        raise ValueError("Duplicate paper variant")
    for row in spec["variants"]:
        for key, value in row["overrides"].items():
            if key not in profile:
                raise ValueError(f"Variant {row['name']} changes a field absent from the fixed Full: {key}")
            _check_type(key, value, profile[key])
    for key, value in changes.items():
        setattr(config, key, value.copy() if isinstance(value, list) else value)
    full_profile.clear()
    full_profile.update(profile)
    formal_env.clear()
    formal_env.update({
        key: ("1" if value else "0") if isinstance(value, bool) else str(value)
        for key, value in profile.items()
    })
    return spec


def seed_paper_runtime(seed=20261030):
    """Seed the libraries used by inference without constructing a detector."""
    random.seed(seed)
    import numpy as np
    import cv2
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    seeded = ["random", "numpy", "opencv"]
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)
        seeded.append("torch")
    return {"seed": seed, "seeded_libraries": seeded}


def paper_environment():
    packages = {}
    for name in ("numpy", "scipy", "supervision", "onnxruntime", "onnxruntime-gpu",
                 "opencv-python", "pytest", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "package_versions": packages}
