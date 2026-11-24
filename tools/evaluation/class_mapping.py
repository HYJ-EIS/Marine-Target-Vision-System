"""Map configured model classes to the one-based classes used by MOT GT."""

from __future__ import annotations

import math
from collections.abc import Mapping

from target_module.image_detect_module.config import Config


def class_names() -> tuple[str, ...]:
    """Return the model's configured class order without a second codebook."""
    names = tuple(Config.CLASSES)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Config.CLASSES must contain unique, non-empty class names")
    return names


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    try:
        numeric = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric != int(numeric):
        raise ValueError(f"{field} must be a finite integer, got {value!r}")
    return int(numeric)


def model_index_to_gt_id(value: object) -> int:
    """Convert a zero-based model index to a one-based MOT class ID."""
    index = _integer(value, "model class_id")
    if not 0 <= index < len(class_names()):
        raise ValueError(f"Unknown model class_id {index}; expected 0..{len(class_names()) - 1}")
    return index + 1


def class_name_from_gt_id(value: object) -> str:
    """Decode a one-based MOT/GT class ID using the configured model order."""
    class_id = _integer(value, "MOT class ID")
    names = class_names()
    if not 1 <= class_id <= len(names):
        raise ValueError(f"Unknown MOT class ID {class_id}; expected 1..{len(names)}")
    return names[class_id - 1]


def normalize_prediction_class(box: Mapping) -> int:
    """Resolve a prediction's class, rejecting missing or conflicting fields.

    ``class`` is a configured name, ``class_id`` is a zero-based model index,
    and ``mot_class_id`` is an explicitly one-based class loaded from MOT.
    """
    candidates = []
    if "class" in box:
        name = box["class"]
        names = class_names()
        if not isinstance(name, str) or name not in names:
            raise ValueError(f"Unknown prediction class {name!r}; expected one of {names!r}")
        candidates.append(names.index(name) + 1)
    if "class_id" in box:
        candidates.append(model_index_to_gt_id(box["class_id"]))
    if "mot_class_id" in box:
        class_id = _integer(box["mot_class_id"], "mot_class_id")
        class_name_from_gt_id(class_id)
        candidates.append(class_id)
    if not candidates:
        raise ValueError("Prediction has no class: provide a configured class name or a zero-based class_id")
    if len(set(candidates)) != 1:
        raise ValueError(f"Conflicting prediction class fields: {dict(box)!r}")
    return candidates[0]
