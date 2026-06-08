"""
Serializable MS-DC-ELT core data structures.

These types describe observations, lifecycle tracks, and lifecycle events only.
They do not implement evidence updates or connect to any baseline tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TrackState(Enum):
    LOW_CANDIDATE = "low_candidate"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


def _state_value(state: TrackState | str | None) -> str | None:
    if state is None:
        return None
    if isinstance(state, TrackState):
        return state.value
    return str(state)


def _box_list(box: Any) -> list[float]:
    arr = np.asarray(box, dtype=np.float64).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"Expected 4 box values, got {arr.size}")
    return [float(v) for v in arr.tolist()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, TrackState):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def xywh_to_xyxy(box: dict) -> np.ndarray:
    x = float(box.get("x", box.get("x1", 0.0)))
    y = float(box.get("y", box.get("y1", 0.0)))
    if "w" in box and "h" in box:
        w = float(box.get("w", 0.0))
        h = float(box.get("h", 0.0))
        return np.array([x, y, x + max(0.0, w), y + max(0.0, h)], dtype=np.float64)
    return np.array([
        x,
        y,
        float(box.get("x2", x)),
        float(box.get("y2", y)),
    ], dtype=np.float64)


def xyxy_to_project_box(
    xyxy,
    track_id: int,
    confidence: float,
    cls: str,
    class_confidence: float | None = None,
    **extras,
) -> dict:
    x1, y1, x2, y2 = _box_list(xyxy)
    x = int(round(x1))
    y = int(round(y1))
    w = max(0, int(round(x2 - x1)))
    h = max(0, int(round(y2 - y1)))
    conf = round(float(confidence), 4)
    result = {
        "track_id": int(track_id),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "confidence": conf,
        "class": str(cls),
        "class_confidence": round(float(class_confidence), 4) if class_confidence is not None else conf,
    }
    for key, value in extras.items():
        result[key] = _json_safe(value)
    return result


@dataclass
class Observation:
    box: Any
    source: str
    score: float
    reliability: float
    modality: str
    frame_idx: int
    class_id: int = -1
    class_name: str = "unknown"
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_project_box(
        cls,
        box: dict,
        source: str,
        frame_idx: int,
        modality: str,
        reliability: float = 1.0,
        class_id: int = -1,
    ) -> "Observation":
        score = float(box.get("confidence", box.get("score", 0.0)))
        return cls(
            box=xywh_to_xyxy(box),
            source=source,
            score=score,
            reliability=float(reliability),
            modality=modality,
            frame_idx=int(frame_idx),
            class_id=int(class_id),
            class_name=str(box.get("class", "unknown")),
            raw=dict(box),
        )

    def to_dict(self) -> dict:
        return observation_to_dict(self)


@dataclass
class EvidenceTrack:
    gid: int
    state: TrackState | str
    box: Any
    public_id: int | None = None
    velocity: Any = field(default_factory=lambda: [0.0, 0.0])
    evidence_score: float = 0.0
    hits: int = 0
    misses: int = 0
    age: int = 0
    last_seen: int = -1
    last_real_det_frame: int = -1
    real_det_hits: int = 0
    template_only_streak: int = 0
    motion_only_streak: int = 0
    last_real_det_box: Any = None
    low_det_history: list[dict] = field(default_factory=list)
    source_history: list[str] = field(default_factory=list)
    template: Any = None
    retired_signature: Any = None
    class_id: int = -1
    class_name: str = "unknown"

    def to_debug_dict(self) -> dict:
        return track_to_dict(self)

    def to_dict(self) -> dict:
        return track_to_dict(self)


@dataclass
class LifecycleEvent:
    frame_idx: int
    gid: int
    event_type: str
    from_state: TrackState | str | None = None
    to_state: TrackState | str | None = None
    reason: str = ""
    evidence_score: float = 0.0
    box: Any | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return event_to_dict(self)


def observation_to_dict(obs: Observation) -> dict:
    return {
        "box": _box_list(obs.box),
        "source": str(obs.source),
        "score": float(obs.score),
        "reliability": float(obs.reliability),
        "modality": str(obs.modality),
        "frame_idx": int(obs.frame_idx),
        "class_id": int(obs.class_id),
        "class_name": str(obs.class_name),
        "raw": _json_safe(obs.raw),
    }


def observation_from_project_box(
    box: dict,
    source: str,
    frame_idx: int,
    modality: str,
    reliability: float = 1.0,
    class_id: int = -1,
) -> Observation:
    return Observation.from_project_box(
        box=box,
        source=source,
        frame_idx=frame_idx,
        modality=modality,
        reliability=reliability,
        class_id=class_id,
    )


def track_to_dict(track: EvidenceTrack) -> dict:
    return {
        "gid": int(track.gid),
        "public_id": None if track.public_id is None else int(track.public_id),
        "state": _state_value(track.state),
        "box": _box_list(track.box),
        "velocity": _json_safe(track.velocity),
        "evidence_score": float(track.evidence_score),
        "hits": int(track.hits),
        "misses": int(track.misses),
        "age": int(track.age),
        "last_seen": int(track.last_seen),
        "last_real_det_frame": int(track.last_real_det_frame),
        "real_det_hits": int(track.real_det_hits),
        "template_only_streak": int(track.template_only_streak),
        "motion_only_streak": int(track.motion_only_streak),
        "last_real_det_box": None if track.last_real_det_box is None else _box_list(track.last_real_det_box),
        "low_det_history": _json_safe(track.low_det_history),
        "source_history": [str(source) for source in track.source_history],
        "template": _json_safe(track.template),
        "retired_signature": _json_safe(track.retired_signature),
        "class_id": int(track.class_id),
        "class_name": str(track.class_name),
    }


def track_to_output_box(track: EvidenceTrack) -> dict:
    output_id = track.public_id if track.public_id is not None else track.gid
    return xyxy_to_project_box(
        track.box,
        track_id=output_id,
        confidence=track.evidence_score,
        cls=track.class_name,
        class_confidence=track.evidence_score,
        gid=track.gid,
        public_id=track.public_id,
        lifecycle_state=_state_value(track.state),
    )


def event_to_dict(event: LifecycleEvent) -> dict:
    return {
        "frame_idx": int(event.frame_idx),
        "gid": int(event.gid),
        "event_type": str(event.event_type),
        "from_state": _state_value(event.from_state),
        "to_state": _state_value(event.to_state),
        "reason": str(event.reason),
        "evidence_score": float(event.evidence_score),
        "box": _box_list(event.box) if event.box is not None else None,
        "extra": _json_safe(event.extra),
    }


def event_to_json_dict(event: LifecycleEvent) -> dict:
    return event_to_dict(event)
