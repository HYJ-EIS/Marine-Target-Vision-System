import json
import sys
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.utils.msdc_types import (
    EvidenceTrack,
    LifecycleEvent,
    Observation,
    TrackState,
    event_to_dict,
    observation_to_dict,
    observation_from_project_box,
    track_to_output_box,
    track_to_dict,
    xywh_to_xyxy,
    xyxy_to_project_box,
)
from tools.evaluation.export_mot_results import format_mot_result_line
from visualization import visualize_detections


def test_observation_from_project_box_round_trips_to_json_dict():
    box = {
        "x": 12,
        "y": 34,
        "w": 20,
        "h": 10,
        "confidence": 0.42,
        "class": "UAV",
        "class_confidence": 0.4,
    }

    obs = observation_from_project_box(
        box,
        source="low_det",
        frame_idx=7,
        modality="visible",
        reliability=0.75,
        class_id=2,
    )
    payload = observation_to_dict(obs)

    assert payload == {
        "box": [12.0, 34.0, 32.0, 44.0],
        "source": "low_det",
        "score": 0.42,
        "reliability": 0.75,
        "modality": "visible",
        "frame_idx": 7,
        "class_id": 2,
        "class_name": "UAV",
        "raw": box,
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_evidence_track_debug_dict_is_json_serializable():
    track = EvidenceTrack(
        gid=3,
        state=TrackState.ACTIVE,
        box=[10, 20, 30, 50],
        velocity=[1.5, -0.5],
        evidence_score=0.875,
        hits=4,
        misses=1,
        age=6,
        last_seen=12,
        last_real_det_frame=11,
        real_det_hits=3,
        last_real_det_box=[9, 19, 29, 49],
        source_history=["high_det", "low_det"],
        class_id=2,
        class_name="UAV",
    )

    payload = track_to_dict(track)

    assert payload["gid"] == 3
    assert payload["public_id"] is None
    assert payload["state"] == "active"
    assert payload["box"] == [10.0, 20.0, 30.0, 50.0]
    assert payload["velocity"] == [1.5, -0.5]
    assert payload["evidence_score"] == 0.875
    assert payload["last_real_det_frame"] == 11
    assert payload["real_det_hits"] == 3
    assert payload["last_real_det_box"] == [9.0, 19.0, 29.0, 49.0]
    assert payload["source_history"] == ["high_det", "low_det"]
    assert payload["retired_signature"] is None
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_lifecycle_event_to_dict_is_json_serializable():
    event = LifecycleEvent(
        frame_idx=12,
        gid=3,
        event_type="transition",
        from_state=TrackState.CANDIDATE,
        to_state=TrackState.ACTIVE,
        reason="confirm",
        evidence_score=2.5,
        box=np.array([10, 20, 30, 50]),
        extra={"hits": 3},
    )

    payload = event_to_dict(event)

    assert payload == {
        "frame_idx": 12,
        "gid": 3,
        "event_type": "transition",
        "from_state": "candidate",
        "to_state": "active",
        "reason": "confirm",
        "evidence_score": 2.5,
        "box": [10.0, 20.0, 30.0, 50.0],
        "extra": {"hits": 3},
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_track_to_output_box_matches_visualization_and_mot_contract():
    track = EvidenceTrack(
        gid=9,
        public_id=2,
        state=TrackState.LOST,
        box=[10.2, 20.6, 30.8, 50.1],
        evidence_score=0.67891,
        class_name="USV",
        source_history=["reacquire"],
    )

    output = track_to_output_box(track)

    assert output == {
        "track_id": 2,
        "x": 10,
        "y": 21,
        "w": 21,
        "h": 30,
        "confidence": 0.6789,
        "class": "USV",
        "class_confidence": 0.6789,
        "gid": 9,
        "public_id": 2,
        "lifecycle_state": "lost",
    }

    mot_line = format_mot_result_line(5, output)
    assert mot_line.startswith("5,2,10.00,21.00,21.00,30.00,0.6789")

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = {"data": {"boxes": [output]}}
    rendered = visualize_detections(image, "video", result)
    assert rendered.shape == image.shape


def test_box_conversion_helpers_preserve_coordinates_and_clip_size():
    box = {"x": 5, "y": 7, "w": 11, "h": 13}

    xyxy = xywh_to_xyxy(box)
    restored = xyxy_to_project_box(
        xyxy,
        track_id=4,
        confidence=0.9,
        cls="fishship",
        class_confidence=0.8,
        lifecycle_state="active",
    )

    assert xyxy.tolist() == [5.0, 7.0, 16.0, 20.0]
    assert restored == {
        "track_id": 4,
        "x": 5,
        "y": 7,
        "w": 11,
        "h": 13,
        "confidence": 0.9,
        "class": "fishship",
        "class_confidence": 0.8,
        "lifecycle_state": "active",
    }


def test_track_states_are_constrained_to_expected_values():
    assert {state.value for state in TrackState} == {"low_candidate", "candidate", "active", "lost", "removed"}

    obs = Observation(
        box=[0, 0, 1, 1],
        source="low_det",
        score=0.1,
        reliability=0.2,
        modality="visible",
        frame_idx=1,
    )
    assert obs.to_dict() == observation_to_dict(obs)
