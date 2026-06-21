import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.ir_reacquire_buffer import IRReacquireBuffer


def _args(**overrides):
    values = {
        "ir_reacquire_confirm_frames": 2,
        "ir_reacquire_max_age": 30,
        "ir_reacquire_iou": 0.05,
        "ir_reacquire_dist_factor": 2.5,
        "ir_reacquire_max_per_frame": 2,
        "ir_reacquire_min_conf": 0.45,
        "ir_reacquire_candidate_iou": 0.10,
        "ir_reacquire_candidate_dist_factor": 1.5,
        "ir_reacquire_max_gap": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ir_box(x=100, y=100, conf=0.8):
    return {"x": x, "y": y, "w": 20, "h": 20, "confidence": conf, "class": "USV"}


def _lost_track(x=102, y=101, age=3, track_id=7, cls="UAV"):
    return {
        "track_id": track_id,
        "x": x,
        "y": y,
        "w": 20,
        "h": 20,
        "class": cls,
        "confidence": 0.6,
        "time_since_update": age,
    }


def test_first_ir_only_frame_buffers_without_emitting():
    buffer = IRReacquireBuffer()

    detections, diagnostics = buffer.update(
        frame_id=1,
        ir_only_boxes=[_ir_box()],
        lost_track_boxes=[_lost_track()],
        args=_args(ir_reacquire_confirm_frames=2),
    )

    assert detections == []
    assert diagnostics[0]["decision"] == "buffered"
    assert diagnostics[0]["reject_reason"] == "not_confirmed"


def test_candidate_emits_after_consecutive_confirmation_near_lost_track():
    buffer = IRReacquireBuffer()
    args = _args(ir_reacquire_confirm_frames=2)
    buffer.update(1, [_ir_box()], [_lost_track()], args)

    detections, diagnostics = buffer.update(2, [_ir_box(x=101)], [_lost_track()], args)

    assert len(detections) == 1
    det = detections[0]
    assert det["support_type"] == "ir_lost_reacquire"
    assert det["allow_new_track"] is False
    assert det["class"] == "UAV"
    assert det["linked_track_id"] == 7
    assert any(row["decision"] == "emitted" for row in diagnostics)


def test_no_lost_track_means_no_emit_even_after_confirmation():
    buffer = IRReacquireBuffer()
    args = _args(ir_reacquire_confirm_frames=2)
    buffer.update(1, [_ir_box()], [], args)

    detections, diagnostics = buffer.update(2, [_ir_box(x=101)], [], args)

    assert detections == []
    assert any(row["reject_reason"] == "no_lost_track" for row in diagnostics)


def test_far_from_lost_track_is_rejected_by_gate():
    buffer = IRReacquireBuffer()
    args = _args(ir_reacquire_confirm_frames=2, ir_reacquire_dist_factor=0.5, ir_reacquire_iou=0.9)
    buffer.update(1, [_ir_box()], [_lost_track(x=300, y=300)], args)

    detections, diagnostics = buffer.update(2, [_ir_box(x=101)], [_lost_track(x=300, y=300)], args)

    assert detections == []
    assert any(row["reject_reason"] == "gate_failed" for row in diagnostics)


def test_low_conf_ir_candidate_is_rejected():
    buffer = IRReacquireBuffer()

    detections, diagnostics = buffer.update(
        frame_id=1,
        ir_only_boxes=[_ir_box(conf=0.2)],
        lost_track_boxes=[_lost_track()],
        args=_args(ir_reacquire_min_conf=0.45),
    )

    assert detections == []
    assert diagnostics[0]["reject_reason"] == "low_conf"


def test_max_per_frame_limits_emitted_candidates():
    buffer = IRReacquireBuffer()
    args = _args(ir_reacquire_confirm_frames=2, ir_reacquire_max_per_frame=1)
    tracks = [_lost_track(track_id=1, x=100), _lost_track(track_id=2, x=200)]
    buffer.update(1, [_ir_box(x=100), _ir_box(x=200)], tracks, args)

    detections, diagnostics = buffer.update(2, [_ir_box(x=101), _ir_box(x=201)], tracks, args)

    assert len(detections) == 1
    assert sum(1 for row in diagnostics if row["reject_reason"] == "max_per_frame") == 1


def test_candidate_gap_prunes_old_candidate():
    buffer = IRReacquireBuffer()
    args = _args(ir_reacquire_confirm_frames=2, ir_reacquire_max_gap=1)
    buffer.update(1, [_ir_box()], [_lost_track()], args)
    buffer.update(3, [], [_lost_track()], args)

    detections, diagnostics = buffer.update(4, [_ir_box()], [_lost_track()], args)

    assert detections == []
    assert any(row["reject_reason"] == "not_confirmed" for row in diagnostics)
