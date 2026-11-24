import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, TrackState, track_to_dict
from target_module.image_detect_module.utils.paper_logging import PAPER_LOG_KINDS, PaperFrameLogger


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_complete_sink_writes_all_streams_and_empty_frames(tmp_path):
    with PaperFrameLogger(tmp_path, "seq", "method") as sink:
        sink.write_frame(1, {"events": [{"gid": np.int64(i)} for i in range(12)],
                             "states": [{"gid": i, "box": np.arange(4)} for i in range(131)]})
        sink.write_frame(2, {})
    for kind in PAPER_LOG_KINDS:
        data = rows(tmp_path / f"{kind}.jsonl")
        assert [(row["seq"], row["method"], row["frame"]) for row in data] == [
            ("seq", "method", 1), ("seq", "method", 2)]
        assert data[1]["records"] == []
    assert len(rows(tmp_path / "events.jsonl")[0]["records"]) == 12
    assert len(rows(tmp_path / "states.jsonl")[0]["records"]) == 131


def test_sink_rejects_noncontiguous_frames(tmp_path):
    with PaperFrameLogger(tmp_path, "seq", "method") as sink:
        with pytest.raises(ValueError, match="contiguous"):
            sink.write_frame(2, {})


class LoggingConfig(Config):
    MSDC_DEBUG_EVENTS = False
    MSDC_DEBUG_DETAIL_LIMIT = 8
    MSDC_DEBUG_TRACK_SNAPSHOT_LIMIT = 128
    MSDC_OUTPUT_NMS_ENABLE = True
    MSDC_OUTPUT_NMS_CENTER_DIST = 5.0
    MSDC_OUTPUT_MIN_BOX_SIZE = 5


def active_track(gid, x=10):
    return EvidenceTrack(gid=gid, state=TrackState.ACTIVE,
                         box=np.array([x, 10, x + 10, 20], dtype=float),
                         public_id=gid, evidence_score=10 - gid / 1000,
                         hits=4, real_det_hits=4, age=4, last_seen=0,
                         last_real_det_frame=0, class_id=2, class_name="UAV")


def test_lifecycle_keeps_deleted_and_nms_objects_beyond_debug_limits(tmp_path):
    tracker = MSDCLifecycleTracker(config=LoggingConfig)
    live = [active_track(gid, 10 if gid <= 12 else gid * 30) for gid in range(1, 130)]
    deleted = active_track(130, 4000)
    snapshots = [dict(track_to_dict(track), retained=True) for track in live]
    snapshots += [dict(track_to_dict(deleted), retained=False, deletion_reason="capacity")]
    trace = {"observations": [], "associations": [], "states": snapshots,
             "events": [{"event_type": "confirmation", "gid": i} for i in range(1, 11)]}

    class Updater:
        paper_trace_enabled = False
        paper_trace_s = 0.0
        last_paper_trace = trace
        def update_tracks(self, tracks, observations, frame_idx):
            return live, []

    tracker.evidence_updater = Updater()
    tracker.enable_paper_logging(tmp_path, "seq", "msdc_v3")
    output = tracker.update(np.zeros((32, 32, 3), np.uint8), 0, "visible", [], [])
    tracker.close_paper_logging()
    outputs = rows(tmp_path / "outputs.jsonl")[0]["records"]
    events = rows(tmp_path / "events.jsonl")[0]["records"]
    assert len(outputs) == 130
    assert len(rows(tmp_path / "states.jsonl")[0]["records"]) == 130
    assert len([row for row in events if row["event_type"] == "output_nms"]) == 11
    assert len(tracker.last_output_nms_debug["suppressed"]) == 8
    assert len([row for row in outputs if row["rejection_reason"] == "output_nms"]) == 11
    assert next(row for row in outputs if row["gid"] == 130)["rejection_reason"] == "capacity"
    emitted = [row for row in outputs if row["emitted"]]
    assert len(emitted) == len(output)
    assert emitted[0]["exported_box_xyxy"] == [10.0, 10.0, 20.0, 20.0]
    assert all(row["model_class_id"] == 2 for row in outputs)
    assert tracker.last_timing_debug["paper_export_s"] > 0
    assert tracker.last_timing_debug["debug_s"] >= tracker.last_timing_debug["paper_export_s"]


def test_lifecycle_records_empty_frame_and_preserves_normal_output(tmp_path):
    tracker = MSDCLifecycleTracker(config=LoggingConfig)
    tracker.enable_paper_logging(tmp_path, "empty", "msdc_v3")
    assert tracker.update(np.zeros((16, 16, 3), np.uint8), 0, "visible", [], []) == []
    tracker.close_paper_logging()
    for kind in PAPER_LOG_KINDS:
        assert rows(tmp_path / f"{kind}.jsonl") == [
            {"seq": "empty", "method": "msdc_v3", "frame": 1, "records": []}]


def test_logging_does_not_change_decisions_or_actual_classes(tmp_path):
    first, second = MSDCLifecycleTracker(config=LoggingConfig), MSDCLifecycleTracker(config=LoggingConfig)
    second.enable_paper_logging(tmp_path, "classes", "msdc_v3")
    frame = np.zeros((64, 64, 3), np.uint8)
    box = {"x": 10, "y": 10, "w": 10, "h": 10, "confidence": .9, "class": "UAV"}
    for idx in range(6):
        assert first.update(frame, idx, "visible", [box], []) == second.update(frame, idx, "visible", [box], [])
    second.close_paper_logging()
    emitted = [r for row in rows(tmp_path / "outputs.jsonl") for r in row["records"] if r["emitted"]]
    assert emitted and all(row["model_class_id"] == 2 for row in emitted)


def test_baseline_metadata_uses_actual_runtime_attributes():
    from tools.evaluation.detection_replay_benchmark import baseline_constructor_parameters
    byte = SimpleNamespace(tracker_type="bytetrack", frame_rate=25, _min_hits=4,
                           _bt=SimpleNamespace(track_activation_threshold=.5, minimum_matching_threshold=.8,
                                               minimum_consecutive_frames=3, det_thresh=.5, max_time_lost=40))
    result = baseline_constructor_parameters(byte)
    assert result["constructor_parameters"]["minimum_consecutive_frames"] == 3
    assert result["constructor_parameters"]["max_time_lost"] == 40
    args = SimpleNamespace(track_high_thresh=.3, match_thresh=.8, with_reid=False)
    bot = SimpleNamespace(tracker_type="official_botsort", frame_rate=29.97, _min_hits=3,
                          _official=SimpleNamespace(frame_rate=29.97,
                              _tracker=SimpleNamespace(args=args, max_time_lost=77, buffer_size=77)))
    result = baseline_constructor_parameters(bot)
    assert result["constructor_parameters"]["match_thresh"] == .8
    assert result["constructor_parameters"]["frame_rate"] == 29
    assert result["applied_max_time_lost"] == 77


def test_synthetic_video_replay_exports_complete_paper_streams(tmp_path):
    import cv2
    from tools.evaluation.detection_replay_benchmark import replay_tracker_from_cache, write_detection_cache
    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 32))
    for _ in range(6):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()
    box = {"x": 10, "y": 10, "w": 10, "h": 10, "confidence": .9, "class": "UAV"}
    cache = tmp_path / "detections.jsonl"
    write_detection_cache(cache, [{"frame_id": i, "high_boxes": [box], "low_boxes": [box]}
                                  for i in range(1, 7)])
    mot = replay_tracker_from_cache(input_video=video, detection_cache=cache,
        output_root=tmp_path / "trackers", tracker_type="msdc_elt", variant="msdc_v3",
        seq_name="synthetic", file_type="visible", frame_rate=25, max_frames=6,
        baseline_input="all")
    logs = mot.parent.parent / "paper_logs" / "synthetic"
    for kind in PAPER_LOG_KINDS:
        assert [row["frame"] for row in rows(logs / f"{kind}.jsonl")] == list(range(1, 7))
    states = rows(logs / "states.jsonl")
    assert states[-1]["records"][0]["public_id"] is not None
    emitted = [record for row in rows(logs / "outputs.jsonl")
               for record in row["records"] if record["emitted"]]
    assert emitted
    assert len(emitted) == len(mot.read_text().splitlines())
    timing = json.loads(mot.with_suffix(".timing.json").read_text())
    assert timing["processed_frames"] == 6
    assert timing["diagnostic_seconds"] > 0
    assert timing["raw_update_seconds"] > timing["diagnostic_seconds"]
