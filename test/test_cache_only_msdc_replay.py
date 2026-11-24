"""Cache-only is an execution mode for the fixed MSDC matrix, never a tracker change."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation import detection_replay_benchmark as drb

KINDS = ("observations", "associations", "states", "events", "outputs")


def arguments(tmp_path, **overrides):
    return {
        "input_video": tmp_path / "must_not_open.mp4",
        "detection_cache": tmp_path / "cache.jsonl",
        "output_root": tmp_path / "trackers",
        "tracker_type": "msdc_elt", "variant": "msdc_v3", "seq_name": "synthetic",
        "file_type": "visible", "frame_rate": 25, "max_frames": 2,
        "frame_source": "cache_only", **overrides,
    }


@pytest.mark.parametrize("tracker", ["bytetrack", "official_botsort", "official_ocsort"])
def test_cache_only_rejects_all_baselines_before_io(tmp_path, monkeypatch, tracker):
    monkeypatch.setattr(drb.cv2, "VideoCapture", lambda *_: pytest.fail("must not open pixels"))
    with pytest.raises(ValueError, match="seventeen fixed"):
        drb.replay_tracker_from_cache(**arguments(tmp_path, tracker_type=tracker, variant=""))
    assert not (tmp_path / "trackers").exists()


@pytest.mark.parametrize("length", [0, -1, 1.5, True])
def test_cache_only_requires_positive_explicit_n(tmp_path, length):
    with pytest.raises(ValueError, match="explicit positive"):
        drb.replay_tracker_from_cache(**arguments(tmp_path, max_frames=length))


@pytest.mark.parametrize("variant", ["", "gated_hungarian", "low_candidate_on", "made_up"])
def test_cache_only_rejects_configuration_outside_fixed_seventeen(tmp_path, variant):
    with pytest.raises(ValueError, match="seventeen fixed"):
        drb.replay_tracker_from_cache(**arguments(tmp_path, variant=variant))


def test_cache_only_rejects_dynamic_configuration_override(tmp_path):
    with pytest.raises(ValueError, match="dynamic"):
        drb.replay_tracker_from_cache(**arguments(tmp_path, env_override={"MSDC_CONFIRM_SCORE": "0"}))


@pytest.mark.parametrize("frames", [[], [1], [1, 3], [2, 1], [1, 1], [0, 1], [1, 2, 3], [1.0, 2], [True, 2], ["1", 2]])
def test_cache_only_requires_complete_ordered_unique_integer_frames(tmp_path, monkeypatch, frames):
    drb.write_detection_cache(tmp_path / "cache.jsonl", [{"frame_id": frame, "boxes": []} for frame in frames])
    monkeypatch.setattr(drb.cv2, "VideoCapture", lambda *_: pytest.fail("must not open pixels"))
    with pytest.raises(ValueError, match="exactly 1..2"):
        drb.replay_tracker_from_cache(**arguments(tmp_path))
    assert not (tmp_path / "trackers").exists()


@pytest.mark.parametrize("variant", sorted(drb.CACHE_ONLY_MSDC_VARIANTS))
def test_all_seventeen_keep_mot_and_five_logs_identical_without_pixel_access(tmp_path, monkeypatch, variant):
    assert len(drb.CACHE_ONLY_MSDC_VARIANTS) == 17
    length = 60
    box = {"x": 1000, "y": 2000, "w": 30, "h": 20, "confidence": 0.9, "class": "USV"}
    rows = [
        {"frame_id": frame, "high_boxes": [box] if frame <= 4 or frame >= 55 else [],
         "low_boxes": [box] if frame <= 4 or frame >= 55 else []}
        for frame in range(1, length + 1)
    ]
    drb.write_detection_cache(tmp_path / "cache.jsonl", rows)
    # The source array has different dimensions/content; MSDC does not clip boxes
    # against them. Empty detections still advance all sixty lifecycle frames.
    rng = np.random.default_rng(123)
    pixels = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(drb.cv2, "VideoCapture", lambda *_: SimpleNamespace(
        isOpened=lambda: True, read=lambda: (True, pixels), release=lambda: None,
    ))
    decoded = drb.replay_tracker_from_cache(**arguments(
        tmp_path, output_root=tmp_path / "decoded/trackers", variant=variant,
        max_frames=length, frame_source="decode",
    ))
    monkeypatch.setattr(drb.cv2, "VideoCapture", lambda *_: pytest.fail("cache_only opened a video"))
    cached = drb.replay_tracker_from_cache(**arguments(
        tmp_path, output_root=tmp_path / "cached/trackers", variant=variant, max_frames=length,
    ))
    assert cached.read_bytes() == decoded.read_bytes()
    for name in KINDS:
        original = decoded.parent.parent / "paper_logs/synthetic" / f"{name}.jsonl"
        optimized = cached.parent.parent / "paper_logs/synthetic" / f"{name}.jsonl"
        assert optimized.read_bytes() == original.read_bytes()
        envelopes = [json.loads(line) for line in optimized.read_text().splitlines()]
        assert [row["frame"] for row in envelopes] == list(range(1, length + 1))
    def stable_frames(path):
        return [(row["frame_id"], row["output_boxes"]) for row in
                map(json.loads, path.with_suffix(".frames.jsonl").read_text().splitlines())]
    assert stable_frames(cached) == stable_frames(decoded)
    timings = [json.loads(path.with_suffix(".timing.json").read_text()) for path in (decoded, cached)]
    assert timings[0]["frame_source"] == "decode" and timings[0]["pixel_frames_decoded"] == length
    assert timings[1]["frame_source"] == "cache_only" and timings[1]["pixel_frames_decoded"] == 0
    assert timings[1]["processed_frames"] == length and timings[1]["stage_seconds"]["read_decode"] == 0
    assert (tmp_path / "decoded/effective_config" / f"{variant}_replay.json").read_bytes() == (
        tmp_path / "cached/effective_config" / f"{variant}_replay.json").read_bytes()


def test_cached_source_metadata_uses_completed_main_without_video_io(tmp_path, monkeypatch):
    sequence = tmp_path / "main/mot_gt/synthetic"
    sequence.mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nframeRate=25\nimWidth=3840\nimHeight=2160\nseqLength=2\n")
    monkeypatch.setattr(drb.cv2, "VideoCapture", lambda *_: pytest.fail("metadata opened video"))
    info = drb._cached_source_video_info(tmp_path / "main/detections", "synthetic", 2)
    assert (info.fps, info.width, info.height, info.frame_count) == (25.0, 3840, 2160, 2)
    with pytest.raises(ValueError, match="requested 3 frames"):
        drb._cached_source_video_info(tmp_path / "main/detections", "synthetic", 3)


def test_cache_only_explicit_n_is_not_truncated_by_container_frame_count():
    assert drb.expected_replay_frames(13311, 13304, frame_source="cache_only") == 13311
    assert drb.expected_replay_frames(13311, 13304) == 13304
    assert drb.expected_replay_frames(287, 0, frame_source="cache_only") == 287


def test_cli_preserves_decode_default_and_requires_explicit_cached_mode_scope():
    assert drb.parse_args(["--dataset-root", "unused"]).frame_source == "decode"
    with pytest.raises(SystemExit):
        drb.parse_args(["--dataset-root", "unused", "--frame-source", "cache_only"])
    with pytest.raises(SystemExit):
        drb.parse_args(["--dataset-root", "unused", "--frame-source", "cache_only",
                        "--formal-frame-limit", "2", "--source-detection-cache-root", "main/detections"])
    args = drb.parse_args([
        "--dataset-root", "unused", "--frame-source", "cache_only", "--trackers", "msdc_elt",
        "--formal-frame-limit", "2", "--source-detection-cache-root", "main/detections",
    ])
    assert args.frame_source == "cache_only" and args.formal_frame_limit == 2
