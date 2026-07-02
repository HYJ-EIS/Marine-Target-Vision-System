import csv
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.constants import DATASET_EXPORT_TRACKER_CHOICES
from tools.evaluation import detection_replay_benchmark as drb
from tools.evaluation.detection_replay_benchmark import (
    effective_max_frames,
    expected_replay_frames,
    high_boxes_from_cache_row,
    is_detection_cache_contiguous,
    is_detection_cache_complete,
    is_mot_result_complete,
    is_render_video_complete,
    load_detection_cache,
    low_boxes_from_cache_row,
    msdc_variant_context,
    parse_args,
    temporary_env,
    tracker_output_name,
    write_detection_cache,
    write_effective_msdc_config,
    write_replay_summary,
)


def test_detection_cache_round_trips_jsonl_rows(tmp_path):
    cache_path = tmp_path / "detections.jsonl"
    rows = [
        {
            "frame_id": 1,
            "boxes": [
                {"x": 10, "y": 20, "w": 30, "h": 40, "confidence": 0.9, "class": "USV"},
            ],
        },
        {"frame_id": 2, "boxes": []},
    ]

    write_detection_cache(cache_path, rows)

    assert load_detection_cache(cache_path) == rows
    raw_lines = cache_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(raw_lines[0])["frame_id"] == 1
    assert json.loads(raw_lines[1])["boxes"] == []


def test_load_detection_cache_can_clip_to_frame_count(tmp_path):
    cache_path = tmp_path / "detections.jsonl"
    write_detection_cache(
        cache_path,
        [
            {"frame_id": 1, "boxes": [{"x": 1}]},
            {"frame_id": 2, "boxes": [{"x": 2}]},
            {"frame_id": 3, "boxes": [{"x": 3}]},
        ],
    )

    rows = load_detection_cache(cache_path, max_frames=2)

    assert [row["frame_id"] for row in rows] == [1, 2]


def test_detection_cache_complete_requires_last_requested_frame(tmp_path):
    cache_path = tmp_path / "detections.jsonl"
    write_detection_cache(cache_path, [{"frame_id": 1}, {"frame_id": 3}])

    assert is_detection_cache_complete(cache_path, max_frames=3)
    assert not is_detection_cache_complete(cache_path, max_frames=4)
    assert not is_detection_cache_complete(tmp_path / "missing.jsonl", max_frames=3)


def test_detection_cache_contiguous_requires_every_requested_frame(tmp_path):
    cache_path = tmp_path / "detections.jsonl"
    write_detection_cache(cache_path, [{"frame_id": 1}, {"frame_id": 3}])

    assert not is_detection_cache_contiguous(cache_path, max_frames=3)
    assert is_detection_cache_contiguous(cache_path, max_frames=1)
    assert not is_detection_cache_contiguous(tmp_path / "missing.jsonl", max_frames=3)


def test_mot_result_complete_requires_last_requested_frame(tmp_path):
    mot_path = tmp_path / "seq.txt"
    mot_path.write_text(
        "1,1,10,10,20,20,0.9,-1,-1,-1\n"
        "3,1,12,10,20,20,0.8,-1,-1,-1\n",
        encoding="utf-8",
    )

    assert is_mot_result_complete(mot_path, max_frames=3)
    assert not is_mot_result_complete(mot_path, max_frames=4)
    assert not is_mot_result_complete(tmp_path / "missing.txt", max_frames=3)


def test_render_video_complete_checks_frame_count(tmp_path):
    video_path = tmp_path / "render.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (8, 8))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    writer.release()

    assert is_render_video_complete(video_path, max_frames=3)
    assert not is_render_video_complete(video_path, max_frames=4)
    assert not is_render_video_complete(tmp_path / "missing.mp4", max_frames=3)


def test_tracker_output_name_marks_replay_mode():
    assert tracker_output_name("bytetrack") == "bytetrack_replay"
    assert tracker_output_name("botsort") == "botsort_replay"
    assert tracker_output_name("ocsort") == "ocsort_replay"
    assert tracker_output_name("msdc_elt", "msdc_v3") == "msdc_v3_replay"
    assert tracker_output_name("msdc_elt", "no_low_candidate") == "no_low_candidate_replay"
    assert DATASET_EXPORT_TRACKER_CHOICES == ("bytetrack", "botsort", "ocsort", "msdc_elt")


def test_high_and_low_boxes_read_named_cache_columns():
    high_boxes = [{"x": 1, "confidence": 0.7}]
    low_boxes = [{"x": 2, "confidence": 0.2}]
    row = {"frame_id": 1, "high_boxes": high_boxes, "low_boxes": low_boxes}

    assert high_boxes_from_cache_row(row) == high_boxes
    assert low_boxes_from_cache_row(row) == low_boxes


def test_high_and_low_boxes_fall_back_to_legacy_boxes_column():
    boxes = [{"x": 3, "confidence": 0.8}]
    row = {"frame_id": 1, "boxes": boxes}

    assert high_boxes_from_cache_row(row) == boxes
    assert low_boxes_from_cache_row(row) == boxes


def test_write_replay_summary_writes_trackeval_fields(tmp_path):
    summary_path = write_replay_summary(
        tmp_path,
        {
            "botsort_replay": {"HOTA": 1.1, "MOTA": 2.2, "IDF1": 3.3, "IDSW": 4, "FP": 5, "FN": 6},
            "ocsort_replay": {"HOTA": 7.7, "MOTA": 8.8, "IDF1": 9.9, "IDSW": 10, "FP": 11, "FN": 12},
        },
    )

    with summary_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert summary_path == tmp_path / "summary" / "replay_summary.csv"
    assert set(["tracker", "variant", "replay_detections", "MOTA", "IDF1"]).issubset(rows[0])
    assert rows[0]["tracker"] == "botsort_replay"
    assert rows[0]["variant"] == ""
    assert rows[0]["replay_detections"] == "True"
    assert rows[0]["HOTA"] == "1.1"
    assert rows[1]["tracker"] == "ocsort_replay"


def test_write_effective_msdc_config_writes_sorted_json_under_run_root(tmp_path):
    path = write_effective_msdc_config(
        tmp_path,
        "no_low_candidate_replay",
        {"MSDC_USE_REACQUIRE": True, "MSDC_LOW_CANDIDATE_ENABLE": False},
    )

    assert path == tmp_path / "effective_config" / "no_low_candidate_replay.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "MSDC_LOW_CANDIDATE_ENABLE": False,
        "MSDC_USE_REACQUIRE": True,
    }
    assert path.read_text(encoding="utf-8").splitlines()[1].startswith('  "MSDC_LOW_CANDIDATE_ENABLE"')


def test_max_frames_controls_debug_replay_when_formal_limit_is_omitted():
    args = parse_args(["--dataset-root", "dataset_a", "--max-frames", "10"])

    assert args.formal_frame_limit == 0
    assert effective_max_frames(args) == 10


def test_parse_args_accepts_source_detection_cache_root():
    args = parse_args([
        "--dataset-root",
        "dataset_a",
        "--source-detection-cache-root",
        "previous_run/detections",
    ])

    assert args.source_detection_cache_root == "previous_run/detections"


def _patch_replay_run_dependencies(monkeypatch, video_path, expected_tracker_root):
    def fake_replay_tracker_from_cache(**kwargs):
        tracker_file = expected_tracker_root / "bytetrack_replay" / "data" / f"{kwargs['seq_name']}.txt"
        tracker_file.parent.mkdir(parents=True, exist_ok=True)
        tracker_file.write_text("1,1,10,10,20,20,0.9,-1,-1,-1\n", encoding="utf-8")
        return tracker_file

    monkeypatch.setattr(
        drb,
        "resolve_single_sequence_dataset",
        lambda dataset_root, video=None, seq_name=None: SimpleNamespace(
            dataset_root=Path(dataset_root),
            video_path=video_path,
            seq_name=seq_name or "seq_a",
        ),
    )
    monkeypatch.setattr(drb, "read_video_info", lambda path: SimpleNamespace(frame_count=3, fps=30.0))
    monkeypatch.setattr(drb, "write_motchallenge_gt_sequence", lambda *args, **kwargs: None)
    monkeypatch.setattr(drb, "replay_tracker_from_cache", fake_replay_tracker_from_cache)
    monkeypatch.setattr(
        drb,
        "write_diagnostics_for_tracker",
        lambda *args, **kwargs: (Path(args[2]) / f"{args[3]}.csv", None),
    )
    monkeypatch.setattr(drb, "write_stage_coverage_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(drb, "summarize_msdc_diagnostics", lambda *args, **kwargs: {})
    monkeypatch.setattr(drb, "write_msdc_diagnostic_summary", lambda *args, **kwargs: Path(args[0]))
    monkeypatch.setattr(
        drb,
        "run_motchallenge_eval",
        lambda **kwargs: {"bytetrack_replay": {"MOTA": 1.0, "IDF1": 1.0, "IDSW": 0, "FP": 0, "FN": 0}},
    )


def test_run_reuses_complete_source_detection_cache_without_dumping(monkeypatch, tmp_path):
    video_path = tmp_path / "seq_a.mp4"
    video_path.write_bytes(b"placeholder")
    source_root = tmp_path / "source_detections"
    source_root.mkdir()
    write_detection_cache(
        source_root / "seq_a_high_low_detections.jsonl",
        [{"frame_id": 1}, {"frame_id": 2}, {"frame_id": 3}],
    )
    output_root = tmp_path / "runs"
    run_id = "reuse_complete"
    _patch_replay_run_dependencies(monkeypatch, video_path, output_root / run_id / "trackers")
    monkeypatch.setattr(
        drb,
        "dump_video_detections",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("detector dump should not be called")),
    )

    summary_path = drb.run_detection_replay_benchmark(
        parse_args([
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--input",
            str(video_path),
            "--seq-name",
            "seq_a",
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
            "--max-frames",
            "3",
            "--trackers",
            "bytetrack",
            "--source-detection-cache-root",
            str(source_root),
        ])
    )

    linked_cache = output_root / run_id / "detections" / "seq_a_high_low_detections.jsonl"
    assert linked_cache.is_symlink()
    assert linked_cache.resolve() == (source_root / "seq_a_high_low_detections.jsonl").resolve()
    assert summary_path.is_file()


def test_run_with_incomplete_source_detection_cache_raises_before_dumping(monkeypatch, tmp_path):
    video_path = tmp_path / "seq_a.mp4"
    video_path.write_bytes(b"placeholder")
    source_root = tmp_path / "source_detections"
    source_root.mkdir()
    write_detection_cache(source_root / "seq_a_high_low_detections.jsonl", [{"frame_id": 1}, {"frame_id": 2}])
    output_root = tmp_path / "runs"
    run_id = "reuse_incomplete"
    _patch_replay_run_dependencies(monkeypatch, video_path, output_root / run_id / "trackers")
    monkeypatch.setattr(
        drb,
        "dump_video_detections",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("detector dump should not be called")),
    )

    with pytest.raises(RuntimeError, match="Incomplete source detection cache"):
        drb.run_detection_replay_benchmark(
            parse_args([
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--input",
                str(video_path),
                "--seq-name",
                "seq_a",
                "--output-root",
                str(output_root),
                "--run-id",
                run_id,
                "--max-frames",
                "3",
                "--trackers",
                "bytetrack",
                "--source-detection-cache-root",
                str(source_root),
            ])
        )


def test_run_with_sparse_source_detection_cache_raises_before_dumping(monkeypatch, tmp_path):
    video_path = tmp_path / "seq_a.mp4"
    video_path.write_bytes(b"placeholder")
    source_root = tmp_path / "source_detections"
    source_root.mkdir()
    write_detection_cache(source_root / "seq_a_high_low_detections.jsonl", [{"frame_id": 1}, {"frame_id": 3}])
    output_root = tmp_path / "runs"
    run_id = "reuse_sparse"
    _patch_replay_run_dependencies(monkeypatch, video_path, output_root / run_id / "trackers")
    monkeypatch.setattr(
        drb,
        "dump_video_detections",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("detector dump should not be called")),
    )

    with pytest.raises(RuntimeError, match="Incomplete source detection cache"):
        drb.run_detection_replay_benchmark(
            parse_args([
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--input",
                str(video_path),
                "--seq-name",
                "seq_a",
                "--output-root",
                str(output_root),
                "--run-id",
                run_id,
                "--max-frames",
                "3",
                "--trackers",
                "bytetrack",
                "--source-detection-cache-root",
                str(source_root),
            ])
        )


def test_run_exports_effective_config_when_msdc_replay_mot_is_skipped(monkeypatch, tmp_path):
    video_path = tmp_path / "seq_a.mp4"
    video_path.write_bytes(b"placeholder")
    output_root = tmp_path / "runs"
    run_id = "skip_complete_mot"
    run_root = output_root / run_id
    write_detection_cache(
        run_root / "detections" / "seq_a_high_low_detections.jsonl",
        [{"frame_id": 1}, {"frame_id": 2}, {"frame_id": 3}],
    )
    tracker_file = run_root / "trackers" / "no_low_candidate_replay" / "data" / "seq_a.txt"
    tracker_file.parent.mkdir(parents=True)
    tracker_file.write_text("3,1,10,10,20,20,0.9,-1,-1,-1\n", encoding="utf-8")

    monkeypatch.setattr(
        drb,
        "resolve_single_sequence_dataset",
        lambda dataset_root, video=None, seq_name=None: SimpleNamespace(
            dataset_root=Path(dataset_root),
            video_path=video_path,
            seq_name="seq_a",
        ),
    )
    monkeypatch.setattr(drb, "read_video_info", lambda path: SimpleNamespace(frame_count=3, fps=30.0))
    monkeypatch.setattr(drb, "write_motchallenge_gt_sequence", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        drb,
        "dump_video_detections",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("detector dump should not be called")),
    )
    monkeypatch.setattr(
        drb,
        "replay_tracker_from_cache",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("replay should not be called")),
    )
    monkeypatch.setattr(
        drb,
        "write_diagnostics_for_tracker",
        lambda *args, **kwargs: (Path(args[2]) / f"{args[3]}.csv", None),
    )
    monkeypatch.setattr(drb, "write_stage_coverage_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(drb, "write_msdc_diagnostic_summary", lambda *args, **kwargs: Path(args[0]))
    monkeypatch.setattr(
        drb,
        "run_motchallenge_eval",
        lambda **kwargs: {
            "no_low_candidate_replay": {"MOTA": 1.0, "IDF1": 1.0, "IDSW": 0, "FP": 0, "FN": 0},
        },
    )

    drb.run_detection_replay_benchmark(
        parse_args([
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--input",
            str(video_path),
            "--seq-name",
            "seq_a",
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
            "--max-frames",
            "3",
            "--trackers",
            "msdc_elt",
            "--variants",
            "no_low_candidate",
        ])
    )

    effective_config = json.loads(
        (run_root / "effective_config" / "no_low_candidate_replay.json").read_text(encoding="utf-8")
    )
    assert effective_config["MSDC_LOW_CANDIDATE_ENABLE"] is False


def test_expected_replay_frames_caps_formal_limit_to_video_length():
    assert expected_replay_frames(max_frames=5400, video_frame_count=4770) == 4770
    assert expected_replay_frames(max_frames=5400, video_frame_count=9000) == 5400
    assert expected_replay_frames(max_frames=5400, video_frame_count=0) == 5400


def test_temporary_env_can_isolate_and_restore_ambient_msdc_keys(monkeypatch):
    monkeypatch.setenv("MSDC_AMBIENT_ONLY", "leak")
    monkeypatch.setenv("MSDC_LOW_CANDIDATE_ENABLE", "ambient")
    monkeypatch.setenv("NON_MSDC_KEY", "keep")

    with temporary_env({"MSDC_LOW_CANDIDATE_ENABLE": "1"}, isolate_msdc=True):
        assert os.environ["MSDC_LOW_CANDIDATE_ENABLE"] == "1"
        assert "MSDC_AMBIENT_ONLY" not in os.environ
        assert os.environ["NON_MSDC_KEY"] == "keep"

    assert os.environ["MSDC_AMBIENT_ONLY"] == "leak"
    assert os.environ["MSDC_LOW_CANDIDATE_ENABLE"] == "ambient"
    assert os.environ["NON_MSDC_KEY"] == "keep"


def test_msdc_variant_context_resets_polluted_config_to_formal_baseline(monkeypatch):
    monkeypatch.setenv("MSDC_CONFIRM_REQUIRE_HIGH_DET", "0")
    monkeypatch.setattr(Config, "MSDC_CONFIRM_REQUIRE_HIGH_DET", False)
    polluted_baseline = dict(getattr(drb, "_MSDC_CONFIG_IMPORT_BASELINE", {}))
    polluted_baseline["MSDC_CONFIRM_REQUIRE_HIGH_DET"] = False
    monkeypatch.setattr(drb, "_MSDC_CONFIG_IMPORT_BASELINE", polluted_baseline, raising=False)

    with msdc_variant_context("msdc_v3"):
        assert os.environ["MSDC_CONFIRM_REQUIRE_HIGH_DET"] == "1"
        assert Config.MSDC_CONFIRM_REQUIRE_HIGH_DET is True

    assert os.environ["MSDC_CONFIRM_REQUIRE_HIGH_DET"] == "0"
    assert Config.MSDC_CONFIRM_REQUIRE_HIGH_DET is False


def test_msdc_variant_context_resets_missing_env_backed_config_key(monkeypatch):
    monkeypatch.setenv("MSDC_LOW_INHERIT_CLASS_MATCH", "0")
    monkeypatch.setattr(Config, "MSDC_LOW_INHERIT_CLASS_MATCH", False)
    polluted_baseline = dict(getattr(drb, "_MSDC_CONFIG_IMPORT_BASELINE", {}))
    polluted_baseline["MSDC_LOW_INHERIT_CLASS_MATCH"] = False
    monkeypatch.setattr(drb, "_MSDC_CONFIG_IMPORT_BASELINE", polluted_baseline, raising=False)

    with msdc_variant_context("msdc_v3"):
        assert os.environ["MSDC_LOW_INHERIT_CLASS_MATCH"] == "1"
        assert Config.MSDC_LOW_INHERIT_CLASS_MATCH is True

    assert os.environ["MSDC_LOW_INHERIT_CLASS_MATCH"] == "0"
    assert Config.MSDC_LOW_INHERIT_CLASS_MATCH is False


def test_msdc_variant_context_resets_removed_guard_alias(monkeypatch):
    monkeypatch.setattr(Config, "MSDC_REMOVED_GUARD_FRAMES", 44)
    monkeypatch.setattr(Config, "MSDC_removed_GUARD_FRAMES", 44)
    monkeypatch.setattr(Config, "MSDC_REMOVED_GUARD_IOU_THRESH", 0.9)
    monkeypatch.setattr(Config, "MSDC_REMOVED_GUARD_CENTER_DIST", 12.0)

    with msdc_variant_context("msdc_v3"):
        assert Config.MSDC_REMOVED_GUARD_FRAMES == 80
        assert Config.MSDC_removed_GUARD_FRAMES == 80
        assert Config.MSDC_REMOVED_GUARD_IOU_THRESH == 0.3
        assert Config.MSDC_REMOVED_GUARD_CENTER_DIST == 80.0

    assert Config.MSDC_REMOVED_GUARD_FRAMES == 44
    assert Config.MSDC_removed_GUARD_FRAMES == 44
    assert Config.MSDC_REMOVED_GUARD_IOU_THRESH == 0.9
    assert Config.MSDC_REMOVED_GUARD_CENTER_DIST == 12.0


def test_dynamic_empty_msdc_env_override_uses_formal_v3_baseline(tmp_path):
    path = drb.write_effective_msdc_variant_config(
        tmp_path,
        "msdc_v3_replay",
        "msdc_v3",
        env_override={},
    )

    effective_config = json.loads(path.read_text(encoding="utf-8"))
    assert effective_config["MSDC_MAX_ACTIVE_TRACKS"] == 128
    assert effective_config["MSDC_MAX_TOTAL_TRACKS"] == 256
    assert effective_config["MSDC_DEBUG_EVENTS"] is True
    assert effective_config["MSDC_LOW_CANDIDATE_ENABLE"] is True
