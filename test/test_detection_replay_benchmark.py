import csv
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.constants import DATASET_EXPORT_TRACKER_CHOICES
from tools.evaluation import detection_replay_benchmark as drb
from tools.evaluation.detection_replay_benchmark import (
    effective_max_frames,
    high_boxes_from_cache_row,
    load_detection_cache,
    low_boxes_from_cache_row,
    msdc_variant_context,
    parse_args,
    temporary_env,
    tracker_output_name,
    write_detection_cache,
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


def test_max_frames_controls_debug_replay_when_formal_limit_is_omitted():
    args = parse_args(["--dataset-root", "dataset_a", "--max-frames", "10"])

    assert args.formal_frame_limit == 0
    assert effective_max_frames(args) == 10


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
