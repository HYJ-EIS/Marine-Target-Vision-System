import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.detection_replay_benchmark import (
    DETECTION_REPLAY_TRACKERS,
    load_detection_cache,
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
    assert tracker_output_name("botsort") == "botsort_replay"
    assert tracker_output_name("ocsort") == "ocsort_replay"
    assert tracker_output_name("msdc_elt") == "msdc_elt_high_replay"
    assert DETECTION_REPLAY_TRACKERS == ["botsort", "ocsort", "msdc_elt"]


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
    assert rows[0]["tracker"] == "botsort_replay"
    assert rows[0]["HOTA"] == "1.1"
    assert rows[1]["tracker"] == "ocsort_replay"
