import csv
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.validation.tracker_effect_test import (
    TRACK_CSV_FIELDS,
    annotated_video_filename,
    build_frame_record,
    build_video_jobs,
    count_jsonl_rows,
    normalize_box,
    resolve_tracker_resume_frame,
    write_tracks_csv_row,
)


def test_normalize_box_keeps_project_tracking_fields():
    box = {
        "track_id": 7,
        "x": 12,
        "y": 34,
        "w": 20,
        "h": 10,
        "confidence": 0.91234,
        "class": "USV",
        "class_confidence": 0.88,
    }

    assert normalize_box(box) == {
        "id": 7,
        "x": 12,
        "y": 34,
        "w": 20,
        "h": 10,
        "confidence": 0.9123,
        "class": "USV",
    }


def test_build_frame_record_writes_empty_frames():
    record = build_frame_record(frame_index=3, timestamp_sec=1.25, boxes=[])

    assert record == {
        "frame_index": 3,
        "timestamp_sec": 1.25,
        "boxes": [],
    }
    assert json.loads(json.dumps(record, ensure_ascii=False)) == record


def test_write_tracks_csv_row_uses_expected_header(tmp_path):
    output = tmp_path / "tracks.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_CSV_FIELDS)
        writer.writeheader()
        write_tracks_csv_row(
            writer,
            frame_index=2,
            box={
                "track_id": 5,
                "x": 10,
                "y": 20,
                "w": 30,
                "h": 40,
                "confidence": 0.99,
                "class": "UAV",
            },
        )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "frame,id,x,y,w,h,confidence,class",
        "2,5,10,20,30,40,0.9900,UAV",
    ]


def test_build_video_jobs_can_limit_modalities():
    jobs = build_video_jobs("ir.mp4", "rgb.mp4", modalities=["RGB"])

    assert len(jobs) == 1
    assert jobs[0].modality == "RGB"
    assert jobs[0].file_type == "visible"
    assert str(jobs[0].video_path) == "rgb.mp4"


def test_resume_helpers_count_existing_frame_rows(tmp_path):
    frames = tmp_path / "IR" / "dist_tracker" / "frames.jsonl"
    frames.parent.mkdir(parents=True)
    frames.write_text('{"frame_index": 1}\n{"frame_index": 2}\n', encoding="utf-8")

    assert count_jsonl_rows(frames) == 2
    assert resolve_tracker_resume_frame(tmp_path, "IR", "dist_tracker") == 2
    assert resolve_tracker_resume_frame(tmp_path, "RGB", "dist_tracker") == 0


def test_annotated_video_filename_uses_segment_for_resume():
    assert annotated_video_filename(0) == "annotated.mp4"
    assert annotated_video_filename(14450) == "annotated_part_014451.mp4"
