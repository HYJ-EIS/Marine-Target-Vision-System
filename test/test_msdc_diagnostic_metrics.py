import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.msdc_diagnostic_metrics import (
    DIAGNOSTIC_FIELDS,
    compute_break_count,
    summarize_msdc_diagnostics,
    write_msdc_diagnostic_summary,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_compute_break_count_counts_non_contiguous_matched_segments():
    assert compute_break_count("1-3;5-8;10-10") == 2
    assert compute_break_count("1-10") == 0
    assert compute_break_count("") == 0


def test_summarize_msdc_diagnostics_reads_events_and_per_gt_rows(tmp_path):
    events_path = tmp_path / "lifecycle_events.jsonl"
    stage_path = tmp_path / "stage_observations.jsonl"
    per_gt_path = tmp_path / "msdc_per_gt_diagnostics.csv"
    _write_jsonl(
        events_path,
        [
            {"frame_idx": 1, "gid": 10, "event_type": "NEW_LOW_CANDIDATE"},
            {"frame_idx": 2, "gid": 11, "event_type": "NEW_LOW_CANDIDATE"},
            {"frame_idx": 3, "gid": 12, "event_type": "NEW_LOW_CANDIDATE"},
            {"frame_idx": 4, "gid": 10, "event_type": "LOW_CANDIDATE_CONFIRMED"},
            {
                "frame_idx": 6,
                "gid": 99,
                "event_type": "LOW_CANDIDATE_INHERITED",
                "extra": {"low_candidate_gid": 11, "inherit_result": "correct"},
            },
            {"frame_idx": 7, "gid": 20, "event_type": "REACQUIRED"},
        ],
    )
    _write_jsonl(stage_path, [{"frame_idx": 1, "boxes": []}])
    with per_gt_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gt_id", "predicted_id_count", "matched_segments"])
        writer.writeheader()
        writer.writerow({"gt_id": 1, "predicted_id_count": 3, "matched_segments": "1-3;5-8;10-10"})
        writer.writerow({"gt_id": 2, "predicted_id_count": 1, "matched_segments": "2-4"})

    row = summarize_msdc_diagnostics(events_path, stage_path, per_gt_path)

    assert row["low_candidate_confirmed"] == 2
    assert row["low_candidate_precision"] == round(2 / 3, 4)
    assert row["low_candidate_recall"] == "N/A"
    assert row["low_candidate_avg_confirm_delay"] == 3.5
    assert row["inherit_correct"] == 1
    assert row["inherit_wrong"] == 0
    assert row["inherit_ambiguous"] == 0
    assert row["reacquire_success"] == 1
    assert row["fragmentation_count"] == 2
    assert row["track_break_count"] == 2
    assert row["idsw_before_reacquire_inherit"] == "N/A"
    assert row["idsw_after_reacquire_inherit"] == "N/A"


def test_summarize_msdc_diagnostics_supports_current_event_aliases(tmp_path):
    events_path = tmp_path / "lifecycle_events.jsonl"
    per_gt_path = tmp_path / "per_gt.csv"
    _write_jsonl(
        events_path,
        [
            {"frame_idx": 0, "gid": 3, "event_type": "NEW_LOW_CANDIDATE"},
            {"frame_idx": 2, "gid": 3, "event_type": "CONFIRM_LOW_ACTIVE"},
            {"frame_idx": 4, "gid": 4, "event_type": "NEW_LOW_CANDIDATE"},
            {
                "frame_idx": 8,
                "gid": 20,
                "event_type": "LOW_CANDIDATE_INHERITED_LOST",
                "extra": {"low_candidate_gid": 4},
            },
            {"frame_idx": 9, "gid": 20, "event_type": "LOST_REACQUIRED"},
            {"frame_idx": 10, "gid": 20, "event_type": "REACQUIRE_LOST"},
        ],
    )
    per_gt_path.write_text("gt_id,predicted_id_count,matched_segments\n", encoding="utf-8")

    row = summarize_msdc_diagnostics(events_path, tmp_path / "missing_stage.jsonl", per_gt_path)

    assert row["low_candidate_confirmed"] == 2
    assert row["low_candidate_precision"] == 1.0
    assert row["low_candidate_avg_confirm_delay"] == 3.0
    assert row["inherit_ambiguous"] == 1
    assert row["reacquire_success"] == 2


def test_write_msdc_diagnostic_summary_writes_header_and_rows(tmp_path):
    output_path = tmp_path / "diagnostics" / "msdc_diagnostic_summary.csv"
    row = {field: "" for field in DIAGNOSTIC_FIELDS}
    row.update({"seq_name": "seq_a", "tracker": "msdc_v3", "low_candidate_confirmed": 1})

    written = write_msdc_diagnostic_summary(output_path, [row])

    assert written == output_path
    with output_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert reader.fieldnames == DIAGNOSTIC_FIELDS
    assert rows[0]["seq_name"] == "seq_a"
    assert rows[0]["tracker"] == "msdc_v3"
    assert rows[0]["low_candidate_confirmed"] == "1"


def test_write_msdc_diagnostic_summary_writes_header_for_empty_rows(tmp_path):
    output_path = tmp_path / "empty.csv"

    write_msdc_diagnostic_summary(output_path, [])

    assert output_path.read_text(encoding="utf-8-sig").splitlines() == [",".join(DIAGNOSTIC_FIELDS)]
