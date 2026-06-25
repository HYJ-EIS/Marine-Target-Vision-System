import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import METRIC_FIELDS
from tools.evaluation import msdc_slice_metrics as slice_metrics


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_frame_in_any_slice_matches_inclusive_bounds():
    slices = [
        {"start_frame": "10", "end_frame": "12"},
        {"start_frame": "30", "end_frame": "30"},
    ]

    assert slice_metrics.frame_in_any_slice(10, slices)
    assert slice_metrics.frame_in_any_slice(12, slices)
    assert slice_metrics.frame_in_any_slice(30, slices)
    assert not slice_metrics.frame_in_any_slice(13, slices)


def test_reindex_mot_line_preserves_columns_except_frame():
    assert slice_metrics.reindex_mot_line("12,7,1,2,3,4,0.5,-1,-1,-1", 10) == (
        "3,7,1,2,3,4,0.5,-1,-1,-1"
    )


def test_write_slice_metric_rows_writes_required_fields(tmp_path):
    output = tmp_path / "slice_metrics.csv"
    row = {
        "seq_name": "seq_a",
        "tracker": "msdc_v3",
        "num_slices": 2,
        "total_slice_frames": 5,
        "MOTA": 100.0,
        "IDF1": 95.0,
    }

    written = slice_metrics.write_slice_metric_rows(output, [row])

    assert written == output
    with output.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["seq_name", "tracker", "num_slices", "total_slice_frames", *METRIC_FIELDS]
        rows = list(reader)
    assert rows[0]["seq_name"] == "seq_a"
    assert rows[0]["MOTA"] == "100.0"
    assert rows[0]["HOTA"] == ""


def test_build_slice_metrics_clips_and_reindexes_cumulative_frames(tmp_path, monkeypatch):
    manifest = tmp_path / "slice_manifest.csv"
    manifest.write_text(
        "seq_name,tracker,gt_id,start_frame,end_frame,length,slice_type\n"
        "seq_a,msdc_v3,1,10,11,2,short_miss\n"
        "seq_a,msdc_v3,1,20,21,2,short_miss\n",
        encoding="utf-8",
    )
    ablation_root = tmp_path / "ablation"
    _write_text(
        ablation_root / "mot_gt" / "seq_a" / "gt" / "gt.txt",
        "\n".join([
            "9,1,0,0,10,10,1,1,1",
            "10,1,10,10,10,10,1,1,1",
            "11,1,11,10,10,10,1,1,1",
            "20,1,20,10,10,10,1,1,1",
            "21,1,21,10,10,10,1,1,1",
            "22,1,22,10,10,10,1,1,1",
            "",
        ]),
    )
    _write_text(
        ablation_root / "trackers" / "msdc_v3" / "data" / "seq_a.txt",
        "\n".join([
            "10,5,10,10,10,10,0.9,-1,-1,-1",
            "11,5,11,10,10,10,0.8,-1,-1,-1",
            "20,5,20,10,10,10,0.7,-1,-1,-1",
            "21,5,21,10,10,10,0.6,-1,-1,-1",
            "",
        ]),
    )
    output_root = tmp_path / "slice"

    calls = []

    def fake_eval(gt_root, trackers_root, output_root, trackers, sequences, print_results):
        calls.append({
            "gt_root": Path(gt_root),
            "trackers_root": Path(trackers_root),
            "output_root": Path(output_root),
            "trackers": list(trackers),
            "sequences": list(sequences),
            "print_results": print_results,
        })
        return {"msdc_v3": {"MOTA": 50.0, "IDF1": 60.0, "IDSW": 1, "FP": 2, "FN": 3}}

    monkeypatch.setattr(slice_metrics, "run_motchallenge_eval", fake_eval)

    metrics_csv = slice_metrics.build_slice_metrics(
        slice_manifest=manifest,
        ablation_root=ablation_root,
        output_root=output_root,
    )

    synthetic_seq = "seq_a__msdc_v3__short_miss"
    assert metrics_csv == output_root / "slice_metrics.csv"
    assert (output_root / "mot_gt" / synthetic_seq / "gt" / "gt.txt").read_text(encoding="utf-8") == (
        "1,1,10,10,10,10,1,1,1\n"
        "2,1,11,10,10,10,1,1,1\n"
        "3,1,20,10,10,10,1,1,1\n"
        "4,1,21,10,10,10,1,1,1\n"
    )
    assert (output_root / "trackers" / "msdc_v3" / "data" / f"{synthetic_seq}.txt").read_text(
        encoding="utf-8"
    ) == (
        "1,5,10,10,10,10,0.9,-1,-1,-1\n"
        "2,5,11,10,10,10,0.8,-1,-1,-1\n"
        "3,5,20,10,10,10,0.7,-1,-1,-1\n"
        "4,5,21,10,10,10,0.6,-1,-1,-1\n"
    )
    assert "seqLength=4" in (output_root / "mot_gt" / synthetic_seq / "seqinfo.ini").read_text(encoding="utf-8")
    assert calls == [
        {
            "gt_root": output_root / "mot_gt",
            "trackers_root": output_root / "trackers",
            "output_root": output_root / "eval" / synthetic_seq,
            "trackers": ["msdc_v3"],
            "sequences": [synthetic_seq],
            "print_results": False,
        }
    ]
    rows = list(csv.DictReader((output_root / "slice_metrics.csv").read_text(encoding="utf-8-sig").splitlines()))
    assert rows[0]["seq_name"] == "seq_a"
    assert rows[0]["tracker"] == "msdc_v3"
    assert rows[0]["num_slices"] == "2"
    assert rows[0]["total_slice_frames"] == "4"
    assert rows[0]["MOTA"] == "50.0"
    eval_rows = list(
        csv.DictReader((output_root / "eval" / "motchallenge_summary.csv").read_text(encoding="utf-8-sig").splitlines())
    )
    assert eval_rows[0]["seq_name"] == "seq_a"
    assert eval_rows[0]["tracker"] == "msdc_v3"
    assert eval_rows[0]["MOTA"] == "50.0"


def test_build_slice_metrics_keeps_multiple_groups_in_aggregate_outputs(tmp_path, monkeypatch):
    manifest = tmp_path / "slice_manifest.csv"
    manifest.write_text(
        "seq_name,tracker,gt_id,start_frame,end_frame,length,slice_type\n"
        "seq_a,msdc_v3,1,10,10,1,short_miss\n"
        "seq_b,no_low_candidate,2,5,5,1,short_miss\n",
        encoding="utf-8",
    )
    ablation_root = tmp_path / "ablation"
    _write_text(ablation_root / "mot_gt" / "seq_a" / "gt" / "gt.txt", "10,1,10,10,10,10,1,1,1\n")
    _write_text(
        ablation_root / "trackers" / "msdc_v3" / "data" / "seq_a.txt",
        "10,5,10,10,10,10,0.9,-1,-1,-1\n",
    )
    _write_text(ablation_root / "mot_gt" / "seq_b" / "gt" / "gt.txt", "5,2,20,20,10,10,1,1,1\n")
    _write_text(
        ablation_root / "trackers" / "no_low_candidate" / "data" / "seq_b.txt",
        "5,7,20,20,10,10,0.8,-1,-1,-1\n",
    )
    output_root = tmp_path / "slice"
    calls = []

    def fake_eval(gt_root, trackers_root, output_root, trackers, sequences, print_results):
        calls.append({
            "output_root": Path(output_root),
            "trackers": list(trackers),
            "sequences": list(sequences),
        })
        tracker = list(trackers)[0]
        return {tracker: {"MOTA": 90.0 if tracker == "msdc_v3" else 80.0, "IDF1": 70.0}}

    monkeypatch.setattr(slice_metrics, "run_motchallenge_eval", fake_eval)

    slice_metrics.build_slice_metrics(
        slice_manifest=manifest,
        ablation_root=ablation_root,
        output_root=output_root,
    )

    seq_a = "seq_a__msdc_v3__short_miss"
    seq_b = "seq_b__no_low_candidate__short_miss"
    assert calls == [
        {"output_root": output_root / "eval" / seq_a, "trackers": ["msdc_v3"], "sequences": [seq_a]},
        {"output_root": output_root / "eval" / seq_b, "trackers": ["no_low_candidate"], "sequences": [seq_b]},
    ]
    metric_rows = list(csv.DictReader((output_root / "slice_metrics.csv").read_text(encoding="utf-8-sig").splitlines()))
    eval_rows = list(
        csv.DictReader((output_root / "eval" / "motchallenge_summary.csv").read_text(encoding="utf-8-sig").splitlines())
    )
    assert [(row["seq_name"], row["tracker"], row["MOTA"]) for row in metric_rows] == [
        ("seq_a", "msdc_v3", "90.0"),
        ("seq_b", "no_low_candidate", "80.0"),
    ]
    assert [(row["seq_name"], row["tracker"], row["MOTA"]) for row in eval_rows] == [
        ("seq_a", "msdc_v3", "90.0"),
        ("seq_b", "no_low_candidate", "80.0"),
    ]
