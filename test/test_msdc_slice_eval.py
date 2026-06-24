import csv
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation import msdc_slice_eval as slice_eval


def test_parse_segments_tolerates_ranges_singletons_and_malformed_parts():
    assert slice_eval._parse_segments("10-12; 30 ; ; bad ; 50-x ; 7-5") == [
        (10, 12),
        (30, 30),
    ]


def test_find_short_missing_segments_filters_by_length():
    rows = [
        {"gt_id": "1", "missed_segments": "10-12;30;50-90"},
        {"gt_id": "2", "missed_segments": "5-6"},
    ]

    slices = slice_eval.find_short_missing_segments(rows, min_len=2, max_len=30)

    assert slices == [
        {"gt_id": "1", "start_frame": 10, "end_frame": 12, "length": 3, "slice_type": "short_miss"},
        {"gt_id": "2", "start_frame": 5, "end_frame": 6, "length": 2, "slice_type": "short_miss"},
    ]


def test_write_slice_manifest_writes_header_when_empty(tmp_path):
    output = tmp_path / "slice" / "slice_manifest.csv"

    slice_eval.write_slice_manifest(output, [])

    assert output.read_text(encoding="utf-8").splitlines() == [",".join(slice_eval.SLICE_FIELDS)]


def test_build_slice_manifest_from_diagnostics(tmp_path):
    diagnostics_root = tmp_path / "diagnostics"
    seq_dir = diagnostics_root / "seq_a"
    seq_dir.mkdir(parents=True)
    (seq_dir / "msdc_elt_per_gt_diagnostics.csv").write_text(
        "gt_id,missed_segments\n"
        "1,10-12;50-85\n"
        "2,20\n",
        encoding="utf-8",
    )
    output = tmp_path / "slice_manifest.csv"

    written = slice_eval.build_slice_manifest_from_diagnostics(diagnostics_root, output, max_len=30)

    assert written == output
    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert rows == [
        {
            "seq_name": "seq_a",
            "tracker": "msdc_elt",
            "gt_id": "1",
            "start_frame": "10",
            "end_frame": "12",
            "length": "3",
            "slice_type": "short_miss",
        },
        {
            "seq_name": "seq_a",
            "tracker": "msdc_elt",
            "gt_id": "2",
            "start_frame": "20",
            "end_frame": "20",
            "length": "1",
            "slice_type": "short_miss",
        },
    ]


def test_cli_runs_directly_as_script(tmp_path):
    diagnostics_root = tmp_path / "diagnostics"
    seq_dir = diagnostics_root / "seq_b"
    seq_dir.mkdir(parents=True)
    (seq_dir / "bytetrack_per_gt_diagnostics.csv").write_text(
        "\ufeffgt_id,missed_segments\n3,100-101\n",
        encoding="utf-8",
    )
    output = tmp_path / "slice_manifest.csv"
    script = _ROOT / "tools" / "evaluation" / "msdc_slice_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--diagnostics-root",
            str(diagnostics_root),
            "--output",
            str(output),
            "--max-len",
            "30",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[OK] slice manifest:" in result.stdout
    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["tracker"] == "bytetrack"
