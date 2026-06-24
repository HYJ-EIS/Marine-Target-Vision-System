import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import (
    BASELINE_TRACKER_CHOICES,
    DATASET_EXPORT_TRACKER_CHOICES,
    FORMAL_FRAME_LIMIT,
    FORMAL_MSDC_VARIANT,
    METHOD_LABELS,
    OPTIONAL_BASELINE_TRACKER_CHOICES,
    OPTIONAL_TRACKER_CHOICES,
    PAPER_TRACKER_CHOICES,
    SPEED_FIELDS,
    TRACKER_CHOICES,
)
from target_module.image_detect_module import constants as shared_constants
from tools.evaluation import msdc_experiment_summary, msdc_speed_benchmark


def test_tracker_choice_constants_preserve_cli_contracts():
    assert TRACKER_CHOICES == (
        "bytetrack",
        "ocsort",
        "botsort",
        "dist_tracker",
        "official_ocsort",
        "official_botsort",
        "msdc_elt",
    )
    assert OPTIONAL_TRACKER_CHOICES == ("", *TRACKER_CHOICES)
    assert BASELINE_TRACKER_CHOICES == TRACKER_CHOICES[:-1]
    assert OPTIONAL_BASELINE_TRACKER_CHOICES == ("", *BASELINE_TRACKER_CHOICES)
    assert PAPER_TRACKER_CHOICES == ("bytetrack", "ocsort", "botsort", "msdc_elt")
    assert DATASET_EXPORT_TRACKER_CHOICES == ("bytetrack", "botsort", "ocsort", "msdc_elt")


def test_formal_v3_and_baseline_tracker_contract():
    assert FORMAL_MSDC_VARIANT == "msdc_v3"
    assert PAPER_TRACKER_CHOICES == ("bytetrack", "ocsort", "botsort", "msdc_elt")
    assert FORMAL_FRAME_LIMIT == 5400


def test_tracker_choices_are_only_defined_in_shared_constants():
    production_files = [
        path for path in (_ROOT / "tools").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    production_files.extend([
        _ROOT / "video_main.py",
    ])
    constants_path = _ROOT / "target_module" / "image_detect_module" / "constants.py"

    forbidden_fragments = [
        "TRACKER_CHOICES =",
        "DETECTION_REPLAY_TRACKERS =",
        'choices=["bytetrack", "ocsort", "botsort"',
        'choices=["", "bytetrack", "ocsort", "botsort"',
        'choices=["bytetrack", "botsort", "ocsort", "msdc_elt"]',
        'default=["bytetrack", "botsort", "ocsort", "msdc_elt"]',
        '["bytetrack", "botsort", "ocsort", "msdc_elt"]',
        '["bytetrack", "ocsort", "botsort", "msdc_elt"]',
    ]

    for path in production_files:
        if path == constants_path:
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in text, f"{path.relative_to(_ROOT)} still hard-codes {fragment}"


def test_formal_msdc_variant_has_method_label():
    assert FORMAL_MSDC_VARIANT == "msdc_v3"
    assert METHOD_LABELS["bytetrack"] == "FFCA-YOLO + ByteTrack"
    assert METHOD_LABELS[FORMAL_MSDC_VARIANT] == "FFCA-YOLO + MS-DC-ELT v3"
    assert METHOD_LABELS["msdc_elt"] == "FFCA-YOLO + MS-DC-ELT"


def test_speed_fields_include_required_phase_timing_columns():
    assert "mean_read_ms" in SPEED_FIELDS
    assert "mean_high_det_ms" in SPEED_FIELDS
    assert "mean_low_det_ms" in SPEED_FIELDS
    assert "mean_tracker_ms" in SPEED_FIELDS
    assert "mean_msdc_total_update_ms" in SPEED_FIELDS
    assert "mean_render_ms" in SPEED_FIELDS
    assert "mean_write_ms" in SPEED_FIELDS


def test_speed_and_method_reporting_constants_are_shared_by_runtime_modules():
    assert msdc_speed_benchmark.SPEED_FIELDS is shared_constants.SPEED_FIELDS
    assert msdc_experiment_summary.SPEED_FIELDS is shared_constants.SPEED_FIELDS
    assert msdc_experiment_summary.SPEED_OUTPUT_FIELDS is shared_constants.SPEED_OUTPUT_FIELDS
    assert msdc_speed_benchmark.METHOD_LABELS is shared_constants.METHOD_LABELS
    assert msdc_experiment_summary.METHOD_LABELS is shared_constants.METHOD_LABELS
    assert msdc_experiment_summary.METRIC_FIELDS is shared_constants.METRIC_FIELDS
    assert "mean_msdc_total_update_ms" in msdc_experiment_summary.SPEED_FIELDS


def test_speed_and_method_reporting_constants_are_only_defined_once():
    production_files = [
        path for path in (_ROOT / "tools").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    production_files.extend([_ROOT / "video_main.py"])
    constants_path = _ROOT / "target_module" / "image_detect_module" / "constants.py"
    forbidden_fragments = [
        "SPEED_FIELDS =",
        "SPEED_OUTPUT_FIELDS =",
        "METHOD_LABELS =",
        "METRIC_FIELDS =",
        '"FFCA-YOLO +',
    ]

    for path in production_files:
        if path == constants_path:
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in text, f"{path.relative_to(_ROOT)} still duplicates {fragment}"
