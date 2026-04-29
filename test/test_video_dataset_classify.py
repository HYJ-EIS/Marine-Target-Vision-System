import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from video_dataset_classify import (
    WINDOW_RESULT_COLUMNS,
    build_time_windows,
    build_video_summary,
    classify_scene,
    compute_date_concentration,
    enrich_report_rows,
    evaluate_video_usability,
    evaluate_window_usability,
    find_best_overlap_window,
    is_t_scene_low_confidence,
    merge_scene_and_target_results,
    max_consecutive_false,
    normalize_worker_count,
    VideoAnalysisResult,
    window_overlap_ratio,
)


def make_window(unit_id: str, modality: str, start: float, end: float, **overrides):
    row = {
        "unit_id": unit_id,
        "date_batch": "0711",
        "modality": modality,
        "pair_status": "paired",
        "window_start": start,
        "window_end": end,
        "frame_count": 12,
        "low_fps_window": False,
        "duration_from_metadata": True,
        "decode_error": False,
        "primary_scene": "sea_sky",
        "sea_ratio": 0.5,
        "sky_ratio": 0.4,
        "shoreline_land_ratio": 0.1,
        "scene_classification_failed": False,
        "scene_classification_low_confidence": False,
        "scene_source": "native",
        "target_presence": True,
        "target_count": 2,
        "track_count": 2,
        "bbox_area_ratio_stats": json.dumps(
            {"count": 2, "min": 0.001, "max": 0.003, "mean": 0.002, "median": 0.002}
        ),
        "scale_bin_counts": json.dumps({"small": 2, "medium": 0, "large": 0}),
        "small_target_count": 2,
        "detector_not_run": False,
        "usable": True,
    }
    row.update(overrides)
    return row


def test_classify_scene_respects_fixed_rules():
    assert classify_scene(0.92, 0.04, 0.04) == "pure_sea"
    assert classify_scene(0.05, 0.93, 0.02) == "pure_sky"
    assert classify_scene(0.45, 0.46, 0.09) == "sea_sky"
    assert classify_scene(0.55, 0.25, 0.20) == "nearshore_sea_sky"
    assert classify_scene(0.72, 0.12, 0.16) == "nearshore_sea"
    assert classify_scene(0.25, 0.10, 0.65) == "shoreline_mixed"


def test_t_scene_low_confidence_uses_explicit_thresholds():
    assert is_t_scene_low_confidence(0.34, 0.33, 0.33, 400.0, 2.0)
    assert is_t_scene_low_confidence(0.70, 0.20, 0.10, 20.0, 2.0)
    assert is_t_scene_low_confidence(0.45, 0.15, 0.40, 400.0, 1.0)
    assert not is_t_scene_low_confidence(0.72, 0.18, 0.10, 400.0, 2.0)


def test_build_time_windows_degrades_to_two_or_four_seconds():
    timestamps = [0.0, 0.1, 0.2, 0.3, 1.4, 1.5, 1.6, 1.7]
    windows = build_time_windows(
        timestamps_sec=timestamps,
        decoded_duration_sec=2.0,
        base_window_seconds=1.0,
        min_frames_per_window=8,
        max_window_seconds=4.0,
    )
    assert len(windows) == 1
    assert windows[0].start_sec == 0.0
    assert windows[0].end_sec == 2.0
    assert not windows[0].low_fps_window

    sparse = build_time_windows(
        timestamps_sec=[0.0, 2.0, 4.0],
        decoded_duration_sec=5.0,
        base_window_seconds=1.0,
        min_frames_per_window=8,
        max_window_seconds=4.0,
    )
    assert sparse[0].low_fps_window


def test_overlap_alignment_uses_overlap_ratio():
    a = make_window("u1", "V", 0.0, 1.0)
    b = make_window("u1", "T", 0.5, 1.5)
    c = make_window("u1", "T", 1.0, 2.0)
    assert window_overlap_ratio(a, b) == 0.5
    assert window_overlap_ratio(a, c) == 0.0
    assert find_best_overlap_window(a, [c, b]) == b


def test_usable_rules_require_both_ratio_and_streak():
    assert not evaluate_window_usability(3, False, False, False)
    assert not evaluate_window_usability(12, True, False, False)
    assert evaluate_window_usability(12, False, False, False)

    assert max_consecutive_false([True, False, False, True, False]) == 2
    assert evaluate_video_usability(0.9, 2)
    assert not evaluate_video_usability(0.79, 1)
    assert not evaluate_video_usability(0.9, 4)


def test_build_video_summary_and_report_include_expected_metrics():
    windows = [
        make_window("unit_a", "V", 0.0, 1.0, primary_scene="sea_sky", usable=True),
        make_window("unit_a", "T", 0.0, 1.0, primary_scene="shoreline_mixed", usable=False, scene_source="paired_visible_proxy"),
    ]
    video_row = build_video_summary(
        unit_id="unit_a",
        date_batch="0711",
        pair_status="paired",
        windows=windows,
        modalities_present=["V", "T"],
        metadata_unreadable=False,
        t_scale_calibration="fallback",
        t_scale_factor=1.5,
    )
    assert video_row["primary_scene_by_time"] in {"sea_sky", "shoreline_mixed"}
    assert video_row["usable_window_ratio"] == 0.5
    assert video_row["max_consecutive_unusable_windows"] == 1

    report_rows = enrich_report_rows(
        [
            {
                **video_row,
                "paired_visible_proxy_count": 1,
            },
            {
                **video_row,
                "unit_id": "unit_b",
                "date_batch": "0812",
                "pair_status": "V_only",
            },
        ],
        windows,
    )
    assert report_rows[0]["paired_visible_proxy_count"] == 1
    assert "dominant_date_share" in report_rows[0]
    assert "date_entropy" in report_rows[0]
    assert "top1_date_batch" in report_rows[0]


def test_compute_date_concentration_returns_global_observation_fields():
    dominant, entropy, top1 = compute_date_concentration(
        [
            {"date_batch": "0711"},
            {"date_batch": "0711"},
            {"date_batch": "0812"},
        ]
    )
    assert dominant == 0.666667
    assert entropy > 0
    assert top1 == "0711"


def test_window_result_columns_match_documented_output():
    assert WINDOW_RESULT_COLUMNS == [
        "unit_id",
        "date_batch",
        "modality",
        "pair_status",
        "window_start",
        "window_end",
        "frame_count",
        "low_fps_window",
        "duration_from_metadata",
        "decode_error",
        "primary_scene",
        "sea_ratio",
        "sky_ratio",
        "shoreline_land_ratio",
        "scene_classification_failed",
        "scene_classification_low_confidence",
        "scene_source",
        "target_presence",
        "target_count",
        "track_count",
        "bbox_area_ratio_stats",
        "scale_bin_counts",
        "small_target_count",
        "detector_not_run",
        "usable",
    ]


def test_normalize_worker_count_clamps_requested_parallelism():
    assert normalize_worker_count(0) == 1
    assert normalize_worker_count(-3) == 1
    assert normalize_worker_count(1) == 1
    assert normalize_worker_count(9999) >= 1


def test_merge_scene_and_target_results_combines_both_sides():
    scene = VideoAnalysisResult(
        unit_id="u1",
        date_batch="0711",
        modality="V",
        pair_status="paired",
        modality_present="V",
        duration_from_metadata=True,
        metadata_duration_sec=10.0,
        decoded_duration_sec=10.0,
        windows=[
            {
                **make_window("u1", "V", 0.0, 1.0),
                "target_presence": False,
                "target_count": 0,
                "track_count": 0,
                "bbox_area_ratio_stats": json.dumps({"count": 0, "median": 0.0}),
                "scale_bin_counts": json.dumps({"small": 0, "medium": 0, "large": 0}),
                "small_target_count": 0,
                "detector_not_run": True,
                "usable": False,
            }
        ],
        open_failed=False,
    )
    target = VideoAnalysisResult(
        unit_id="u1",
        date_batch="0711",
        modality="V",
        pair_status="paired",
        modality_present="V",
        duration_from_metadata=True,
        metadata_duration_sec=10.0,
        decoded_duration_sec=10.0,
        windows=[
            {
                "unit_id": "u1",
                "date_batch": "0711",
                "modality": "V",
                "pair_status": "paired",
                "window_start": 0.0,
                "window_end": 1.0,
                "frame_count": 12,
                "low_fps_window": False,
                "duration_from_metadata": True,
                "decode_error": False,
                "target_presence": True,
                "target_count": 3,
                "track_count": 2,
                "bbox_area_ratio_stats": json.dumps({"count": 3, "median": 0.002}),
                "scale_bin_counts": json.dumps({"small": 3, "medium": 0, "large": 0}),
                "small_target_count": 3,
                "detector_not_run": False,
            }
        ],
        open_failed=False,
    )
    merged = merge_scene_and_target_results(scene, target)
    assert merged.windows[0]["primary_scene"] == "sea_sky"
    assert merged.windows[0]["target_count"] == 3
    assert merged.windows[0]["usable"] is True
