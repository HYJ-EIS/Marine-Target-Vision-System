import os
import sys
from argparse import Namespace

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from extract_tracking_frames import (
    angular_difference_deg,
    build_class_flags,
    build_frame_id,
    build_video_job,
    build_yolo_lines,
    compute_frame_phash,
    extraction_confidence_override,
    estimate_pose_angle,
    evaluate_detection_keep_reasons,
    make_export_record,
    merge_raw_and_tracked_boxes,
    nearest_frame_index,
    phash_hamming_distance,
    prune_representative_records,
    sanitize_export_boxes,
    select_records_by_reference,
    should_keep_detection_frame,
    should_keep_by_phash,
    has_strong_detection_novelty,
    VideoExtractionResult,
)
from target_module.image_detect_module.config import Config


def draw_rotated_rect(angle_deg: float, size=(200, 200), rect_size=(120, 30)) -> np.ndarray:
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    rect = ((size[0] / 2, size[1] / 2), rect_size, angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(img, box, (255, 255, 255))
    return img


def test_build_video_job_and_frame_id():
    input_root = r"D:\Desktop\烟台项目数据\原始数据集\视频"
    video_path = os.path.join(
        input_root,
        "0711",
        "DJI_202507111023_001",
        "DJI_20250711104519_0004_V.MP4",
    )
    job = build_video_job(input_root, video_path)
    assert job.session_id == "0711__DJI_202507111023_001"
    assert job.video_id == "DJI_20250711104519_0004"
    assert job.modality == "rgb"
    assert build_frame_id(job.session_id, job.video_id, 45.6, job.modality) == (
        "0711__DJI_202507111023_001__DJI_20250711104519_0004__045600"
    )
    assert build_frame_id(job.session_id, job.video_id, 45.6, "ir").endswith("_ir")


def test_angular_difference_wraps_symmetrically():
    assert angular_difference_deg(5.0, 175.0) == 10.0
    assert angular_difference_deg(20.0, 30.0) == 10.0
    assert angular_difference_deg(None, 30.0) is None


def test_estimate_pose_angle_for_elongated_shape():
    img = draw_rotated_rect(32.0)
    box = {"x": 20, "y": 20, "w": 160, "h": 160}
    angle = estimate_pose_angle(img, box)
    assert angle is not None
    assert angular_difference_deg(angle, 32.0) <= 10.0


def test_estimate_pose_angle_rejects_square_shape():
    img = draw_rotated_rect(45.0, rect_size=(80, 80))
    box = {"x": 30, "y": 30, "w": 140, "h": 140}
    angle = estimate_pose_angle(img, box)
    assert angle is None


def test_evaluate_detection_keep_reasons_for_motion_scale_and_new_track():
    frame = np.zeros((220, 220, 3), dtype=np.uint8)
    previous_states = {
        1: type("State", (), {
            "center": (60.0, 60.0),
            "size": (100.0, 50.0),
            "area": 5000.0,
            "pose_angle_deg": None,
        })()
    }
    tracked_boxes = [
        {"track_id": 1, "x": 100, "y": 35, "w": 140, "h": 90, "class": "USV"},
        {"track_id": 2, "x": 10, "y": 10, "w": 40, "h": 20, "class": "UAV"},
    ]
    reasons, states = evaluate_detection_keep_reasons(
        tracked_boxes,
        frame,
        previous_states,
        motion_threshold=0.15,
        pose_angle_threshold_deg=12.0,
        area_threshold=0.12,
        seen_detection_frames=True,
    )
    assert "motion_change" in reasons
    assert "scale_change" in reasons
    assert "new_track" in reasons
    assert set(states) == {1, 2}


def test_phash_dedup_respects_exempt_reasons():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    hash_a = compute_frame_phash(img)
    hash_b = compute_frame_phash(img.copy())
    assert phash_hamming_distance(hash_a, hash_b) == 0
    assert not should_keep_by_phash(hash_b, hash_a, {"motion_change"}, 4)
    assert should_keep_by_phash(hash_b, hash_a, {"new_track"}, 4)


def test_detection_frame_dedup_requires_global_or_target_change():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:30, 20:30] = 255
    hash_a = compute_frame_phash(img)
    hash_b = compute_frame_phash(img.copy())
    assert not should_keep_detection_frame(
        hash_b,
        hash_a,
        hash_b,
        hash_a,
        {"motion_change"},
        global_threshold=8,
        target_threshold=6,
    )
    assert should_keep_detection_frame(
        hash_b,
        hash_a,
        hash_b,
        hash_a,
        {"new_track"},
        global_threshold=8,
        target_threshold=6,
    )


def test_strong_detection_novelty_blocks_small_motion():
    prev = {
        1: type("State", (), {
            "center": (100.0, 100.0),
            "size": (80.0, 40.0),
            "area": 3200.0,
            "pose_angle_deg": 10.0,
        })()
    }
    cur = {
        1: type("State", (), {
            "center": (110.0, 104.0),
            "size": (82.0, 40.0),
            "area": 3280.0,
            "pose_angle_deg": 14.0,
        })()
    }
    assert not has_strong_detection_novelty(
        cur,
        prev,
        {"motion_change"},
        motion_threshold=0.30,
        area_threshold=0.20,
        pose_threshold_deg=18.0,
    )


def test_class_flags_and_yolo_lines():
    boxes = [
        {"x": 10, "y": 20, "w": 50, "h": 40, "class": "USV"},
        {"x": 30, "y": 50, "w": 10, "h": 15, "class": "UAV"},
    ]
    assert build_class_flags(boxes) == (1, 0, 1)
    lines = build_yolo_lines((100, 200, 3), boxes)
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("2 ")


def test_merge_raw_and_tracked_boxes_prefers_raw_geometry_and_track_id():
    raw_boxes = [{"x": 10, "y": 10, "w": 20, "h": 10, "class": "USV", "confidence": 0.6}]
    tracked_boxes = [{"x": 11, "y": 11, "w": 20, "h": 10, "class": "USV", "confidence": 0.7, "track_id": 9}]
    merged = merge_raw_and_tracked_boxes(raw_boxes, tracked_boxes)
    assert len(merged) == 1
    assert merged[0]["x"] == 10
    assert merged[0]["track_id"] == 9


def test_extraction_confidence_override_by_modality():
    assert extraction_confidence_override("visible") == Config.EXTERNAL_FRAMES_VISIBLE_CONF_THRESH
    assert extraction_confidence_override("infrared") == Config.EXTERNAL_FRAMES_INFRARED_CONF_THRESH


def test_sanitize_export_boxes_drops_tracker_predictions_by_default():
    boxes = [
        {"x": 10, "y": 10, "w": 20, "h": 10, "class": "USV", "confidence": 0.6},
        {
            "x": 12,
            "y": 10,
            "w": 20,
            "h": 10,
            "class": "USV",
            "confidence": 0.6,
            "is_tracker_prediction": True,
        },
    ]
    sanitized = sanitize_export_boxes(boxes, allow_tracker_fill_labels=False)
    assert len(sanitized) == 1
    assert sanitized[0]["x"] == 10


def test_make_export_record_skips_empty_records_for_clean_training_set():
    job = build_video_job(
        r"D:\Desktop\烟台项目数据\原始数据集\视频",
        r"D:\Desktop\烟台项目数据\原始数据集\视频\0711\DJI_202507111023_001\DJI_20250711104519_0004_V.MP4",
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert make_export_record(
        output_root="results/tmp",
        job=job,
        frame_idx=0,
        timestamp_sec=0.0,
        frame=frame,
        boxes=[],
        keep_reasons={"detection_lost"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    ) is None


def test_prune_representative_records_keeps_high_value_representative():
    job = build_video_job(
        r"D:\Desktop\烟台项目数据\原始数据集\视频",
        r"D:\Desktop\烟台项目数据\原始数据集\视频\0711\DJI_202507111023_001\DJI_20250711104519_0004_V.MP4",
    )
    frame_a = np.zeros((96, 96, 3), dtype=np.uint8)
    frame_a[30:42, 30:42] = 255
    frame_b = frame_a.copy()
    frame_c = np.zeros((96, 96, 3), dtype=np.uint8)
    frame_c[55:72, 55:72] = 255

    box_a = {"x": 30, "y": 30, "w": 12, "h": 12, "class": "USV"}
    box_b = {"x": 30, "y": 30, "w": 12, "h": 12, "class": "USV"}
    box_c = {"x": 55, "y": 55, "w": 17, "h": 17, "class": "USV"}

    record_a = make_export_record(
        output_root="results/tmp",
        job=job,
        frame_idx=0,
        timestamp_sec=0.0,
        frame=frame_a,
        boxes=[box_a],
        keep_reasons={"paired_alignment"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    record_b = make_export_record(
        output_root="results/tmp",
        job=job,
        frame_idx=1,
        timestamp_sec=1.0,
        frame=frame_b,
        boxes=[box_b],
        keep_reasons={"motion_change"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    record_c = make_export_record(
        output_root="results/tmp",
        job=job,
        frame_idx=2,
        timestamp_sec=8.0,
        frame=frame_c,
        boxes=[box_c],
        keep_reasons={"motion_change", "scale_change"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    args = Namespace(
        representative_global_phash_threshold=14,
        representative_target_phash_threshold=14,
        representative_motion_threshold=0.55,
        representative_area_threshold=0.35,
        representative_max_cluster_span_sec=6.0,
        representative_min_time_gap_sec=0.0,
    )

    pruned, anchors = prune_representative_records(
        {
            record_a.frame_id: record_a,
            record_b.frame_id: record_b,
            record_c.frame_id: record_c,
        },
        args,
    )

    assert len(pruned) == 2
    assert record_b.frame_id in pruned
    assert record_c.frame_id in pruned
    assert anchors == [
        (record_b.timestamp_sec, record_b.frame_id),
        (record_c.timestamp_sec, record_c.frame_id),
    ]


def test_prune_representative_records_enforces_min_time_gap():
    job = build_video_job(
        r"D:\Desktop\烟台项目数据\原始数据集\视频",
        r"D:\Desktop\烟台项目数据\原始数据集\视频\0711\DJI_202507111023_001\DJI_20250711104519_0004_V.MP4",
    )
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[30:42, 30:42] = 255
    box = {"x": 30, "y": 30, "w": 12, "h": 12, "class": "USV"}
    a = make_export_record(
        output_root="results/tmp",
        job=job,
        frame_idx=0,
        timestamp_sec=10.0,
        frame=frame,
        boxes=[box],
        keep_reasons={"paired_alignment"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    b = make_export_record(
        output_root="results/tmp",
        job=job,
        frame_idx=1,
        timestamp_sec=10.2,
        frame=frame,
        boxes=[box],
        keep_reasons={"motion_change", "scale_change"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    args = Namespace(
        representative_global_phash_threshold=14,
        representative_target_phash_threshold=14,
        representative_motion_threshold=0.0,
        representative_area_threshold=1.0,
        representative_max_cluster_span_sec=0.0,
        representative_min_time_gap_sec=0.6,
    )
    pruned, anchors = prune_representative_records(
        {a.frame_id: a, b.frame_id: b},
        args,
    )
    assert list(pruned) == [b.frame_id]
    assert anchors == [(b.timestamp_sec, b.frame_id)]


def test_select_records_by_reference_uses_rgb_timestamps_for_ir():
    job_rgb = build_video_job(
        r"D:\Desktop\烟台项目数据\原始数据集\视频",
        r"D:\Desktop\烟台项目数据\原始数据集\视频\0711\DJI_202507111023_001\DJI_20250711104519_0004_V.MP4",
    )
    job_ir = build_video_job(
        r"D:\Desktop\烟台项目数据\原始数据集\视频",
        r"D:\Desktop\烟台项目数据\原始数据集\视频\0711\DJI_202507111023_001\DJI_20250711104519_0004_T.MP4",
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    box = {"x": 12, "y": 12, "w": 10, "h": 8, "class": "USV"}

    rgb_a = make_export_record(
        output_root="results/tmp",
        job=job_rgb,
        frame_idx=0,
        timestamp_sec=1.0,
        frame=frame,
        boxes=[box],
        keep_reasons={"motion_change"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    rgb_b = make_export_record(
        output_root="results/tmp",
        job=job_rgb,
        frame_idx=1,
        timestamp_sec=3.0,
        frame=frame,
        boxes=[box],
        keep_reasons={"scale_change"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    ir_a = make_export_record(
        output_root="results/tmp",
        job=job_ir,
        frame_idx=0,
        timestamp_sec=1.02,
        frame=frame,
        boxes=[box],
        keep_reasons={"motion_change"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    ir_b = make_export_record(
        output_root="results/tmp",
        job=job_ir,
        frame_idx=1,
        timestamp_sec=2.98,
        frame=frame,
        boxes=[box],
        keep_reasons={"motion_change"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    ir_extra = make_export_record(
        output_root="results/tmp",
        job=job_ir,
        frame_idx=2,
        timestamp_sec=4.5,
        frame=frame,
        boxes=[box],
        keep_reasons={"motion_change"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )

    rgb_result = VideoExtractionResult(
        job=job_rgb,
        fps=30.0,
        frame_count=100,
        records={rgb_a.frame_id: rgb_a, rgb_b.frame_id: rgb_b},
        anchor_timestamps=[(rgb_a.timestamp_sec, rgb_a.frame_id), (rgb_b.timestamp_sec, rgb_b.frame_id)],
    )
    ir_result = VideoExtractionResult(
        job=job_ir,
        fps=30.0,
        frame_count=100,
        records={ir_a.frame_id: ir_a, ir_b.frame_id: ir_b, ir_extra.frame_id: ir_extra},
        anchor_timestamps=[],
    )

    selected, anchors = select_records_by_reference(
        rgb_result,
        ir_result,
        pair_tolerance_sec=0.1,
        processor=None,
        output_root="results/tmp",
        args=Namespace(
            allow_tracker_fill_labels=False,
            include_empty_frames=False,
        ),
    )
    assert list(selected) == [ir_a.frame_id, ir_b.frame_id]
    assert anchors == [(ir_a.timestamp_sec, ir_a.frame_id), (ir_b.timestamp_sec, ir_b.frame_id)]


def test_select_records_by_reference_materializes_empty_ir_when_requested():
    job_rgb = build_video_job(
        r"D:\Desktop\烟台项目数据\原始数据集\视频",
        r"D:\Desktop\烟台项目数据\原始数据集\视频\0711\DJI_202507111023_001\DJI_20250711104519_0004_V.MP4",
    )
    job_ir = build_video_job(
        r"D:\Desktop\烟台项目数据\原始数据集\视频",
        r"D:\Desktop\烟台项目数据\原始数据集\视频\0711\DJI_202507111023_001\DJI_20250711104519_0004_T.MP4",
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    box = {"x": 12, "y": 12, "w": 10, "h": 8, "class": "USV"}
    rgb = make_export_record(
        output_root="results/tmp",
        job=job_rgb,
        frame_idx=0,
        timestamp_sec=1.0,
        frame=frame,
        boxes=[box],
        keep_reasons={"motion_change"},
        allow_tracker_fill_labels=False,
        include_empty_frames=False,
    )
    ir = make_export_record(
        output_root="results/tmp",
        job=job_ir,
        frame_idx=0,
        timestamp_sec=1.02,
        frame=frame,
        boxes=[],
        keep_reasons={"paired_alignment"},
        allow_tracker_fill_labels=False,
        include_empty_frames=True,
    )
    rgb_result = VideoExtractionResult(
        job=job_rgb,
        fps=30.0,
        frame_count=100,
        records={rgb.frame_id: rgb},
        anchor_timestamps=[(rgb.timestamp_sec, rgb.frame_id)],
    )
    ir_result = VideoExtractionResult(
        job=job_ir,
        fps=30.0,
        frame_count=100,
        records={ir.frame_id: ir},
        anchor_timestamps=[],
    )

    selected, anchors = select_records_by_reference(
        rgb_result,
        ir_result,
        pair_tolerance_sec=0.1,
        processor=None,
        output_root="results/tmp",
        args=Namespace(
            allow_tracker_fill_labels=False,
            include_empty_frames=False,
            representative_min_time_gap_sec=0.6,
            paired_reference_include_empty_target=True,
        ),
    )
    assert list(selected) == [ir.frame_id]
    assert anchors == [(ir.timestamp_sec, ir.frame_id)]


def test_nearest_frame_index_clamps_bounds():
    assert nearest_frame_index(0.0, 25.0, 100) == 0
    assert nearest_frame_index(0.5, 20.0, 5) == 4
