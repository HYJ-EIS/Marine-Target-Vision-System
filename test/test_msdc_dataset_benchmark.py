import csv
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation import export_mot_results
from tools.evaluation.msdc_dataset_benchmark import (
    VideoInfo,
    _build_render_command,
    build_run_metadata,
    compute_per_gt_diagnostics,
    compute_per_gt_stage_coverage,
    compute_tracker_box_stats,
    initialize_failures_file,
    main,
    resolve_single_sequence_dataset,
    run_logged_command,
    windows_path_to_wsl_path,
    write_command_record,
    write_failure_record,
    write_motchallenge_gt_sequence,
)


def test_formal_frame_limit_controls_printed_export_limit(monkeypatch, tmp_path, capsys):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    (dataset_root / "gt.txt").write_text("1,1,10,10,20,20,1,1,1\n", encoding="utf-8")
    (dataset_root / "原视频地址.txt").write_text(str(video_path), encoding="utf-8")
    output_root = tmp_path / "runs"

    monkeypatch.setattr(sys, "argv", [
        "msdc_dataset_benchmark.py",
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--run-id",
        "formal",
        "--trackers",
        "ocsort",
        "--formal-frame-limit",
        "5400",
        "--render",
    ])

    main()

    captured = capsys.readouterr()
    assert "--max-frames 5400" in captured.out
    assert "[INFO] formal frame limit: first 5400 frames" in captured.out
    assert not (output_root / "formal").exists()


def test_windows_path_to_wsl_path_handles_drive_and_wsl_unc():
    assert windows_path_to_wsl_path(r'"D:\dataset\video.mp4"') == Path("/mnt/d/dataset/video.mp4")
    assert windows_path_to_wsl_path(
        r"\\wsl.localhost\Ubuntu-D\home\hyj\Anti_Drone_Project\USV_MOT标注数据集"
    ) == Path("/home/hyj/Anti_Drone_Project/USV_MOT标注数据集")


def test_metadata_and_failure_records_are_written(tmp_path):
    metadata = build_run_metadata(
        run_id="paper_20260608_120000",
        mode="main",
        commit_hash="abcdef0",
        dataset_roots=[tmp_path / "dataset_a", tmp_path / "dataset_b"],
        output_root=tmp_path / "out",
        trackers=["ocsort", "botsort", "msdc_elt"],
        variants=["v3_candidate_topk_no_roi_no_motion"],
        max_frames=0,
        duration_seconds=0.0,
        render=True,
        formal=True,
    )

    assert metadata["run_id"] == "paper_20260608_120000"
    assert metadata["formal"] is True
    assert metadata["max_frames"] == 0
    assert metadata["duration_seconds"] == 0.0
    assert metadata["commit_hash"] == "abcdef0"

    commands_path = tmp_path / "commands.jsonl"
    write_command_record(
        commands_path,
        label="EXPORT:ocsort",
        command=["python", "script.py"],
        env_delta={"A": "1"},
        status="planned",
        start_time="2026-06-08T12:00:00+08:00",
        end_time="2026-06-08T12:00:01+08:00",
        returncode=0,
    )
    command_record = json.loads(commands_path.read_text(encoding="utf-8").strip())
    assert command_record["label"] == "EXPORT:ocsort"
    assert command_record["env_delta"] == {"A": "1"}

    failures_path = tmp_path / "failures.csv"
    write_failure_record(
        failures_path,
        stage="export",
        label="ocsort",
        command="python script.py",
        returncode=2,
        message="failed",
    )
    rows = list(csv.DictReader(failures_path.open(encoding="utf-8-sig")))
    assert rows == [{
        "stage": "export",
        "label": "ocsort",
        "command": "python script.py",
        "returncode": "2",
        "message": "failed",
    }]


def test_initialize_failures_file_writes_empty_header(tmp_path):
    failures_path = tmp_path / "metadata" / "failures.csv"

    initialize_failures_file(failures_path)
    initialize_failures_file(failures_path)

    assert failures_path.read_text(encoding="utf-8-sig").splitlines() == [
        "stage,label,command,returncode,message"
    ]


def test_print_only_benchmark_does_not_create_run_metadata(monkeypatch, tmp_path, capsys):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    (dataset_root / "gt.txt").write_text("1,1,10,10,20,20,1,1,1\n", encoding="utf-8")
    (dataset_root / "原视频地址.txt").write_text(str(video_path), encoding="utf-8")
    output_root = tmp_path / "runs"

    monkeypatch.setattr(sys, "argv", [
        "msdc_dataset_benchmark.py",
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--run-id",
        "print_only",
        "--trackers",
        "ocsort",
    ])

    main()

    captured = capsys.readouterr()
    assert "[INFO] print-only mode" in captured.out
    assert not (output_root / "print_only").exists()


def test_run_logged_command_records_launch_exception(monkeypatch, tmp_path):
    def raise_os_error(*args, **kwargs):
        raise OSError("cannot launch")

    monkeypatch.setattr("tools.evaluation.msdc_dataset_benchmark.subprocess.run", raise_os_error)
    commands_path = tmp_path / "metadata" / "commands.jsonl"
    failures_path = tmp_path / "metadata" / "failures.csv"

    with pytest.raises(OSError, match="cannot launch"):
        run_logged_command(
            "render",
            "ocsort",
            ["missing-command"],
            {},
            commands_path,
            failures_path,
        )

    command_record = json.loads(commands_path.read_text(encoding="utf-8").strip())
    assert command_record["label"] == "RENDER:ocsort"
    assert command_record["status"] == "failed"
    assert command_record["returncode"] is None

    rows = list(csv.DictReader(failures_path.open(encoding="utf-8-sig")))
    assert rows == [{
        "stage": "render",
        "label": "ocsort",
        "command": "missing-command",
        "returncode": "",
        "message": "cannot launch",
    }]


def test_resolve_single_sequence_dataset_supports_nested_and_flat_gt(tmp_path):
    nested = tmp_path / "USV_MOT标注数据集"
    (nested / "gt").mkdir(parents=True)
    (nested / "gt" / "gt.txt").write_text("1,1,10,10,20,20,1,1,1\n", encoding="utf-8")
    (nested / "gt" / "labels.txt").write_text("UAV\nUSV\n", encoding="utf-8")
    (nested / "原视频地址.txt").write_text(r"D:\videos\usv.mp4", encoding="utf-8")

    flat = tmp_path / "UAV_USV_MOT标注数据集"
    flat.mkdir()
    (flat / "gt.txt").write_text("1,1,10,10,20,20,1,1,1\n", encoding="utf-8")
    (flat / "labels.txt").write_text("UAV\nUSV\nplatform\nfishship\n", encoding="utf-8")
    (flat / "原视频文件地址.txt").write_text('"D:\\videos\\uav_usv.mp4"', encoding="utf-8")

    nested_spec = resolve_single_sequence_dataset(nested)
    flat_spec = resolve_single_sequence_dataset(flat)

    assert nested_spec.gt_file == nested / "gt" / "gt.txt"
    assert nested_spec.labels_file == nested / "gt" / "labels.txt"
    assert nested_spec.video_path == Path("/mnt/d/videos/usv.mp4")
    assert nested_spec.seq_name == "usv"
    assert flat_spec.gt_file == flat / "gt.txt"
    assert flat_spec.labels_file == flat / "labels.txt"
    assert flat_spec.video_path == Path("/mnt/d/videos/uav_usv.mp4")
    assert flat_spec.seq_name == "uav_usv"


def test_resolve_single_sequence_dataset_falls_back_to_root_video_file(tmp_path):
    dataset_root = tmp_path / "UAV_USV_MOT标注数据集"
    dataset_root.mkdir()
    video_path = dataset_root / "DJI_20250916100639_0001_V.MP4"
    video_path.write_bytes(b"fake")
    (dataset_root / "gt.txt").write_text("1,1,10,10,20,20,1,1,1\n", encoding="utf-8")
    (dataset_root / "labels.txt").write_text("UAV\nUSV\n", encoding="utf-8")

    spec = resolve_single_sequence_dataset(dataset_root)

    assert spec.video_ref_file is None
    assert spec.video_path == video_path
    assert spec.seq_name == "DJI_20250916100639_0001_V"


def test_write_motchallenge_gt_sequence_copies_gt_and_writes_seqinfo(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    gt_file = dataset_root / "gt.txt"
    gt_file.write_text("1,1,10,10,20,20,1,1,1\n", encoding="utf-8")
    spec = resolve_single_sequence_dataset(dataset_root, video="/mnt/d/sample.mp4", seq_name="seq_a")
    gt_root = tmp_path / "mot_gt"

    seq_dir = write_motchallenge_gt_sequence(
        spec,
        output_gt_root=gt_root,
        video_info=VideoInfo(fps=29.97, width=3840, height=2160, frame_count=123),
    )

    assert seq_dir == gt_root / "seq_a"
    assert (seq_dir / "gt" / "gt.txt").read_text(encoding="utf-8") == "1,1,10,10,20,20,1,1,1\n"
    seqinfo = (seq_dir / "seqinfo.ini").read_text(encoding="utf-8")
    assert "name=seq_a" in seqinfo
    assert "frameRate=29.97" in seqinfo
    assert "seqLength=123" in seqinfo
    assert "imWidth=3840" in seqinfo
    assert "imHeight=2160" in seqinfo


def test_write_motchallenge_gt_sequence_can_clip_gt_to_max_frames(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    gt_file = dataset_root / "gt.txt"
    gt_file.write_text(
        "\n".join([
            "1,1,10,10,20,20,1,1,1",
            "2,1,11,10,20,20,1,1,1",
            "3,1,12,10,20,20,1,1,1",
            "",
        ]),
        encoding="utf-8",
    )
    spec = resolve_single_sequence_dataset(dataset_root, video="/mnt/d/sample.mp4", seq_name="seq_clip")
    gt_root = tmp_path / "mot_gt"

    seq_dir = write_motchallenge_gt_sequence(
        spec,
        output_gt_root=gt_root,
        video_info=VideoInfo(fps=30.0, width=100, height=100, frame_count=3),
        max_frames=2,
    )

    gt_rows = (seq_dir / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines()
    assert gt_rows == [
        "1,1,10,10,20,20,1,1,1",
        "2,1,11,10,20,20,1,1,1",
    ]
    assert "seqLength=2" in (seq_dir / "seqinfo.ini").read_text(encoding="utf-8")


def test_build_render_command_uses_effective_export_max_frames(tmp_path):
    cmd = _build_render_command(
        input_video=tmp_path / "video.mp4",
        tracker_file=tmp_path / "trackers" / "v3_candidate_topk_no_roi_no_motion" / "data" / "seq.txt",
        output_file=tmp_path / "visualizations" / "seq" / "v3_candidate_topk_no_roi_no_motion.mp4",
        max_frames=1798,
        progress_interval=300,
    )

    assert "--max-frames" in cmd
    assert cmd[cmd.index("--max-frames") + 1] == "1798"
    assert "--progress-interval" in cmd
    assert cmd[cmd.index("--progress-interval") + 1] == "300"
    assert cmd[cmd.index("--class-source") + 1] == "detector"


def test_build_render_command_accepts_class_source_none(tmp_path):
    cmd = _build_render_command(
        input_video=tmp_path / "video.mp4",
        tracker_file=tmp_path / "trackers" / "v3_candidate_topk_no_roi_no_motion" / "data" / "seq.txt",
        output_file=tmp_path / "visualizations" / "seq" / "v3_candidate_topk_no_roi_no_motion.mp4",
        max_frames=1798,
        progress_interval=300,
        class_source="none",
    )

    assert cmd[cmd.index("--class-source") + 1] == "none"


def test_compute_per_gt_diagnostics_reports_fragmented_identity(tmp_path):
    gt_file = tmp_path / "gt.txt"
    tracker_file = tmp_path / "tracker.txt"
    gt_file.write_text(
        "\n".join([
            "1,3,10,10,10,10,1,1,1",
            "2,3,11,10,10,10,1,1,1",
            "3,3,12,10,10,10,1,1,1",
            "4,3,13,10,10,10,1,1,1",
            "",
        ]),
        encoding="utf-8",
    )
    tracker_file.write_text(
        "\n".join([
            "1,4,10,10,10,10,0.9,-1,-1,-1",
            "2,4,11,10,10,10,0.9,-1,-1,-1",
            "4,5,13,10,10,10,0.9,-1,-1,-1",
            "",
        ]),
        encoding="utf-8",
    )

    rows = compute_per_gt_diagnostics(gt_file, tracker_file)

    assert rows == [
        {
            "gt_id": 3,
            "gt_frame_count": 4,
            "tracked_frame_count": 3,
            "missed_frame_count": 1,
            "coverage_ratio": 0.75,
            "predicted_id_count": 2,
            "predicted_ids": "4;5",
            "matched_segments": "1-2;4-4",
            "missed_segments": "3-3",
        }
    ]


def test_compute_tracker_box_stats_counts_duplicates_and_far_boxes(tmp_path):
    gt_file = tmp_path / "gt.txt"
    tracker_file = tmp_path / "tracker.txt"
    gt_file.write_text("1,1,10,10,10,10,1,1,1\n", encoding="utf-8")
    tracker_file.write_text(
        "\n".join([
            "1,1,10,10,10,10,0.9,-1,-1,-1",
            "1,2,10,10,10,10,0.8,-1,-1,-1",
            "1,3,200,200,10,10,0.7,-1,-1,-1",
            "2,4,1,1,0,10,0.6,-1,-1,-1",
            "",
        ]),
        encoding="utf-8",
    )

    stats = compute_tracker_box_stats(gt_file, tracker_file)

    assert stats["rows"] == 4
    assert stats["unique_ids"] == 4
    assert stats["max_id"] == 4
    assert stats["duplicate_pairs_iou05"] == 1
    assert stats["zero_area_count"] == 1
    assert stats["far_from_gt_count"] == 1


def test_compute_per_gt_stage_coverage_reports_detection_stage_gaps(tmp_path):
    gt_file = tmp_path / "gt.txt"
    stage_file = tmp_path / "stage_observations.jsonl"
    gt_file.write_text(
        "\n".join([
            "1,3,10,10,10,10,1,1,1",
            "2,3,11,10,10,10,1,1,1",
            "3,3,12,10,10,10,1,1,1",
            "",
        ]),
        encoding="utf-8",
    )
    stage_file.write_text(
        "\n".join([
            '{"frame_idx": 0, "boxes": [{"stage": "high_det", "x": 10, "y": 10, "w": 10, "h": 10}]}',
            '{"frame_idx": 1, "boxes": [{"stage": "low_det", "x": 11, "y": 10, "w": 10, "h": 10}, {"stage": "output", "track_id": 4, "x": 11, "y": 10, "w": 10, "h": 10}]}',
            '{"frame_idx": 2, "boxes": []}',
            "",
        ]),
        encoding="utf-8",
    )

    rows = compute_per_gt_stage_coverage(gt_file, stage_file)

    assert rows == [
        {
            "gt_id": 3,
            "gt_frame_count": 3,
            "high_det_frame_count": 1,
            "low_det_frame_count": 1,
            "low_only_frame_count": 0,
            "output_frame_count": 1,
            "high_det_coverage": 0.3333,
            "low_det_coverage": 0.3333,
            "low_only_coverage": 0.0,
            "output_coverage": 0.3333,
            "output_predicted_id_count": 1,
            "output_predicted_ids": "4",
            "missing_after_stage_segments": "3-3",
        }
    ]


def test_export_video_to_mot_results_uses_tracker_output_name(monkeypatch, tmp_path):
    calls = {}

    class DummyCapture:
        def __init__(self, path):
            calls["video_path"] = path
            self.frame = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return 25.0

        def read(self):
            return False, None

        def release(self):
            calls["released"] = True

    class DummyDetector:
        pass

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(export_mot_results.cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(export_mot_results.td, "get_detector", lambda: DummyDetector())

    output_file = export_mot_results.export_video_to_mot_results(
        input_video=video,
        output_root=tmp_path / "results",
        tracker_type="botsort",
        tracker_output_name="baseline_botsort",
        seq_name="seq1",
    )

    assert output_file == tmp_path / "results" / "baseline_botsort" / "data" / "seq1.txt"
    assert output_file.is_file()
