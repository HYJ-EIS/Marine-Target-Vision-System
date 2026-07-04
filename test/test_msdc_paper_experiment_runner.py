import sys
import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import FORMAL_FRAME_LIMIT, FORMAL_MSDC_VARIANT
from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS, FORMAL_V3_ENV
from tools.evaluation import run_msdc_paper_experiments as runner
from tools.evaluation.run_msdc_paper_experiments import (
    build_ablation_cache_only_command,
    build_ablation_command,
    build_main_command,
    build_sensitivity_matrix_command,
    build_sensitivity_command,
    build_slice_command,
    build_slice_metrics_command,
    build_speed_command,
)

UNSUPPORTED_REPLAY_FLAGS = {
    "--mode",
    "--commit-hash",
    "--run",
    "--duration-seconds",
}


def assert_replay_command(cmd, run_id):
    assert cmd[1] == str(runner._ROOT / "tools" / "evaluation" / "detection_replay_benchmark.py")
    assert cmd[cmd.index("--run-id") + 1] == run_id
    assert "--dataset-root" in cmd
    assert "--output-root" in cmd
    assert "--formal-frame-limit" in cmd
    assert "--progress-interval" in cmd
    assert "--render" in cmd
    assert "--render-class-source" in cmd
    assert not (UNSUPPORTED_REPLAY_FLAGS & set(cmd))


def test_main_command_uses_full_video_and_render():
    cmd = build_main_command(
        dataset_roots=["/data/a", "/data/b"],
        output_root=Path("results/main"),
        run_id="main_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=True,
    )
    assert_replay_command(cmd, "main_full")
    assert "--max-frames" not in cmd
    assert "--render" in cmd
    assert cmd[cmd.index("--trackers") + 1:cmd.index("--variants")] == [
        "bytetrack",
        "ocsort",
        "botsort",
        "msdc_elt",
    ]
    assert cmd[cmd.index("--variants") + 1] == FORMAL_MSDC_VARIANT


def test_formal_v3_env_freezes_expected_switches():
    assert FORMAL_MSDC_VARIANT == "msdc_v3"
    assert ABLATION_VARIANTS[FORMAL_MSDC_VARIANT] == FORMAL_V3_ENV
    assert FORMAL_V3_ENV["MSDC_LOW_CANDIDATE_ENABLE"] == "1"
    assert FORMAL_V3_ENV["MSDC_USE_REACQUIRE"] == "1"
    assert FORMAL_V3_ENV["MSDC_LOW_INHERIT_ENABLE"] == "1"
    assert FORMAL_V3_ENV["MSDC_OUTPUT_NMS_ENABLE"] == "1"
    assert FORMAL_V3_ENV["MSDC_OUTPUT_MAX_REAL_DET_AGE"] == "3"
    assert FORMAL_V3_ENV["MSDC_LOW_OBS_TOPK"] == "32"


def test_main_command_uses_formal_frame_limit_and_all_main_trackers(tmp_path):
    cmd = build_main_command(
        dataset_roots=["/data/a", "/data/b"],
        output_root=tmp_path,
        run_id="main_full",
        commit_hash="abc123",
        progress_interval=500,
        run=True,
        formal_frame_limit=FORMAL_FRAME_LIMIT,
        render_class_source="detector",
    )
    assert_replay_command(cmd, "main_full")
    assert "--formal-frame-limit" in cmd
    assert cmd[cmd.index("--formal-frame-limit") + 1] == "5400"
    assert cmd[cmd.index("--trackers") + 1:cmd.index("--variants")] == [
        "bytetrack",
        "ocsort",
        "botsort",
        "msdc_elt",
    ]


def test_formal_replay_commands_ignore_duration_limit():
    main_cmd = build_main_command(
        dataset_roots=["/data/a"],
        output_root=Path("results/main"),
        run_id="main_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=True,
        duration_seconds=120.0,
        render_class_source="none",
    )
    ablation_cmd = build_ablation_command(
        dataset_roots=["/data/a"],
        output_root=Path("results/ablation"),
        run_id="ablation_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=True,
        variants=["low_budget_topk16"],
        duration_seconds=120.0,
        render_class_source="none",
    )

    assert "--duration-seconds" not in main_cmd
    assert "--duration-seconds" not in ablation_cmd
    assert main_cmd[main_cmd.index("--render-class-source") + 1] == "none"
    assert ablation_cmd[ablation_cmd.index("--render-class-source") + 1] == "none"
    assert main_cmd[main_cmd.index("--variants") + 1] == FORMAL_MSDC_VARIANT


def test_ablation_command_includes_required_variants():
    cmd = build_ablation_command(
        dataset_roots=["/data/a"],
        output_root=Path("results/ablation"),
        run_id="ablation_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=True,
    )
    assert_replay_command(cmd, "ablation_full")
    required_variants = {
        FORMAL_MSDC_VARIANT,
        "no_low_candidate",
        "no_direct_reacquire",
        "no_low_inheritance",
        "hits_only_no_evidence",
        "no_output_nms",
        "no_output_real_det_age_gate",
        "low_budget_off",
        "low_budget_topk16",
        "low_budget_topk64",
        "removed_recovery_off",
    }
    variants = cmd[cmd.index("--variants") + 1:cmd.index("--formal-frame-limit")]
    assert variants[0] == FORMAL_MSDC_VARIANT
    assert required_variants <= set(variants)


def test_ablation_command_uses_formal_frame_limit():
    cmd = build_ablation_command(
        dataset_roots=["/data/a"],
        output_root=Path("results/ablation"),
        run_id="ablation_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=True,
        formal_frame_limit=FORMAL_FRAME_LIMIT,
    )

    assert_replay_command(cmd, "ablation_full")
    assert "--formal-frame-limit" in cmd
    assert cmd[cmd.index("--formal-frame-limit") + 1] == "5400"


def test_ablation_cache_only_command_reuses_detection_cache_without_rendering():
    cmd = build_ablation_cache_only_command(
        dataset_roots=["/data/a", "/data/b"],
        output_root=Path("results/ablation_cache_only"),
        run_id="ablation_cache_only",
        source_detection_cache_root=Path("results/main/main_full/detections"),
        formal_frame_limit=FORMAL_FRAME_LIMIT,
        progress_interval=500,
    )

    assert cmd[1] == str(runner._ROOT / "tools" / "evaluation" / "detection_replay_benchmark.py")
    assert cmd[cmd.index("--run-id") + 1] == "ablation_cache_only"
    assert cmd[cmd.index("--trackers") + 1:cmd.index("--variants")] == ["msdc_elt"]
    variants = cmd[cmd.index("--variants") + 1:cmd.index("--formal-frame-limit")]
    assert FORMAL_MSDC_VARIANT in variants
    assert "no_low_candidate" in variants
    assert "no_direct_reacquire" in variants
    assert "hits_only_no_evidence" in variants
    assert "--source-detection-cache-root" in cmd
    assert cmd[cmd.index("--source-detection-cache-root") + 1] == "results/main/main_full/detections"
    assert "--render-class-source" in cmd
    assert cmd[cmd.index("--render-class-source") + 1] == "cache"
    assert "--formal-frame-limit" in cmd
    assert cmd[cmd.index("--formal-frame-limit") + 1] == "5400"
    assert "--render" not in cmd


def test_ablation_command_accepts_selected_variants():
    args = runner.parse_args([
        "--ablation-variants",
        "no_low_candidate",
        "hits_only_no_evidence",
        FORMAL_MSDC_VARIANT,
        "low_budget_topk64",
    ])
    cmd = build_ablation_command(
        dataset_roots=["/data/a"],
        output_root=Path("results/ablation"),
        run_id="ablation_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=False,
        variants=list(args.ablation_variants),
    )
    joined = " ".join(cmd)
    assert "no_low_candidate" in joined
    assert "hits_only_no_evidence" in joined
    assert FORMAL_MSDC_VARIANT in joined
    assert "low_budget_topk64" in joined


def test_formal_main_and_ablation_commands_are_replay_commands(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_commit_hash", lambda: "abcdef0")
    monkeypatch.setattr(runner, "run_command", lambda label, cmd: calls.append((label, cmd)))

    runner.main(["--run-formal", "--output-root", str(tmp_path / "runs"), "--run-id", "formal", "--speed-frames", "0"])

    main_cmd = calls[0][1]
    ablation_cmd = calls[1][1]
    assert_replay_command(main_cmd, "main_full")
    assert_replay_command(ablation_cmd, "ablation_full")
    assert main_cmd[main_cmd.index("--variants") + 1] == FORMAL_MSDC_VARIANT
    assert ablation_cmd[ablation_cmd.index("--trackers") + 1:ablation_cmd.index("--variants")] == ["msdc_elt"]


def test_speed_command_records_fixed_frame_count():
    cmd = build_speed_command(
        dataset_root="/data/a",
        output_root=Path("results/speed"),
        run_id="speed_1000",
        commit_hash="abcdef0",
        frames=1000,
        progress_interval=100,
    )
    assert cmd[cmd.index("--frames") + 1] == "1000"
    assert "bytetrack" in cmd
    assert "ocsort" in cmd
    assert "botsort" in cmd
    assert "msdc_elt" in cmd


def test_sensitivity_matrix_command_writes_matrix_under_run_root(tmp_path):
    cmd = build_sensitivity_matrix_command(
        output_root=tmp_path / "sensitivity",
        run_id="sensitivity_full",
    )

    assert cmd == [
        sys.executable,
        str(runner._ROOT / "tools" / "evaluation" / "msdc_sensitivity_matrix.py"),
        "--output",
        str(tmp_path / "sensitivity" / "sensitivity_full" / "sensitivity_matrix.csv"),
    ]


def test_sensitivity_command_runs_metrics_from_matrix_and_source_cache(tmp_path):
    cmd = build_sensitivity_command(
        dataset_roots=["/data/a", "/data/b"],
        matrix=tmp_path / "sensitivity" / "sensitivity_full" / "sensitivity_matrix.csv",
        output_root=tmp_path / "sensitivity",
        run_id="sensitivity_metrics",
        source_detection_cache_root=tmp_path / "main" / "main_full" / "detections",
        formal_frame_limit=FORMAL_FRAME_LIMIT,
        progress_interval=500,
        render_class_source="cache",
    )

    assert cmd[1] == str(runner._ROOT / "tools" / "evaluation" / "run_msdc_sensitivity_benchmark.py")
    assert cmd[cmd.index("--dataset-root") + 1:cmd.index("--matrix")] == ["/data/a", "/data/b"]
    assert cmd[cmd.index("--matrix") + 1] == str(
        tmp_path / "sensitivity" / "sensitivity_full" / "sensitivity_matrix.csv"
    )
    assert cmd[cmd.index("--source-detection-cache-root") + 1] == str(
        tmp_path / "main" / "main_full" / "detections"
    )
    assert cmd[cmd.index("--run-id") + 1] == "sensitivity_metrics"
    assert cmd[cmd.index("--formal-frame-limit") + 1] == "5400"
    assert cmd[cmd.index("--render-class-source") + 1] == "cache"


def test_slice_command_writes_manifest_under_slice_root(tmp_path):
    cmd = build_slice_command(
        diagnostics_root=tmp_path / "ablation" / "ablation_full" / "diagnostics",
        output_root=tmp_path / "slice",
    )

    assert cmd == [
        sys.executable,
        str(runner._ROOT / "tools" / "evaluation" / "msdc_slice_eval.py"),
        "--diagnostics-root",
        str(tmp_path / "ablation" / "ablation_full" / "diagnostics"),
        "--output",
        str(tmp_path / "slice" / "slice_manifest.csv"),
        "--max-len",
        "30",
    ]


def test_slice_metrics_command_uses_ablation_full_by_default(tmp_path):
    cmd = build_slice_metrics_command(run_root=tmp_path / "formal")

    assert cmd == [
        sys.executable,
        str(runner._ROOT / "tools" / "evaluation" / "msdc_slice_metrics.py"),
        "--slice-manifest",
        str(tmp_path / "formal" / "slice" / "slice_manifest.csv"),
        "--ablation-root",
        str(tmp_path / "formal" / "ablation" / "ablation_full"),
        "--output-root",
        str(tmp_path / "formal" / "slice" / "slice_metrics"),
    ]


def test_slice_metrics_command_can_target_ablation_cache_root(tmp_path):
    cmd = build_slice_metrics_command(
        run_root=tmp_path / "formal",
        ablation_run_id="ablation_cache_full",
    )

    assert cmd == [
        sys.executable,
        str(runner._ROOT / "tools" / "evaluation" / "msdc_slice_metrics.py"),
        "--slice-manifest",
        str(tmp_path / "formal" / "slice" / "slice_manifest.csv"),
        "--ablation-root",
        str(tmp_path / "formal" / "ablation" / "ablation_cache_full"),
        "--output-root",
        str(tmp_path / "formal" / "slice" / "slice_metrics"),
    ]


def test_default_dry_run_does_not_create_output_or_latest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_commit_hash", lambda: "abcdef0")
    output_root = tmp_path / "paper_runs"

    runner.main(["--output-root", str(output_root), "--run-id", "dry"])

    captured = capsys.readouterr()
    assert "[PLAN]" in captured.out
    assert not output_root.exists()
    assert not (output_root / "latest_run.json").exists()


def test_relative_output_root_is_resolved_under_repo_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_commit_hash", lambda: "abcdef0")

    runner.main(["--output-root", "relative_runs", "--run-id", "dry"])

    captured = capsys.readouterr()
    expected_root = tmp_path / "relative_runs" / "dry"
    assert str(expected_root / "main") in captured.out
    assert str(expected_root / "ablation") in captured.out
    assert str(expected_root / "slice") in captured.out
    assert str(expected_root / "speed") in captured.out
    assert str(expected_root / "sensitivity") in captured.out
    assert str(expected_root / "summary") in captured.out


def test_smoke_does_not_update_latest_formal_metadata(tmp_path, monkeypatch):
    output_root = tmp_path / "runs"
    latest_path = output_root / "latest_run.json"
    original = {"run_root": str(tmp_path / "old_run")}
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text(json.dumps(original), encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_commit_hash", lambda: "abcdef0")
    monkeypatch.setattr(runner, "run_command", lambda label, cmd: calls.append((label, cmd)))

    runner.main(["--smoke", "--output-root", str(output_root), "--run-id", "smoke"])

    assert [label for label, _ in calls] == ["smoke"]
    assert json.loads(latest_path.read_text(encoding="utf-8")) == original


def test_smoke_and_run_formal_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        runner.parse_args(["--smoke", "--run-formal"])


def test_formal_success_writes_latest_after_all_commands(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_commit_hash", lambda: "abcdef0")
    monkeypatch.setattr(runner, "run_command", lambda label, cmd: calls.append((label, cmd)))

    runner.main(["--run-formal", "--output-root", str(tmp_path / "runs"), "--run-id", "formal", "--speed-frames", "7"])

    assert [label for label, _ in calls] == [
        "main",
        "ablation",
        "slice",
        "slice_metrics",
        "speed",
        "sensitivity_matrix",
        "sensitivity_metrics",
        "summary",
    ]
    latest = json.loads((tmp_path / "runs" / "latest_run.json").read_text(encoding="utf-8"))
    assert latest["run_root"] == str(tmp_path / "runs" / "formal")
    assert latest["main_root"] == str(tmp_path / "runs" / "formal" / "main" / "main_full")
    assert latest["ablation_root"] == str(tmp_path / "runs" / "formal" / "ablation" / "ablation_full")
    assert latest["slice_manifest_csv"] == str(tmp_path / "runs" / "formal" / "slice" / "slice_manifest.csv")
    assert latest["speed_root"] == str(tmp_path / "runs" / "formal" / "speed" / "speed_7")
    assert latest["sensitivity_matrix_csv"] == str(
        tmp_path / "runs" / "formal" / "sensitivity" / "sensitivity_full" / "sensitivity_matrix.csv"
    )
    assert Path(latest["main_results_csv"]).is_absolute()


def test_formal_failure_does_not_write_latest(tmp_path, monkeypatch):
    output_root = tmp_path / "runs"
    calls = []

    def fail_on_speed(label, cmd):
        calls.append((label, cmd))
        if label == "speed":
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_commit_hash", lambda: "abcdef0")
    monkeypatch.setattr(runner, "run_command", fail_on_speed)

    with pytest.raises(subprocess.CalledProcessError):
        runner.main(["--run-formal", "--output-root", str(output_root), "--run-id", "formal"])

    assert [label for label, _ in calls] == ["main", "ablation", "slice", "slice_metrics", "speed"]
    assert not (output_root / "latest_run.json").exists()


def test_print_latest_and_check_latest_use_existing_metadata(tmp_path, capsys):
    run_root = tmp_path / "run"
    main_csv = run_root / "summary" / "main_results.csv"
    ablation_csv = run_root / "summary" / "ablation_results.csv"
    speed_csv = run_root / "summary" / "speed_results.csv"
    report = run_root / "MSDC_EXPERIMENT_REPORT.md"
    docs = tmp_path / "docs" / "MSDC_EXPERIMENT_RESULT.md"
    main_root = run_root / "main" / "main_full"
    ablation_root = run_root / "ablation" / "ablation_full"
    speed_root = run_root / "speed" / "speed_1000"
    for path in [main_csv, ablation_csv, speed_csv, report, docs]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for path in [main_root, ablation_root, speed_root]:
        path.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_root": str(run_root),
        "main_results_csv": str(main_csv),
        "ablation_results_csv": str(ablation_csv),
        "speed_results_csv": str(speed_csv),
        "report_path": str(report),
        "docs_result_path": str(docs),
        "main_root": str(main_root),
        "ablation_root": str(ablation_root),
        "speed_root": str(speed_root),
    }
    output_root = tmp_path / "runs"
    runner.write_latest_run(output_root / "latest_run.json", payload)

    runner.main(["--output-root", str(output_root), "--print-latest"])
    print_out = capsys.readouterr().out
    assert "main_results_csv:" in print_out
    assert str(main_csv) in print_out

    runner.main(["--output-root", str(output_root), "--check-latest"])
    assert "[OK]" in capsys.readouterr().out


def test_check_latest_cli_prints_error_without_traceback(tmp_path, capsys):
    output_root = tmp_path / "runs"
    runner.write_latest_run(
        output_root / "latest_run.json",
        {"run_root": str(tmp_path / "missing")},
    )

    with pytest.raises(SystemExit) as exc:
        runner.main(["--output-root", str(output_root), "--check-latest"])

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "Missing latest run paths" in captured.err
    assert "Traceback" not in captured.err


def test_run_command_executes_child_from_repo_root(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_ROOT", tmp_path)

    def fake_run(cmd, check, cwd):
        calls.append({"cmd": cmd, "check": check, "cwd": cwd})

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner.run_command("label", ["python", "script.py"])

    assert calls == [{"cmd": ["python", "script.py"], "check": True, "cwd": tmp_path}]


def test_formal_v3_ablation_variants_are_available():
    from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS, _variant_env

    expected = {
        FORMAL_MSDC_VARIANT,
        "no_low_candidate",
        "no_direct_reacquire",
        "no_low_inheritance",
        "hits_only_no_evidence",
        "no_output_nms",
        "no_output_real_det_age_gate",
        "low_budget_off",
        "low_budget_topk16",
        "low_budget_topk64",
        "removed_recovery_off",
        "reacquire_every_frame_low_score",
        "pending_recovery_candidate",
    }
    assert expected <= set(ABLATION_VARIANTS)

    env = _variant_env("low_budget_topk16")
    assert env["MSDC_USE_REACQUIRE"] == "1"
    assert env["MSDC_LOW_OBS_TOPK"] == "16"
    assert env["MSDC_LOW_OBS_GLOBAL_TOPK"] == "16"
    assert env["MSDC_LOW_OBS_MAX_PER_FRAME"] == "32"

    formal_env = _variant_env(FORMAL_MSDC_VARIANT)
    assert formal_env["MSDC_LOW_OBS_TOPK"] == "32"
    assert formal_env["MSDC_LOW_OBS_GLOBAL_TOPK"] == "32"
    assert formal_env["MSDC_LOW_OBS_MIN_CONF"] == "0.25"
    assert formal_env["MSDC_DEBUG_EVENTS"] == "1"
    assert formal_env["MSDC_REMOVED_GUARD_IOU_THRESH"] == "0.3"
    assert formal_env["MSDC_REMOVED_GUARD_CENTER_DIST"] == "80"
    assert formal_env["MSDC_REMOVED_RECOVERY_ENABLE"] == "1"
    assert formal_env["MSDC_REMOVED_RECOVERY_MIN_IOU"] == "0.20"
    assert formal_env["MSDC_REACQUIRE_INTERVAL"] == "1"
    assert formal_env["MSDC_REACQUIRE_SCORE"] == "0.8"
    assert formal_env["MSDC_PENDING_RECOVERY_ENABLE"] == "0"

    recovery_off_env = _variant_env("removed_recovery_off")
    assert recovery_off_env["MSDC_REMOVED_GUARD_IOU_THRESH"] == "0.3"
    assert recovery_off_env["MSDC_REMOVED_GUARD_CENTER_DIST"] == "80"
    assert recovery_off_env["MSDC_REMOVED_RECOVERY_ENABLE"] == "0"
    assert recovery_off_env["MSDC_REMOVED_RECOVERY_MIN_IOU"] == "0.20"

    every_frame_env = _variant_env("reacquire_every_frame_low_score")
    assert every_frame_env["MSDC_REACQUIRE_INTERVAL"] == "1"
    assert every_frame_env["MSDC_REACQUIRE_SCORE"] == "1.0"
    assert every_frame_env["MSDC_REMOVED_RECOVERY_ENABLE"] == "1"

    pending_env = _variant_env("pending_recovery_candidate")
    assert pending_env["MSDC_PENDING_RECOVERY_ENABLE"] == "1"
    assert pending_env["MSDC_PENDING_RECOVERY_FRAMES"] == "1"
    assert pending_env["MSDC_PENDING_RECOVERY_REQUIRE_CLASS_MATCH"] == "1"


def test_formal_v3_ablation_execution_env_ignores_ambient_msdc_overrides(monkeypatch):
    from tools.experiments.run_msdc_ablation import _execution_env, _variant_env

    monkeypatch.setenv("MSDC_CONFIRM_REQUIRE_HIGH_DET", "0")
    monkeypatch.setenv("UNRELATED_FLAG", "keep")

    env = _execution_env(FORMAL_MSDC_VARIANT, _variant_env(FORMAL_MSDC_VARIANT))

    assert "MSDC_CONFIRM_REQUIRE_HIGH_DET" not in env
    assert env["UNRELATED_FLAG"] == "keep"
