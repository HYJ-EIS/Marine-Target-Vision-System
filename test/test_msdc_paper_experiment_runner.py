import sys
import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation import run_msdc_paper_experiments as runner
from tools.evaluation.run_msdc_paper_experiments import (
    build_ablation_command,
    build_main_command,
    build_speed_command,
)


def test_main_command_uses_full_video_and_render():
    cmd = build_main_command(
        dataset_roots=["/data/a", "/data/b"],
        output_root=Path("results/main"),
        run_id="main_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=True,
    )
    assert "--max-frames" not in cmd
    assert "--duration-seconds" not in cmd
    assert "--render" in cmd
    assert "--run" in cmd
    assert cmd[cmd.index("--trackers") + 1:cmd.index("--variants")] == ["ocsort", "botsort", "msdc_elt"]


def test_ablation_command_includes_required_variants():
    cmd = build_ablation_command(
        dataset_roots=["/data/a"],
        output_root=Path("results/ablation"),
        run_id="ablation_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=True,
    )
    assert "Ours-full" in cmd
    assert "Ours-lite-no-motion" in cmd
    assert "Ours-lite-no-low-det" in cmd
    assert "Ours-no-template" in cmd
    assert "Ours-no-reacquire" in cmd
    assert "Ours-no-removed-guard" in cmd


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
    assert "ocsort" in cmd
    assert "botsort" in cmd
    assert "msdc_elt" in cmd


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
    assert str(expected_root / "speed") in captured.out
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

    assert [label for label, _ in calls] == ["main", "ablation", "speed", "summary"]
    latest = json.loads((tmp_path / "runs" / "latest_run.json").read_text(encoding="utf-8"))
    assert latest["run_root"] == str(tmp_path / "runs" / "formal")
    assert latest["main_root"] == str(tmp_path / "runs" / "formal" / "main" / "main_full")
    assert latest["ablation_root"] == str(tmp_path / "runs" / "formal" / "ablation" / "ablation_full")
    assert latest["speed_root"] == str(tmp_path / "runs" / "formal" / "speed" / "speed_7")
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

    assert [label for label, _ in calls] == ["main", "ablation", "speed"]
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


def test_v2_ablation_variants_are_available():
    from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS

    expected = {
        "v2_template_off",
        "v2_shared_det",
        "v2_low_clean",
        "v2_output_age5_size8",
        "v2_output_age8_size8",
        "v2_output_age12_size8",
        "v2_candidate_low3_window6",
        "v2_candidate_low4_window8",
        "v2_candidate_real2_age8",
        "v2_roi_budget_active8_max2",
        "v2_roi_budget_active10_max2",
        "v2_reacquire_interval1",
        "v2_reacquire_interval2",
        "v2_reacquire_interval5_center240",
    }
    assert expected <= set(ABLATION_VARIANTS)
    assert ABLATION_VARIANTS["v2_template_off"]["MSDC_USE_TEMPLATE"] == "0"
    assert ABLATION_VARIANTS["v2_shared_det"]["MSDC_EXPORT_SHARE_LOW_HIGH_DET"] == "1"
    assert ABLATION_VARIANTS["v2_roi_budget_active8_max2"]["MSDC_ROI_REDETECT_MAX_TRACKS"] == "2"
