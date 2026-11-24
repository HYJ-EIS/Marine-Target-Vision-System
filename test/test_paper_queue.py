import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments import paper_queue as pq


def manager():
    # This test file lives in the isolated snapshot beside the task-local copy.
    path = Path(os.environ.get("PAPER_QUEUE_MANAGER_PATH", ROOT / "tools/experiments/paper_queue_manager.py"))
    module_spec = importlib.util.spec_from_file_location("paper_local_queue", path)
    result = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(result)
    return result


def synthetic_manifest(tmp_path):
    return {"project": "synthetic", "cwd": str(tmp_path), "max_parallel": 1,
            "gpus": [0], "oom_retry": {"delay": 120, "max_attempts": 3},
            "phases": [
                {"name": "first", "depends_on": [], "jobs": [
                    {"id": "a", "cmd": "never execute", "expected_output": str(tmp_path / "a.done")},
                    {"id": "b", "cmd": "never execute", "expected_output": str(tmp_path / "b.done")}]},
                {"name": "second", "depends_on": ["first"], "jobs": [
                    {"id": "c", "cmd": "never execute", "expected_output": str(tmp_path / "c.done")}]}]}


def init(qm, manifest, tmp_path):
    state = qm.load_state(tmp_path / "state.json", manifest)
    qm.assign_jobs_to_phases(manifest, state)
    return state


def test_stuck_phase_never_counts_success_and_blocks_downstream(tmp_path, monkeypatch):
    qm, manifest = manager(), synthetic_manifest(tmp_path)
    state = init(qm, manifest, tmp_path)
    state["jobs"][0].update(status="stuck", attempts=1, error="bad input")
    monkeypatch.setattr(qm, "free_gpus", lambda *args: [0])
    monkeypatch.setattr(qm, "launch_job", lambda *args: pytest.fail("Failure may not launch any sibling or dependent"))
    qm.step(manifest, state, tmp_path / "state.json", tmp_path)
    assert not qm.phase_complete("first", state)
    assert state["phases"][0]["status"] == "failed"
    assert [job["status"] for job in state["jobs"]] == ["stuck", "blocked", "blocked"]
    summary = json.loads((tmp_path / "queue_summary.json").read_text())
    assert summary["failed"][0]["id"] == "a"
    assert len(summary["blocked"]) == 2


def test_success_requires_exit_zero_and_marker(tmp_path):
    qm = manager()
    exit_file, marker = tmp_path / "exit", tmp_path / "marker"
    job = {"exit_file": str(exit_file), "expected_output": str(marker), "pid": None}
    marker.write_text("{}")
    assert qm.job_status_check(job, tmp_path, tmp_path)[0] == "failed_other"
    exit_file.write_text("1")
    assert qm.job_status_check(job, tmp_path, tmp_path)[0] == "failed_other"
    exit_file.write_text("0")
    assert qm.job_status_check(job, tmp_path, tmp_path)[0] == "completed"
    marker.unlink()
    assert qm.job_status_check(job, tmp_path, tmp_path)[0] == "failed_other"


def test_only_one_process_launch_and_resume_reuses_running_job(tmp_path, monkeypatch):
    qm, manifest = manager(), synthetic_manifest(tmp_path)
    state = init(qm, manifest, tmp_path)
    launches = []
    monkeypatch.setattr(qm, "free_gpus", lambda *args: [0, 1])
    monkeypatch.setattr(qm, "launch_job", lambda job, *args: (launches.append(job["id"]) or None, 12345))
    monkeypatch.setattr(qm, "job_status_check", lambda *args: ("running", None))
    qm.step(manifest, state, tmp_path / "state.json", tmp_path)
    assert launches == ["a"]
    reloaded = qm.load_state(tmp_path / "state.json", manifest)
    qm.step(manifest, reloaded, tmp_path / "state.json", tmp_path)
    assert launches == ["a"]
    assert reloaded["jobs"][0]["pid"] == 12345


def test_manual_repair_releases_dependencies_only_once(tmp_path):
    qm, manifest = manager(), synthetic_manifest(tmp_path)
    state = init(qm, manifest, tmp_path)
    state["jobs"][0].update(status="stuck", attempts=1)
    qm.update_phases(state)
    qm.retry_job(state, "a", "Fixed generated wrapper path")
    assert state["jobs"][0]["status"] == "pending"
    assert not qm.phase_ready("second", state)
    state["jobs"][0]["status"] = "stuck"
    with pytest.raises(ValueError):
        qm.retry_job(state, "a", "Another retry")


def test_oom_wait_and_three_total_attempt_limit(tmp_path, monkeypatch):
    qm, manifest = manager(), synthetic_manifest(tmp_path)
    state = init(qm, manifest, tmp_path)
    job = state["jobs"][0]
    job.update(status="failed_oom", attempts=2, completed=qm.now(), error="OOM")
    monkeypatch.setattr(qm, "free_gpus", lambda *args: [])
    qm.step(manifest, state, tmp_path / "state.json", tmp_path)
    assert job["status"] == "failed_oom"
    job["completed"] = "2020-01-01T00:00:00Z"
    qm.step(manifest, state, tmp_path / "state.json", tmp_path)
    assert job["status"] == "pending"
    job.update(status="failed_oom", completed=qm.now())
    job.update(attempts=3)
    qm.step(manifest, state, tmp_path / "state.json", tmp_path)
    assert job["status"] == "stuck"
    with pytest.raises(ValueError):
        qm.retry_job(state, "a", "Must not exceed three total attempts")


def test_free_memory_threshold_ignores_display_utilization(monkeypatch):
    qm = manager()
    monkeypatch.setattr(qm, "gpu_snapshot", lambda: [
        {"free_mib": 5000, "used_mib": 3000, "utilization_percent": 98},
        {"free_mib": 4000, "used_mib": 4000, "utilization_percent": 0}])
    assert qm.free_gpus([0, 1], 4096) == [0]


def test_manifest_has_exact_fixed_21_jobs_and_dependencies(tmp_path):
    spec = {"sequences": [{"sequence": f"seq{i}"} for i in range(9)],
            "core_variants": [f"core{i}" for i in range(8)],
            "sensitivity_variants": [f"sens{i}" for i in range(8)]}
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(spec))
    manifest = pq.build_manifest(tmp_path / "run", source, ROOT)
    assert [len(p["jobs"]) for p in manifest["phases"]] == [1, 9, 1, 9, 1]
    assert [p["depends_on"] for p in manifest["phases"]] == [
        [], ["main"], ["core"], ["core_stats"], ["sensitivity"]]
    assert manifest["formal_unique_results"] == 180 and manifest["max_parallel"] == 1
    jobs = [j for p in manifest["phases"] for j in p["jobs"]]
    assert len({j["id"] for j in jobs}) == 21
    assert all(j["expected_output"].endswith(".completed.json") for j in jobs)
    assert manifest["oom_retry"] == {"delay": 120, "max_attempts": 3}


def test_existing_main_is_preserved_and_failure_recorded(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    (tmp_path / "main").mkdir()
    witness = tmp_path / "main/witness"
    witness.write_text("preserve")
    args = SimpleNamespace(run_root=tmp_path, spec=spec, stage="main", sequence="", job_id="main")
    with pytest.raises(RuntimeError, match="targeted resume"):
        pq.run_job(args, execute=lambda _: pytest.fail("Must not relaunch main into existing directory"))
    assert witness.read_text() == "preserve"
    assert not pq.marker_path(tmp_path, "main").exists()
    assert len(list((tmp_path / "queue/failures").glob("*.json"))) == 1


def test_failed_job_cannot_emit_completed_marker(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    args = SimpleNamespace(run_root=tmp_path, spec=spec, stage="core_stats", sequence="", job_id="core_stats")
    calls = []
    def fail(command):
        calls.append(command)
        raise RuntimeError("Synthetic command error")
    with pytest.raises(RuntimeError, match="Synthetic"):
        pq.run_job(args, execute=fail)
    assert len(calls) == 1
    assert not pq.marker_path(tmp_path, "core_stats").exists()


def test_partial_preservation_moves_only_the_failed_result(tmp_path):
    failed = tmp_path / "core/trackers/fail_replay/data/seq.txt"
    valid = tmp_path / "core/trackers/valid_replay/data/seq.txt"
    for path in (failed, valid):
        path.parent.mkdir(parents=True)
        path.write_text("x")
    pq.preserve_partial_result(tmp_path, "core", "fail_replay", "seq", "core_01")
    assert not failed.exists() and valid.read_text() == "x"
    assert len(list((tmp_path / "queue/partial_results").rglob("seq.txt"))) == 1


def test_real_cpu_runner_persists_exit_and_marker_without_gpu_task(tmp_path):
    qm = manager()
    marker = tmp_path / "ok.json"
    job = {"id": "noop", "attempts": 0, "cmd": f"printf '{{}}' > {marker}",
           "expected_output": str(marker)}
    screen, pid = qm.launch_job(job, 0, "unused", tmp_path, tmp_path / "logs")
    assert screen is None and pid > 0
    job["pid"] = pid
    for _ in range(100):
        if Path(job["exit_file"]).exists():
            break
        time.sleep(.01)
    assert Path(job["exit_file"]).read_text().strip() == "0"
    assert qm.job_status_check(job, tmp_path, tmp_path) == ("completed", None)


def test_stats_marker_written_only_after_both_commands_and_validation(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    calls = []
    monkeypatch.setattr(pq, "validate_protocol", lambda _: calls.append("protocol"))
    monkeypatch.setattr(pq, "validate_statistics", lambda *_: calls.append("validated") or {"rows": 108})
    monkeypatch.setattr(pq, "validate_cases", lambda *_: calls.append("cases_validated") or {"cases": 0})
    args = SimpleNamespace(run_root=tmp_path, spec=spec, stage="core_stats", sequence="", job_id="core_stats")
    marker = pq.run_job(args, execute=lambda cmd: calls.append(cmd))
    assert len([item for item in calls if isinstance(item, list)]) == 3
    assert calls[-1] == "cases_validated"
    assert json.loads(marker.read_text())["status"] == "completed"


def write_complete_result(run, batch, tracker, seq, length, last_output):
    import cv2
    import numpy as np
    mot = run / batch / "trackers" / tracker / "data" / f"{seq}.txt"
    mot.parent.mkdir(parents=True, exist_ok=True)
    mot.write_text(f"{last_output},1,1,1,8,8,.9,3,-1,-1\n" if last_output else "")
    timing = {"processed_frames": length, "stage_seconds": dict.fromkeys(pq.STAGES, 0)}
    mot.with_suffix(".timing.json").write_text(json.dumps(timing))
    mot.with_suffix(".frames.jsonl").write_text("".join(json.dumps({"frame_id": i}) + "\n" for i in range(1, length+1)))
    logs = mot.parent.parent / "paper_logs" / seq
    logs.mkdir(parents=True, exist_ok=True)
    for name in pq.LOGS:
        (logs / f"{name}.jsonl").write_text("".join(json.dumps(
            {"seq": seq, "method": tracker, "frame": i, "records": []}) + "\n" for i in range(1, length+1)))
    video = run / batch / "visualizations" / seq / f"{tracker}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (16, 16))
    for _ in range(length):
        writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    video.with_suffix(".timing.json").write_text(json.dumps(timing))
    return mot, video


def test_repairing_one_variant_never_replays_valid_peer_with_no_last_frame_output(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"core_variants": ["good", "bad"]}))
    (tmp_path / "main").mkdir()
    (tmp_path / "main/inventory.json").write_text(json.dumps([{"sequence": "seq", "frames": 100}]))
    good, video = write_complete_result(tmp_path, "core", "good_replay", "seq", 100, 99)
    existing = [good, good.with_suffix(".timing.json"), good.with_suffix(".frames.jsonl"), video]
    existing += list((good.parent.parent / "paper_logs/seq").glob("*.jsonl"))
    before = {path: path.read_bytes() for path in existing}
    (tmp_path / "core/eval").mkdir(parents=True)
    (tmp_path / "core/eval/evaluation_protocol.json").write_text(json.dumps(
        {"class_consistency_required": True, "matching": "same_class_and_iou"}))
    (tmp_path / "core/eval/old_tracker").mkdir()
    (tmp_path / "core/eval/old_tracker/pedestrian_detailed.csv").write_text("seq\nprevious_sequence\n")
    (tmp_path / "core/summary").mkdir()
    (tmp_path / "core/summary/replay_summary.csv").write_text("tracker\nprevious_tracker\n")
    calls = []
    def scoped_eval(command):
        assert command[command.index("--sequences")+1] == "seq"
        trackers = command[command.index("--trackers")+1:command.index("--sequences")]
        assert trackers == ["good_replay", "bad_replay"]
        directory = Path(command[command.index("--output-root")+1])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "evaluation_protocol.json").write_text(json.dumps(
            {"class_consistency_required": True, "matching": "same_class_and_iou"}))
        for tracker in trackers:
            (directory / tracker).mkdir(exist_ok=True)
            (directory / tracker / "pedestrian_detailed.csv").write_text("seq\nseq\n")
    def execute(command):
        if "tools/evaluation/motchallenge_eval.py" in command:
            scoped_eval(command)
            return
        variants = command[command.index("--variants")+1:command.index("--formal-frame-limit")]
        assert variants == ["bad"]
        calls.append(variants)
        write_complete_result(tmp_path, "core", "bad_replay", "seq", 100, 0)
        (tmp_path / "core/summary/replay_summary.csv").write_text("tracker\nbad_replay\n")
    args = SimpleNamespace(run_root=tmp_path, spec=spec_path, stage="core", sequence="seq", job_id="core_01")
    pq.run_job(args, execute=execute)
    assert calls == [["bad"]]
    assert all(path.read_bytes() == data for path, data in before.items())
    metadata = tmp_path / "core/metadata/seq"
    assert not (metadata / "eval/old_tracker").exists()
    assert not (metadata / "summary").exists()
    scope = json.loads((metadata / "metadata_scope.json").read_text())
    assert scope["replayed_variants"] == ["bad"] and scope["reused_trackers"] == ["good_replay"]
    assert (metadata / "eval/good_replay/pedestrian_detailed.csv").read_text() == "seq\nseq\n"
    calls_after = []
    def reuse_execute(command):
        assert "tools/evaluation/motchallenge_eval.py" in command
        calls_after.append(command)
        scoped_eval(command)
    pq.run_job(args, execute=reuse_execute)
    assert len(calls_after) == 1
    assert json.loads((metadata / "metadata_scope.json").read_text())["replayed_variants"] == []
    assert all(path.read_bytes() == data for path, data in before.items())


def test_all_stats_reuses_core_case_videos(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    monkeypatch.setattr(pq, "validate_protocol", lambda _: None)
    monkeypatch.setattr(pq, "validate_statistics", lambda *_: {"rows": 180})
    monkeypatch.setattr(pq, "validate_cases", lambda *_: {"cases": 2})
    commands = []
    args = SimpleNamespace(run_root=tmp_path, spec=spec, stage="all_stats", sequence="", job_id="all_stats")
    pq.run_job(args, execute=lambda cmd: commands.append(cmd))
    assert len(commands) == 2
    assert not any("export_paper_cases.py" in str(cmd) for cmd in commands)
