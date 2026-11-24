#!/usr/bin/env python3
"""Task-local ARIS experiment-queue adapter for direct WSL execution.

Adapted from ARIS/skills/experiment-queue/scripts/queue_manager.py, copied as
queue_manager_upstream.py. Preserve its manifest/state interface; require
successful dependencies, persisted process exits, and validated output markers.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

OOM_RE = re.compile(r"out of memory|OutOfMemoryError|failed to allocate memory", re.I)
TERMINAL = {"completed", "stuck", "blocked"}
LAST_GPU_SNAPSHOT = []


def now():
    return datetime.utcnow().isoformat() + "Z"


def save_state(state, state_file):
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)
    summary = {"updated": now(), "counts": dict(Counter(j["status"] for j in state["jobs"])),
               "completed": [j["id"] for j in state["jobs"] if j["status"] == "completed"],
               "failed": [j for j in state["jobs"] if j["status"] == "stuck"],
               "blocked": [j for j in state["jobs"] if j["status"] == "blocked"],
               "phases": state["phases"]}
    (path.parent / "queue_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_state(state_file, manifest):
    if Path(state_file).exists():
        return json.loads(Path(state_file).read_text(encoding="utf-8"))
    phases = []
    for phase in manifest["phases"]:
        deps = phase.get("depends_on", [])
        phases.append({"name": phase["name"], "depends_on": [deps] if isinstance(deps, str) else deps,
                       "status": "pending"})
    return {"meta": {"project": manifest.get("project"), "started": now(),
                     "manifest_path": manifest.get("_path"), "backend": "local_wsl"},
            "phases": phases, "jobs": []}


def assign_jobs_to_phases(manifest, state):
    for phase in manifest["phases"]:
        for job in phase["jobs"]:
            if not any(j["id"] == job["id"] for j in state["jobs"]):
                state["jobs"].append({**job, "phase": phase["name"], "status": "pending",
                    "gpu": None, "screen_name": None, "pid": None, "attempts": 0,
                    "manual_retries": 0, "started": None, "completed": None,
                    "error": None, "attempt_history": []})


def phase_complete(name, state):
    jobs = [j for j in state["jobs"] if j["phase"] == name]
    return bool(jobs) and all(j["status"] == "completed" for j in jobs)


def phase_ready(name, state):
    phases = {p["name"]: p for p in state["phases"]}
    phase = phases[name]
    return phase["status"] not in {"failed", "blocked"} and all(
        phases[dep]["status"] == "completed" for dep in phase["depends_on"])


def update_phases(state):
    for phase in state["phases"]:
        jobs = [j for j in state["jobs"] if j["phase"] == phase["name"]]
        phase["status"] = ("completed" if phase_complete(phase["name"], state) else
                           "failed" if any(j["status"] == "stuck" for j in jobs) else
                           "running" if any(j["status"] == "running" for j in jobs) else "pending")
    changed = True
    while changed:
        changed = False
        phases = {p["name"]: p for p in state["phases"]}
        for phase in state["phases"]:
            bad = [d for d in phase["depends_on"] if phases[d]["status"] in {"failed", "blocked"}]
            if bad and phase["status"] not in {"completed", "failed", "blocked"}:
                phase["status"] = "blocked"
                changed = True
            if phase["status"] in {"failed", "blocked"}:
                for job in state["jobs"]:
                    if job["phase"] == phase["name"] and job["status"] == "pending":
                        job.update(status="blocked", error="Dependency/phase failed: " + ", ".join(bad))
                        changed = True


def pending_jobs_in_active_phases(state, manifest):
    return [j for j in state["jobs"] if j["status"] == "pending" and phase_ready(j["phase"], state)]


def gpu_snapshot():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [dict(zip(("free_mib", "used_mib", "utilization_percent"), map(int, row.split(","))))
            for row in out.stdout.strip().splitlines()]


def free_gpus(allowed, minimum_free_mib=4096):
    global LAST_GPU_SNAPSHOT
    values = gpu_snapshot()
    LAST_GPU_SNAPSHOT = values
    return [gpu for gpu in allowed if gpu < len(values) and values[gpu]["free_mib"] >= minimum_free_mib]


def process_running(pid):
    try:
        if not pid:
            return False
        os.kill(int(pid), 0)
        path = Path(f"/proc/{int(pid)}/stat")
        return not (path.exists() and path.read_text().split(") ", 1)[1].split()[0] == "Z")
    except OSError:
        return False


def output_exists(path, cwd):
    path = Path(path)
    return (path if path.is_absolute() else Path(cwd) / path).is_file()


def detect_oom_in_log(path):
    return bool(path and Path(path).exists() and OOM_RE.search(Path(path).read_text(errors="replace")[-30000:]))


def launch_job(job, gpu, conda_env, cwd, log_dir, conda_hook=None):
    # Commands already include the validated absolute conda executable.
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base = log_dir / f'{job["id"]}.attempt{job["attempts"] + 1}'
    exit_file, log_file, script = (Path(str(base) + suffix) for suffix in (".exit", ".log", ".sh"))
    marker = Path(job["expected_output"])
    marker = marker if marker.is_absolute() else Path(cwd) / marker
    if marker.exists():
        marker.unlink()
    command = job["cmd"].replace("$" + "{GPU}", str(gpu))
    script.write_text("#!/usr/bin/env bash\nset +e\ncd " + shlex.quote(str(cwd)) +
        "\nbash -o pipefail -c " + shlex.quote(command) + "\nrc=$?\nprintf '%s\\n' \"$rc\" > " +
        shlex.quote(str(exit_file)) + "\nexit \"$rc\"\n", encoding="utf-8")
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(["bash", str(script)], cwd=cwd,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    job.update(log_file=str(log_file), exit_file=str(exit_file), runner_script=str(script), exit_code=None)
    return None, process.pid


def job_status_check(job, log_dir, cwd):
    exit_file = Path(job["exit_file"]) if job.get("exit_file") else None
    if exit_file and exit_file.exists():
        code = int(exit_file.read_text().strip())
        job["exit_code"] = code
        if code == 0 and output_exists(job["expected_output"], cwd):
            return "completed", None
        if detect_oom_in_log(job.get("log_file")):
            return "failed_oom", f"OOM, exit {code}"
        return "failed_other", f"Process exit {code}; validated success marker missing or command failed"
    if process_running(job.get("pid")):
        return "running", None
    if detect_oom_in_log(job.get("log_file")):
        return "failed_oom", "Process exited with OOM, no exit receipt"
    return "failed_other", "Process exited without an exit receipt"


def step(manifest, state, state_file, log_dir):
    for job in state["jobs"]:
        if job["status"] == "running":
            status, error = job_status_check(job, log_dir, manifest["cwd"])
            if status != "running":
                job.update(status=status, error=error, completed=now())
                job["attempt_history"].append({k: job.get(k) for k in
                    ("attempts", "started", "completed", "status", "error", "log_file", "exit_file", "pid")})
    for job in state["jobs"]:
        if job["status"] == "failed_other":
            job["status"] = "stuck"
        elif job["status"] == "failed_oom":
            retry = manifest.get("oom_retry", {})
            if job.get("oom_requires_manual_resume") or job["attempts"] >= min(3, int(retry.get("max_attempts", 3))):
                job["status"] = "stuck"
                if job.get("oom_requires_manual_resume"):
                    job["error"] += "; partial main requires targeted resume and is preserved"
            elif (datetime.utcnow() - datetime.fromisoformat(job["completed"].rstrip("Z"))).total_seconds() >= int(retry.get("delay", 120)):
                job["status"] = "pending"
    update_phases(state)
    running = [j for j in state["jobs"] if j["status"] == "running"]
    pending = pending_jobs_in_active_phases(state, manifest)
    if pending and not running:
        free = free_gpus(manifest.get("gpus", [0]), int(manifest.get("gpu_minimum_free_mib", 4096)))
        state["meta"]["gpu_last_observation"] = {"time": now(), "gpus": LAST_GPU_SNAPSHOT}
        if free:
            job = pending[0]
            screen, pid = launch_job(job, free[0], manifest.get("conda"), manifest["cwd"], log_dir)
            job.update(status="running", gpu=free[0], screen_name=screen, pid=pid,
                       attempts=job["attempts"] + 1, started=now(), completed=None, error=None)
    update_phases(state)
    save_state(state, state_file)


def all_done(state):
    return bool(state["jobs"]) and all(j["status"] in TERMINAL for j in state["jobs"])


def retry_job(state, job_id, reason):
    job = next(j for j in state["jobs"] if j["id"] == job_id)
    if job["status"] != "stuck" or job.get("manual_retries", 0) >= 1 or job["attempts"] >= 3 or not reason.strip():
        raise ValueError("A stuck job allows one targeted retry with a concrete repair reason")
    job.update(status="pending", manual_retries=job.get("manual_retries", 0) + 1,
               error=None, completed=None, repair_reason=reason)
    for item in state["jobs"]:
        if item["status"] == "blocked":
            item.update(status="pending", error=None)
    update_phases(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--poll", type=float, default=60)
    parser.add_argument("--retry-job")
    parser.add_argument("--retry-reason", default="")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if int(manifest.get("max_parallel", 1)) != 1:
        raise ValueError("Paper queue requires max_parallel=1")
    manifest["_path"] = str(Path(args.manifest).resolve())
    state = load_state(args.state, manifest)
    assign_jobs_to_phases(manifest, state)
    if args.retry_job:
        retry_job(state, args.retry_job, args.retry_reason)
    save_state(state, args.state)
    print(f"[{now()}] Local ARIS queue: {len(state['jobs'])} jobs, serial", flush=True)
    while not all_done(state):
        step(manifest, state, args.state, args.log_dir)
        if not all_done(state):
            time.sleep(args.poll)
    failed = [j for j in state["jobs"] if j["status"] != "completed"]
    print(f"[{now()}] Terminal: {len(state['jobs'])-len(failed)} completed, {len(failed)} failed/blocked", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
