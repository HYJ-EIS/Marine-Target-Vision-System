# MSDC Paper Phase 1 Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the first phase of paper experiments for MS-DC-ELT: main comparison, ablation, speed/complexity, and a truthful report from real outputs only.

**Architecture:** Keep detector, baseline trackers, and MS-DC-ELT algorithm code unchanged. Add experiment-layer scripts that orchestrate existing MOT export, TrackEval, diagnostics, rendering, timing, metadata capture, CSV aggregation, and report generation into timestamped run directories.

**Tech Stack:** Python 3.11 via `conda run -n ship_detect`, OpenCV, vendored TrackEval, existing `target_module.image_detect_module` APIs, CSV/JSON/Markdown artifacts.

---

## Scope Rules

- Do not modify core algorithm files: detector implementation, tracker implementation, `MSDCLifecycleTracker`, evidence state logic, motion seed logic, ROI redetection logic, template lock logic.
- Allowed changes are limited to experiment scripts, result aggregation scripts, tests for those scripts, README/documentation, and generated experiment report files.
- Every formal number must come from a real command output in the current run directory.
- Smoke runs use `--max-frames` or `--duration-seconds` and must be recorded as smoke only.
- Formal main and ablation runs must not pass `--max-frames` or `--duration-seconds`.
- Speed runs may pass a fixed frame count, and the exact processed frame count must be written to `speed_results.csv` and the report.
- All user-facing Python and pytest commands must use `conda run -n ship_detect ...`.

## Required Dataset Roots

Use these exact POSIX paths:

- `/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集`
- `/home/hyj/Anti_Drone_Project/USV_MOT标注数据集`

The existing dataset resolver may read `原视频地址.txt` or `原视频文件地址.txt` inside each root and convert Windows/WSL path forms to local POSIX paths. The experiment runner must fail clearly if a referenced full video is not accessible.

## File Structure

- Modify `tools/evaluation/motchallenge_eval.py`
  - Add reliable `DetA` and `AssA` extraction from TrackEval HOTA output when keys are present.
  - Keep existing HOTA, MOTA, IDF1, IDSW, FP, FN, IDTP, IDFP, IDFN behavior.
- Modify `tools/evaluation/msdc_dataset_benchmark.py`
  - Add deterministic `--run-id`.
  - Write `run_metadata.json`, `commands.jsonl`, and `failures.csv`.
  - Keep existing export/eval/render behavior and existing full-video semantics.
- Create `tools/evaluation/msdc_experiment_summary.py`
  - Read raw benchmark outputs.
  - Write `main_results.csv`, `ablation_results.csv`, `MSDC_EXPERIMENT_REPORT.md`, and `docs/MSDC_EXPERIMENT_RESULT.md`.
  - Refuse to mark formal results complete when required paths or metrics are missing.
- Create `tools/evaluation/msdc_speed_benchmark.py`
  - Run OC-SORT, BoT-SORT, and MS-DC-ELT on the same video and same frame count.
  - Disable rendering, MQ, TrackEval, and MS-DC-ELT debug JSONL.
  - Measure read/decode, detection, low-threshold detection, tracker/update, total latency, FPS, P50, P95, peak memory, and detector call count.
- Create `tools/evaluation/run_msdc_paper_experiments.py`
  - Orchestrate smoke, formal main, formal ablation, speed, summary, and report generation.
  - Use timestamped directories under `results/msdc_paper_phase1/<run-id>/`.
- Create tests:
  - `test/test_msdc_experiment_summary.py`
  - `test/test_msdc_speed_benchmark.py`
  - `test/test_msdc_paper_experiment_runner.py`
  - Modify `test/test_motchallenge_eval.py`
  - Modify `test/test_msdc_dataset_benchmark.py`
- Modify docs:
  - `README.md`
  - `tools/README.md`
  - Create/update `docs/MSDC_EXPERIMENT_RESULT.md` only with real results after the formal run.
  - Create/update root `MSDC_EXPERIMENT_REPORT.md` only after the formal run.

## Output Layout

Use this layout for every execution:

```text
results/msdc_paper_phase1/<run-id>/
  metadata/
    run_metadata.json
    commands.jsonl
    failures.csv
  smoke/
  main/
    <benchmark-run-id>/
      mot_gt/
      trackers/
      eval/motchallenge_summary.csv
      diagnostics/
      visualizations/
  ablation/
    <benchmark-run-id>/
      mot_gt/
      trackers/
      eval/motchallenge_summary.csv
      diagnostics/
      visualizations/
  speed/
    speed_results.csv
    speed_timings.jsonl
    speed_metadata.json
  summary/
    main_results.csv
    ablation_results.csv
    speed_results.csv
    path_manifest.csv
    analysis_main.md
    analysis_ablation.md
    analysis_speed.md
  MSDC_EXPERIMENT_REPORT.md
```

---

### Task 1: Extend TrackEval Summary Metrics

**Files:**
- Modify: `tools/evaluation/motchallenge_eval.py`
- Modify: `test/test_motchallenge_eval.py`

- [ ] **Step 1: Write the failing test**

Append these assertions to `test_motchallenge_eval_reports_formal_metrics_for_perfect_track` in `test/test_motchallenge_eval.py`:

```python
    assert perfect["DetA"] == 100.0
    assert perfect["AssA"] == 100.0

    summary_text = (output_root / "motchallenge_summary.csv").read_text(encoding="utf-8-sig")
    assert "DetA" in summary_text
    assert "AssA" in summary_text
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
conda run -n ship_detect pytest test/test_motchallenge_eval.py::test_motchallenge_eval_reports_formal_metrics_for_perfect_track -q
```

Expected: FAIL because `DetA` and `AssA` are not yet included in the summary dict or CSV.

- [ ] **Step 3: Implement DetA/AssA extraction**

In `tools/evaluation/motchallenge_eval.py`, add this helper above `_summary_from_trackeval`:

```python
def _mean_hota_field(combined: dict, field: str) -> float | str:
    hota_values = combined.get("HOTA", {})
    if field not in hota_values:
        return "N/A"
    return round(float(np.mean(hota_values[field])) * 100.0, 5)
```

Then update each tracker summary row inside `_summary_from_trackeval`:

```python
        summary[tracker] = {
            "HOTA": round(hota, 5),
            "MOTA": round(mota, 5),
            "IDF1": round(idf1, 5),
            "DetA": _mean_hota_field(combined, "DetA"),
            "AssA": _mean_hota_field(combined, "AssA"),
            "IDSW": int(combined["CLEAR"]["IDSW"]),
            "FP": int(combined["CLEAR"]["CLR_FP"]),
            "FN": int(combined["CLEAR"]["CLR_FN"]),
            "IDTP": int(combined["Identity"]["IDTP"]),
            "IDFP": int(combined["Identity"]["IDFP"]),
            "IDFN": int(combined["Identity"]["IDFN"]),
        }
```

Update `_write_summary_csv` fields:

```python
    fields = [
        "tracker",
        "HOTA",
        "DetA",
        "AssA",
        "MOTA",
        "IDF1",
        "IDSW",
        "FP",
        "FN",
        "IDTP",
        "IDFP",
        "IDFN",
    ]
```

Update the CLI print block:

```python
            f"{tracker}: HOTA={values['HOTA']}, DetA={values['DetA']}, "
            f"AssA={values['AssA']}, MOTA={values['MOTA']}, "
            f"IDF1={values['IDF1']}, IDSW={values['IDSW']}, "
            f"FP={values['FP']}, FN={values['FN']}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
conda run -n ship_detect pytest test/test_motchallenge_eval.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/evaluation/motchallenge_eval.py test/test_motchallenge_eval.py
git commit -m "test: include DetA and AssA in mot summary"
```

---

### Task 2: Add Run Metadata and Failure Logging to Dataset Benchmark

**Files:**
- Modify: `tools/evaluation/msdc_dataset_benchmark.py`
- Modify: `test/test_msdc_dataset_benchmark.py`

- [ ] **Step 1: Write failing tests for run IDs and metadata helpers**

Add these imports to `test/test_msdc_dataset_benchmark.py`:

```python
import json
from tools.evaluation.msdc_dataset_benchmark import (
    build_run_metadata,
    write_command_record,
    write_failure_record,
)
```

Add this test:

```python
def test_metadata_and_failure_records_are_written(tmp_path):
    metadata = build_run_metadata(
        run_id="paper_20260608_120000",
        mode="main",
        commit_hash="abcdef0",
        dataset_roots=[tmp_path / "dataset_a", tmp_path / "dataset_b"],
        output_root=tmp_path / "out",
        trackers=["ocsort", "botsort", "msdc_elt"],
        variants=["Ours-full"],
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
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_dataset_benchmark.py::test_metadata_and_failure_records_are_written -q
```

Expected: FAIL because the helper functions do not exist.

- [ ] **Step 3: Add metadata helpers**

In `tools/evaluation/msdc_dataset_benchmark.py`, add these imports:

```python
import datetime as _dt
```

Add these helpers after `_env_prefix`:

```python
def iso_now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def build_run_metadata(
    run_id: str,
    mode: str,
    commit_hash: str,
    dataset_roots: list[str | Path],
    output_root: str | Path,
    trackers: list[str],
    variants: list[str],
    max_frames: int,
    duration_seconds: float,
    render: bool,
    formal: bool,
) -> dict:
    return {
        "run_id": run_id,
        "mode": mode,
        "commit_hash": commit_hash,
        "dataset_roots": [str(Path(path)) for path in dataset_roots],
        "output_root": str(Path(output_root)),
        "trackers": list(trackers),
        "variants": list(variants),
        "max_frames": int(max_frames),
        "duration_seconds": float(duration_seconds),
        "render": bool(render),
        "formal": bool(formal),
        "started_at": iso_now(),
    }


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_command_record(
    path: str | Path,
    label: str,
    command: list[str],
    env_delta: dict[str, str],
    status: str,
    start_time: str,
    end_time: str,
    returncode: int | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "label": label,
        "command": list(command),
        "env_delta": dict(env_delta),
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "returncode": returncode,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_failure_record(
    path: str | Path,
    stage: str,
    label: str,
    command: str,
    returncode: int | None,
    message: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8-sig") as fh:
        fields = ["stage", "label", "command", "returncode", "message"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "stage": stage,
            "label": label,
            "command": command,
            "returncode": "" if returncode is None else int(returncode),
            "message": message,
        })
```

Add parser args:

```python
    parser.add_argument("--run-id", default="", help="Timestamp/run folder name; default uses current time")
    parser.add_argument("--mode", default="benchmark", choices=["benchmark", "main", "ablation", "smoke"])
    parser.add_argument("--commit-hash", default="", help="Commit hash recorded in run metadata")
```

Change run id initialization:

```python
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
```

Create these paths after `root` is initialized:

```python
    metadata_dir = root / "metadata"
    commands_path = metadata_dir / "commands.jsonl"
    failures_path = metadata_dir / "failures.csv"
    metadata = build_run_metadata(
        run_id=run_id,
        mode=args.mode,
        commit_hash=args.commit_hash,
        dataset_roots=[Path(path) for path in args.dataset_root],
        output_root=root,
        trackers=list(args.trackers),
        variants=list(args.variants),
        max_frames=int(args.max_frames),
        duration_seconds=float(args.duration_seconds),
        render=bool(args.render),
        formal=not bool(args.max_frames) and not bool(args.duration_seconds),
    )
    write_json(metadata_dir / "run_metadata.json", metadata)
```

Wrap every `subprocess.run(...)` call with command record and failure record. Use this exact pattern:

```python
                started = iso_now()
                try:
                    completed = subprocess.run(cmd, cwd=str(_ROOT), env=env, check=True)
                    write_command_record(commands_path, f"EXPORT:{tracker_name}", cmd, env_delta, "success", started, iso_now(), completed.returncode)
                except subprocess.CalledProcessError as exc:
                    write_command_record(commands_path, f"EXPORT:{tracker_name}", cmd, env_delta, "failed", started, iso_now(), exc.returncode)
                    write_failure_record(failures_path, "export", tracker_name, _quote_command(cmd), exc.returncode, str(exc))
                    raise
```

Use the same pattern for render and eval stages, with labels `RENDER:<tracker_name>` and `EVAL`.

- [ ] **Step 4: Run dataset benchmark tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_dataset_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/evaluation/msdc_dataset_benchmark.py test/test_msdc_dataset_benchmark.py
git commit -m "feat: record benchmark run metadata"
```

---

### Task 3: Add Experiment Summary and Report Generator

**Files:**
- Create: `tools/evaluation/msdc_experiment_summary.py`
- Create: `test/test_msdc_experiment_summary.py`

- [ ] **Step 1: Write tests for summary CSV generation**

Create `test/test_msdc_experiment_summary.py`:

```python
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.msdc_experiment_summary import (
    METHOD_LABELS,
    build_analysis_text,
    load_metric_rows,
    write_ablation_results,
    write_main_results,
    write_path_manifest,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_main_results_maps_trackers_to_method_labels(tmp_path):
    summary = tmp_path / "eval" / "motchallenge_summary.csv"
    _write_csv(summary, [
        {"tracker": "ocsort", "HOTA": "10", "DetA": "20", "AssA": "30", "MOTA": "40", "IDF1": "50", "IDSW": "6", "FP": "7", "FN": "8", "IDTP": "9", "IDFP": "10", "IDFN": "11"},
        {"tracker": "botsort", "HOTA": "11", "DetA": "21", "AssA": "31", "MOTA": "41", "IDF1": "51", "IDSW": "5", "FP": "6", "FN": "7", "IDTP": "8", "IDFP": "9", "IDFN": "10"},
        {"tracker": "Ours-full", "HOTA": "12", "DetA": "22", "AssA": "32", "MOTA": "42", "IDF1": "52", "IDSW": "4", "FP": "5", "FN": "6", "IDTP": "7", "IDFP": "8", "IDFN": "9"},
    ])

    out = tmp_path / "main_results.csv"
    rows = write_main_results(
        metric_rows=load_metric_rows(summary),
        output_csv=out,
        run_name="main_full",
        benchmark_root=tmp_path,
    )

    assert [row["method"] for row in rows] == [
        METHOD_LABELS["ocsort"],
        METHOD_LABELS["botsort"],
        METHOD_LABELS["Ours-full"],
    ]
    assert out.read_text(encoding="utf-8-sig").splitlines()[0].startswith("run_name,method,tracker")


def test_ablation_results_keeps_variant_names(tmp_path):
    summary = tmp_path / "eval" / "motchallenge_summary.csv"
    _write_csv(summary, [
        {"tracker": "Ours-full", "HOTA": "60", "DetA": "61", "AssA": "62", "MOTA": "63", "IDF1": "64", "IDSW": "1", "FP": "2", "FN": "3", "IDTP": "4", "IDFP": "5", "IDFN": "6"},
        {"tracker": "Ours-no-template", "HOTA": "50", "DetA": "51", "AssA": "52", "MOTA": "53", "IDF1": "54", "IDSW": "9", "FP": "8", "FN": "7", "IDTP": "6", "IDFP": "5", "IDFN": "4"},
    ])

    rows = write_ablation_results(
        metric_rows=load_metric_rows(summary),
        output_csv=tmp_path / "ablation_results.csv",
        run_name="ablation_full",
        benchmark_root=tmp_path,
    )

    assert rows[0]["variant"] == "Ours-full"
    assert rows[1]["variant"] == "Ours-no-template"


def test_analysis_text_reports_direction_without_inventing_values():
    rows = [
        {"method": "FFCA-YOLO + OC-SORT", "IDF1": "50", "IDSW": "10", "HOTA": "20", "AssA": "30", "FP": "5", "FN": "100"},
        {"method": "FFCA-YOLO + MS-DC-ELT", "IDF1": "55", "IDSW": "8", "HOTA": "25", "AssA": "35", "FP": "7", "FN": "90"},
    ]
    text = build_analysis_text(rows, focus_name="FFCA-YOLO + MS-DC-ELT", baseline_names=["FFCA-YOLO + OC-SORT"])
    assert "IDF1" in text
    assert "55" in text
    assert "10" in text
    assert "8" in text


def test_path_manifest_lists_existing_outputs(tmp_path):
    mot_file = tmp_path / "trackers" / "ocsort" / "data" / "seq.txt"
    mot_file.parent.mkdir(parents=True)
    mot_file.write_text("", encoding="utf-8")
    rows = write_path_manifest(tmp_path, tmp_path / "manifest.csv")
    assert any(row["kind"] == "mot_result" and row["path"].endswith("seq.txt") for row in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_experiment_summary.py -q
```

Expected: FAIL because `tools/evaluation/msdc_experiment_summary.py` does not exist.

- [ ] **Step 3: Implement summary module**

Create `tools/evaluation/msdc_experiment_summary.py` with these top-level constants and functions:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_FIELDS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "IDTP", "IDFP", "IDFN"]

METHOD_LABELS = {
    "ocsort": "FFCA-YOLO + OC-SORT",
    "botsort": "FFCA-YOLO + BoT-SORT",
    "Ours-full": "FFCA-YOLO + MS-DC-ELT",
    "msdc_elt": "FFCA-YOLO + MS-DC-ELT",
}


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: str | Path, rows: list[dict], fields: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_metric_rows(summary_csv: str | Path) -> list[dict]:
    rows = _read_csv(summary_csv)
    if not rows:
        raise ValueError(f"No metric rows found: {summary_csv}")
    return rows


def _metric_value(row: dict, field: str) -> str:
    value = row.get(field, "")
    return value if value != "" else "N/A"


def _artifact_path(root: Path, tracker: str, suffix: str) -> str:
    matches = sorted(root.glob(f"**/{tracker}/{suffix}"))
    return str(matches[0]) if matches else "N/A"


def write_main_results(metric_rows: list[dict], output_csv: str | Path, run_name: str, benchmark_root: str | Path) -> list[dict]:
    benchmark_root = Path(benchmark_root)
    rows = []
    for metric in metric_rows:
        tracker = metric["tracker"]
        if tracker not in {"ocsort", "botsort", "Ours-full", "msdc_elt"}:
            continue
        method = METHOD_LABELS.get(tracker, tracker)
        row = {
            "run_name": run_name,
            "method": method,
            "tracker": tracker,
            **{field: _metric_value(metric, field) for field in METRIC_FIELDS},
            "mot_result_path": _artifact_path(benchmark_root, tracker, "data/*.txt"),
            "trackeval_output_root": str(benchmark_root / "eval"),
        }
        rows.append(row)
    fields = ["run_name", "method", "tracker", *METRIC_FIELDS, "mot_result_path", "trackeval_output_root"]
    _write_csv(output_csv, rows, fields)
    return rows


def write_ablation_results(metric_rows: list[dict], output_csv: str | Path, run_name: str, benchmark_root: str | Path) -> list[dict]:
    benchmark_root = Path(benchmark_root)
    rows = []
    for metric in metric_rows:
        variant = metric["tracker"]
        row = {
            "run_name": run_name,
            "variant": variant,
            "MSDC_USE_LOW_DET": "False" if variant == "Ours-lite-no-low-det" else "True",
            "MSDC_USE_MOTION": "False" if variant == "Ours-lite-no-motion" else "True",
            "MSDC_USE_TEMPLATE": "False" if variant == "Ours-no-template" else "True",
            "MSDC_USE_REACQUIRE": "False" if variant == "Ours-no-reacquire" else "True",
            "MSDC_REUSE_GUARD_ENABLE": "False" if variant == "Ours-no-removed-guard" else "True",
            **{field: _metric_value(metric, field) for field in METRIC_FIELDS},
            "mot_result_path": _artifact_path(benchmark_root, variant, "data/*.txt"),
            "trackeval_output_root": str(benchmark_root / "eval"),
        }
        rows.append(row)
    fields = [
        "run_name",
        "variant",
        "MSDC_USE_LOW_DET",
        "MSDC_USE_MOTION",
        "MSDC_USE_TEMPLATE",
        "MSDC_USE_REACQUIRE",
        "MSDC_REUSE_GUARD_ENABLE",
        *METRIC_FIELDS,
        "mot_result_path",
        "trackeval_output_root",
    ]
    _write_csv(output_csv, rows, fields)
    return rows


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_analysis_text(rows: list[dict], focus_name: str, baseline_names: list[str]) -> str:
    focus = next((row for row in rows if row.get("method", row.get("variant")) == focus_name), None)
    if focus is None:
        return f"{focus_name} is missing from the real metric table. No conclusion is reported."
    lines = [f"Focus method: {focus_name}."]
    for baseline_name in baseline_names:
        baseline = next((row for row in rows if row.get("method", row.get("variant")) == baseline_name), None)
        if baseline is None:
            lines.append(f"{baseline_name}: missing; comparison is N/A.")
            continue
        for metric in ["IDF1", "IDSW", "HOTA", "AssA", "FP", "FN"]:
            focus_value = focus.get(metric, "N/A")
            base_value = baseline.get(metric, "N/A")
            lines.append(f"{metric} versus {baseline_name}: {focus_value} vs {base_value}.")
    return "\n".join(lines)


def write_path_manifest(root: str | Path, output_csv: str | Path) -> list[dict]:
    root = Path(root)
    patterns = {
        "mot_result": "**/trackers/*/data/*.txt",
        "trackeval_summary": "**/eval/motchallenge_summary.csv",
        "diagnostic_csv": "**/diagnostics/**/*.csv",
        "diagnostic_jsonl": "**/trackers/*/diagnostics/**/*.jsonl",
        "visualization_mp4": "**/visualizations/**/*.mp4",
        "speed_csv": "**/speed_results.csv",
        "metadata_json": "**/run_metadata.json",
    }
    rows = []
    for kind, pattern in patterns.items():
        for path in sorted(root.glob(pattern)):
            rows.append({"kind": kind, "path": str(path)})
    _write_csv(output_csv, rows, ["kind", "path"])
    return rows
```

Add a CLI `main()` that accepts `--main-root`, `--ablation-root`, `--speed-csv`, `--output-root`, `--report-output`, and `--docs-output`, calls the functions above, reads `speed_results.csv` if present, and writes the report markdown. The report must write `N/A` for missing metrics instead of inventing values.

- [ ] **Step 4: Run summary tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_experiment_summary.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/evaluation/msdc_experiment_summary.py test/test_msdc_experiment_summary.py
git commit -m "feat: summarize msdc paper experiment results"
```

---

### Task 4: Add Speed and Complexity Benchmark

**Files:**
- Create: `tools/evaluation/msdc_speed_benchmark.py`
- Create: `test/test_msdc_speed_benchmark.py`

- [ ] **Step 1: Write unit tests for latency statistics and processor counting**

Create `test/test_msdc_speed_benchmark.py`:

```python
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.msdc_speed_benchmark import (
    CountingProcessor,
    percentile,
    summarize_latencies,
)


class DummyProcessor:
    def process_frame(self, frame, file_type, conf_override=None):
        return {"boxes": [], "processing_time": 0.01}


def test_percentile_uses_sorted_nearest_rank():
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 95) == 38.5


def test_summarize_latencies_reports_mean_p50_p95():
    summary = summarize_latencies([0.01, 0.02, 0.03], processed_frames=3)
    assert summary["processed_frames"] == 3
    assert round(summary["mean_latency_ms"], 4) == 20.0
    assert round(summary["p50_latency_ms"], 4) == 20.0
    assert round(summary["p95_latency_ms"], 4) == 29.0


def test_counting_processor_records_calls_by_stage():
    processor = CountingProcessor(DummyProcessor())
    processor.stage = "high_det"
    processor.process_frame(frame=object(), file_type="visible", conf_override=None)
    processor.stage = "low_det"
    processor.process_frame(frame=object(), file_type="visible", conf_override=0.18)
    assert processor.call_counts == {"high_det": 1, "low_det": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_speed_benchmark.py -q
```

Expected: FAIL because `tools/evaluation/msdc_speed_benchmark.py` does not exist.

- [ ] **Step 3: Implement speed benchmark module**

Create `tools/evaluation/msdc_speed_benchmark.py` with these units:

```python
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.file_utils import get_file_type
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from tools.evaluation.export_mot_results import (
    _is_msdc_tracker,
    _run_msdc_high_threshold_detection,
    _run_msdc_low_threshold_detection,
    _update_tracking_for_frame,
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_latencies(latencies: list[float], processed_frames: int) -> dict:
    total = float(sum(latencies))
    mean = statistics.mean(latencies) if latencies else 0.0
    fps = processed_frames / total if total > 0 else 0.0
    return {
        "processed_frames": int(processed_frames),
        "total_time_s": round(total, 6),
        "mean_fps": round(fps, 6),
        "mean_latency_ms": round(mean * 1000.0, 6),
        "p50_latency_ms": round(percentile(latencies, 50) * 1000.0, 6),
        "p95_latency_ms": round(percentile(latencies, 95) * 1000.0, 6),
    }


class CountingProcessor:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.stage = "unknown"
        self.call_counts: dict[str, int] = {}
        self.stage_seconds: dict[str, float] = {}

    @contextmanager
    def use_stage(self, stage: str):
        previous = self.stage
        self.stage = stage
        try:
            yield
        finally:
            self.stage = previous

    def process_frame(self, frame, file_type, conf_override=None):
        start = time.perf_counter()
        result = self.wrapped.process_frame(frame, file_type, conf_override=conf_override)
        elapsed = time.perf_counter() - start
        self.call_counts[self.stage] = self.call_counts.get(self.stage, 0) + 1
        self.stage_seconds[self.stage] = self.stage_seconds.get(self.stage, 0.0) + elapsed
        return result

    def __getattr__(self, name):
        return getattr(self.wrapped, name)
```

In the benchmark loop:

- Set `Config.MSDC_DEBUG_EVENTS = False`.
- For baseline trackers, call `detector.processor.process_frame(frame, file_type)` once per frame and then `MultiObjectTracker.update(...)`.
- For `msdc_elt`, call `_run_msdc_high_threshold_detection(...)`, `_run_msdc_low_threshold_detection(...)`, and `_update_tracking_for_frame(...)`.
- Wrap detection calls in `CountingProcessor.use_stage("high_det")` and `CountingProcessor.use_stage("low_det")`.
- Wrap the tracker update call with `CountingProcessor.use_stage("tracker_update")`; any ROI redetect detector calls inside tracker update are then counted under `tracker_update`.
- Write per-frame timing JSONL with fields `frame_id`, `tracker`, `read_s`, `high_det_s`, `low_det_s`, `tracker_s`, and `total_s`.
- Write `speed_results.csv` with fields:

```python
[
    "run_id",
    "commit_hash",
    "video_path",
    "seq_name",
    "tracker",
    "method",
    "resolution",
    "requested_frames",
    "processed_frames",
    "total_time_s",
    "mean_fps",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "peak_memory_mb",
    "detector_calls_total",
    "detector_calls_high_det",
    "detector_calls_low_det",
    "detector_calls_tracker_update",
    "mean_read_ms",
    "mean_high_det_ms",
    "mean_low_det_ms",
    "mean_tracker_ms",
]
```

For peak memory, use this helper:

```python
def peak_memory_mb() -> str:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"{usage / 1024.0:.3f}"
    except Exception:
        return "N/A"
```

- [ ] **Step 4: Run speed unit tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_speed_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/evaluation/msdc_speed_benchmark.py test/test_msdc_speed_benchmark.py
git commit -m "feat: add msdc speed benchmark"
```

---

### Task 5: Add Paper Experiment Runner

**Files:**
- Create: `tools/evaluation/run_msdc_paper_experiments.py`
- Create: `test/test_msdc_paper_experiment_runner.py`

- [ ] **Step 1: Write tests for command planning**

Create `test/test_msdc_paper_experiment_runner.py`:

```python
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py -q
```

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement runner command builders**

Create `tools/evaluation/run_msdc_paper_experiments.py` with:

```python
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

MAIN_TRACKERS = ["ocsort", "botsort", "msdc_elt"]
ABLATION_VARIANTS = [
    "Ours-full",
    "Ours-lite-no-motion",
    "Ours-lite-no-low-det",
    "Ours-no-template",
    "Ours-no-reacquire",
    "Ours-no-removed-guard",
]


def build_main_command(dataset_roots: list[str], output_root: Path, run_id: str, commit_hash: str, progress_interval: int, run: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "msdc_dataset_benchmark.py"),
        "--dataset-root",
        *dataset_roots,
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--mode",
        "main",
        "--commit-hash",
        commit_hash,
        "--trackers",
        *MAIN_TRACKERS,
        "--variants",
        "Ours-full",
        "--progress-interval",
        str(progress_interval),
        "--render",
    ]
    if run:
        cmd.append("--run")
    return cmd


def build_ablation_command(dataset_roots: list[str], output_root: Path, run_id: str, commit_hash: str, progress_interval: int, run: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "msdc_dataset_benchmark.py"),
        "--dataset-root",
        *dataset_roots,
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--mode",
        "ablation",
        "--commit-hash",
        commit_hash,
        "--trackers",
        "msdc_elt",
        "--variants",
        *ABLATION_VARIANTS,
        "--progress-interval",
        str(progress_interval),
        "--render",
    ]
    if run:
        cmd.append("--run")
    return cmd


def build_speed_command(dataset_root: str, output_root: Path, run_id: str, commit_hash: str, frames: int, progress_interval: int) -> list[str]:
    return [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "msdc_speed_benchmark.py"),
        "--dataset-root",
        dataset_root,
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--commit-hash",
        commit_hash,
        "--frames",
        str(frames),
        "--trackers",
        "ocsort",
        "botsort",
        "msdc_elt",
        "--progress-interval",
        str(progress_interval),
    ]
```

The CLI must:

- Default `--dataset-root` to the two required paths.
- Default `--output-root` to `results/msdc_paper_phase1`.
- Default `--speed-frames` to `1000`.
- Require `--run-formal` before running formal main/ablation/speed commands.
- Support `--smoke` to run a separate `smoke` benchmark with `--max-frames 100`; the smoke command must write under `results/msdc_paper_phase1/<run-id>/smoke`.
- After formal commands finish, call `msdc_experiment_summary.py` to write summary CSVs and reports.
- Write `results/msdc_paper_phase1/latest_run.json` after a formal run with fields `run_root`, `main_results_csv`, `ablation_results_csv`, `speed_results_csv`, `report_path`, `docs_result_path`, `main_root`, `ablation_root`, and `speed_root`.
- Support `--print-latest` to print those paths without starting a run.
- Support `--check-latest` to read `latest_run.json`, verify every recorded path exists, and exit nonzero with a clear message if any path is missing.

- [ ] **Step 4: Run runner tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/evaluation/run_msdc_paper_experiments.py test/test_msdc_paper_experiment_runner.py
git commit -m "feat: orchestrate msdc paper experiments"
```

---

### Task 6: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `tools/README.md`
- Create: `docs/MSDC_EXPERIMENT_RESULT.md`

- [ ] **Step 1: Add README section for Phase 1 paper experiments**

In `README.md`, add a subsection under MS-DC-ELT evaluation:

```markdown
### 10.23 MS-DC-ELT paper phase-1 experiments

`tools/evaluation/run_msdc_paper_experiments.py` orchestrates the paper phase-1 experiment bundle: main comparison, ablation, speed/complexity, and report generation.

Formal main and ablation runs process full videos. Do not pass `--max-frames` or `--duration-seconds` for formal results. Smoke runs are only workflow checks and are written under a separate `smoke/` directory.

Example smoke:

```powershell
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --smoke --progress-interval 50
```

Example formal run:

```powershell
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --run-formal --speed-frames 1000 --progress-interval 500
```

Default dataset roots:

- `/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集`
- `/home/hyj/Anti_Drone_Project/USV_MOT标注数据集`

Primary outputs:

- `results/msdc_paper_phase1/<run-id>/summary/main_results.csv`
- `results/msdc_paper_phase1/<run-id>/summary/ablation_results.csv`
- `results/msdc_paper_phase1/<run-id>/summary/speed_results.csv`
- `results/msdc_paper_phase1/<run-id>/MSDC_EXPERIMENT_REPORT.md`
- `docs/MSDC_EXPERIMENT_RESULT.md`
```

- [ ] **Step 2: Update tools README**

In `tools/README.md`, add bullets:

```markdown
- `msdc_speed_benchmark.py`：在不渲染、不发 MQ、不跑 TrackEval 的条件下，对 OC-SORT、BoT-SORT、MS-DC-ELT 做固定帧数速度统计。
- `msdc_experiment_summary.py`：把 TrackEval summary、速度结果和输出路径汇总为论文实验 CSV 与 Markdown 报告。
- `run_msdc_paper_experiments.py`：编排 MS-DC-ELT 论文第一阶段主结果、消融、速度与报告生成。
```

- [ ] **Step 3: Create initial result doc without invented metrics**

Create `docs/MSDC_EXPERIMENT_RESULT.md` with:

```markdown
# MS-DC-ELT Experiment Result

No formal phase-1 experiment has been run for this document yet.

Rows below must only be filled by `tools/evaluation/msdc_experiment_summary.py` after a real run. Missing or unavailable metrics must remain `N/A`.

## Current Formal Run

- Run ID: N/A
- Commit hash: N/A
- Dataset roots: N/A
- Output root: N/A

## Main Results

N/A

## Ablation Results

N/A

## Speed and Complexity

N/A

## Failures and Missing Metrics

N/A
```

- [ ] **Step 4: Commit**

```bash
git add README.md tools/README.md docs/MSDC_EXPERIMENT_RESULT.md
git commit -m "docs: document msdc paper experiment workflow"
```

---

### Task 7: Run Unit Test Verification

**Files:**
- No file changes.

- [ ] **Step 1: Run focused tests**

Run:

```bash
conda run -n ship_detect pytest test/test_motchallenge_eval.py test/test_msdc_dataset_benchmark.py test/test_msdc_experiment_summary.py test/test_msdc_speed_benchmark.py test/test_msdc_paper_experiment_runner.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run existing MS-DC-ELT tests**

Run:

```bash
conda run -n ship_detect pytest test/test_mot_result_export.py test/test_msdc_dataset_benchmark.py test/test_msdc_evidence_state.py test/test_msdc_lifecycle_tracker.py test/test_msdc_low_conf_debug.py test/test_msdc_motion_seed.py test/test_msdc_roi_redetect.py test/test_msdc_template_lock.py test/test_msdc_types.py test/test_render_mot_video.py test/test_video_main_msdc.py -q
```

Expected: all tests pass.

---

### Task 8: Run Smoke Workflow Check

**Files:**
- Runtime outputs only under `results/msdc_paper_phase1/<run-id>/smoke/`.

- [ ] **Step 1: Confirm datasets and referenced videos are accessible**

Run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_dataset_benchmark.py --dataset-root "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集" "/home/hyj/Anti_Drone_Project/USV_MOT标注数据集" --trackers ocsort botsort msdc_elt --variants Ours-full --max-frames 5
```

Expected: print-only output lists both datasets, resolved full video paths, and commands. If a video path is missing, stop and record the failure; do not start formal runs.

- [ ] **Step 2: Run smoke**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --smoke --progress-interval 50
```

Expected:

- A timestamped smoke run directory is created.
- MOT export, TrackEval, diagnostics, and visualizations run on 100 frames only.
- Smoke output is clearly under `smoke/`.
- The report does not treat smoke metrics as formal results.

---

### Task 9: Run Formal Phase-1 Experiment Bundle

**Files:**
- Create/update: `MSDC_EXPERIMENT_REPORT.md`
- Create/update: `docs/MSDC_EXPERIMENT_RESULT.md`
- Runtime outputs under `results/msdc_paper_phase1/<run-id>/`

- [ ] **Step 1: Run full-video main, full-video ablation, fixed-frame speed, and report generation**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --run-formal --speed-frames 1000 --progress-interval 500
```

Expected:

- Main comparison runs on full videos with trackers `ocsort`, `botsort`, and `Ours-full`.
- Ablation runs on full videos with variants `Ours-full`, `Ours-lite-no-motion`, `Ours-lite-no-low-det`, `Ours-no-template`, `Ours-no-reacquire`, and `Ours-no-removed-guard`.
- Speed comparison uses exactly 1000 requested frames from one resolved full video, or records fewer processed frames if the video is shorter.
- No formal main or ablation command passes `--max-frames` or `--duration-seconds`.
- Every formal main and ablation tracker/variant has MOT txt, TrackEval output, diagnostics, and visualization MP4 paths.
- `results/msdc_paper_phase1/latest_run.json` points to the timestamped formal run directory and summary files.
- `MSDC_EXPERIMENT_REPORT.md` and `docs/MSDC_EXPERIMENT_RESULT.md` are updated from real outputs only.

---

### Task 10: Validate Formal Output Paths

**Files:**
- No planned code changes.

- [ ] **Step 1: Check latest formal artifact paths**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --check-latest
```

Expected:

- The command exits 0.
- It verifies `main_results.csv`, `ablation_results.csv`, `speed_results.csv`, `MSDC_EXPERIMENT_REPORT.md`, and `docs/MSDC_EXPERIMENT_RESULT.md`.
- It verifies the timestamped run root still exists.

- [ ] **Step 2: Print latest paths for the final response**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --print-latest
```

Expected: output includes absolute or repository-relative paths for `main_results_csv`, `ablation_results_csv`, `speed_results_csv`, and `report_path`.

---

### Task 11: Review Formal Metrics Before Reporting

**Files:**
- No planned code changes.

- [ ] **Step 1: Read the generated report and CSVs**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --print-latest
```

Then open the printed CSV/report paths with `sed -n '1,220p'` followed by the path printed by `--print-latest`, or use a direct file read in the current environment.

Expected:

- `main_results.csv` includes HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN, IDTP, IDFP, IDFN for OC-SORT, BoT-SORT, and MS-DC-ELT; DetA/AssA may be `N/A` only if TrackEval did not expose them.
- `ablation_results.csv` includes every required ablation variant.
- `speed_results.csv` includes requested frames, processed frames, FPS, latency, peak memory, detector call count, and stage timing fields.
- The effectiveness conclusion in `MSDC_EXPERIMENT_REPORT.md` matches the actual direction of IDF1, IDSW, HOTA/AssA, FP, and FN.

---

### Task 12: Commit Reports

- [ ] **Step 1: Commit reports**

Commit only scripts, docs, and report markdown/CSV summaries that are intended for the repository. Do not commit heavy `results/` videos or MOT outputs unless the user explicitly asks.

```bash
git add MSDC_EXPERIMENT_REPORT.md docs/MSDC_EXPERIMENT_RESULT.md README.md tools/README.md
git commit -m "docs: add msdc phase one experiment report"
```

---

### Task 13: Final Verification and Status

**Files:**
- No planned code changes.

- [ ] **Step 1: Verify no algorithm files were modified**

Run:

```bash
git diff --name-only origin/MSDC..HEAD
```

Expected modified paths are limited to:

- `README.md`
- `tools/README.md`
- `tools/evaluation/*`
- `test/test_msdc_*`
- `test/test_motchallenge_eval.py`
- `MSDC_EXPERIMENT_REPORT.md`
- `docs/MSDC_EXPERIMENT_RESULT.md`

If any core algorithm path appears, stop and inspect before pushing.

- [ ] **Step 2: Verify tests**

Run:

```bash
conda run -n ship_detect pytest test/test_motchallenge_eval.py test/test_msdc_dataset_benchmark.py test/test_msdc_experiment_summary.py test/test_msdc_speed_benchmark.py test/test_msdc_paper_experiment_runner.py test/test_mot_result_export.py test/test_msdc_evidence_state.py test/test_msdc_lifecycle_tracker.py test/test_msdc_low_conf_debug.py test/test_msdc_motion_seed.py test/test_msdc_roi_redetect.py test/test_msdc_template_lock.py test/test_msdc_types.py test/test_render_mot_video.py test/test_video_main_msdc.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Verify required final artifacts exist**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --check-latest
```

Expected: the command exits 0 and prints that every recorded latest-run artifact exists.

- [ ] **Step 4: Prepare final user response**

The final response must include:

- Completed experiments.
- Failed experiments and exact failure reasons.
- Path to `main_results.csv`.
- Path to `ablation_results.csv`.
- Path to `speed_results.csv`.
- Path to `MSDC_EXPERIMENT_REPORT.md`.
- Whether current real results preliminarily support MS-DC-ELT effectiveness.

The effectiveness statement must be based only on real `IDF1`, `IDSW`, `HOTA`, `AssA`, `FP`, and `FN` values in the generated CSV files.

---

## Self-Review

- Spec coverage:
  - Main comparison is handled by Task 9 and summarized by Task 12.
  - Ablation is handled by Task 10 and summarized by Task 12.
  - Speed/complexity is handled by Task 11 and summarized by Task 12.
  - Metadata, commit hash, dataset paths, output paths, failures, and run time are handled by Tasks 2, 4, 5, and 12.
  - Smoke isolation is handled by Tasks 5 and 8.
  - Full-video formal requirements are enforced by Tasks 9 and 10.
  - No core algorithm modifications are enforced by Scope Rules and Task 13.
- Placeholder scan:
  - No invented metric values are present.
  - `N/A` is reserved for unavailable metrics or not-yet-run report fields.
  - Dynamic formal output paths are resolved through `results/msdc_paper_phase1/latest_run.json`.
- Type consistency:
  - Metrics use CSV row dictionaries with string values.
  - Run metadata uses JSON-serializable dictionaries.
  - Speed summaries use seconds internally and milliseconds in CSV fields.
  - Tracker labels match current script choices: `ocsort`, `botsort`, `msdc_elt`, and ablation aliases.
