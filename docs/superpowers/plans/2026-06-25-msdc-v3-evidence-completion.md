# MS-DC-ELT v3 Evidence Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the formal evaluation evidence gaps after `results/msdc_paper_phase1/20260624_190710` without re-running detector inference: prove effective ablation configs, produce sensitivity metrics, produce slice-level metrics, and add low-candidate/reacquire/inherit opportunity diagnostics from cached high/low detections and existing MOT artifacts.

**Architecture:** Keep the existing replay-detection evaluation stack as the single source of formal metrics. Add explicit per-variant effective config export, allow sensitivity variants to execute as real replay runs from a source detection-cache root, extend slice evaluation from manifest-only to metric-producing clipped MOT evaluation, and extend diagnostics with denominators/opportunities instead of fabricating success events. The supplemental run must never call the detector; it must reuse `results/msdc_paper_phase1/20260624_190710/main/main_full/detections/*.jsonl` or equivalent symlinked copies.

**Tech Stack:** Python, pytest, TrackEval, existing `conda run -n ship_detect ...` workflow, existing `tools/evaluation/detection_replay_benchmark.py` and `tools/evaluation/msdc_experiment_summary.py`.

---

## Root Cause Summary

1. `sensitivity/sensitivity_full/sensitivity_matrix.csv` has only 16 config rows because `tools/evaluation/msdc_sensitivity_matrix.py` only writes parameter grids. `run_msdc_paper_experiments.py` calls that matrix writer, but never executes TrackEval for those variants.
2. `slice/slice_manifest.csv` has only 143 short-miss segments because `tools/evaluation/msdc_slice_eval.py` only extracts segments from per-GT diagnostics. It does not create clipped GT/pred files and does not run TrackEval or compute IDF1/IDSW/FN/FP.
3. `low_candidate_recall` is `N/A` because `tools/evaluation/msdc_diagnostic_metrics.py` hard-codes `"low_candidate_recall": "N/A"`; there is no denominator definition yet.
4. `reacquire_success = 0` and `inherit_* = 0` are not summary bugs by themselves. Current lifecycle event logs did not contain success events for full v3. The missing piece is opportunity diagnostics: how many LOST/reacquire or LOW_CANDIDATE/inherit opportunities existed, how many were attempted, rejected, or matched.
5. The ablation config columns are wrong because `tools/evaluation/msdc_experiment_summary.py` reads TrackEval rows named like `no_low_candidate_replay`, but `_ablation_switches()` checks keys like `no_low_candidate`. The suffix is not stripped, so it falls back to default full-v3-looking values.
6. The ablation results are not all identical. MOT file hashes show `no_low_candidate`, `no_direct_reacquire`, `hits_only_no_evidence`, and `no_output_nms` differ from full v3. `no_low_inheritance`, `low_budget_*`, and `no_output_real_det_age_gate` match full v3 in this run, most likely because inheritance/output-age/budget gates were not active or not binding on these sequences. This still needs explicit effective-config and event-count evidence.

## Cache-Only Constraint

Do not run the full formal experiment again. Do not re-run detector inference. Reuse these already-generated high/low detection caches:

```text
results/msdc_paper_phase1/20260624_190710/main/main_full/detections/DJI_20250711140128_0002_V_high_low_detections.jsonl
results/msdc_paper_phase1/20260624_190710/main/main_full/detections/DJI_20250916100639_0001_V_high_low_detections.jsonl
```

The same cache files also exist under `ablation/ablation_full/detections/` with identical sizes. Prefer the `main/main_full/detections/` root as the canonical source.

All supplemental outputs should be written to a new directory, for example:

```text
results/msdc_paper_phase1/20260624_190710_supplement_cache_only
```

This directory should contain only newly computed supplemental evidence and copied/symlinked summaries; it must not overwrite `20260624_190710`.

## File Structure

- Modify `tools/evaluation/msdc_experiment_summary.py`: strip `_replay` suffix when mapping ablation rows to config deltas; include `variant_key`.
- Modify `tools/evaluation/detection_replay_benchmark.py`: write `effective_config/<tracker>.json` for every MS-DC variant; optionally load dynamic variant env files for sensitivity runs; accept `--source-detection-cache-root` and skip detector dumping when source cache is available.
- Create `tools/evaluation/cache_reuse.py`: locate and symlink/copy existing high/low detection JSONL files into a supplemental replay run root.
- Modify `tools/evaluation/msdc_sensitivity_matrix.py`: keep matrix generation, expose rows for execution.
- Create `tools/evaluation/run_msdc_sensitivity_benchmark.py`: execute each sensitivity point as replay + TrackEval and write metric CSV.
- Modify `tools/evaluation/run_msdc_paper_experiments.py`: add cache-only ablation command helpers and call the real sensitivity benchmark after ablation.
- Create `tools/evaluation/msdc_slice_metrics.py`: build clipped slice GT/pred sets and compute slice metrics.
- Modify `tools/evaluation/msdc_slice_eval.py`: keep manifest generation, optionally invoke slice metrics.
- Modify `tools/evaluation/msdc_diagnostic_metrics.py`: add low-candidate recall denominator and opportunity counts for reacquire/inherit.
- Modify `tools/evaluation/validate_msdc_formal_run.py`: require sensitivity metrics, slice metrics, effective config JSON, and extended diagnostic fields.
- Create `tools/evaluation/validate_msdc_supplemental_run.py`: validate a cache-only supplemental root against the original base run without requiring detector rerun, speed rerun, or re-rendered main videos.
- Modify `tools/evaluation/msdc_experiment_summary.py`: include new output paths and new tables in the report.
- Modify `README.md` and `tools/README.md`: document the corrected formal evidence package.
- Add tests under `test/`: summary variant mapping, sensitivity execution plan, slice metrics, diagnostic denominator/opportunity counts, validator requirements.

## Task 0: Add Detection Cache Reuse Contract

**Files:**
- Create: `tools/evaluation/cache_reuse.py`
- Modify: `tools/evaluation/detection_replay_benchmark.py`
- Test: `test/test_cache_reuse.py`
- Test: `test/test_detection_replay_benchmark.py`

- [ ] **Step 1: Write cache reuse tests**

Create `test/test_cache_reuse.py`:

```python
from tools.evaluation.cache_reuse import link_detection_cache_for_sequence


def test_link_detection_cache_for_sequence_symlinks_existing_cache(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    cache = source / "seq_a_high_low_detections.jsonl"
    cache.write_text('{"frame_id": 1, "high_boxes": [], "low_boxes": []}\n', encoding="utf-8")

    linked = link_detection_cache_for_sequence(source, target, "seq_a")

    assert linked == target / "seq_a_high_low_detections.jsonl"
    assert linked.is_file()
    assert linked.read_text(encoding="utf-8").startswith('{"frame_id"')
```

- [ ] **Step 2: Run failing cache reuse test**

Run:

```bash
conda run -n ship_detect pytest test/test_cache_reuse.py -q
```

Expected before implementation: FAIL because `tools.evaluation.cache_reuse` does not exist.

- [ ] **Step 3: Create cache reuse helper**

Create `tools/evaluation/cache_reuse.py`:

```python
"""Helpers for reusing high/low detection caches in supplemental evaluations."""

from __future__ import annotations

import shutil
from pathlib import Path


def detection_cache_name(seq_name: str) -> str:
    return f"{seq_name}_high_low_detections.jsonl"


def link_detection_cache_for_sequence(
    source_cache_root: str | Path,
    target_cache_root: str | Path,
    seq_name: str,
    *,
    symlink: bool = True,
) -> Path:
    source = Path(source_cache_root) / detection_cache_name(seq_name)
    if not source.is_file():
        raise FileNotFoundError(f"Missing source detection cache: {source}")
    target_root = Path(target_cache_root)
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / source.name
    if target.exists() or target.is_symlink():
        return target
    if symlink:
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)
    return target
```

- [ ] **Step 4: Add source cache argument to replay benchmark**

In `tools/evaluation/detection_replay_benchmark.py`, import:

```python
from tools.evaluation.cache_reuse import link_detection_cache_for_sequence
```

Add CLI argument:

```python
parser.add_argument(
    "--source-detection-cache-root",
    default="",
    help="Optional existing high/low detection cache root. When set, symlink caches and do not run detector dumping.",
)
```

Inside `run_detection_replay_benchmark()`, before `is_detection_cache_complete(...)`, add:

```python
source_cache_root = Path(str(getattr(args, "source_detection_cache_root", "") or ""))
if str(source_cache_root):
    cache_path = link_detection_cache_for_sequence(source_cache_root, detections_root, spec.seq_name)
```

Then keep the existing completeness check. If the linked cache is complete, the current code prints `[SKIP] detection cache complete` and never calls `dump_video_detections()`.

- [ ] **Step 5: Add no-detector guard for cache-only runs**

In `run_detection_replay_benchmark()`, immediately before `dump_video_detections(...)`, add:

```python
if str(source_cache_root):
    raise RuntimeError(f"Source detection cache was requested but cache is incomplete: {cache_path}")
```

This prevents accidental detector calls in supplemental runs.

- [ ] **Step 6: Run tests**

Run:

```bash
conda run -n ship_detect pytest test/test_cache_reuse.py test/test_detection_replay_benchmark.py -q
```

Expected: PASS.

## Task 1: Fix Ablation Config Reporting and Export Effective Runtime Config

**Files:**
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Modify: `tools/evaluation/detection_replay_benchmark.py`
- Test: `test/test_msdc_experiment_summary.py`
- Test: `test/test_detection_replay_benchmark.py`

- [ ] **Step 1: Write failing summary mapping test**

Append to `test/test_msdc_experiment_summary.py`:

```python
from tools.evaluation.msdc_experiment_summary import _ablation_switches


def test_ablation_switches_strip_replay_suffix_for_formal_variants():
    no_low = _ablation_switches("no_low_candidate_replay")
    assert no_low["MSDC_LOW_CANDIDATE_ENABLE"] == "0"

    no_reacquire = _ablation_switches("no_direct_reacquire_replay")
    assert no_reacquire["MSDC_USE_REACQUIRE"] == "0"

    hits_only = _ablation_switches("hits_only_no_evidence_replay")
    assert hits_only["MSDC_EVIDENCE_MODE"] == "hits_only"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_experiment_summary.py::test_ablation_switches_strip_replay_suffix_for_formal_variants -q
```

Expected before fix: FAIL because `_ablation_switches("no_low_candidate_replay")` falls through to defaults.

- [ ] **Step 3: Normalize replay tracker names**

In `tools/evaluation/msdc_experiment_summary.py`, add:

```python
def _variant_key_from_tracker(tracker: str) -> str:
    if tracker.endswith("_replay"):
        return tracker[: -len("_replay")]
    return tracker
```

Change `_ablation_switches()`:

```python
def _ablation_switches(variant: str) -> dict[str, str]:
    variant_key = _variant_key_from_tracker(variant)
    switches = {field: "N/A" for field in ABLATION_FIELDS}
    if variant_key in FORMAL_ABLATION_VARIANTS:
        switches.update({
            "MSDC_REUSE_GUARD_ENABLE": "True",
            "MSDC_EVIDENCE_MODE": "score",
        })
        switches.update({key: str(value) for key, value in FORMAL_ABLATION_VARIANTS[variant_key].items()})
        return switches

    switches.update(ABLATION_SWITCH_DEFAULTS)
    switches.update(ABLATION_SWITCH_OVERRIDES.get(variant_key, {}))
    return switches
```

In `write_ablation_results()`, include `variant_key`:

```python
"variant_key": _variant_key_from_tracker(variant),
```

and add `"variant_key"` after `"variant"` in `fields`.

- [ ] **Step 4: Export effective runtime config per MS-DC variant**

In `tools/evaluation/detection_replay_benchmark.py`, add:

```python
def write_effective_msdc_config(output_root: str | Path, tracker_name: str, config_snapshot: dict[str, object]) -> Path:
    output_path = Path(output_root) / "effective_config" / f"{tracker_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
```

Inside `replay_tracker_from_cache()`, immediately after `MSDCLifecycleTracker(...)` is constructed for `tracker_type == "msdc_elt"`, call:

```python
write_effective_msdc_config(Path(output_root).parent, tracker_name, _snapshot_msdc_config())
```

- [ ] **Step 5: Write effective config test**

Append to `test/test_detection_replay_benchmark.py`:

```python
import json

from tools.evaluation.detection_replay_benchmark import write_effective_msdc_config


def test_write_effective_msdc_config(tmp_path):
    output = write_effective_msdc_config(
        tmp_path,
        "no_low_candidate_replay",
        {"MSDC_LOW_CANDIDATE_ENABLE": False, "MSDC_USE_REACQUIRE": True},
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["MSDC_LOW_CANDIDATE_ENABLE"] is False
    assert output.name == "no_low_candidate_replay.json"
```

- [ ] **Step 6: Run tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_experiment_summary.py::test_ablation_switches_strip_replay_suffix_for_formal_variants \
  test/test_detection_replay_benchmark.py::test_write_effective_msdc_config \
  -q
```

Expected after fix: PASS.

## Task 1A: Recompute Ablation Metrics From Cached Detections

**Files:**
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Modify: `tools/evaluation/detection_replay_benchmark.py`
- Test: `test/test_msdc_paper_experiment_runner.py`
- Test: `test/test_detection_replay_benchmark.py`

This task is required because Task 1 only fixes reporting and exports effective config. It does not recalculate MOT txt or TrackEval metrics. To answer “correct configuration, correct metrics”, this task must replay every ablation variant from the existing high/low detection cache and rerun TrackEval.

- [ ] **Step 1: Add a cache-only ablation command builder test**

Append to `test/test_msdc_paper_experiment_runner.py`:

```python
from pathlib import Path

from tools.evaluation.run_msdc_paper_experiments import build_ablation_cache_only_command


def test_build_ablation_cache_only_command_reuses_source_detection_cache(tmp_path):
    cmd = build_ablation_cache_only_command(
        dataset_roots=["/data/uav", "/data/usv"],
        output_root=tmp_path / "ablation",
        run_id="ablation_cache_full",
        source_detection_cache_root=Path("results/msdc_paper_phase1/20260624_190710/main/main_full/detections"),
        formal_frame_limit=5400,
        progress_interval=500,
    )
    assert "tools/evaluation/detection_replay_benchmark.py" in cmd[1]
    assert "--source-detection-cache-root" in cmd
    assert cmd[cmd.index("--source-detection-cache-root") + 1].endswith("main/main_full/detections")
    assert "--trackers" in cmd
    assert cmd[cmd.index("--trackers") + 1] == "msdc_elt"
    assert "--variants" in cmd
    variant_args = cmd[cmd.index("--variants") + 1:cmd.index("--formal-frame-limit")]
    assert "msdc_v3" in variant_args
    assert "no_low_candidate" in variant_args
    assert "no_direct_reacquire" in variant_args
    assert "hits_only_no_evidence" in variant_args
```

- [ ] **Step 2: Run failing command builder test**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py::test_build_ablation_cache_only_command_reuses_source_detection_cache -q
```

Expected before implementation: FAIL because `build_ablation_cache_only_command` does not exist.

- [ ] **Step 3: Add the cache-only ablation command builder**

In `tools/evaluation/run_msdc_paper_experiments.py`, add:

```python
def build_ablation_cache_only_command(
    dataset_roots: list[str],
    output_root: Path,
    run_id: str,
    source_detection_cache_root: Path,
    formal_frame_limit: int,
    progress_interval: int,
) -> list[str]:
    return [
        sys.executable,
        _detection_replay_script(),
        "--dataset-root",
        *[str(path) for path in dataset_roots],
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--trackers",
        "msdc_elt",
        "--variants",
        *ABLATION_VARIANTS,
        "--formal-frame-limit",
        str(int(formal_frame_limit)),
        "--source-detection-cache-root",
        str(source_detection_cache_root),
        "--progress-interval",
        str(int(progress_interval)),
        "--render-class-source",
        "cache",
    ]
```

Do not include `--render` by default for the supplemental ablation replay; the goal is metric completion, not re-rendering all videos.

- [ ] **Step 4: Ensure cache-only replay never calls detector**

After Task 0, `detection_replay_benchmark.py` must behave as follows:

```text
--source-detection-cache-root set
  -> symlink/copy <seq>_high_low_detections.jsonl into this run's detections/
  -> if complete: print [SKIP] detection cache complete
  -> if incomplete: raise RuntimeError
  -> never call dump_video_detections()
```

Add or update a test in `test/test_detection_replay_benchmark.py` that monkeypatches `dump_video_detections` to raise and verifies a cache-complete run does not call it.

- [ ] **Step 5: Expected cache-only ablation outputs**

After execution, the supplemental root must contain:

```text
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/msdc_v3_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/no_low_candidate_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/no_direct_reacquire_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/no_low_inheritance_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/hits_only_no_evidence_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/no_output_nms_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/no_output_real_det_age_gate_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/low_budget_off_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/low_budget_topk16_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/trackers/low_budget_topk64_replay/data/*.txt
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/eval/motchallenge_summary.csv
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/summary/replay_summary.csv
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/effective_config/*_replay.json
```

- [ ] **Step 6: Run tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_paper_experiment_runner.py::test_build_ablation_cache_only_command_reuses_source_detection_cache \
  test/test_detection_replay_benchmark.py \
  -q
```

Expected after implementation: PASS.

## Task 2: Produce Real Hyperparameter Sensitivity Metrics

**Files:**
- Create: `tools/evaluation/run_msdc_sensitivity_benchmark.py`
- Modify: `tools/evaluation/detection_replay_benchmark.py`
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Test: `test/test_msdc_sensitivity_benchmark.py`

- [ ] **Step 1: Write sensitivity runner tests**

Create `test/test_msdc_sensitivity_benchmark.py`:

```python
from tools.evaluation.run_msdc_sensitivity_benchmark import (
    build_dynamic_variant_envs,
    sensitivity_tracker_name,
)


def test_sensitivity_tracker_name_is_replay_safe():
    assert sensitivity_tracker_name("msdc_confirm_score_2p0") == "msdc_confirm_score_2p0_replay"


def test_build_dynamic_variant_envs_parses_matrix_rows():
    rows = [
        {"variant": "msdc_v3", "parameter": "baseline", "value": "baseline", "env_json": ""},
        {"variant": "msdc_confirm_score_2p0", "parameter": "MSDC_CONFIRM_SCORE", "value": "2.0", "env_json": '{"MSDC_CONFIRM_SCORE": "2.0"}'},
    ]
    envs = build_dynamic_variant_envs(rows)
    assert envs["msdc_confirm_score_2p0"]["MSDC_CONFIRM_SCORE"] == "2.0"
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_sensitivity_benchmark.py -q
```

Expected before implementation: FAIL because module does not exist.

- [ ] **Step 3: Create sensitivity benchmark runner**

Create `tools/evaluation/run_msdc_sensitivity_benchmark.py` with:

```python
"""Run real replay + TrackEval metrics for MS-DC sensitivity variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.detection_replay_benchmark import run_dynamic_msdc_replay_benchmark


def sensitivity_tracker_name(variant: str) -> str:
    return f"{variant}_replay"


def read_matrix(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_dynamic_variant_envs(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    envs: dict[str, dict[str, str]] = {}
    for row in rows:
        variant = row["variant"]
        if row.get("parameter") == "baseline":
            envs[variant] = {}
            continue
        env_json = row.get("env_json") or "{}"
        envs[variant] = {key: str(value) for key, value in json.loads(env_json).items()}
    return envs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run MS-DC sensitivity replay metrics")
    parser.add_argument("--dataset-root", nargs="+", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default="sensitivity_metrics")
    parser.add_argument("--formal-frame-limit", type=int, default=5400)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--render-class-source", choices=["cache", "detector", "none"], default="cache")
    parser.add_argument("--source-detection-cache-root", required=True)
    args = parser.parse_args(argv)

    rows = read_matrix(args.matrix)
    envs = build_dynamic_variant_envs(rows)
    summary_path = run_dynamic_msdc_replay_benchmark(
        dataset_roots=list(args.dataset_root),
        output_root=Path(args.output_root),
        run_id=str(args.run_id),
        variant_envs=envs,
        formal_frame_limit=int(args.formal_frame_limit),
        progress_interval=int(args.progress_interval),
        render=False,
        render_class_source=str(args.render_class_source),
        source_detection_cache_root=str(args.source_detection_cache_root),
    )
    print(f"[OK] sensitivity metrics: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add dynamic replay support**

In `tools/evaluation/detection_replay_benchmark.py`, add:

```python
def run_dynamic_msdc_replay_benchmark(
    *,
    dataset_roots: list[str],
    output_root: Path,
    run_id: str,
    variant_envs: dict[str, dict[str, str]],
    formal_frame_limit: int,
    progress_interval: int,
    render: bool,
    render_class_source: str,
    source_detection_cache_root: str,
) -> Path:
    args = argparse.Namespace(
        dataset_root=dataset_roots,
        output_root=str(output_root),
        run_id=run_id,
        max_frames=300,
        formal_frame_limit=formal_frame_limit,
        trackers=["msdc_elt"],
        variants=[],
        dynamic_variant_envs=variant_envs,
        progress_interval=progress_interval,
        render=render,
        render_class_source=render_class_source,
        input="",
        seq_name="",
        file_type="",
        source_detection_cache_root=source_detection_cache_root,
    )
    return run_detection_replay_benchmark(args)
```

Modify the variant validation and context selection in `run_detection_replay_benchmark()`:

```python
dynamic_variant_envs = getattr(args, "dynamic_variant_envs", {}) or {}
...
if tracker == "msdc_elt":
    variant_names = list(dynamic_variant_envs) if dynamic_variant_envs else list(args.variants)
```

Modify `msdc_variant_context()` to accept an optional env override:

```python
def msdc_variant_context(variant: str, env_override: dict[str, str] | None = None):
    env_delta = dict(env_override) if env_override is not None else _complete_msdc_variant_env(variant)
```

When replaying:

```python
env_override = dynamic_variant_envs.get(variant)
tracker_file = replay_tracker_from_cache(..., variant=variant, env_override=env_override)
```

Add `env_override` to `replay_tracker_from_cache()` and pass it into `msdc_variant_context()`.

- [ ] **Step 5: Wire real sensitivity into paper runner**

In `tools/evaluation/run_msdc_paper_experiments.py`, change `build_sensitivity_command()` to run the benchmark, not only the matrix writer:

```python
def build_sensitivity_command(dataset_roots: list[str], output_root: Path, run_id: str, matrix_path: Path, formal_frame_limit: int, progress_interval: int, source_detection_cache_root: Path) -> list[str]:
    return [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "run_msdc_sensitivity_benchmark.py"),
        "--dataset-root",
        *[str(path) for path in dataset_roots],
        "--matrix",
        str(matrix_path),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--formal-frame-limit",
        str(int(formal_frame_limit)),
        "--progress-interval",
        str(int(progress_interval)),
        "--source-detection-cache-root",
        str(source_detection_cache_root),
    ]
```

Keep a separate matrix command:

```python
sensitivity_matrix_cmd = build_sensitivity_matrix_command(sensitivity_output_root, "sensitivity_full")
sensitivity_metrics_cmd = build_sensitivity_command(
    dataset_roots=list(args.dataset_root),
    output_root=sensitivity_output_root,
    run_id="sensitivity_metrics",
    matrix_path=sensitivity_output_root / "sensitivity_full" / "sensitivity_matrix.csv",
    formal_frame_limit=int(args.formal_frame_limit),
    progress_interval=int(args.progress_interval),
    source_detection_cache_root=Path("results/msdc_paper_phase1/20260624_190710/main/main_full/detections"),
)
```

Order:

```python
("sensitivity_matrix", sensitivity_matrix_cmd),
("sensitivity_metrics", sensitivity_metrics_cmd),
```

- [ ] **Step 6: Expected new outputs**

After running, these files must exist:

```text
results/msdc_paper_phase1/<new_run>/sensitivity/sensitivity_full/sensitivity_matrix.csv
results/msdc_paper_phase1/<new_run>/sensitivity/sensitivity_metrics/summary/replay_summary.csv
results/msdc_paper_phase1/<new_run>/sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv
results/msdc_paper_phase1/<new_run>/summary/sensitivity_results.csv
```

- [ ] **Step 7: Run tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_sensitivity_benchmark.py \
  test/test_detection_replay_benchmark.py \
  test/test_msdc_paper_experiment_runner.py \
  -q
```

Expected: PASS.

## Task 3: Produce Slice-Level Metrics

**Files:**
- Create: `tools/evaluation/msdc_slice_metrics.py`
- Modify: `tools/evaluation/msdc_slice_eval.py`
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Test: `test/test_msdc_slice_metrics.py`

- [ ] **Step 1: Write slice metrics tests**

Create `test/test_msdc_slice_metrics.py`:

```python
from tools.evaluation.msdc_slice_metrics import frame_in_any_slice, reindex_mot_line


def test_frame_in_any_slice():
    slices = [{"start_frame": "10", "end_frame": "12"}, {"start_frame": "20", "end_frame": "21"}]
    assert frame_in_any_slice(10, slices)
    assert frame_in_any_slice(21, slices)
    assert not frame_in_any_slice(13, slices)


def test_reindex_mot_line_for_slice_start():
    line = "12,4,10,20,30,40,1,-1,-1,-1"
    assert reindex_mot_line(line, slice_start=10).startswith("3,4,")
```

- [ ] **Step 2: Run failing test**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_slice_metrics.py -q
```

Expected before implementation: FAIL because module does not exist.

- [ ] **Step 3: Create slice metric module**

Create `tools/evaluation/msdc_slice_metrics.py` with helpers:

```python
"""Build clipped MOT slice datasets and evaluate slice-level tracking metrics."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import METRIC_FIELDS
from tools.evaluation.motchallenge_eval import run_motchallenge_eval


def frame_in_any_slice(frame_id: int, slices: list[dict[str, str]]) -> bool:
    for row in slices:
        if int(row["start_frame"]) <= int(frame_id) <= int(row["end_frame"]):
            return True
    return False


def reindex_mot_line(line: str, slice_start: int) -> str:
    parts = line.rstrip("\n").split(",")
    parts[0] = str(int(float(parts[0])) - int(slice_start) + 1)
    return ",".join(parts)
```

Then implement:

```python
def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_slice_metric_rows(output_csv: str | Path, rows: list[dict[str, str]]) -> Path:
    fields = ["seq_name", "tracker", "num_slices", "total_slice_frames", *METRIC_FIELDS]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv
```

Build clipped datasets by grouping `slice_manifest.csv` rows by `(seq_name, tracker)`. For each group, write one synthetic sequence named `<seq_name>__<tracker>__short_miss` under:

```text
slice/slice_metrics/mot_gt/<synthetic_seq>/gt/gt.txt
slice/slice_metrics/trackers/<tracker>/data/<synthetic_seq>.txt
```

For each slice, copy only GT/pred MOT lines whose frame is in `[start_frame, end_frame]` and reindex frame numbers to a cumulative synthetic timeline. Write `seqinfo.ini` with `seqLength=<total_slice_frames>`.

Run TrackEval:

```python
summary = run_motchallenge_eval(
    gt_root=gt_root,
    trackers_root=trackers_root,
    output_root=eval_root,
    trackers=trackers,
    sequences=synthetic_sequences,
    print_results=False,
)
```

Write:

```text
slice/slice_metrics/slice_metrics.csv
slice/slice_metrics/eval/motchallenge_summary.csv
```

- [ ] **Step 4: Wire into paper runner**

In `tools/evaluation/run_msdc_paper_experiments.py`, after slice manifest command, add:

```python
def build_slice_metrics_command(run_root: Path) -> list[str]:
    return [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "msdc_slice_metrics.py"),
        "--slice-manifest",
        str(run_root / "slice" / "slice_manifest.csv"),
        "--ablation-root",
        str(run_root / "ablation" / "ablation_cache_full"),
        "--output-root",
        str(run_root / "slice" / "slice_metrics"),
    ]
```

Order:

```python
("slice", slice_cmd),
("slice_metrics", build_slice_metrics_command(run_root)),
```

- [ ] **Step 5: Expected new outputs**

```text
results/msdc_paper_phase1/<new_run>/slice/slice_manifest.csv
results/msdc_paper_phase1/<new_run>/slice/slice_metrics/slice_metrics.csv
results/msdc_paper_phase1/<new_run>/slice/slice_metrics/eval/motchallenge_summary.csv
```

- [ ] **Step 6: Run tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_slice_eval.py test/test_msdc_slice_metrics.py -q
```

Expected: PASS.

## Task 4: Add Low-Candidate Recall and Reacquire/Inherit Opportunity Diagnostics

**Files:**
- Modify: `tools/evaluation/msdc_diagnostic_metrics.py`
- Modify: `tools/evaluation/detection_replay_benchmark.py`
- Test: `test/test_msdc_diagnostic_metrics.py`

- [ ] **Step 1: Define diagnostic denominators**

Use these definitions in docs and code:

- `low_candidate_precision = confirmed_low_candidates / created_low_candidates`.
- `low_candidate_recall = confirmed_low_candidate_gt_matches / low_candidate_opportunity_gt_matches`.
- `low_candidate_opportunity_gt_matches`: GT instances or short-miss slice targets that have low-only stage observations overlapping GT with IoU >= 0.3 within the low confirmation window.
- `reacquire_opportunities`: LOST tracks with at least one unmatched observation inside the configured reacquire center/IoU gate on an attempted reacquire frame.
- `reacquire_success_rate = reacquire_success / reacquire_opportunities`; if denominator is 0, report `N/A`, not 0.
- `inherit_opportunities`: ready LOW_CANDIDATE tracks and LOST tracks passing class/age coarse gates before score threshold.
- `inherit_success_rate = inherit_correct / inherit_opportunities`; if denominator is 0, report `N/A`.

- [ ] **Step 2: Write failing diagnostic tests**

Append to `test/test_msdc_diagnostic_metrics.py`:

```python
from tools.evaluation.msdc_diagnostic_metrics import summarize_msdc_diagnostics


def test_diagnostic_summary_reports_opportunity_denominators(tmp_path):
    events = tmp_path / "events.jsonl"
    stage = tmp_path / "stage.jsonl"
    per_gt = tmp_path / "per_gt.csv"

    events.write_text(
        '{"frame_idx": 1, "gid": 10, "event_type": "NEW_LOW_CANDIDATE"}\n'
        '{"frame_idx": 5, "gid": 10, "event_type": "CONFIRM_LOW_ACTIVE"}\n'
        '{"frame_idx": 8, "gid": 11, "event_type": "REACQUIRE_ATTEMPT"}\n',
        encoding="utf-8",
    )
    stage.write_text(
        '{"frame_idx": 2, "stage": "low_only", "gt_id": 1, "iou": 0.5, "gid": 10}\n',
        encoding="utf-8",
    )
    per_gt.write_text(
        "gt_id,predicted_id_count,matched_segments,missed_segments\n"
        "1,1,1-10,\n",
        encoding="utf-8",
    )

    summary = summarize_msdc_diagnostics(events, stage, per_gt)
    assert summary["low_candidate_opportunities"] == 1
    assert summary["low_candidate_recall"] == 1.0
    assert summary["reacquire_attempts"] == 1
```

- [ ] **Step 3: Extend diagnostic fields**

In `tools/evaluation/msdc_diagnostic_metrics.py`, extend `DIAGNOSTIC_FIELDS`:

```python
"low_candidate_created",
"low_candidate_opportunities",
"reacquire_attempts",
"reacquire_opportunities",
"reacquire_success_rate",
"inherit_opportunities",
"inherit_success_rate",
```

Implement robust parsing from events and stage observations. If current stage JSONL lacks `gt_id` or IoU fields, write `low_candidate_recall = "N/A"` and add a note field `low_candidate_recall_reason = "missing_stage_gt_overlap"`.

- [ ] **Step 4: Preserve honest zero-success interpretation**

Do not force a successful reacquire/inherit. The expected formal output can validly be:

```text
reacquire_opportunities = 0, reacquire_success_rate = N/A
inherit_opportunities = 0, inherit_success_rate = N/A
```

or:

```text
reacquire_opportunities > 0, reacquire_success = 0, reacquire_success_rate = 0.0
```

This distinction decides whether V6 writes “not triggered” or “triggered but failed”.

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_diagnostic_metrics.py -q
```

Expected: PASS.

## Task 5: Strengthen Formal Validation and Summary Outputs

**Files:**
- Modify: `tools/evaluation/validate_msdc_formal_run.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Modify: `README.md`
- Modify: `tools/README.md`
- Test: `test/test_validate_msdc_formal_run.py`
- Test: `test/test_msdc_experiment_summary.py`

- [ ] **Step 1: Add validation requirements**

In `tools/evaluation/validate_msdc_formal_run.py`, require:

```python
("effective_config", "**/effective_config/*.json"),
("sensitivity_metrics", "sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv"),
("slice_metrics", "slice/slice_metrics/slice_metrics.csv"),
```

Require new diagnostic fields:

```python
REQUIRED_DIAGNOSTIC = [
    "low_candidate_created",
    "low_candidate_confirmed",
    "low_candidate_precision",
    "low_candidate_opportunities",
    "low_candidate_recall",
    "reacquire_attempts",
    "reacquire_opportunities",
    "reacquire_success",
    "inherit_opportunities",
    "inherit_correct",
    "inherit_wrong",
    "inherit_ambiguous",
]
```

- [ ] **Step 2: Add summary materialization**

In `tools/evaluation/msdc_experiment_summary.py`, copy or materialize:

```text
summary/sensitivity_results.csv
summary/slice_metrics.csv
summary/diagnostic_results.csv
```

The summary must not invent missing metrics. If a required file is absent, write a row with `status=missing` and `failure=<path>`.

- [ ] **Step 3: Update docs**

In `README.md` and `tools/README.md`, document:

```text
Formal v3 completion now requires:
- main_results.csv
- ablation_results.csv with correct variant_key and effective config JSON
- sensitivity_results.csv with HOTA/IDF1/AssA/MOTA/IDSW/FP/FN per point
- slice_metrics.csv with HOTA/IDF1/AssA or IDF1/IDSW/FN/FP
- diagnostic_results.csv with low_candidate recall denominator and reacquire/inherit opportunities
```

- [ ] **Step 4: Run tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_validate_msdc_formal_run.py \
  test/test_msdc_experiment_summary.py \
  -q
```

Expected: PASS.

## Task 6: Run Cache-Only Supplemental Completion

**Files/Outputs:**
- Supplemental cache-only root: `results/msdc_paper_phase1/20260624_190710_supplement_cache_only`
- Do not overwrite `results/msdc_paper_phase1/20260624_190710`.
- Do not run detector inference.

- [ ] **Step 1: Run targeted tests before experiments**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_detection_replay_benchmark.py \
  test/test_msdc_experiment_summary.py \
  test/test_msdc_diagnostic_metrics.py \
  test/test_msdc_slice_eval.py \
  test/test_msdc_slice_metrics.py \
  test/test_msdc_sensitivity_benchmark.py \
  test/test_validate_msdc_formal_run.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Create the supplemental root**

Run:

```bash
mkdir -p results/msdc_paper_phase1/20260624_190710_supplement_cache_only
```

Expected: creates an empty supplemental root without touching the original run.

- [ ] **Step 3: Recompute ablation metrics from cached detections only**

Run:

```bash
conda run -n ship_detect python tools/evaluation/detection_replay_benchmark.py \
  --dataset-root \
  /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 \
  /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --output-root results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation \
  --run-id ablation_cache_full \
  --trackers msdc_elt \
  --variants \
  msdc_v3 \
  no_low_candidate \
  no_direct_reacquire \
  no_low_inheritance \
  hits_only_no_evidence \
  no_output_nms \
  no_output_real_det_age_gate \
  low_budget_off \
  low_budget_topk16 \
  low_budget_topk64 \
  --formal-frame-limit 5400 \
  --source-detection-cache-root results/msdc_paper_phase1/20260624_190710/main/main_full/detections \
  --progress-interval 500 \
  --render-class-source cache
```

Expected:

```text
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/eval/motchallenge_summary.csv
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/summary/replay_summary.csv
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full/effective_config/*_replay.json
```

The command must print `[SKIP] detection cache complete` for both sequences. It must not call `dump_video_detections()`.

- [ ] **Step 4: Generate sensitivity matrix**

Run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_sensitivity_matrix.py \
  --output results/msdc_paper_phase1/20260624_190710_supplement_cache_only/sensitivity/sensitivity_full/sensitivity_matrix.csv
```

Expected: writes the 16-row matrix.

- [ ] **Step 5: Run sensitivity metrics from cached detections only**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_sensitivity_benchmark.py \
  --dataset-root \
  /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 \
  /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --matrix results/msdc_paper_phase1/20260624_190710_supplement_cache_only/sensitivity/sensitivity_full/sensitivity_matrix.csv \
  --output-root results/msdc_paper_phase1/20260624_190710_supplement_cache_only/sensitivity \
  --run-id sensitivity_metrics \
  --formal-frame-limit 5400 \
  --source-detection-cache-root results/msdc_paper_phase1/20260624_190710/main/main_full/detections \
  --progress-interval 500
```

Expected:

```text
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/sensitivity/sensitivity_metrics/summary/replay_summary.csv
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv
```

The command must print `[SKIP] detection cache complete` for both sequences. It must not call `dump_video_detections()`.

- [ ] **Step 6: Generate slice metrics from recomputed ablation MOT and existing GT**

Run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_slice_metrics.py \
  --slice-manifest results/msdc_paper_phase1/20260624_190710/slice/slice_manifest.csv \
  --ablation-root results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full \
  --output-root results/msdc_paper_phase1/20260624_190710_supplement_cache_only/slice/slice_metrics
```

Expected:

```text
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/slice/slice_metrics/slice_metrics.csv
results/msdc_paper_phase1/20260624_190710_supplement_cache_only/slice/slice_metrics/eval/motchallenge_summary.csv
```

- [ ] **Step 7: Regenerate corrected ablation summary from recomputed TrackEval outputs**

Run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_experiment_summary.py \
  --main-root results/msdc_paper_phase1/20260624_190710/main/main_full \
  --ablation-root results/msdc_paper_phase1/20260624_190710_supplement_cache_only/ablation/ablation_cache_full \
  --speed-csv results/msdc_paper_phase1/20260624_190710/summary/speed_results.csv \
  --output-root results/msdc_paper_phase1/20260624_190710_supplement_cache_only/summary \
  --report-output results/msdc_paper_phase1/20260624_190710_supplement_cache_only/MSDC_EXPERIMENT_REPORT.md
```

Expected: `summary/ablation_results.csv` has `variant_key`, correct config columns, and metrics from `ablation_cache_full`. This step does not re-run detector; it only reads existing main results, recomputed ablation TrackEval CSVs, and speed CSV.

- [ ] **Step 8: Validate the supplemental run**

Run:

```bash
conda run -n ship_detect python tools/evaluation/validate_msdc_formal_run.py \
  --run-root results/msdc_paper_phase1/20260624_190710_supplement_cache_only \
  --json-output results/msdc_paper_phase1/20260624_190710_supplement_cache_only/summary/formal_validation.json
```

Expected:

```json
{
  "ok": true,
  "missing": [],
  "run_root": "results/msdc_paper_phase1/20260624_190710_supplement_cache_only"
}
```

- [ ] **Step 9: Regenerate evidence package**

Update only:

```text
paper-prepare/实验后证据包.md
```

Do not modify:

```text
paper-prepare/9 个科研问题 V5.md
```

Do not generate V6 until evidence package says which claims are safe.

## Verification Checklist

- [ ] `ablation/ablation_cache_full/eval/motchallenge_summary.csv` exists and was produced by cache-only replay of every ablation variant.
- [ ] `summary/ablation_results.csv` has `variant_key`, config columns reflect `no_low_candidate`, `no_direct_reacquire`, `hits_only_no_evidence`, etc., and metrics come from `ablation_cache_full`, not old `ablation_full`.
- [ ] Every MS-DC replay variant has `effective_config/<variant>_replay.json`.
- [ ] `sensitivity_results.csv` contains HOTA, IDF1, AssA, MOTA, IDSW, FP, FN for every sensitivity point.
- [ ] `slice_metrics.csv` contains slice-level HOTA/IDF1/AssA or at least IDF1/IDSW/FN/FP.
- [ ] `msdc_diagnostic_summary.csv` contains low-candidate recall denominator and reacquire/inherit opportunity counts.
- [ ] Zero reacquire/inherit success is interpreted honestly as either no opportunity or failed opportunities; do not fabricate success.
- [ ] Supplemental cache-only run validates with `ok: true`.
- [ ] Supplemental logs show detection cache reuse and no detector dumping.
- [ ] `paper-prepare/实验后证据包.md` is updated with the supplemental cache-only results only; old v2 is not used.
