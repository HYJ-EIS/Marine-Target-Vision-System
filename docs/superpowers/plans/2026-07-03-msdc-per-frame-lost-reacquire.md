# MS-DC Per-Frame Lost Reacquire Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a comparable MS-DC variant that attempts LOST-track reacquire on every frame against unmatched high/low detections with a lower reacquire score threshold, then run full-length no-render MOT and speed comparisons against `msdc_v3`.

**Architecture:** The existing update path already computes unmatched observation groups after ACTIVE/CANDIDATE association and sends them to `_reacquire_lost_track_indices()`. Keep current `msdc_v3` as the baseline and add `reacquire_every_frame_low_score` with `MSDC_REACQUIRE_INTERVAL=1` and `MSDC_REACQUIRE_SCORE=1.0`, so the effect is isolated and reversible.

**Tech Stack:** Python, pytest, MS-DC evidence-state tracker, MOT replay benchmark, MS-DC speed benchmark, cached high/low detections.

---

### Task 1: Add Variant Tests

**Files:**
- Modify: `test/test_msdc_paper_experiment_runner.py`

- [ ] **Step 1: Extend variant availability test**

Add `"reacquire_every_frame_low_score"` to the expected ablation variants, then assert:

```python
every_frame_env = _variant_env("reacquire_every_frame_low_score")
assert every_frame_env["MSDC_REACQUIRE_INTERVAL"] == "1"
assert every_frame_env["MSDC_REACQUIRE_SCORE"] == "1.0"
assert every_frame_env["MSDC_REMOVED_RECOVERY_ENABLE"] == "1"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py::test_formal_v3_ablation_variants_are_available -q
```

Expected: fail because the new variant does not exist.

### Task 2: Add the Reacquire Variant

**Files:**
- Modify: `tools/experiments/run_msdc_ablation.py`

- [ ] **Step 1: Add the variant**

Add:

```python
"reacquire_every_frame_low_score": {
    **FORMAL_V3_ENV,
    "MSDC_REACQUIRE_INTERVAL": "1",
    "MSDC_REACQUIRE_SCORE": "1.0",
},
```

- [ ] **Step 2: Verify GREEN**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py::test_formal_v3_ablation_variants_are_available -q
```

Expected: pass.

### Task 3: Make Speed Benchmark Variant-Aware

**Files:**
- Modify: `tools/evaluation/msdc_speed_benchmark.py`
- Modify: `test/test_msdc_speed_benchmark.py`

- [ ] **Step 1: Write failing speed variant test**

Add a test that calls `run_speed_benchmark()` with `msdc_variant="reacquire_every_frame_low_score"` and a fake `_run_tracker_benchmark()`, then asserts `Config.MSDC_REACQUIRE_INTERVAL == 1` and `Config.MSDC_REACQUIRE_SCORE == 1.0` inside the benchmark and original config values are restored afterward.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_speed_benchmark.py::test_run_speed_benchmark_applies_selected_msdc_variant -q
```

Expected: fail because `msdc_speed_benchmark.py` has no `--msdc-variant` support.

- [ ] **Step 3: Implement variant support**

Import `ABLATION_VARIANTS` and `FORMAL_MSDC_VARIANT`, add `MSDC_REACQUIRE_SCORE` to the formal float keys, add:

```python
def _config_overrides_for_variant(variant: str) -> dict[str, bool | int | float]:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown MS-DC variant: {variant}")
    overrides = dict(FORMAL_V3_CONFIG_OVERRIDES)
    supported_keys = _FORMAL_V3_BOOL_KEYS | _FORMAL_V3_INT_KEYS | _FORMAL_V3_FLOAT_KEYS
    for key, value in ABLATION_VARIANTS[variant].items():
        if key in supported_keys:
            overrides[key] = _coerce_formal_v3_value(key, value)
    overrides["MSDC_DEBUG_EVENTS"] = False
    return overrides
```

Use `args.msdc_variant` when applying `_temporary_config_overrides()`, and add:

```python
parser.add_argument("--msdc-variant", choices=list(ABLATION_VARIANTS), default=FORMAL_MSDC_VARIANT)
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_speed_benchmark.py::test_run_speed_benchmark_applies_selected_msdc_variant -q
```

Expected: pass.

### Task 4: Docs and Full Test

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the variant**

Add a concise note that `reacquire_every_frame_low_score` keeps the rest of formal `msdc_v3` fixed but changes LOST reacquire to every frame and lowers the score threshold to `1.0`.

- [ ] **Step 2: Run full tests**

Run:

```bash
conda run -n ship_detect pytest test -q
```

Expected: pass.

### Task 5: Full-Length No-Render MOT Replay

**Output root:**
- `results/msdc_reacquire_every_frame/20260703_full_no_render_reacquire_every_frame`

- [ ] **Step 1: Run full replay**

Run:

```bash
conda run -n ship_detect python tools/evaluation/detection_replay_benchmark.py \
  --dataset-root /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --output-root results/msdc_reacquire_every_frame \
  --run-id 20260703_full_no_render_reacquire_every_frame \
  --formal-frame-limit 5400 \
  --trackers msdc_elt \
  --variants msdc_v3 reacquire_every_frame_low_score \
  --source-detection-cache-root results/msdc_paper_phase1/20260624_190710/main/main_full/detections \
  --progress-interval 500 \
  --render-class-source cache
```

This intentionally omits `--render`.

### Task 6: Full-Length No-Render Speed Comparison

- [ ] **Step 1: Run speed for both videos and both variants**

Run `tools/evaluation/msdc_speed_benchmark.py` for each dataset root with:

```bash
--frames 5400 --trackers msdc_elt --msdc-variant msdc_v3
--frames 5400 --trackers msdc_elt --msdc-variant reacquire_every_frame_low_score
```

Write results under:

```text
results/msdc_reacquire_every_frame/20260703_full_no_render_reacquire_every_frame/speed/
```

- [ ] **Step 2: Summarize**

Report combined frames, total seconds, FPS, mean latency, and stage timing fields from the four `speed_results.csv` files.
