# MS-DC Reacquire Grid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the best values for `MSDC_REACQUIRE_INTERVAL` and `MSDC_REACQUIRE_SCORE` by running a full-length no-render grid search and comparing tracking metrics and speed against `msdc_v3`.

**Architecture:** Use the existing dynamic sensitivity replay path instead of adding new tracker code. Generate a matrix of `env_json` overrides on top of formal `msdc_v3`, replay cached high/low detections for all grid points on both full videos, rank by HOTA/IDF1/MOTA with IDSW/FP/FN as guard metrics, then run speed only for `msdc_v3` and the top candidates.

**Tech Stack:** Python, CSV sensitivity matrix, `tools/evaluation/run_msdc_sensitivity_benchmark.py`, cached detection replay, `tools/evaluation/msdc_speed_benchmark.py`.

---

### Task 1: Build Search Matrix

**Files:**
- Create output artifact: `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/reacquire_grid.csv`

- [x] **Step 1: Use this grid**

Search:

```text
MSDC_REACQUIRE_INTERVAL: 1, 2, 3, 5, 8
MSDC_REACQUIRE_SCORE:    0.8, 1.0, 1.2, 1.5
```

This includes `msdc_v3` as `(interval=5, score=1.5)`.

- [x] **Step 2: Write rows**

Each row must contain:

```text
variant,parameter,value,env_json
```

Variant names use:

```text
reacq_i<interval>_s<score-with-p>
```

Example:

```text
reacq_i1_s1p0,reacquire_grid,i=1,s=1.0,"{""MSDC_REACQUIRE_INTERVAL"": ""1"", ""MSDC_REACQUIRE_SCORE"": ""1.0""}"
```

### Task 2: Full-Length No-Render MOT Replay

**Output root:**
- `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/replay`

- [x] **Step 1: Run dynamic replay**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_sensitivity_benchmark.py \
  --dataset-root /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --matrix results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/reacquire_grid.csv \
  --output-root results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/replay \
  --run-id grid_full \
  --formal-frame-limit 5400 \
  --source-detection-cache-root results/msdc_paper_phase1/20260624_190710/main/main_full/detections \
  --progress-interval 500 \
  --render-class-source cache
```

This intentionally omits rendering.

### Task 3: Rank the Grid

**Files:**
- Read: `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/replay/grid_full/summary/replay_summary.csv`
- Create output artifact: `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/grid_ranked.csv`

- [x] **Step 1: Rank by metrics**

Primary sort:

```text
HOTA desc, IDF1 desc, MOTA desc, IDSW asc, FP asc, FN asc
```

- [x] **Step 2: Compute deltas vs baseline**

Use baseline variant `reacq_i5_s1p5`.

Fields:

```text
delta_HOTA, delta_IDF1, delta_MOTA, delta_IDSW, delta_FP, delta_FN
```

- [x] **Step 3: Select speed candidates**

Run speed for:

```text
reacq_i5_s1p5
top 3 variants by ranking
```

Deduplicate if baseline is in the top 3.

### Task 4: Full-Length Speed for Baseline and Top Candidates

**Output root:**
- `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/speed`

- [x] **Step 1: Run speed benchmark**

For each selected variant and both dataset roots, run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_speed_benchmark.py \
  --dataset-root <dataset-root> \
  --output-root results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/speed \
  --run-id speed_<seq>_<variant> \
  --frames 5400 \
  --trackers msdc_elt \
  --msdc-variant <variant> \
  --progress-interval 500
```

If a selected dynamic grid variant is not available to `--msdc-variant`, use a temporary environment override with `MSDC_REACQUIRE_INTERVAL` and `MSDC_REACQUIRE_SCORE`, or add named ablation variants for the selected candidates.

### Task 5: Final Recommendation

- [x] **Step 1: Report raw table**

Include at least top 10 grid rows plus baseline.

- [x] **Step 2: Report speed**

Report combined frames, total seconds, FPS, mean latency, and stage timings for baseline and selected top candidates.

- [x] **Step 3: State recommendation**

Recommend the best parameter pair only if it improves HOTA and IDF1 without increasing IDSW or FP relative to baseline. If multiple pairs are statistically close, prefer the larger interval and higher score for stability.

## Execution Results

Artifacts:

- Matrix: `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/reacquire_grid.csv`
- Ranked grid: `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/grid_ranked.csv`
- Replay summary: `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/replay/grid_full/summary/replay_summary.csv`
- TrackEval summary: `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/replay/grid_full/eval/motchallenge_summary.csv`
- Selected speed summary: `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/selected_candidate_summary.csv`
- Speed runs: `results/msdc_reacquire_param_search/20260703_full_no_render_reacquire_grid/speed/`

Top metric candidate:

```text
reacq_i1_s0p8: interval=1, score=0.8
HOTA=77.99963 (+0.13112), IDF1=85.96941 (+0.09904), MOTA=78.09394 (+0.39625),
IDSW=10 (+0), FP=1812 (+3), FN=4591 (-119)
```

Best guarded candidate under `IDSW` and `FP` non-increase:

```text
reacq_i8_s1p0: interval=8, score=1.0
HOTA=77.99336 (+0.12485), IDF1=85.93596 (+0.06559), MOTA=77.80700 (+0.10931),
IDSW=10 (+0), FP=1809 (+0), FN=4678 (-32)
```

Speed was run for baseline, the top metric candidate, and the best guarded candidate. This differs from the initial "top 3 by ranking" note because the top 4 ranked candidates all use `score=0.8` and increase FP; the guarded candidate is the one needed for the stated recommendation rule.
