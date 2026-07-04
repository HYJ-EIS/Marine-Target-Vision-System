# MS-DC Pending Recovery Frames Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the best default value for `MSDC_PENDING_RECOVERY_FRAMES` by comparing full-length no-render tracking metrics and speed against current `msdc_v3`.

**Architecture:** Use the existing dynamic sensitivity replay path with cached high/low detections. Keep all formal `msdc_v3` settings fixed, enable `MSDC_PENDING_RECOVERY_ENABLE=1`, and sweep only `MSDC_PENDING_RECOVERY_FRAMES=1..5`; include `msdc_v3` with pending recovery disabled as the baseline. Rank variants by tracking metrics first, then run speed for baseline and the best candidate(s).

**Tech Stack:** Python, CSV sensitivity matrix, `tools/evaluation/run_msdc_sensitivity_benchmark.py`, cached detection replay, `tools/evaluation/msdc_speed_benchmark.py`.

---

### Task 1: Build Search Matrix

**Files:**
- Create: `results/msdc_pending_recovery_frames/20260703_full_no_render/pending_frames_matrix.csv`

- [x] **Step 1: Use this grid**

Search:

```text
baseline: msdc_v3, pending recovery disabled
MSDC_PENDING_RECOVERY_FRAMES: 1, 2, 3, 4, 5
```

- [x] **Step 2: Write matrix rows**

Each row must contain:

```text
variant,parameter,value,env_json
```

Variant names:

```text
msdc_v3
pending_frames_1
pending_frames_2
pending_frames_3
pending_frames_4
pending_frames_5
```

### Task 2: Full-Length No-Render Replay

**Output root:**
- `results/msdc_pending_recovery_frames/20260703_full_no_render/replay`

- [x] **Step 1: Run replay**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_sensitivity_benchmark.py \
  --dataset-root /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --matrix results/msdc_pending_recovery_frames/20260703_full_no_render/pending_frames_matrix.csv \
  --output-root results/msdc_pending_recovery_frames/20260703_full_no_render/replay \
  --run-id frames_full \
  --formal-frame-limit 5400 \
  --source-detection-cache-root results/msdc_paper_phase1/20260624_190710/main/main_full/detections \
  --progress-interval 500 \
  --render-class-source cache
```

This intentionally omits visualization.

### Task 3: Rank Tracking Metrics

**Files:**
- Read: `results/msdc_pending_recovery_frames/20260703_full_no_render/replay/frames_full/summary/replay_summary.csv`
- Create: `results/msdc_pending_recovery_frames/20260703_full_no_render/pending_frames_ranked.csv`

- [x] **Step 1: Rank**

Primary sort:

```text
HOTA desc, IDF1 desc, MOTA desc, IDSW asc, FP asc, FN asc
```

- [x] **Step 2: Compute deltas**

Use baseline `msdc_v3`.

Fields:

```text
delta_HOTA, delta_IDF1, delta_MOTA, delta_IDSW, delta_FP, delta_FN
```

### Task 4: Speed for Baseline and Best Candidate

**Output root:**
- `results/msdc_pending_recovery_frames/20260703_full_no_render/speed`

- [x] **Step 1: Select candidates**

Run speed for:

```text
msdc_v3
top tracking candidate
best no-IDSW/FP-regression candidate, if different
```

- [x] **Step 2: Run speed**

For each selected candidate and both dataset roots, run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_speed_benchmark.py \
  --dataset-root <dataset-root> \
  --output-root results/msdc_pending_recovery_frames/20260703_full_no_render/speed \
  --run-id <seq>_<variant> \
  --frames 5400 \
  --trackers msdc_elt \
  --msdc-variant msdc_v3 \
  --msdc-env-json <candidate-env-json> \
  --progress-interval 500
```

### Task 5: Final Recommendation

- [x] **Step 1: Report tracking table**

Include all frame settings plus baseline.

- [x] **Step 2: Report speed table**

Include total frames, total seconds, FPS, mean latency, and MS-DC update timing for selected candidates.

- [x] **Step 3: Choose default**

Recommend a default only if it improves HOTA or IDF1 and does not increase IDSW or FP relative to `msdc_v3`. If several settings are close, prefer the smaller frame count because it delays output less.

## Execution Results

Artifacts:

- Matrix: `results/msdc_pending_recovery_frames/20260703_full_no_render/pending_frames_matrix.csv`
- Replay summary: `results/msdc_pending_recovery_frames/20260703_full_no_render/replay/frames_full/summary/replay_summary.csv`
- Ranked table: `results/msdc_pending_recovery_frames/20260703_full_no_render/pending_frames_ranked.csv`
- Selected speed table: `results/msdc_pending_recovery_frames/20260703_full_no_render/pending_frames_selected_speed.csv`
- Speed runs: `results/msdc_pending_recovery_frames/20260703_full_no_render/speed/`

Tracking ranking:

```text
frames=1: HOTA=78.08988, IDF1=85.96941, MOTA=78.09735, IDSW=9, FP=1812, FN=4591
frames=2: HOTA=78.08894, IDF1=85.97095, MOTA=78.09394, IDSW=9, FP=1812, FN=4592
frames=3: HOTA=78.08801, IDF1=85.97249, MOTA=78.09052, IDSW=9, FP=1812, FN=4593
frames=4: HOTA=78.08707, IDF1=85.97404, MOTA=78.08711, IDSW=9, FP=1812, FN=4594
frames=5: HOTA=78.08614, IDF1=85.97558, MOTA=78.08369, IDSW=9, FP=1812, FN=4595
baseline: HOTA=77.99963, IDF1=85.96941, MOTA=78.09394, IDSW=10, FP=1812, FN=4591
```

Selected speed:

```text
msdc_v3:          FPS=10.471772, mean_latency_ms=95.494818
pending_frames_1: FPS=10.182386, mean_latency_ms=98.208807
delta:            FPS=-0.289386
```

Recommendation:

```text
Use MSDC_PENDING_RECOVERY_FRAMES=1.
```

Reason: it gives the best HOTA and MOTA in the sweep, reduces IDSW by 1, does not increase FP or FN, and avoids the extra delayed-output FN cost seen when frames >= 2.
