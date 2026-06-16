# MS-DC-ELT v2 Baseline Gap Experiment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine why `v2_candidate_topk_no_roi` underperforms BoT-SORT / OC-SORT and separate detector, candidate filtering, lifecycle association, and timing causes.

**Architecture:** Run a same-slice benchmark first, then use existing per-GT, stage coverage, candidate pool, MOT output, speed timing, and visualization artifacts to localize the failure mode. Only after the same-slice comparison is complete, add replay/oracle tooling if the first pass still cannot distinguish detector effects from tracker association effects.

**Tech Stack:** `conda run -n ship_detect`, `tools/evaluation/msdc_dataset_benchmark.py`, `tools/evaluation/msdc_speed_benchmark.py`, vendored TrackEval, MS-DC-ELT diagnostics JSONL/CSV, MOTChallenge txt.

---

## Known Starting Point

Existing run `results/msdc_metric_ablation/20260614_163810_metric_1000` is not a fair baseline comparison because `metadata/run_metadata.json` lists only `trackers=["msdc_elt"]`. Its summary shows:

- `v2_candidate_topk_no_roi`: `HOTA=83.17215`, `IDF1=83.73967`, `MOTA=74.47265`, `IDSW=5`, `FP=39`, `FN=670`.
- `v2_no_roi_redetect`: `HOTA=82.88882`, `IDF1=83.81331`, `MOTA=74.97319`, `IDSW=5`, `FP=68`, `FN=627`.
- Candidate topK reduced FP by `29` but added `43` FN, which means the current candidate filter is buying precision with recall loss.
- The same report only gives speed baseline rows for OC-SORT / BoT-SORT, not TrackEval metrics, so it cannot prove where the accuracy gap comes from.

## Hypotheses

1. **Lifecycle association gap:** MS-DC-ELT lifecycle state machine fragments or suppresses difficult GT tracks more than OC-SORT / BoT-SORT.
2. **Candidate topK recall loss:** `MSDC_LOW_OBS_TOPK=32` and `MSDC_LOW_OBS_MIN_CONF=0.25` filter useful low-only evidence for small/far targets.
3. **ROI is not the accuracy lever on this slice:** `v2_no_roi_redetect`, `v2_roi_interval10`, `v2_roi_interval15`, and `v2_roi_max1` had identical metrics in the existing run.
4. **Detector is not sufficient evidence by itself:** GT3 in `v2_candidate_topk` had `low_det_coverage=0.6424` but `output_coverage=0.2146`, so lifecycle/output gating is likely losing available candidates.
5. **BoT-SORT identity advantage:** If BoT-SORT has lower IDSW or higher IDF1 under the same detections/slice, ReID/association is the likely differentiator.

## Task 1: Preflight and Same-Slice Baseline Comparison

**Files and outputs:**
- Read: `/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集/gt.txt`
- Read: `/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集/原视频文件地址.txt`
- Create: `results/msdc_gap_analysis/<run-id>/`

- [ ] **Step 1: Record current code state**

Run:

```bash
git rev-parse --short HEAD
git status --short
```

Expected: commit hash recorded; dirty files noted in the experiment log. Do not reset unrelated local changes.

- [ ] **Step 2: Check dataset and video accessibility**

Run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_dataset_benchmark.py \
  --dataset-root "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集" \
  --output-root "results/msdc_gap_analysis" \
  --run-id "preflight_print_only" \
  --trackers botsort ocsort msdc_elt \
  --variants v2_candidate_topk_no_roi \
  --max-frames 1000 \
  --render \
  --render-class-source none
```

Expected: print-only mode shows resolved sequence `DJI_20250916100639_0001_V`, GT path, video path, export commands, render commands, and eval command.

- [ ] **Step 3: Run the 1000-frame fair comparison**

Run:

```bash
RUN_ID="20260615_gap_baseline_1000"
conda run -n ship_detect python tools/evaluation/msdc_dataset_benchmark.py \
  --dataset-root "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集" \
  --output-root "results/msdc_gap_analysis" \
  --run-id "$RUN_ID" \
  --mode benchmark \
  --trackers botsort ocsort msdc_elt \
  --variants v2_candidate_topk_no_roi \
  --max-frames 1000 \
  --progress-interval 100 \
  --render \
  --render-class-source none \
  --run
```

Expected artifacts:

- `results/msdc_gap_analysis/20260615_gap_baseline_1000/eval/motchallenge_summary.csv`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/trackers/botsort/data/DJI_20250916100639_0001_V.txt`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/trackers/ocsort/data/DJI_20250916100639_0001_V.txt`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/trackers/v2_candidate_topk_no_roi/data/DJI_20250916100639_0001_V.txt`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/visualizations/DJI_20250916100639_0001_V/botsort.mp4`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/visualizations/DJI_20250916100639_0001_V/ocsort.mp4`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/visualizations/DJI_20250916100639_0001_V/v2_candidate_topk_no_roi.mp4`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/diagnostics/DJI_20250916100639_0001_V/*_per_gt_diagnostics.csv`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/diagnostics/DJI_20250916100639_0001_V/*_per_gt_stage_coverage.csv`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/metadata/commands.jsonl`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/metadata/failures.csv`

- [ ] **Step 4: Run same-slice speed benchmark**

Run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_speed_benchmark.py \
  --dataset-root "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集" \
  --trackers ocsort botsort msdc_elt \
  --frames 1000 \
  --output-root "results/msdc_gap_analysis/20260615_gap_baseline_1000/speed" \
  --run-id "speed_1000" \
  --progress-interval 100
```

Expected artifacts:

- `results/msdc_gap_analysis/20260615_gap_baseline_1000/speed/speed_1000/speed_results.csv`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/speed/speed_1000/speed_timings.jsonl`

## Task 2: First-Pass Diagnosis From Existing Artifacts

**Files and outputs:**
- Read: `results/msdc_gap_analysis/20260615_gap_baseline_1000/eval/motchallenge_summary.csv`
- Read: `results/msdc_gap_analysis/20260615_gap_baseline_1000/diagnostics/DJI_20250916100639_0001_V/*_per_gt_diagnostics.csv`
- Read: `results/msdc_gap_analysis/20260615_gap_baseline_1000/diagnostics/DJI_20250916100639_0001_V/*_per_gt_stage_coverage.csv`
- Read: `results/msdc_gap_analysis/20260615_gap_baseline_1000/trackers/v2_candidate_topk_no_roi/diagnostics/DJI_20250916100639_0001_V/candidate_pool_stats.jsonl`

- [ ] **Step 1: Compare headline metrics**

Extract columns: `tracker,HOTA,DetA,AssA,MOTA,IDF1,IDSW,FP,FN,IDTP,IDFP,IDFN`.

Decision rules:

- If `v2_candidate_topk_no_roi` has much higher `FN` but similar `FP`, prioritize candidate/lifecycle recall.
- If it has similar `FN` but worse `IDF1` or higher `IDSW`, prioritize association/identity recovery.
- If it has worse `DetA` while `AssA` is close, prioritize detector/candidate recall.
- If it has worse `AssA` while `DetA` is close, prioritize tracker association and lifecycle fragmentation.

- [ ] **Step 2: Compare per-GT coverage**

For each GT ID, compare `coverage_ratio`, `predicted_id_count`, `matched_segments`, and `missed_segments` across `botsort`, `ocsort`, and `v2_candidate_topk_no_roi`.

Decision rules:

- GT missed by all trackers: detector/visibility or annotation difficulty.
- GT covered by OC-SORT/BoT-SORT but not MS-DC: lifecycle/candidate filtering issue.
- GT covered by MS-DC with many `predicted_ids`: fragmentation/ID management issue.
- GT covered by MS-DC only in short bursts: promotion, lost, or output gate issue.

- [ ] **Step 3: Compare stage coverage for MS-DC only**

Use `v2_candidate_topk_no_roi_per_gt_stage_coverage.csv`.

Decision rules:

- `low_det_coverage` high but `output_coverage` low: lifecycle/output gate is dropping usable detections.
- `low_only_coverage` high but output low: low candidate confirmation/topK/min_conf policy is too strict or noisy.
- `high_det_coverage` high but output low: high-det association/promotion state machine is the issue.
- `roi_low_det_coverage=0` is expected for no-ROI variant and should not be treated as missing evidence.

- [ ] **Step 4: Summarize candidate pool behavior**

Use `candidate_pool_stats.jsonl` to calculate:

- Frames where `num_low_only > 0`.
- Mean and P95 of `num_low_only`.
- Mean and P95 of `suppressed_spawn_groups`.
- Total `low_inherit_matches`.
- Total `low_inherit_active_conflicts`.
- Total `removed_guard_vetoes`.
- Frames where `reacquire_attempted=true`.

Decision rules:

- High `suppressed_spawn_groups` around missed GT segments: active-neighbor suppression is too aggressive.
- Low `low_inherit_matches` despite high `low_only_coverage`: low detections are not associating to existing tracks.
- Frequent `removed_guard_vetoes`: reuse guard is blocking legitimate ID recovery.

## Task 3: Candidate TopK Sweep

**Purpose:** Check whether the current `topK=32,min_conf=0.25` is the reason FN increases.

**Implementation choice:** The current benchmark runner only accepts named variants from `tools/experiments/run_msdc_ablation.py`. Add named variants before running this task, or run `export_mot_results.py` directly with explicit environment variables and then run `motchallenge_eval.py`. For repeatability, prefer named variants committed to `run_msdc_ablation.py` and documented in `README.md`.

**Variant matrix:**

| Variant | `MSDC_LOW_OBS_TOPK` | `MSDC_LOW_OBS_MIN_CONF` | `MSDC_USE_ROI_REDETECT` |
|---|---:|---:|---:|
| `v2_topk16_conf018_no_roi` | 16 | 0.18 | 0 |
| `v2_topk16_conf025_no_roi` | 16 | 0.25 | 0 |
| `v2_topk32_conf018_no_roi` | 32 | 0.18 | 0 |
| `v2_topk32_conf025_no_roi` | 32 | 0.25 | 0 |
| `v2_topk64_conf018_no_roi` | 64 | 0.18 | 0 |
| `v2_topk64_conf025_no_roi` | 64 | 0.25 | 0 |
| `v2_topk64_conf030_no_roi` | 64 | 0.30 | 0 |

- [ ] **Step 1: Add variants if missing**

Modify `tools/experiments/run_msdc_ablation.py` by adding each variant to `ABLATION_VARIANTS` with:

```python
"v2_topk32_conf018_no_roi": {
    "MSDC_LOW_OBS_TOPK": "32",
    "MSDC_LOW_OBS_MIN_CONF": "0.18",
    "MSDC_USE_ROI_REDETECT": "0",
    "MSDC_USE_TEMPLATE": "0",
    "MSDC_TEMPLATE_ENABLE": "0",
    "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
},
```

Repeat the same structure for the remaining rows in the matrix, changing only `MSDC_LOW_OBS_TOPK` and `MSDC_LOW_OBS_MIN_CONF`.

- [ ] **Step 2: Update README variant list**

Add the sweep variants to the MS-DC-ELT v2 experiment variants section in `README.md`, because repository instructions require README updates after code changes.

- [ ] **Step 3: Run focused 1000-frame sweep**

Run:

```bash
RUN_ID="20260615_topk_sweep_1000"
conda run -n ship_detect python tools/evaluation/msdc_dataset_benchmark.py \
  --dataset-root "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集" \
  --output-root "results/msdc_gap_analysis" \
  --run-id "$RUN_ID" \
  --mode ablation \
  --trackers msdc_elt \
  --variants \
    v2_topk16_conf018_no_roi \
    v2_topk16_conf025_no_roi \
    v2_topk32_conf018_no_roi \
    v2_topk32_conf025_no_roi \
    v2_topk64_conf018_no_roi \
    v2_topk64_conf025_no_roi \
    v2_topk64_conf030_no_roi \
  --max-frames 1000 \
  --progress-interval 100 \
  --render \
  --render-class-source none \
  --run
```

Expected: one MOT txt, TrackEval summary row, per-GT diagnostics, candidate pool stats, and MP4 for each variant.

- [ ] **Step 4: Pick candidate setting**

Decision rule:

- Reject any setting with `FN` higher than `v2_no_roi_redetect` unless `FP` reduction improves `MOTA` and `IDF1` does not regress.
- Prefer the setting with best `IDF1` among variants whose `MOTA` is within `0.2` points of the best sweep result.
- If lower `min_conf=0.18` recovers GT3 coverage without large FP growth, candidate filter was too strict.
- If `topK=64` still misses the same GT3 segments, topK is not the primary issue.

## Task 4: High-Only Lifecycle Isolation

**Purpose:** Determine whether MS-DC lifecycle is weaker even without low-det branches.

**Variants to add before running:**

| Variant | Low | Motion | Template | Reacquire | ROI |
|---|---:|---:|---:|---:|---:|
| `v2_high_only_lifecycle` | 0 | 0 | 0 | 0 | 0 |
| `v2_high_motion_lifecycle` | 0 | 1 | 0 | 0 | 0 |
| `v2_high_reacquire_lifecycle` | 0 | 0 | 0 | 1 | 0 |

- [ ] **Step 1: Add lifecycle isolation variants**

Modify `tools/experiments/run_msdc_ablation.py`:

```python
"v2_high_only_lifecycle": {
    "MSDC_USE_LOW_DET": "0",
    "MSDC_USE_MOTION": "0",
    "MSDC_USE_TEMPLATE": "0",
    "MSDC_TEMPLATE_ENABLE": "0",
    "MSDC_USE_REACQUIRE": "0",
    "MSDC_USE_ROI_REDETECT": "0",
    "MSDC_REUSE_GUARD_ENABLE": "1",
    "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
},
"v2_high_motion_lifecycle": {
    "MSDC_USE_LOW_DET": "0",
    "MSDC_USE_MOTION": "1",
    "MSDC_USE_TEMPLATE": "0",
    "MSDC_TEMPLATE_ENABLE": "0",
    "MSDC_USE_REACQUIRE": "0",
    "MSDC_USE_ROI_REDETECT": "0",
    "MSDC_REUSE_GUARD_ENABLE": "1",
    "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
},
"v2_high_reacquire_lifecycle": {
    "MSDC_USE_LOW_DET": "0",
    "MSDC_USE_MOTION": "0",
    "MSDC_USE_TEMPLATE": "0",
    "MSDC_TEMPLATE_ENABLE": "0",
    "MSDC_USE_REACQUIRE": "1",
    "MSDC_USE_ROI_REDETECT": "0",
    "MSDC_REUSE_GUARD_ENABLE": "1",
    "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
},
```

- [ ] **Step 2: Update README variant list**

Document that these variants are diagnostic only and not proposed defaults.

- [ ] **Step 3: Run lifecycle isolation**

Run:

```bash
RUN_ID="20260615_lifecycle_isolation_1000"
conda run -n ship_detect python tools/evaluation/msdc_dataset_benchmark.py \
  --dataset-root "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集" \
  --output-root "results/msdc_gap_analysis" \
  --run-id "$RUN_ID" \
  --mode ablation \
  --trackers botsort ocsort msdc_elt \
  --variants \
    v2_high_only_lifecycle \
    v2_high_motion_lifecycle \
    v2_high_reacquire_lifecycle \
    v2_candidate_topk_no_roi \
  --max-frames 1000 \
  --progress-interval 100 \
  --render \
  --render-class-source none \
  --run
```

Decision rules:

- If all high-only MS-DC variants are below OC-SORT / BoT-SORT on `AssA`, `IDF1`, or `IDSW`, lifecycle association is the core gap.
- If high-only MS-DC is close to OC-SORT but `v2_candidate_topk_no_roi` regresses, low-candidate integration is the gap.
- If motion improves `FN` but worsens `IDSW`, motion evidence is helping recall but harming identity.

## Task 5: Visual Timeline Review

**Purpose:** Confirm metric-level findings on frames, especially GT3 missed/fragmented segments.

- [ ] **Step 1: Open the three same-slice videos**

Inspect:

- `results/msdc_gap_analysis/20260615_gap_baseline_1000/visualizations/DJI_20250916100639_0001_V/botsort.mp4`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/visualizations/DJI_20250916100639_0001_V/ocsort.mp4`
- `results/msdc_gap_analysis/20260615_gap_baseline_1000/visualizations/DJI_20250916100639_0001_V/v2_candidate_topk_no_roi.mp4`

- [ ] **Step 2: Review specific missed segments**

Use the `missed_segments` and `missing_after_roi_segments` columns from diagnostics. Prioritize segments where:

- BoT-SORT or OC-SORT tracks continuously.
- MS-DC has `low_det_coverage` or `low_only_coverage` evidence.
- MS-DC output is missing or changes ID repeatedly.

Expected output: a short table with columns `gt_id`, `frame_range`, `botsort_status`, `ocsort_status`, `msdc_status`, `likely_cause`.

## Task 6: Optional Shared Detection Replay / Oracle Study

**Purpose:** Fully isolate tracker association from detector variability.

Only do this after Tasks 1-5 if the first-pass diagnostics cannot decide between detector and lifecycle causes.

- [ ] **Step 1: Add detection dump mode**

Create or extend a tool so a single detector pass writes per-frame detections to JSONL:

- `frame_idx`
- `high_det` boxes with class/confidence
- `low_det` boxes with class/confidence
- image size
- model/config metadata

Suggested output:

```text
results/msdc_gap_analysis/20260615_detection_replay_1000/detections/DJI_20250916100639_0001_V.jsonl
```

- [ ] **Step 2: Add replay input mode**

Add a replay runner that feeds the exact same detection JSONL into:

- OC-SORT
- BoT-SORT
- MS-DC high-only
- MS-DC candidate topK no ROI

The runner must not call the detector during replay.

- [ ] **Step 3: Run replay metrics**

Expected decision rules:

- If MS-DC still trails under identical detections, the gap is tracker association/lifecycle.
- If MS-DC closes the gap under identical detections, the gap is detector invocation/config/output preprocessing.
- If BoT-SORT still wins IDF1 but not MOTA, identity association is the differentiator.

## Formal Reporting Checklist

Do not call a run "formal" unless all items below exist in a new timestamped run directory:

- MOT txt for every tracker/variant.
- TrackEval summary with at least `MOTA`, `IDF1`, `IDSW`, `FN`, `FP`; include `HOTA`, `IDTP`, `IDFP`, `IDFN` when present.
- Annotated visualization MP4s with boxes, class labels, and track IDs under that run's `visualizations/` directory.
- Speed summary with processed frames, total elapsed time, mean frame time, and FPS.
- Stage timing file with read/decode, detection, low-threshold detection or ROI redetect, motion/template/lifecycle tracker, render, and write/export stages. Use `0` or `N/A` for inapplicable stages.
- Path manifest listing MOT txt, TrackEval summary, diagnostics JSONL/CSV, visualization MP4, and speed/timing files.
- `metadata/commands.jsonl` and `metadata/failures.csv`.

## Final Interpretation Template

Use this structure for the final write-up:

```markdown
## Conclusion

Primary cause: <lifecycle association | candidate filtering | detector recall | identity/ReID | mixed>.

Evidence:
- Same-slice metrics: <tracker table with HOTA/MOTA/IDF1/IDSW/FP/FN>.
- Per-GT diagnosis: <GT IDs and frame ranges>.
- Stage coverage: <high/low/low-only/output coverage deltas>.
- Candidate pool: <topK/min_conf/suppression/reuse guard evidence>.
- Speed cost: <FPS and mean stage times>.

Decision:
- Keep / reject `v2_candidate_topk_no_roi` as default.
- Next code change to test: <one change only>.
```
