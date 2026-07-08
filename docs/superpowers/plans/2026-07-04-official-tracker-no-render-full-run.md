# Official Tracker No-Render Full Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch paper main comparisons to official OC-SORT, official BoT-SORT, ByteTrack, and MS-DC-ELT v3, then execute a full-video no-render experiment plan without slice or qualitative analysis.

**Architecture:** Keep the tracker key contract centralized in `target_module/image_detect_module/constants.py`. The formal replay and speed scripts should consume those constants, while summary generation should report official baseline labels. The no-render run will use `detection_replay_benchmark.py` and `msdc_speed_benchmark.py` directly so slice metrics and visualization are not produced.

**Tech Stack:** Python, pytest, MOTChallenge/TrackEval replay scripts, Conda environment `ship_detect`.

---

### Task 1: Update Official Baseline Contracts

**Files:**
- Modify: `test/test_msdc_constants.py`
- Modify: `test/test_msdc_paper_experiment_runner.py`
- Modify: `test/test_detection_replay_benchmark.py`
- Modify: `test/test_msdc_speed_benchmark.py`
- Modify: `target_module/image_detect_module/constants.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`

- [ ] **Step 1: Write failing tests for official paper tracker choices**

Update tests so `PAPER_TRACKER_CHOICES` and `DATASET_EXPORT_TRACKER_CHOICES` are:

```python
("official_ocsort", "official_botsort", "bytetrack", "msdc_elt")
```

Update command expectations so main and speed commands use:

```python
["official_ocsort", "official_botsort", "bytetrack", "msdc_elt"]
```

Update replay output expectations to use:

```python
"official_ocsort_replay"
"official_botsort_replay"
"bytetrack_replay"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_constants.py \
  test/test_msdc_paper_experiment_runner.py \
  test/test_detection_replay_benchmark.py \
  test/test_msdc_speed_benchmark.py \
  -q
```

Expected: failures showing old `ocsort` / `botsort` paper choices are still active.

- [ ] **Step 3: Implement official baseline constants and summary labels**

Change `target_module/image_detect_module/constants.py`:

```python
PAPER_TRACKER_CHOICES = ("official_ocsort", "official_botsort", "bytetrack", "msdc_elt")
DATASET_EXPORT_TRACKER_CHOICES = ("official_ocsort", "official_botsort", "bytetrack", "msdc_elt")
SUMMARY_MAIN_TRACKERS = {
    "official_ocsort",
    "official_botsort",
    "bytetrack",
    "msdc_elt",
    MAIN_MSDC_TRACKER,
}
```

Add official method labels while keeping legacy labels:

```python
"official_ocsort": "FFCA-YOLO + OC-SORT (official)",
"official_botsort": "FFCA-YOLO + BoT-SORT (official)",
```

Update summary baseline assumptions in `tools/evaluation/msdc_experiment_summary.py` from lite `ocsort` / `botsort` labels to official labels.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same pytest command from Step 2. Expected: all selected tests pass.

### Task 2: Document the Experiment Contract

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Document that paper main comparison uses official OC-SORT, official BoT-SORT, ByteTrack, and MS-DC-ELT v3. Also document that no-render full-video runs are diagnostic runs and do not satisfy the formal visualization requirement.

- [ ] **Step 2: Verify focused docs-related tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_constants.py -q
```

Expected: pass.

### Task 3: Execute Full-Video No-Render Experiments

**Files:**
- Produce results under `results/msdc_official_no_render/<timestamp>/`

- [ ] **Step 1: Run main full-video replay without visualization**

Run:

```bash
conda run -n ship_detect python tools/evaluation/detection_replay_benchmark.py \
  --dataset-root /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --trackers official_ocsort official_botsort bytetrack msdc_elt \
  --variants msdc_v3 \
  --output-root results/msdc_official_no_render \
  --run-id <timestamp>_main_full_no_render \
  --formal-frame-limit 0 \
  --max-frames 999999 \
  --progress-interval 500 \
  --render-class-source cache
```

Expected outputs: MOT txt, detection cache, diagnostics CSV/JSONL, TrackEval summary, replay summary. No visualization MP4 is expected.

- [ ] **Step 2: Run selected ablation replay without visualization**

Run:

```bash
conda run -n ship_detect python tools/evaluation/detection_replay_benchmark.py \
  --dataset-root /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --trackers msdc_elt \
  --variants msdc_v3 no_low_candidate no_direct_reacquire no_low_inheritance removed_recovery_off hits_only_no_evidence no_output_nms no_output_real_det_age_gate low_budget_off low_budget_topk16 low_budget_topk64 \
  --output-root results/msdc_official_no_render \
  --run-id <timestamp>_ablation_full_no_render \
  --formal-frame-limit 0 \
  --max-frames 999999 \
  --source-detection-cache-root results/msdc_official_no_render/<timestamp>_main_full_no_render/detections \
  --progress-interval 500 \
  --render-class-source cache
```

Expected outputs: ablation MOT txt, diagnostics, TrackEval summary. No slice metrics and no visualization MP4.

- [ ] **Step 3: Run speed benchmark without visualization**

Run one speed benchmark per dataset root:

```bash
conda run -n ship_detect python tools/evaluation/msdc_speed_benchmark.py \
  --dataset-root <dataset_root> \
  --output-root results/msdc_official_no_render/<timestamp>_speed \
  --run-id <seq_name>_full \
  --frames 999999 \
  --trackers official_ocsort official_botsort bytetrack msdc_elt \
  --progress-interval 500
```

Expected outputs: `speed_results.csv` and `speed_timings.jsonl`.

- [ ] **Step 4: Report no-render run outputs**

Report paths for replay summary, TrackEval summary, diagnostics, MOT txt roots, and speed CSVs. State explicitly that visualization and slice/qualitative outputs were intentionally skipped, so this is a full-video no-render run rather than a complete formal run under the README formal checklist.
