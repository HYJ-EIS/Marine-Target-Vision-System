# UAV/USV Till1700 RGB-Only Baseline Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a formal RGB-only BoT-SORT baseline on the same UAV/USV till1700 RGB sequence used by the RGB/IR comparison.

**Architecture:** Create a new timestamped result directory, reuse the normalized 1700-frame GT from the previous piecewise RGB/IR run, export RGB-only MOT results, then run TrackEval, detection-level evaluation, visualization rendering, and output verification. No detector/tracker code changes are required.

**Tech Stack:** `conda run -n ship_detect`, `tools/evaluation/export_dual_modal_mot_results.py`, `tools/evaluation/motchallenge_eval.py`, `tools/evaluation/evaluate_detection_results.py`, `tools/evaluation/render_mot_video.py`, OpenCV verification.

---

### Task 1: Prepare Formal Run Directory

**Files:**
- Create: `results/rgb_ir_support_ablation/uav_usv_20250916100639_0001_till1700_rgb_only_<timestamp>/`
- Reuse/copy GT from: `results/rgb_ir_support_ablation/uav_usv_20250916100639_0001_till1700_piecewise_20260620_152316/gt/`

- [ ] **Step 1: Create a new run directory**

Run:
```bash
RUN_DIR="results/rgb_ir_support_ablation/uav_usv_20250916100639_0001_till1700_rgb_only_<timestamp>"
mkdir -p "$RUN_DIR"
```

Expected: directory exists and does not overwrite the previous RGB/IR formal run.

- [ ] **Step 2: Copy the normalized 1700-frame GT**

Run:
```bash
cp -a results/rgb_ir_support_ablation/uav_usv_20250916100639_0001_till1700_piecewise_20260620_152316/gt "$RUN_DIR/gt"
```

Expected: `$RUN_DIR/gt/DJI_20250916100639_0001_V/gt/gt.txt` and `seqinfo.ini` exist.

### Task 2: Export RGB-Only BoT-SORT MOT Results

**Files:**
- Create: `$RUN_DIR/trackers/botsort_rgb/data/DJI_20250916100639_0001_V.txt`
- Create: `$RUN_DIR/trackers/botsort_rgb/detections/DJI_20250916100639_0001_V.txt`
- Create: `$RUN_DIR/trackers/botsort_rgb/stats.json`
- Create: `$RUN_DIR/trackers/botsort_rgb/stage_timings.json`
- Create: `$RUN_DIR/trackers/botsort_rgb/diagnostics.csv`

- [ ] **Step 1: Run RGB-only export**

Run:
```bash
conda run -n ship_detect python tools/evaluation/export_dual_modal_mot_results.py \
  --visible-input "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集/DJI_20250916100639_0001_V.MP4" \
  --tracker botsort \
  --fusion rgb_only \
  --run-name botsort_rgb \
  --seq-name DJI_20250916100639_0001_V \
  --output-root "$RUN_DIR/trackers" \
  --rgb-high-conf 0.50 \
  --rgb-low-conf 0.30 \
  --max-frames 1700 \
  --progress-interval 100
```

Expected: export exits with code 0, `processed_frames` in `stats.json` is `1700`.

### Task 3: Run Formal Metrics

**Files:**
- Create: `$RUN_DIR/trackeval/motchallenge_summary.csv`
- Create: `$RUN_DIR/detection_eval_summary.csv`

- [ ] **Step 1: Run TrackEval**

Run:
```bash
conda run -n ship_detect python tools/evaluation/motchallenge_eval.py \
  --gt-root "$RUN_DIR/gt" \
  --trackers-root "$RUN_DIR/trackers" \
  --output-root "$RUN_DIR/trackeval" \
  --trackers botsort_rgb \
  --sequences DJI_20250916100639_0001_V
```

Expected: summary contains `MOTA`, `IDF1`, `IDSW`, `FN`, `FP`, `HOTA`, `IDTP`, `IDFP`, and `IDFN`.

- [ ] **Step 2: Run detection-level evaluation**

Run:
```bash
conda run -n ship_detect python tools/evaluation/evaluate_detection_results.py \
  --gt-file "$RUN_DIR/gt/DJI_20250916100639_0001_V/gt/gt.txt" \
  --det-file rgb_only="$RUN_DIR/trackers/botsort_rgb/detections/DJI_20250916100639_0001_V.txt" \
  --output-csv "$RUN_DIR/detection_eval_summary.csv"
```

Expected: summary contains precision, recall, F1, FP, FN, FP/frame, and small recall.

### Task 4: Render and Verify Visualization

**Files:**
- Create: `$RUN_DIR/visualizations/botsort_rgb.mp4`

- [ ] **Step 1: Render annotated MP4**

Run:
```bash
mkdir -p "$RUN_DIR/visualizations"
conda run -n ship_detect python tools/evaluation/render_mot_video.py \
  --input "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集/DJI_20250916100639_0001_V.MP4" \
  --mot-results "$RUN_DIR/trackers/botsort_rgb/data/DJI_20250916100639_0001_V.txt" \
  --output "$RUN_DIR/visualizations/botsort_rgb.mp4" \
  --file-type visible \
  --class-source none \
  --scale 0.5 \
  --max-frames 1700 \
  --progress-interval 300
```

Expected: MP4 exists in `visualizations/`.

- [ ] **Step 2: Verify output files**

Run:
```bash
conda run -n ship_detect python -c "import cv2, json; from pathlib import Path; run=Path('$RUN_DIR'); stats=json.loads((run/'trackers/botsort_rgb/stats.json').read_text()); assert stats['processed_frames']==1700; cap=cv2.VideoCapture(str(run/'visualizations/botsort_rgb.mp4')); assert cap.isOpened(); assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==1700; ok,_=cap.read(); assert ok; cap.release(); print('verified')"
```

Expected: prints `verified`.

### Task 5: Compare Against RGB/IR Runs

**Files:**
- Read: `results/rgb_ir_support_ablation/uav_usv_20250916100639_0001_till1700_piecewise_20260620_152316/trackeval/motchallenge_summary.csv`
- Read: `results/rgb_ir_support_ablation/uav_usv_20250916100639_0001_till1700_piecewise_20260620_152316/detection_eval_summary.csv`

- [ ] **Step 1: Build comparison table**

Compare `botsort_rgb` with:
- `botsort_rgb_ir_proben_score_only`
- `botsort_rgb_ir_proben_lost_reacquire_n3`
- `botsort_rgb_ir_presence_roi_redetect`

Expected: final report states whether IR support improves HOTA, MOTA, IDF1, IDSW, FP, FN, detection precision, detection recall, and speed.
