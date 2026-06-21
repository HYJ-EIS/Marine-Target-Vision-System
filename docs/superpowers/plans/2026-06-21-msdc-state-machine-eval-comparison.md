# MS-DC State Machine Evaluation Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare MS-DC-ELT tracking behavior before and after the state-machine `assign_state` change on the first 1800 frames of `UAV_USV_MOT标注数据集` and `USV_MOT标注数据集`.

**Architecture:** Treat the current `MSDC` `HEAD` as "before" and the current uncommitted state-machine patch as "after". Run both versions in isolated worktrees against the same two dataset roots, capped to the same first 1800 frames per sequence, write timestamped outputs under `results/msdc_state_machine_compare/<run-id>/`, and produce a final metric/speed/path comparison report.

**Tech Stack:** Python 3.11 via `conda run -n ship_detect`, existing `tools/evaluation/msdc_dataset_benchmark.py`, `tools/evaluation/msdc_speed_benchmark.py`, TrackEval output CSVs, MOT txt, JSONL/CSV diagnostics, rendered MP4 visualizations.

---

## File Structure

- Read only: `/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集`
- Read only: `/home/hyj/Anti_Drone_Project/USV_MOT标注数据集`
- Read only: `tools/evaluation/msdc_dataset_benchmark.py`
- Read only: `tools/evaluation/msdc_speed_benchmark.py`
- Read only: `tools/evaluation/validate_msdc_formal_run.py`
- Read only: `target_module/image_detect_module/utils/evidence_state.py`
- Read only: `test/test_msdc_evidence_state.py`
- Create output directory: `results/msdc_state_machine_compare/<run-id>/`
- Create temporary worktrees: `/tmp/msdc_state_machine_compare_<run-id>/before` and `/tmp/msdc_state_machine_compare_<run-id>/after`
- Create temporary patch: `/tmp/msdc_state_machine_compare_<run-id>/state_machine_after.patch`
- Create temporary dataset wrappers: `/tmp/msdc_state_machine_compare_<run-id>/datasets/UAV_USV_MOT标注数据集` and `/tmp/msdc_state_machine_compare_<run-id>/datasets/USV_MOT标注数据集`
- Create report artifact: `results/msdc_state_machine_compare/<run-id>/summary/state_machine_before_after_report.md`
- Create comparison CSV: `results/msdc_state_machine_compare/<run-id>/summary/state_machine_before_after_metrics.csv`
- Create path manifest: `results/msdc_state_machine_compare/<run-id>/summary/path_manifest.csv`

## Definitions

- `before`: clean `MSDC` `HEAD` before the state-machine modification.
- `after`: `MSDC` `HEAD` plus the current working-tree diff for:
  - `README.md`
  - `target_module/image_detect_module/utils/evidence_state.py`
  - `test/test_msdc_evidence_state.py`
- Dataset roots:
  - Original UAV/USV annotations: `/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集`
  - Original USV annotations: `/home/hyj/Anti_Drone_Project/USV_MOT标注数据集`
  - Runtime dataset wrapper roots must be used for benchmark commands because the original annotation directories do not contain `原视频地址.txt`.
- Tracker/variant under comparison:
  - `msdc_elt`
  - `v2_low_clean`
- This is an 1800-frame partial comparison requested by the user, not a full-video formal run. Do not report it as full formal completion.
- Every before/after dataset run must use `--max-frames 1800` so both videos are evaluated on the same requested prefix.

---

### Task 1: Preflight Current Workspace and Datasets

**Files:**
- Read: `target_module/image_detect_module/utils/evidence_state.py`
- Read: `test/test_msdc_evidence_state.py`
- Read: `/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集`
- Read: `/home/hyj/Anti_Drone_Project/USV_MOT标注数据集`

- [ ] **Step 1: Confirm branch and working-tree diff scope**

Run:

```bash
git status -sb
git diff --name-only
```

Expected:

```text
## MSDC...origin/MSDC
README.md
target_module/image_detect_module/utils/evidence_state.py
test/test_msdc_evidence_state.py
```

- [ ] **Step 2: Verify the after patch tests still pass in the current workspace**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py test/test_msdc_lifecycle_tracker.py test/test_msdc_roi_redetect.py test/test_msdc_types.py -q
conda run -n ship_detect python -m py_compile target_module/image_detect_module/utils/evidence_state.py test/test_msdc_evidence_state.py
git diff --check
```

Expected:

```text
66 passed
```

The `py_compile` command and `git diff --check` must exit `0`.

- [ ] **Step 3: Confirm dataset directories exist**

Run:

```bash
ls -ld /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 /home/hyj/Anti_Drone_Project/USV_MOT标注数据集
```

Expected:

```text
/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集
/home/hyj/Anti_Drone_Project/USV_MOT标注数据集
```

### Task 2: Create Isolated Before/After Worktrees

**Files:**
- Create: `/tmp/msdc_state_machine_compare_<run-id>/before`
- Create: `/tmp/msdc_state_machine_compare_<run-id>/after`
- Create: `/tmp/msdc_state_machine_compare_<run-id>/state_machine_after.patch`

- [ ] **Step 1: Define run identifiers**

Run:

```bash
RUN_ID="state_machine_compare_$(date +%Y%m%d_%H%M%S)"
WORK_ROOT="/tmp/msdc_state_machine_compare_${RUN_ID}"
BEFORE_TREE="${WORK_ROOT}/before"
AFTER_TREE="${WORK_ROOT}/after"
PATCH_FILE="${WORK_ROOT}/state_machine_after.patch"
OUTPUT_ROOT="$(pwd)/results/msdc_state_machine_compare/${RUN_ID}"
mkdir -p "${WORK_ROOT}" "${OUTPUT_ROOT}/summary"
printf '%s\n' "${RUN_ID}" > "${OUTPUT_ROOT}/summary/run_id.txt"
```

Expected:

```text
results/msdc_state_machine_compare/<run-id>/summary/run_id.txt
```

- [ ] **Step 2: Save the after patch from the current working tree**

Run:

```bash
git diff -- README.md target_module/image_detect_module/utils/evidence_state.py test/test_msdc_evidence_state.py > "${PATCH_FILE}"
test -s "${PATCH_FILE}"
```

Expected: `test -s` exits `0`.

- [ ] **Step 3: Create clean worktrees from current `MSDC` HEAD**

Run:

```bash
git worktree add "${BEFORE_TREE}" HEAD
git worktree add "${AFTER_TREE}" HEAD
```

Expected:

```text
Preparing worktree
HEAD is now at ...
```

- [ ] **Step 4: Apply after patch only to after worktree**

Run:

```bash
git -C "${AFTER_TREE}" apply "${PATCH_FILE}"
git -C "${BEFORE_TREE}" status -sb
git -C "${AFTER_TREE}" status -sb
```

Expected:

```text
## ... clean for before
 M README.md
 M target_module/image_detect_module/utils/evidence_state.py
 M test/test_msdc_evidence_state.py
```

- [ ] **Step 5: Record code provenance**

Run:

```bash
git -C "${BEFORE_TREE}" rev-parse HEAD > "${OUTPUT_ROOT}/summary/before_commit.txt"
git -C "${AFTER_TREE}" rev-parse HEAD > "${OUTPUT_ROOT}/summary/after_base_commit.txt"
cp "${PATCH_FILE}" "${OUTPUT_ROOT}/summary/state_machine_after.patch"
```

Expected:

```text
before_commit.txt
after_base_commit.txt
state_machine_after.patch
```

- [ ] **Step 6: Create temporary dataset wrappers with video reference files**

Run:

```bash
DATASET_ROOT="${WORK_ROOT}/datasets"
UAV_DATASET="${DATASET_ROOT}/UAV_USV_MOT标注数据集"
USV_DATASET="${DATASET_ROOT}/USV_MOT标注数据集"
mkdir -p "${UAV_DATASET}" "${USV_DATASET}"
ln -sf /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集/gt.txt "${UAV_DATASET}/gt.txt"
ln -sf /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集/labels.txt "${UAV_DATASET}/labels.txt"
printf '%s\n' /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集/DJI_20250916100639_0001_V.MP4 > "${UAV_DATASET}/原视频地址.txt"
ln -sf /home/hyj/Anti_Drone_Project/USV_MOT标注数据集/gt.txt "${USV_DATASET}/gt.txt"
ln -sf /home/hyj/Anti_Drone_Project/USV_MOT标注数据集/labels.txt "${USV_DATASET}/labels.txt"
printf '%s\n' /home/hyj/Anti_Drone_Project/USV_MOT标注数据集/DJI_20250711140128_0002_V.MP4 > "${USV_DATASET}/原视频地址.txt"
printf '%s\n' "${UAV_DATASET}" "${USV_DATASET}" > "${OUTPUT_ROOT}/summary/runtime_dataset_roots.txt"
```

Expected:

```text
runtime_dataset_roots.txt
```

- [ ] **Step 7: Dry-run dataset command planning against wrappers**

Run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_dataset_benchmark.py \
  --dataset-root "${UAV_DATASET}" "${USV_DATASET}" \
  --trackers msdc_elt \
  --variants v2_low_clean \
  --max-frames 5 \
  --progress-interval 5 \
  --render \
  --render-class-source none
```

Expected:

```text
[EXPORT:v2_low_clean]
[EVAL]
[DIAG]
[RENDER:v2_low_clean]
[INFO] print-only mode; add --run to execute exports/evaluation/diagnostics.
```

Stop and report if either wrapper cannot resolve `gt.txt` or video path.

---

### Task 3: Smoke Test Before/After on 100 Frames

**Files:**
- Create: `results/msdc_state_machine_compare/<run-id>/smoke/before`
- Create: `results/msdc_state_machine_compare/<run-id>/smoke/after`

- [ ] **Step 1: Run before smoke**

Run:

```bash
conda run -n ship_detect python "${BEFORE_TREE}/tools/evaluation/msdc_dataset_benchmark.py" \
  --dataset-root "${UAV_DATASET}" "${USV_DATASET}" \
  --output-root "${OUTPUT_ROOT}/smoke" \
  --run-id before_smoke \
  --mode smoke \
  --commit-hash "$(cat "${OUTPUT_ROOT}/summary/before_commit.txt")" \
  --trackers msdc_elt \
  --variants v2_low_clean \
  --max-frames 100 \
  --progress-interval 50 \
  --render \
  --render-class-source none \
  --run
```

Expected:

```text
before_smoke/eval/motchallenge_summary.csv
before_smoke/visualizations/**/*.mp4
before_smoke/metadata/failures.csv
```

- [ ] **Step 2: Run after smoke**

Run:

```bash
conda run -n ship_detect python "${AFTER_TREE}/tools/evaluation/msdc_dataset_benchmark.py" \
  --dataset-root "${UAV_DATASET}" "${USV_DATASET}" \
  --output-root "${OUTPUT_ROOT}/smoke" \
  --run-id after_smoke \
  --mode smoke \
  --commit-hash "$(cat "${OUTPUT_ROOT}/summary/after_base_commit.txt")+state-machine-patch" \
  --trackers msdc_elt \
  --variants v2_low_clean \
  --max-frames 100 \
  --progress-interval 50 \
  --render \
  --render-class-source none \
  --run
```

Expected:

```text
after_smoke/eval/motchallenge_summary.csv
after_smoke/visualizations/**/*.mp4
after_smoke/metadata/failures.csv
```

- [ ] **Step 3: Check smoke failures**

Run:

```bash
cat "${OUTPUT_ROOT}/smoke/before_smoke/metadata/failures.csv"
cat "${OUTPUT_ROOT}/smoke/after_smoke/metadata/failures.csv"
```

Expected: both files contain only the CSV header, or no failure rows after the header.

Stop before 1800-frame comparison runs if smoke export, TrackEval, diagnostics, or render fails.

---

### Task 4: Run 1800-Frame MOT and Visualization Comparison

**Files:**
- Create: `results/msdc_state_machine_compare/<run-id>/partial_1800/before_1800`
- Create: `results/msdc_state_machine_compare/<run-id>/partial_1800/after_1800`

- [ ] **Step 1: Run before 1800-frame evaluation**

Run:

```bash
conda run -n ship_detect python "${BEFORE_TREE}/tools/evaluation/msdc_dataset_benchmark.py" \
  --dataset-root "${UAV_DATASET}" "${USV_DATASET}" \
  --output-root "${OUTPUT_ROOT}/partial_1800" \
  --run-id before_1800 \
  --mode main \
  --commit-hash "$(cat "${OUTPUT_ROOT}/summary/before_commit.txt")" \
  --trackers msdc_elt \
  --variants v2_low_clean \
  --max-frames 1800 \
  --progress-interval 500 \
  --render \
  --render-class-source none \
  --run
```

Expected:

```text
partial_1800/before_1800/trackers/v2_low_clean/data/*.txt
partial_1800/before_1800/eval/motchallenge_summary.csv
partial_1800/before_1800/diagnostics/**/*.csv
partial_1800/before_1800/visualizations/**/*.mp4
partial_1800/before_1800/metadata/commands.jsonl
partial_1800/before_1800/metadata/failures.csv
```

- [ ] **Step 2: Run after 1800-frame evaluation**

Run:

```bash
conda run -n ship_detect python "${AFTER_TREE}/tools/evaluation/msdc_dataset_benchmark.py" \
  --dataset-root "${UAV_DATASET}" "${USV_DATASET}" \
  --output-root "${OUTPUT_ROOT}/partial_1800" \
  --run-id after_1800 \
  --mode main \
  --commit-hash "$(cat "${OUTPUT_ROOT}/summary/after_base_commit.txt")+state-machine-patch" \
  --trackers msdc_elt \
  --variants v2_low_clean \
  --max-frames 1800 \
  --progress-interval 500 \
  --render \
  --render-class-source none \
  --run
```

Expected:

```text
partial_1800/after_1800/trackers/v2_low_clean/data/*.txt
partial_1800/after_1800/eval/motchallenge_summary.csv
partial_1800/after_1800/diagnostics/**/*.csv
partial_1800/after_1800/visualizations/**/*.mp4
partial_1800/after_1800/metadata/commands.jsonl
partial_1800/after_1800/metadata/failures.csv
```

- [ ] **Step 3: Confirm 1800-frame run failure logs are empty**

Run:

```bash
cat "${OUTPUT_ROOT}/partial_1800/before_1800/metadata/failures.csv"
cat "${OUTPUT_ROOT}/partial_1800/after_1800/metadata/failures.csv"
```

Expected: both files contain only the CSV header, or no failure rows after the header.

---

### Task 5: Run Speed and Stage-Timing Comparison

**Files:**
- Create: `results/msdc_state_machine_compare/<run-id>/speed/before_speed`
- Create: `results/msdc_state_machine_compare/<run-id>/speed/after_speed`

- [ ] **Step 1: Run before speed benchmark**

Run:

```bash
conda run -n ship_detect python "${BEFORE_TREE}/tools/evaluation/msdc_speed_benchmark.py" \
  --dataset-root "${USV_DATASET}" \
  --output-root "${OUTPUT_ROOT}/speed" \
  --run-id before_speed \
  --commit-hash "$(cat "${OUTPUT_ROOT}/summary/before_commit.txt")" \
  --frames 1000 \
  --trackers msdc_elt \
  --progress-interval 100
```

Expected:

```text
speed/before_speed/speed_results.csv
speed/before_speed/speed_timings.jsonl
```

- [ ] **Step 2: Run after speed benchmark**

Run:

```bash
conda run -n ship_detect python "${AFTER_TREE}/tools/evaluation/msdc_speed_benchmark.py" \
  --dataset-root "${USV_DATASET}" \
  --output-root "${OUTPUT_ROOT}/speed" \
  --run-id after_speed \
  --commit-hash "$(cat "${OUTPUT_ROOT}/summary/after_base_commit.txt")+state-machine-patch" \
  --frames 1000 \
  --trackers msdc_elt \
  --progress-interval 100
```

Expected:

```text
speed/after_speed/speed_results.csv
speed/after_speed/speed_timings.jsonl
```

- [ ] **Step 3: Confirm required speed fields exist**

Run:

```bash
head -n 1 "${OUTPUT_ROOT}/speed/before_speed/speed_results.csv"
head -n 1 "${OUTPUT_ROOT}/speed/after_speed/speed_results.csv"
```

Expected header includes:

```text
processed_frames,total_time_s,mean_latency_ms,mean_fps,mean_read_ms,mean_high_det_ms,mean_low_det_ms,mean_roi_redetect_ms,mean_tracker_ms,mean_render_ms,mean_write_ms
```

---

### Task 6: Build Before/After Comparison Tables

**Files:**
- Create: `results/msdc_state_machine_compare/<run-id>/summary/state_machine_before_after_metrics.csv`
- Create: `results/msdc_state_machine_compare/<run-id>/summary/path_manifest.csv`
- Create: `results/msdc_state_machine_compare/<run-id>/summary/state_machine_before_after_report.md`

- [ ] **Step 1: Generate comparison CSV and report**

Run:

```bash
conda run -n ship_detect python -c '
import csv
from pathlib import Path

root = Path("'"${OUTPUT_ROOT}"'")
summary = root / "summary"
before_metrics = root / "partial_1800" / "before_1800" / "eval" / "motchallenge_summary.csv"
after_metrics = root / "partial_1800" / "after_1800" / "eval" / "motchallenge_summary.csv"
before_speed = root / "speed" / "before_speed" / "speed_results.csv"
after_speed = root / "speed" / "after_speed" / "speed_results.csv"

def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

def key(row):
    return row.get("seq", row.get("sequence", "")) or row.get("Sequence", "") or "ALL"

metric_fields = ["HOTA", "MOTA", "IDF1", "IDSW", "FN", "FP", "IDTP", "IDFP", "IDFN"]
before_by_key = {key(row): row for row in rows(before_metrics)}
after_by_key = {key(row): row for row in rows(after_metrics)}
out_rows = []
for seq in sorted(set(before_by_key) | set(after_by_key)):
    b = before_by_key.get(seq, {})
    a = after_by_key.get(seq, {})
    row = {"section": "mot", "sequence": seq}
    for field in metric_fields:
        bv = b.get(field, "")
        av = a.get(field, "")
        row[f"before_{field}"] = bv
        row[f"after_{field}"] = av
        try:
            row[f"delta_{field}"] = f"{float(av) - float(bv):.6f}"
        except (TypeError, ValueError):
            row[f"delta_{field}"] = ""
    out_rows.append(row)

speed_fields = ["processed_frames", "total_time_s", "mean_latency_ms", "mean_fps", "mean_read_ms", "mean_high_det_ms", "mean_low_det_ms", "mean_roi_redetect_ms", "mean_tracker_ms", "mean_render_ms", "mean_write_ms"]
speed_before_rows = rows(before_speed)
speed_after_rows = rows(after_speed)
if speed_before_rows and speed_after_rows:
    b = speed_before_rows[0]
    a = speed_after_rows[0]
    row = {"section": "speed", "sequence": b.get("dataset", "USV_MOT标注数据集")}
    for field in speed_fields:
        bv = b.get(field, "")
        av = a.get(field, "")
        row[f"before_{field}"] = bv
        row[f"after_{field}"] = av
        try:
            row[f"delta_{field}"] = f"{float(av) - float(bv):.6f}"
        except (TypeError, ValueError):
            row[f"delta_{field}"] = ""
    out_rows.append(row)

all_fields = sorted({field for row in out_rows for field in row})
comparison_csv = summary / "state_machine_before_after_metrics.csv"
with comparison_csv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=all_fields)
    writer.writeheader()
    writer.writerows(out_rows)

manifest_rows = []
for label, pattern in [
    ("before_mot_txt", "partial_1800/before_1800/trackers/*/data/*.txt"),
    ("after_mot_txt", "partial_1800/after_1800/trackers/*/data/*.txt"),
    ("before_trackeval_summary", "partial_1800/before_1800/eval/motchallenge_summary.csv"),
    ("after_trackeval_summary", "partial_1800/after_1800/eval/motchallenge_summary.csv"),
    ("before_diagnostics", "partial_1800/before_1800/diagnostics/**/*.csv"),
    ("after_diagnostics", "partial_1800/after_1800/diagnostics/**/*.csv"),
    ("before_visualization", "partial_1800/before_1800/visualizations/**/*.mp4"),
    ("after_visualization", "partial_1800/after_1800/visualizations/**/*.mp4"),
    ("before_speed", "speed/before_speed/speed_results.csv"),
    ("after_speed", "speed/after_speed/speed_results.csv"),
    ("before_speed_timings", "speed/before_speed/speed_timings.jsonl"),
    ("after_speed_timings", "speed/after_speed/speed_timings.jsonl"),
]:
    for path in root.glob(pattern):
        manifest_rows.append({"label": label, "path": str(path)})
manifest_csv = summary / "path_manifest.csv"
with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["label", "path"])
    writer.writeheader()
    writer.writerows(manifest_rows)

report = summary / "state_machine_before_after_report.md"
report.write_text(
    "# MS-DC State Machine Before/After Comparison\n\n"
    f"- Output root: `{root}`\n"
    f"- Comparison CSV: `{comparison_csv}`\n"
    f"- Path manifest: `{manifest_csv}`\n"
    "- Required metrics: MOTA, IDF1, IDSW, FN, FP; HOTA/IDTP/IDFP/IDFN included when present.\n"
    "- Required speed fields: processed frames, total time, mean latency, mean FPS, and stage timings.\n",
    encoding="utf-8",
)
print(comparison_csv)
print(manifest_csv)
print(report)
'
```

Expected:

```text
state_machine_before_after_metrics.csv
path_manifest.csv
state_machine_before_after_report.md
```

- [ ] **Step 2: Inspect comparison CSV**

Run:

```bash
cat "${OUTPUT_ROOT}/summary/state_machine_before_after_metrics.csv"
```

Expected:

```text
section,sequence,...
mot,...
speed,...
```

The table must include before, after, and delta columns for at least `MOTA`, `IDF1`, `IDSW`, `FN`, `FP`, `processed_frames`, `total_time_s`, `mean_latency_ms`, and `mean_fps`.

---

### Task 7: Validate 1800-Frame Completeness and Prepare Final Report

**Files:**
- Read: `results/msdc_state_machine_compare/<run-id>/summary/state_machine_before_after_metrics.csv`
- Read: `results/msdc_state_machine_compare/<run-id>/summary/path_manifest.csv`
- Modify: `results/msdc_state_machine_compare/<run-id>/summary/state_machine_before_after_report.md`

- [ ] **Step 1: Verify required 1800-frame artifacts exist**

Run:

```bash
find "${OUTPUT_ROOT}/partial_1800" -path "*trackers*data*.txt" -type f -size +0
find "${OUTPUT_ROOT}/partial_1800" -path "*eval/motchallenge_summary.csv" -type f -size +0
find "${OUTPUT_ROOT}/partial_1800" -path "*diagnostics*.csv" -type f -size +0
find "${OUTPUT_ROOT}/partial_1800" -path "*visualizations*.mp4" -type f -size +0
find "${OUTPUT_ROOT}/speed" -name "speed_results.csv" -type f -size +0
find "${OUTPUT_ROOT}/speed" -name "speed_timings.jsonl" -type f -size +0
```

Expected: each command prints at least one path.

- [ ] **Step 2: Verify speed/stage timing fields are non-empty**

Run:

```bash
conda run -n ship_detect python -c '
import csv
from pathlib import Path
root = Path("'"${OUTPUT_ROOT}"'")
required = ["processed_frames", "total_time_s", "mean_latency_ms", "mean_fps", "mean_read_ms", "mean_high_det_ms", "mean_low_det_ms", "mean_roi_redetect_ms", "mean_tracker_ms", "mean_render_ms", "mean_write_ms"]
for path in [root / "speed" / "before_speed" / "speed_results.csv", root / "speed" / "after_speed" / "speed_results.csv"]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    assert rows, path
    missing = [field for field in required if not rows[0].get(field)]
    assert not missing, (path, missing)
    print(path, "OK")
'
```

Expected:

```text
before_speed/speed_results.csv OK
after_speed/speed_results.csv OK
```

- [ ] **Step 3: Append concrete output paths to the report**

Run:

```bash
{
  printf '\n## Output Paths\n\n'
  cat "${OUTPUT_ROOT}/summary/path_manifest.csv"
  printf '\n## Comparison Table\n\n'
  cat "${OUTPUT_ROOT}/summary/state_machine_before_after_metrics.csv"
} >> "${OUTPUT_ROOT}/summary/state_machine_before_after_report.md"
```

Expected:

```text
state_machine_before_after_report.md contains Output Paths and Comparison Table
```

- [ ] **Step 4: Final status check**

Run:

```bash
git status -sb
git -C "${BEFORE_TREE}" status -sb
git -C "${AFTER_TREE}" status -sb
```

Expected:

```text
Current repo still has only the planned state-machine changes plus this plan file if it has not been committed.
Before worktree is clean.
After worktree has the state-machine patch applied.
```

---

## Final Response Requirements

When reporting completion, include:

- Run ID and output root.
- Before code version: commit hash from `before_commit.txt`.
- After code version: base commit hash plus `state_machine_after.patch`.
- MOT metrics summary for each dataset: `MOTA`, `IDF1`, `IDSW`, `FN`, `FP`; include `HOTA`, `IDTP`, `IDFP`, `IDFN` if present.
- Speed summary: total processed frames, total time, mean latency, mean FPS.
- Stage timings: video read/decode, high detection, low detection, ROI redetect, tracker update, render, write/export; write `0` or `N/A` only if the CSV records that value.
- Output path list for MOT txt, TrackEval summary, diagnostics CSV/JSONL, visualization MP4, speed CSV, and timing JSONL.
- State whether smoke was run before the 1800-frame comparison.
- State any failures from `metadata/failures.csv`; if failures exist, do not call the 1800-frame comparison complete.

## Cleanup

After the user confirms the comparison artifacts are no longer needed in `/tmp`, remove only the temporary worktrees:

```bash
git worktree remove "${BEFORE_TREE}"
git worktree remove "${AFTER_TREE}"
git worktree prune
```

Do not remove `results/msdc_state_machine_compare/<run-id>/`.
