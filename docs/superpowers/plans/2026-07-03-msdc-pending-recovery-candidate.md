# MS-DC Pending Recovery Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a comparable MS-DC variant that delays new `public_id` allocation for confirming candidates near recent lost/removed signatures, then compare it with `msdc_v3` on full-length no-render tracking metrics and speed.

**Architecture:** Keep current `msdc_v3` as the baseline and add a `pending_recovery_candidate` variant. A ready `CANDIDATE` that is geometrically close to a class-compatible lost/removed signature enters `pending_recovery` instead of immediately becoming `ACTIVE` with a new `public_id`; after two consecutive near matches to the same old track it inherits that old `public_id`, otherwise it is allowed to confirm with a new ID once the conflict disappears. The implementation stores pending metadata on `EvidenceTrack` and hooks only the candidate-to-active transition path.

**Tech Stack:** Python dataclasses, MS-DC `EvidenceStateUpdater`, MOT replay benchmark, TrackEval summary, speed benchmark.

---

### Task 1: Pending Recovery Unit Tests

**Files:**
- Modify: `test/test_msdc_evidence_state.py`

- [x] **Step 1: Add a test config**

Add:

```python
class PendingRecoveryConfig(Config):
    MSDC_PENDING_RECOVERY_ENABLE = True
    MSDC_PENDING_RECOVERY_FRAMES = 2
    MSDC_PENDING_RECOVERY_REQUIRE_CLASS_MATCH = True
    MSDC_PENDING_RECOVERY_MAX_LOST_AGE = 40
    MSDC_PENDING_RECOVERY_REMOVED_GUARD_FRAMES = 80
    MSDC_REMOVED_GUARD_IOU_THRESH = 0.1
    MSDC_REMOVED_GUARD_CENTER_DIST = 20.0
    MSDC_REMOVED_RECOVERY_REQUIRE_CLASS_MATCH = True
    MSDC_CONFIRM_REQUIRE_HIGH_DET = True
    MSDC_CONFIRM_SCORE = 2.5
    MSDC_CONFIRM_MIN_HITS = 4
    MSDC_CONFIRM_MIN_REAL_DET_HITS = 4
    MSDC_CANDIDATE_MAX_AGE = 8
    MSDC_ASSOC_CENTER_DIST = 80.0
```

- [x] **Step 2: Add lost-inheritance test**

Create a ready candidate near a `LOST` track with `public_id=7`. On the first confirming frame assert `PENDING_RECOVERY_STARTED`, candidate remains `CANDIDATE`, and `public_id is None`. On the second close confirming frame assert candidate becomes `ACTIVE` with `public_id=7` and the old lost track is removed.

- [x] **Step 3: Add removed-release test**

Create a ready candidate near a removed signature with `public_id=4`. On the first confirming frame assert it remains pending. On the next confirming frame move the candidate outside the removed guard while still within association range; assert it becomes `ACTIVE` with a new `public_id`, not `4`.

- [x] **Step 4: Add class-mismatch test**

Create a ready candidate near a removed signature with a different class. Assert it confirms immediately with a new `public_id` and does not emit pending recovery events.

- [x] **Step 5: Verify RED**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py -q
```

Expected before implementation: failures because `EvidenceTrack.pending_recovery` and pending transition logic do not exist.

### Task 2: Pending Recovery Implementation

**Files:**
- Modify: `target_module/image_detect_module/config.py`
- Modify: `target_module/image_detect_module/utils/msdc_types.py`
- Modify: `target_module/image_detect_module/utils/evidence_state.py`

- [x] **Step 1: Add config defaults**

Add disabled-by-default knobs:

```python
MSDC_PENDING_RECOVERY_ENABLE = _env_bool("MSDC_PENDING_RECOVERY_ENABLE", False)
MSDC_PENDING_RECOVERY_FRAMES = _env_int("MSDC_PENDING_RECOVERY_FRAMES", 2)
MSDC_PENDING_RECOVERY_REQUIRE_CLASS_MATCH = _env_bool("MSDC_PENDING_RECOVERY_REQUIRE_CLASS_MATCH", True)
MSDC_PENDING_RECOVERY_MAX_LOST_AGE = _env_int("MSDC_PENDING_RECOVERY_MAX_LOST_AGE", 40)
MSDC_PENDING_RECOVERY_REMOVED_GUARD_FRAMES = _env_int("MSDC_PENDING_RECOVERY_REMOVED_GUARD_FRAMES", 80)
```

- [x] **Step 2: Add track metadata**

Add `pending_recovery: dict = field(default_factory=dict)` to `EvidenceTrack`, and include it in `track_to_dict()`.

- [x] **Step 3: Intercept candidate confirmation**

Pass `tracks` into `_transition_track()`. When a `CANDIDATE` would become `ACTIVE`, call a new helper that finds a class-compatible lost/removed conflict. If the same conflict has been observed for fewer than `MSDC_PENDING_RECOVERY_FRAMES`, keep the track as `CANDIDATE` and emit `PENDING_RECOVERY_STARTED` or `PENDING_RECOVERY_WAITING`.

- [x] **Step 4: Inherit or release**

If the same conflict persists long enough, set the candidate `public_id` to the old track's `public_id`, mark a lost source as internally removed, and emit `PENDING_RECOVERY_INHERITED_LOST` or `PENDING_RECOVERY_INHERITED_REMOVED`. If the conflict disappears or class is incompatible, clear `pending_recovery` and allow normal `CONFIRM_ACTIVE` with a new `public_id`.

- [x] **Step 5: Verify GREEN**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py -q
```

Expected: all evidence state tests pass.

### Task 3: Variant and Reporting Integration

**Files:**
- Modify: `tools/experiments/run_msdc_ablation.py`
- Modify: `tools/evaluation/msdc_speed_benchmark.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Modify: `test/test_msdc_paper_experiment_runner.py`
- Modify: `test/test_msdc_speed_benchmark.py`
- Modify: `README.md`

- [x] **Step 1: Add ablation variant**

Add:

```python
"pending_recovery_candidate": {
    **FORMAL_V3_ENV,
    "MSDC_PENDING_RECOVERY_ENABLE": "1",
    "MSDC_PENDING_RECOVERY_FRAMES": "2",
    "MSDC_PENDING_RECOVERY_REQUIRE_CLASS_MATCH": "1",
    "MSDC_PENDING_RECOVERY_MAX_LOST_AGE": "40",
    "MSDC_PENDING_RECOVERY_REMOVED_GUARD_FRAMES": "80",
}
```

- [x] **Step 2: Add speed config support**

Add pending recovery keys to `_FORMAL_V3_BOOL_KEYS` and `_FORMAL_V3_INT_KEYS` so `msdc_speed_benchmark.py --msdc-variant pending_recovery_candidate` applies them.

- [x] **Step 3: Add summary fields**

Add pending recovery keys to ablation switch fields/defaults so summary CSVs record whether the mechanism was enabled.

- [x] **Step 4: Update tests and README**

Assert the new variant exists and speed benchmark applies `MSDC_PENDING_RECOVERY_ENABLE=True`. Document the variant and its purpose in README.

- [x] **Step 5: Run tests**

Run:

```bash
conda run -n ship_detect pytest test -q
```

Expected: all tests pass.

### Task 4: Full-Length No-Render Comparison

**Output root:**
- `results/msdc_pending_recovery_candidate/20260703_full_no_render`

- [x] **Step 1: Run full replay**

Run:

```bash
conda run -n ship_detect python tools/evaluation/detection_replay_benchmark.py \
  --dataset-root /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --trackers msdc_elt \
  --variants msdc_v3 pending_recovery_candidate \
  --output-root results/msdc_pending_recovery_candidate \
  --run-id 20260703_full_no_render \
  --formal-frame-limit 5400 \
  --source-detection-cache-root results/msdc_paper_phase1/20260624_190710/main/main_full/detections \
  --progress-interval 500 \
  --render-class-source cache
```

This intentionally omits visualization.

- [x] **Step 2: Run speed for both variants**

Run `tools/evaluation/msdc_speed_benchmark.py` for both dataset roots and both variants with `--frames 5400 --trackers msdc_elt`, writing to `results/msdc_pending_recovery_candidate/20260703_full_no_render/speed`.

- [x] **Step 3: Summarize**

Report MOT metrics (`HOTA`, `IDF1`, `MOTA`, `IDSW`, `FP`, `FN`) and speed (`processed_frames`, `total_time_s`, `mean_fps`, `mean_latency_ms`, stage timing) for `msdc_v3` and `pending_recovery_candidate`.

## Execution Results

Artifacts:

- Replay summary: `results/msdc_pending_recovery_candidate/20260703_full_no_render/summary/replay_summary.csv`
- TrackEval summary: `results/msdc_pending_recovery_candidate/20260703_full_no_render/eval/motchallenge_summary.csv`
- Combined comparison: `results/msdc_pending_recovery_candidate/20260703_full_no_render/summary/pending_recovery_comparison.csv`
- Speed runs: `results/msdc_pending_recovery_candidate/20260703_full_no_render/speed/`
- MOT outputs: `results/msdc_pending_recovery_candidate/20260703_full_no_render/trackers/`
- Diagnostics: `results/msdc_pending_recovery_candidate/20260703_full_no_render/diagnostics/` and tracker `diagnostics/`

Tracking result:

```text
msdc_v3:                    HOTA=77.99963, IDF1=85.96941, MOTA=78.09394, IDSW=10, FP=1812, FN=4591
pending_recovery_candidate: HOTA=78.08894, IDF1=85.97095, MOTA=78.09394, IDSW=9,  FP=1812, FN=4592
delta:                      HOTA=+0.08931, IDF1=+0.00154, MOTA=+0.00000, IDSW=-1, FP=+0, FN=+1
```

Speed result:

```text
msdc_v3:                    frames=10172, FPS=9.996719, mean_latency_ms=100.032818
pending_recovery_candidate: frames=10172, FPS=9.884358, mean_latency_ms=101.169949
delta:                      FPS=-0.112361
```

Pending recovery event count:

```text
pending_recovery_candidate: total=2, started=1, inherited_lost=1, inherited_removed=0
```
