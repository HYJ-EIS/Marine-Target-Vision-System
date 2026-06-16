# MS-DC Replay And Speed Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 300-frame same-detection replay on both MOT datasets and produce before/after speed comparison for MS-DC shared high/low detection.

**Architecture:** Add one focused replay utility that caches detector boxes to JSONL, replays those boxes through OC-SORT, BoT-SORT, and MS-DC lifecycle, then evaluates with existing MOTChallenge helpers. Extend the speed benchmark with a comparison mode that runs MS-DC twice with `MSDC_EXPORT_SHARE_LOW_HIGH_DET=0/1` and writes a compact delta CSV.

**Tech Stack:** Python, OpenCV, existing `MultiObjectTracker`, `MSDCLifecycleTracker`, TrackEval wrapper, pytest under `conda run -n ship_detect`.

---

### Task 1: Detection Replay Utility

**Files:**
- Create: `tools/evaluation/detection_replay_benchmark.py`
- Test: `test/test_detection_replay_benchmark.py`

- [ ] Write tests for JSONL box cache read/write, MOT row export, and 300-frame GT normalization.
- [ ] Implement cache generation with `processor.process_frame(frame, file_type)` once per frame.
- [ ] Implement replay for `ocsort`, `botsort`, and `msdc_elt`; MS-DC replay uses cached boxes as `high_boxes` and an empty `low_boxes` list to isolate tracker behavior.
- [ ] Run TrackEval through existing `run_motchallenge_eval`.
- [ ] Write `summary/replay_summary.csv` and per-tracker diagnostics.

### Task 2: Shared High/Low Speed Compare

**Files:**
- Modify: `tools/evaluation/msdc_speed_benchmark.py`
- Modify: `test/test_msdc_speed_benchmark.py`

- [ ] Add CLI flags `--compare-msdc-share-low-high` and `--compare-label-prefix`.
- [ ] Run two MS-DC speed passes: before with `MSDC_EXPORT_SHARE_LOW_HIGH_DET=False`, after with `True`.
- [ ] Write `speed_compare.csv` with total time, FPS, latency, detector call counts, and delta/speedup columns.
- [ ] Keep existing single-run speed benchmark behavior unchanged.

### Task 3: Docs And Verification

**Files:**
- Modify: `README.md`

- [ ] Document the replay command and speed comparison command.
- [ ] Run focused pytest for the new and changed tests.
- [ ] Run replay on both requested dataset videos with `--max-frames 300`.
- [ ] Run speed comparison on both datasets and report before/after speed.
