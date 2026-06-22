# Remove MS-DC Two-Pass Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make shared low/high MS-DC detection the only supported MS-DC detection path.

**Architecture:** `resolve_msdc_high_low_boxes()` will always run the low-threshold detector once and derive high-threshold boxes from that result. The old `MSDC_EXPORT_SHARE_LOW_HIGH_DET` switch, two-pass benchmarking comparison, ablation switch metadata, and tests for disabled sharing will be removed.

**Tech Stack:** Python, pytest via `conda run -n ship_detect`, existing MS-DC evaluation scripts.

---

### Task 1: Runtime Path

**Files:**
- Modify: `target_module/image_detect_module/utils/msdc_detection.py`
- Modify: `target_module/image_detect_module/config.py`
- Test: `test/test_msdc_detection.py`
- Test: `test/test_config_paths.py`

- [ ] Remove `Config.MSDC_EXPORT_SHARE_LOW_HIGH_DET`.
- [ ] Change `resolve_msdc_high_low_boxes()` so it always calls `run_msdc_low_threshold_detection()` and derives high boxes with `split_msdc_high_from_low_boxes()`.
- [ ] Remove tests that force `MSDC_EXPORT_SHARE_LOW_HIGH_DET=False`.
- [ ] Keep `MSDC_USE_LOW_DET` only if existing ablation tests still require it; do not reintroduce a high+low two-pass fallback.

### Task 2: Evaluation And Experiment Cleanup

**Files:**
- Modify: `tools/evaluation/msdc_speed_benchmark.py`
- Modify: `tools/experiments/run_msdc_ablation.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Test: `test/test_msdc_speed_benchmark.py`
- Test: `test/test_msdc_experiment_summary.py`
- Test: `test/test_msdc_paper_experiment_runner.py`

- [ ] Delete `--compare-msdc-share-low-high`, `run_msdc_share_low_high_comparison()`, and speed compare rows that mention `before_two_pass`.
- [ ] Remove `v2_shared_det` because sharing is no longer a variant.
- [ ] Remove `MSDC_EXPORT_SHARE_LOW_HIGH_DET` from ablation env dictionaries and summary switch tables.
- [ ] Update tests to assert shared detection is the only MS-DC path.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] Update README wording so MS-DC high boxes are always derived from one low-threshold detector pass.
- [ ] Run targeted tests:
  `conda run -n ship_detect pytest test/test_msdc_detection.py test/test_config_paths.py test/test_msdc_speed_benchmark.py test/test_msdc_experiment_summary.py test/test_msdc_paper_experiment_runner.py -q`
- [ ] Run stale scan:
  `rg -n "MSDC_EXPORT_SHARE_LOW_HIGH_DET|two-pass high\\+low|before_two_pass|after_shared_low_high|v2_shared_det|compare-msdc-share-low-high" target_module tools test README.md`
- [ ] Run `git diff --check`.
