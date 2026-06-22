# Remove Disabled MS-DC Auxiliary Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the disabled MS-DC motion seed, ROI redetect, and template-lock runtime paths, and align experiment/report/test/docs surfaces with the v3 low-det-only tracker profile.

**Architecture:** `MSDCLifecycleTracker` becomes responsible only for high/low detector observations, low-observation budgeting, evidence update, output filtering, and diagnostics. Experiment runners and summaries keep only variants that are meaningful after removing motion/template/ROI. Speed/report/validation schemas remove ROI/motion/template phase fields instead of emitting permanent zero columns.

**Tech Stack:** Python 3.11, pytest, ONNX Runtime project code, `conda run -n ship_detect` command wrapper.

---

### Task 1: Remove Runtime Auxiliary Modules

**Files:**
- Delete: `target_module/image_detect_module/utils/motion_seed.py`
- Delete: `target_module/image_detect_module/utils/roi_redetect.py`
- Delete: `target_module/image_detect_module/utils/template_lock.py`
- Modify: `target_module/image_detect_module/utils/lifecycle_tracker.py`

- [ ] Remove imports for `MotionSeedGenerator`, `ROIRedetector`, and `TemplateLock`.
- [ ] Remove constructor fields `motion_seed`, `roi_redetector`, `template_lock`, `last_motion_debug`, `last_template_debug`, and `last_roi_redetect_debug`.
- [ ] In `update()`, remove ROI redetect, motion seed, template match, and template sync stages.
- [ ] Keep timing keys only for active stages: `low_filter_s`, `observation_build_s`, `evidence_update_s`, `output_s`, `debug_s`, `total_update_s`.
- [ ] Build observations from `high_det` and budgeted `low_det` only.
- [ ] Update debug payloads to remove `num_roi_low`, `num_motion`, `num_template`, `motion_debug`, `roi_redetect_debug`, `template_score`, `template_updated`, `template_match_box`, and `template_debug`.
- [ ] Remove helper methods that only supported deleted modules: `_template_enabled()`, `_compact_template_debug()`, `_compact_roi_redetect_debug()`, `_sync_active_templates()`, `_template_debug_summary()`, `_empty_template_debug()`.
- [ ] Keep output NMS, low observation budgeting, event writing, and active/lost lifecycle logic unchanged.

### Task 2: Clean Config Surface

**Files:**
- Modify: `target_module/image_detect_module/config.py`
- Modify: `test/test_msdc_detection.py`

- [ ] Remove MS-DC motion seed settings: `MSDC_MOTION_*`.
- [ ] Remove runtime toggles and parameters for `MSDC_USE_MOTION`, `MSDC_USE_TEMPLATE`, `MSDC_TEMPLATE_*`.
- [ ] Remove ROI redetect settings: `MSDC_USE_ROI_REDETECT`, `MSDC_ROI_REDETECT_*`, and `MSDC_ROI_REACQUIRE_CENTER_DIST`.
- [ ] Update config default tests so they assert the retained v3 defaults: shared high/low detection and low observation Top-K/min confidence.

### Task 3: Clean Experiment Variants And Summary Switches

**Files:**
- Modify: `tools/experiments/run_msdc_ablation.py`
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Modify: `test/test_msdc_experiment_summary.py`

- [ ] Remove ablation variants that depend on deleted paths: `Ours-full`, `Ours-lite-no-motion`, `Ours-no-template`, `Ours-no-reacquire`, `Ours-no-removed-guard`, `v2_template_off`, `v2_shared_det`, `v2_low_clean`, `v2_roi_*`, `v2_no_roi_redetect`, `v2_candidate_topk_roi_max1`.
- [ ] Keep variants that still exercise current v3 logic: low-det off, output age/size, low confirmation windows, real detection age, candidate Top-K, formal v3, debug off, reacquire interval/center settings.
- [ ] Remove ablation switch fields/defaults/overrides for motion/template/ROI.
- [ ] Update summary tests to assert retained variants and retained switch columns only.

### Task 4: Clean Speed And Formal Validation Fields

**Files:**
- Modify: `target_module/image_detect_module/constants.py`
- Modify: `tools/evaluation/msdc_speed_benchmark.py`
- Modify: `tools/evaluation/validate_msdc_formal_run.py`
- Modify: `tools/evaluation/detection_replay_benchmark.py`
- Modify: `test/test_msdc_constants.py`
- Modify: `test/test_msdc_speed_benchmark.py`
- Modify: `test/test_validate_msdc_formal_run.py`

- [ ] Remove `detector_calls_roi_redetect`, `mean_roi_redetect_ms`, `mean_msdc_roi_redetect_internal_ms`, `mean_msdc_motion_ms`, `mean_msdc_template_match_ms`, and `mean_msdc_template_sync_ms` from shared speed fields.
- [ ] Remove ROI nested-stage accounting from `msdc_speed_benchmark.py`.
- [ ] Remove `MSDC_USE_MOTION`, `MSDC_USE_TEMPLATE`, `MSDC_TEMPLATE_ENABLE`, `MSDC_USE_ROI_REDETECT`, and ROI env preservation from benchmark helper scripts.
- [ ] Update formal validation required speed fields to match the new schema.
- [ ] Update speed tests and validation tests to assert removed columns are absent.

### Task 5: Delete Obsolete Tests

**Files:**
- Delete: `test/test_msdc_motion_seed.py`
- Delete: `test/test_msdc_roi_redetect.py`
- Delete: `test/test_msdc_template_lock.py`
- Modify: `test/test_msdc_lifecycle_tracker.py`

- [ ] Delete direct unit tests for removed modules.
- [ ] Remove lifecycle tests that monkeypatch or assert motion/template/ROI behavior.
- [ ] Keep lifecycle tests for high/low detection, low observation budgeting, evidence transitions, debug events, output filtering, reset, and timing.

### Task 6: Update README And Verify

**Files:**
- Modify: `README.md`

- [ ] Remove deleted module rows from the MS-DC implementation table.
- [ ] Describe formal v3 as high/low detection with low-observation budgeting, without ROI/motion/template auxiliary modules.
- [ ] Update formal evaluation timing requirement to remove ROI/motion/template phase names.
- [ ] Run targeted tests:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_detection.py \
  test/test_msdc_lifecycle_tracker.py \
  test/test_msdc_constants.py \
  test/test_msdc_speed_benchmark.py \
  test/test_msdc_experiment_summary.py \
  test/test_validate_msdc_formal_run.py \
  -q
```

- [ ] Run stale-reference checks:

```bash
rg -n "motion_seed|roi_redetect|template_lock|MSDC_USE_MOTION|MSDC_USE_TEMPLATE|MSDC_TEMPLATE|MSDC_USE_ROI_REDETECT|MSDC_ROI_REDETECT|detector_calls_roi_redetect|mean_roi_redetect_ms|mean_msdc_motion_ms|mean_msdc_template" target_module tools test README.md
```

Expected: no active references outside historical docs or intentionally retained generic words in unrelated contexts.

### Self-Review

- Spec coverage: The six user requirements map to Tasks 1 through 6.
- Placeholder scan: No placeholder steps are used.
- Type consistency: The plan keeps `MSDCLifecycleTracker.update()` signature unchanged and only removes internal auxiliary observation sources.
