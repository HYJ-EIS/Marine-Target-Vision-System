# MS-DC-ELT V2 Formal Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the remaining MS-DC-ELT v2 reduction/restructure/tuning work and produce formal TrackEval, speed, diagnostics, and visualization outputs for the v2 experiment matrix.

**Architecture:** Keep algorithm changes confined to existing MS-DC-ELT modules and make all tuning values configurable through `Config` environment overrides. Extend the existing dataset benchmark, speed benchmark, and experiment summary tools so the same run directory contains MOT txt, TrackEval metrics, diagnostics, rendered videos, speed timing, and a validation manifest. Run smoke tests before full runs and do not claim formal completion until the artifact validator passes.

**Tech Stack:** Python 3.11, conda env `ship_detect`, pytest, OpenCV, existing ONNX detector, existing TrackEval wrapper, existing MS-DC-ELT benchmark scripts.

---

## File Structure

- Modify `target_module/image_detect_module/config.py`
  - Make remaining MS-DC-ELT tuning knobs environment-overridable.
  - Tighten ROI redetect defaults for v2.
  - Keep formal defaults aligned with TemplateLock off and shared high/low detection on.

- Modify `target_module/image_detect_module/utils/roi_redetect.py`
  - Wrap ROI detector calls in a processor stage context when available.
  - Preserve the existing rule that ROI boxes only support active/lost tracks and do not spawn new candidates.

- Modify `target_module/image_detect_module/utils/evidence_state.py`
  - Add scale-adaptive reacquire gate helpers.
  - Keep one-to-one lost/candidate matching; add explicit debug fields showing conflicts and gating thresholds.
  - Ensure template-only observations cannot reacquire lost tracks.

- Modify `tools/evaluation/msdc_speed_benchmark.py`
  - Add `detector_calls_roi_redetect` and `mean_roi_redetect_ms` to speed CSV.
  - Keep `detector_calls_tracker_update` as non-detector lifecycle overhead after subtracting ROI detector calls.

- Modify `tools/experiments/run_msdc_ablation.py`
  - Add v2 variants for template off, shared detection, low clean, output recall, candidate grid, ROI budget, and reacquire grid.

- Modify `tools/evaluation/run_msdc_paper_experiments.py`
  - Add `--ablation-variants` so formal runs can target v2 groups without editing code.
  - Add `--speed-trackers` only if needed; keep default main trackers unchanged.

- Create `tools/evaluation/validate_msdc_formal_run.py`
  - Validate formal artifact requirements before reporting completion.
  - Check MOT txt, TrackEval summary, visualizations, diagnostics, speed CSV, timing JSONL, and path manifest.

- Modify `tools/evaluation/msdc_experiment_summary.py`
  - Include new v2 variants and artifact validator output in report/docs.
  - Include `detector_calls_roi_redetect` and `mean_roi_redetect_ms` in speed results.

- Modify `README.md`
  - Document v2 ROI budget defaults, v2 experiment variants, formal run commands, and required validation command.

- Tests:
  - Modify `test/test_config_paths.py`
  - Modify `test/test_msdc_roi_redetect.py`
  - Modify `test/test_msdc_evidence_state.py`
  - Modify `test/test_msdc_speed_benchmark.py`
  - Modify `test/test_msdc_paper_experiment_runner.py`
  - Modify `test/test_msdc_experiment_summary.py`
  - Create `test/test_validate_msdc_formal_run.py`

---

### Task 1: Make V2 ROI Budget And Tuning Knobs Configurable

**Files:**
- Modify: `target_module/image_detect_module/config.py`
- Modify: `test/test_config_paths.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing config tests**

Append this test to `test/test_config_paths.py`:

```python
def test_msdc_v2_roi_budget_defaults_are_low_frequency():
    assert Config.MSDC_ROI_REDETECT_MAX_TRACKS == 2
    assert Config.MSDC_ROI_REDETECT_ACTIVE_INTERVAL == 8
    assert Config.MSDC_ROI_REDETECT_LOST_INTERVAL == 3
    assert Config.MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI == 1


def test_msdc_v2_tuning_knobs_are_environment_backed(monkeypatch):
    import importlib
    import target_module.image_detect_module.config as config_module

    monkeypatch.setenv("MSDC_CONFIRM_MIN_HITS", "3")
    monkeypatch.setenv("MSDC_CONFIRM_SCORE", "2.0")
    monkeypatch.setenv("MSDC_CANDIDATE_MAX_AGE", "8")
    monkeypatch.setenv("MSDC_REACQUIRE_INTERVAL", "1")
    monkeypatch.setenv("MSDC_REACQUIRE_SCORE", "1.2")
    monkeypatch.setenv("MSDC_REACQUIRE_CENTER_DIST", "240")

    reloaded = importlib.reload(config_module)
    try:
        assert reloaded.Config.MSDC_CONFIRM_MIN_HITS == 3
        assert reloaded.Config.MSDC_CONFIRM_SCORE == 2.0
        assert reloaded.Config.MSDC_CANDIDATE_MAX_AGE == 8
        assert reloaded.Config.MSDC_REACQUIRE_INTERVAL == 1
        assert reloaded.Config.MSDC_REACQUIRE_SCORE == 1.2
        assert reloaded.Config.MSDC_REACQUIRE_CENTER_DIST == 240.0
    finally:
        importlib.reload(config_module)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_config_paths.py::test_msdc_v2_roi_budget_defaults_are_low_frequency test/test_config_paths.py::test_msdc_v2_tuning_knobs_are_environment_backed -q
```

Expected: FAIL because ROI defaults are still `4/3/3/2`, and some tuning knobs are not environment-backed.

- [ ] **Step 3: Update `Config`**

In `target_module/image_detect_module/config.py`, replace these assignments:

```python
    MSDC_CONFIRM_SCORE = 2.5
    MSDC_CONFIRM_MIN_HITS = 4
```

with:

```python
    MSDC_CONFIRM_SCORE = _env_float("MSDC_CONFIRM_SCORE", 2.5)
    MSDC_CONFIRM_MIN_HITS = _env_int("MSDC_CONFIRM_MIN_HITS", 4)
```

Replace:

```python
    MSDC_CANDIDATE_MAX_AGE = 5
```

with:

```python
    MSDC_CANDIDATE_MAX_AGE = _env_int("MSDC_CANDIDATE_MAX_AGE", 5)
```

Replace:

```python
    MSDC_REACQUIRE_INTERVAL = 5
    MSDC_REACQUIRE_SCORE = 1.5
    MSDC_REACQUIRE_IOU_THRESH = 0.05
    MSDC_REACQUIRE_CENTER_DIST = 160.0
```

with:

```python
    MSDC_REACQUIRE_INTERVAL = _env_int("MSDC_REACQUIRE_INTERVAL", 5)
    MSDC_REACQUIRE_SCORE = _env_float("MSDC_REACQUIRE_SCORE", 1.5)
    MSDC_REACQUIRE_IOU_THRESH = _env_float("MSDC_REACQUIRE_IOU_THRESH", 0.05)
    MSDC_REACQUIRE_CENTER_DIST = _env_float("MSDC_REACQUIRE_CENTER_DIST", 160.0)
```

Replace ROI defaults:

```python
    MSDC_ROI_REDETECT_ACTIVE_INTERVAL = _env_int("MSDC_ROI_REDETECT_ACTIVE_INTERVAL", 3)
    MSDC_ROI_REDETECT_LOST_INTERVAL = _env_int("MSDC_ROI_REDETECT_LOST_INTERVAL", 3)
    MSDC_ROI_REDETECT_MAX_TRACKS = _env_int("MSDC_ROI_REDETECT_MAX_TRACKS", 4)
    MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI = _env_int("MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI", 2)
```

with:

```python
    MSDC_ROI_REDETECT_ACTIVE_INTERVAL = _env_int("MSDC_ROI_REDETECT_ACTIVE_INTERVAL", 8)
    MSDC_ROI_REDETECT_LOST_INTERVAL = _env_int("MSDC_ROI_REDETECT_LOST_INTERVAL", 3)
    MSDC_ROI_REDETECT_MAX_TRACKS = _env_int("MSDC_ROI_REDETECT_MAX_TRACKS", 2)
    MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI = _env_int("MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI", 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run -n ship_detect pytest test/test_config_paths.py -q
```

Expected: PASS.

- [ ] **Step 5: Update README**

In `README.md`, update the MS-DC-ELT ROI line so it states:

```markdown
- MS-DC-ELT ROI 重检：`MSDC_USE_ROI_REDETECT = True`，`MSDC_ROI_REDETECT_LOW_CONF = 0.12`，`MSDC_ROI_REDETECT_ACTIVE_INTERVAL = 8`，`MSDC_ROI_REDETECT_LOST_INTERVAL = 3`，`MSDC_ROI_REDETECT_MAX_TRACKS = 2`，`MSDC_ROI_REDETECT_SEARCH_SCALE = 4.0`，`MSDC_ROI_REDETECT_UPSCALE = 2.0`，`MSDC_ROI_REDETECT_EXISTING_IOU = 0.5`，`MSDC_ROI_REDETECT_MIN_BOX_SIZE = 8`，`MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI = 1`；ROI 重检作为低频重捕工具，只针对已有 active/lost track，过滤 tiny / existing-overlap box，且不单独生成新 candidate
```

- [ ] **Step 6: Commit**

Run:

```bash
git add target_module/image_detect_module/config.py test/test_config_paths.py README.md
git commit -m "config: tighten msdc v2 roi budget defaults"
```

Expected: commit succeeds.

---

### Task 2: Account ROI Detector Calls Separately In Speed Benchmark

**Files:**
- Modify: `target_module/image_detect_module/utils/roi_redetect.py`
- Modify: `tools/evaluation/msdc_speed_benchmark.py`
- Modify: `test/test_msdc_roi_redetect.py`
- Modify: `test/test_msdc_speed_benchmark.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing ROI stage test**

Append this test to `test/test_msdc_roi_redetect.py`:

```python
def test_roi_redetect_uses_processor_stage_context_when_available():
    from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, TrackState
    from target_module.image_detect_module.utils.roi_redetect import ROIRedetector

    class StageProcessor:
        def __init__(self):
            self.stage = "unspecified"
            self.calls = []

        def use_stage(self, stage):
            processor = self

            class _Context:
                def __enter__(self):
                    self.previous = processor.stage
                    processor.stage = stage
                    return processor

                def __exit__(self, exc_type, exc, tb):
                    processor.stage = self.previous
                    return False

            return _Context()

        def process_frame(self, frame, file_type, conf_override=None):
            self.calls.append((self.stage, frame.shape, file_type, conf_override))
            return {"boxes": [{"x": 2, "y": 2, "w": 10, "h": 10, "confidence": 0.8, "class": "UAV"}]}

    class ROIStageConfig(Config):
        MSDC_USE_ROI_REDETECT = True
        MSDC_ROI_REDETECT_ACTIVE_INTERVAL = 1
        MSDC_ROI_REDETECT_MAX_TRACKS = 1
        MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI = 1
        MSDC_ROI_REDETECT_MIN_CROP_SIZE = 8
        MSDC_ROI_REDETECT_EXISTING_IOU = 1.1

    processor = StageProcessor()
    redetector = ROIRedetector(ROIStageConfig, processor=processor)
    track = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.ACTIVE,
        box=[10, 10, 30, 30],
        evidence_score=3.0,
        hits=4,
        misses=0,
        age=4,
        last_seen=0,
        last_real_det_frame=0,
        real_det_hits=4,
        class_id=2,
        class_name="UAV",
    )

    redetector.update(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        tracks=[track],
        frame_idx=1,
        file_type="visible",
        existing_boxes=[],
    )

    assert processor.calls
    assert {call[0] for call in processor.calls} == {"roi_redetect"}
    assert processor.stage == "unspecified"
```

- [ ] **Step 2: Write failing speed CSV field test**

In `test/test_msdc_speed_benchmark.py`, update `fake_run_tracker_benchmark()` in `test_run_speed_benchmark_writes_zero_frame_outputs` so returned rows include:

```python
            "detector_calls_roi_redetect": 0,
            "mean_roi_redetect_ms": "0.000000",
```

Then add assertions after reading rows:

```python
    assert "detector_calls_roi_redetect" in rows[0]
    assert "mean_roi_redetect_ms" in rows[0]
```

Also update `test_msdc_shared_low_high_detection_uses_one_detector_call` to assert:

```python
    assert row["detector_calls_roi_redetect"] == 0
    assert row["mean_roi_redetect_ms"] == "0.000000"
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_roi_redetect.py::test_roi_redetect_uses_processor_stage_context_when_available test/test_msdc_speed_benchmark.py::test_run_speed_benchmark_writes_zero_frame_outputs test/test_msdc_speed_benchmark.py::test_msdc_shared_low_high_detection_uses_one_detector_call -q
```

Expected: FAIL because ROI does not use a dedicated processor stage and speed rows do not expose ROI fields.

- [ ] **Step 4: Implement ROI processor stage context**

In `target_module/image_detect_module/utils/roi_redetect.py`, find the place where `self.processor.process_frame(...)` is called. Replace the direct call with:

```python
        process_context = getattr(self.processor, "use_stage", None)
        if callable(process_context):
            with process_context("roi_redetect"):
                stats = self.processor.process_frame(crop, file_type, conf_override=low_conf)
        else:
            stats = self.processor.process_frame(crop, file_type, conf_override=low_conf)
```

Use the existing local variable names for `crop`, `file_type`, and `low_conf`. If the file currently uses different variable names, keep the same expression values and only wrap the call.

- [ ] **Step 5: Add speed fields**

In `tools/evaluation/msdc_speed_benchmark.py`, add these to `SPEED_FIELDS` immediately after `detector_calls_tracker_update`:

```python
    "detector_calls_roi_redetect",
```

and immediately after `mean_low_det_ms`:

```python
    "mean_roi_redetect_ms",
```

In `_run_tracker_benchmark`, when building the row, add:

```python
        roi_calls = int(processor.call_counts.get("roi_redetect", 0))
        roi_seconds = float(processor.stage_seconds.get("roi_redetect", 0.0))
        tracker_update_calls = int(processor.call_counts.get("tracker_update", 0))
        tracker_update_detector_calls = max(0, tracker_update_calls - roi_calls)
```

Then set row values:

```python
            "detector_calls_tracker_update": tracker_update_detector_calls,
            "detector_calls_roi_redetect": roi_calls,
            "mean_roi_redetect_ms": _format_float((roi_seconds / roi_calls * 1000.0) if roi_calls else 0.0),
```

Keep `detector_calls_total` as the sum of all processor calls, including ROI calls.

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_roi_redetect.py test/test_msdc_speed_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 7: Update README**

In the speed benchmark section, add this sentence:

```markdown
MS-DC-ELT speed CSV separates full-frame detector calls from ROI redetect calls via `detector_calls_roi_redetect` and `mean_roi_redetect_ms`; `detector_calls_tracker_update` excludes ROI detector calls and represents detector calls still hidden inside lifecycle update.
```

- [ ] **Step 8: Commit**

Run:

```bash
git add target_module/image_detect_module/utils/roi_redetect.py tools/evaluation/msdc_speed_benchmark.py test/test_msdc_roi_redetect.py test/test_msdc_speed_benchmark.py README.md
git commit -m "perf: account msdc roi redetect calls separately"
```

Expected: commit succeeds.

---

### Task 3: Add Scale-Adaptive Real-Detection Reacquire Gates

**Files:**
- Modify: `target_module/image_detect_module/utils/evidence_state.py`
- Modify: `target_module/image_detect_module/config.py`
- Modify: `test/test_msdc_evidence_state.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing reacquire tests**

Append these tests to `test/test_msdc_evidence_state.py`:

```python
class ReacquireScaleGateConfig(Config):
    MSDC_USE_REACQUIRE = True
    MSDC_REACQUIRE_INTERVAL = 1
    MSDC_REACQUIRE_SCORE = 0.5
    MSDC_REACQUIRE_IOU_THRESH = 0.1
    MSDC_REACQUIRE_CENTER_DIST = 40.0
    MSDC_REACQUIRE_CENTER_SCALE_FACTOR = 6.0
    MSDC_REACQUIRE_MAX_CENTER_DIST = 240.0


def test_reacquire_center_gate_expands_for_large_tracks():
    updater = EvidenceStateUpdater(ReacquireScaleGateConfig)
    lost = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.LOST,
        box=[100, 100, 180, 180],
        velocity=[0.0, 0.0],
        evidence_score=1.0,
        hits=8,
        misses=3,
        age=20,
        last_seen=10,
        last_real_det_frame=10,
        real_det_hits=8,
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [lost],
        [_obs(11, source="low_det", score=0.8, box=[230, 100, 310, 180])],
        frame_idx=11,
    )

    assert tracks[0].state == TrackState.ACTIVE
    assert [event.event_type for event in events] == ["LOST_REACQUIRED"]
    assert updater.last_reacquire_debug["matches"][0]["center_threshold"] == 240.0


def test_template_only_observation_cannot_reacquire_lost_track():
    updater = EvidenceStateUpdater(ReacquireScaleGateConfig)
    lost = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.LOST,
        box=[100, 100, 120, 120],
        velocity=[0.0, 0.0],
        evidence_score=1.0,
        hits=8,
        misses=3,
        age=20,
        last_seen=10,
        last_real_det_frame=10,
        real_det_hits=8,
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [lost],
        [_obs(11, source="template", score=1.0, box=[100, 100, 120, 120])],
        frame_idx=11,
    )

    assert tracks[0].state == TrackState.LOST
    assert events == []
    assert updater.last_reacquire_debug["num_matches"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py::test_reacquire_center_gate_expands_for_large_tracks test/test_msdc_evidence_state.py::test_template_only_observation_cannot_reacquire_lost_track -q
```

Expected: first test FAILS because center gate is fixed; second should PASS already if template-only reacquire is blocked. If second fails, fix it in Step 4.

- [ ] **Step 3: Add config knobs**

In `target_module/image_detect_module/config.py`, after `MSDC_REACQUIRE_CENTER_DIST`, add:

```python
    MSDC_REACQUIRE_CENTER_SCALE_FACTOR = _env_float("MSDC_REACQUIRE_CENTER_SCALE_FACTOR", 4.0)
    MSDC_REACQUIRE_MAX_CENTER_DIST = _env_float("MSDC_REACQUIRE_MAX_CENTER_DIST", 240.0)
```

- [ ] **Step 4: Implement scale-adaptive center threshold**

In `target_module/image_detect_module/utils/evidence_state.py`, add this helper near `_score_reacquire_candidate`:

```python
    def _reacquire_center_threshold(self, track: EvidenceTrack) -> float:
        base = float(self._cfg("MSDC_REACQUIRE_CENTER_DIST", self._cfg("MSDC_ASSOC_CENTER_DIST", 80.0) * 2.0))
        scale_factor = float(self._cfg("MSDC_REACQUIRE_CENTER_SCALE_FACTOR", 4.0))
        max_dist = float(self._cfg("MSDC_REACQUIRE_MAX_CENTER_DIST", max(base, 240.0)))
        box = _box_array(track.box)
        width = max(1.0, float(box[2] - box[0]))
        height = max(1.0, float(box[3] - box[1]))
        scaled = max(base, max(width, height) * scale_factor)
        return min(max_dist, scaled)
```

In `_score_reacquire_candidate`, replace:

```python
        center_thresh = float(self._cfg("MSDC_REACQUIRE_CENTER_DIST", self._cfg("MSDC_ASSOC_CENTER_DIST", 80.0) * 2.0))
```

with:

```python
        center_thresh = self._reacquire_center_threshold(track)
```

Do not remove this existing real-detection guard:

```python
        if not any(source in source_scores for source in self._real_detection_sources()):
            return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py -q
```

Expected: PASS.

- [ ] **Step 6: Update README**

Add to the reacquire config line:

```markdown
`MSDC_REACQUIRE_CENTER_SCALE_FACTOR = 4.0`，`MSDC_REACQUIRE_MAX_CENTER_DIST = 240.0`
```

Also state:

```markdown
重捕只接受 high/low/roi-low/reacquire 真实检测源，template 不能单独重捕 lost track；中心距离门控按目标尺度扩展并受最大距离限制。
```

- [ ] **Step 7: Commit**

Run:

```bash
git add target_module/image_detect_module/config.py target_module/image_detect_module/utils/evidence_state.py test/test_msdc_evidence_state.py README.md
git commit -m "fix: add scale adaptive msdc reacquire gates"
```

Expected: commit succeeds.

---

### Task 4: Add V2 Experiment Variants

**Files:**
- Modify: `tools/experiments/run_msdc_ablation.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Modify: `test/test_msdc_paper_experiment_runner.py`
- Modify: `test/test_msdc_experiment_summary.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing variant tests**

Append this test to `test/test_msdc_paper_experiment_runner.py`:

```python
def test_v2_ablation_variants_are_available():
    from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS

    expected = {
        "v2_template_off",
        "v2_shared_det",
        "v2_low_clean",
        "v2_output_age5_size8",
        "v2_output_age8_size8",
        "v2_output_age12_size8",
        "v2_candidate_low3_window6",
        "v2_candidate_low4_window8",
        "v2_candidate_real2_age8",
        "v2_roi_budget_active8_max2",
        "v2_roi_budget_active10_max2",
        "v2_reacquire_interval1",
        "v2_reacquire_interval2",
        "v2_reacquire_interval5_center240",
    }
    assert expected <= set(ABLATION_VARIANTS)
    assert ABLATION_VARIANTS["v2_template_off"]["MSDC_USE_TEMPLATE"] == "0"
    assert ABLATION_VARIANTS["v2_shared_det"]["MSDC_EXPORT_SHARE_LOW_HIGH_DET"] == "1"
    assert ABLATION_VARIANTS["v2_roi_budget_active8_max2"]["MSDC_ROI_REDETECT_MAX_TRACKS"] == "2"
```

Append this test to `test/test_msdc_experiment_summary.py`:

```python
def test_v2_ablation_switches_are_reported():
    from tools.evaluation.msdc_experiment_summary import _ablation_switches

    switches = _ablation_switches("v2_candidate_real2_age8")
    assert switches["MSDC_USE_TEMPLATE"] == "False"
    assert switches["MSDC_EXPORT_SHARE_LOW_HIGH_DET"] == "True"
    assert switches["MSDC_CONFIRM_MIN_REAL_DET_HITS"] == "2"
    assert switches["MSDC_CANDIDATE_MAX_AGE"] == "8"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py::test_v2_ablation_variants_are_available test/test_msdc_experiment_summary.py::test_v2_ablation_switches_are_reported -q
```

Expected: FAIL because v2 variants and summary fields are missing.

- [ ] **Step 3: Add v2 variants**

In `tools/experiments/run_msdc_ablation.py`, append these entries to `ABLATION_VARIANTS`:

```python
    "v2_template_off": {
        "MSDC_USE_LOW_DET": "1",
        "MSDC_USE_MOTION": "1",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_USE_REACQUIRE": "1",
        "MSDC_USE_ROI_REDETECT": "1",
        "MSDC_REUSE_GUARD_ENABLE": "1",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "0",
    },
    "v2_shared_det": {
        "MSDC_USE_LOW_DET": "1",
        "MSDC_USE_MOTION": "1",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_USE_REACQUIRE": "1",
        "MSDC_USE_ROI_REDETECT": "1",
        "MSDC_REUSE_GUARD_ENABLE": "1",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_low_clean": {
        "MSDC_USE_LOW_DET": "1",
        "MSDC_USE_MOTION": "1",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_USE_REACQUIRE": "1",
        "MSDC_USE_ROI_REDETECT": "1",
        "MSDC_REUSE_GUARD_ENABLE": "1",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_output_age5_size8": {
        "MSDC_OUTPUT_MAX_REAL_DET_AGE": "5",
        "MSDC_OUTPUT_MIN_BOX_SIZE": "8",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_output_age8_size8": {
        "MSDC_OUTPUT_MAX_REAL_DET_AGE": "8",
        "MSDC_OUTPUT_MIN_BOX_SIZE": "8",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_output_age12_size8": {
        "MSDC_OUTPUT_MAX_REAL_DET_AGE": "12",
        "MSDC_OUTPUT_MIN_BOX_SIZE": "8",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_candidate_low3_window6": {
        "MSDC_LOW_CONFIRM_MIN_HITS": "3",
        "MSDC_LOW_CONFIRM_WINDOW": "6",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_candidate_low4_window8": {
        "MSDC_LOW_CONFIRM_MIN_HITS": "4",
        "MSDC_LOW_CONFIRM_WINDOW": "8",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_candidate_real2_age8": {
        "MSDC_CONFIRM_MIN_REAL_DET_HITS": "2",
        "MSDC_CANDIDATE_MAX_AGE": "8",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_roi_budget_active8_max2": {
        "MSDC_ROI_REDETECT_ACTIVE_INTERVAL": "8",
        "MSDC_ROI_REDETECT_LOST_INTERVAL": "3",
        "MSDC_ROI_REDETECT_MAX_TRACKS": "2",
        "MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI": "1",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_roi_budget_active10_max2": {
        "MSDC_ROI_REDETECT_ACTIVE_INTERVAL": "10",
        "MSDC_ROI_REDETECT_LOST_INTERVAL": "3",
        "MSDC_ROI_REDETECT_MAX_TRACKS": "2",
        "MSDC_ROI_REDETECT_MAX_BOXES_PER_ROI": "1",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_reacquire_interval1": {
        "MSDC_REACQUIRE_INTERVAL": "1",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_reacquire_interval2": {
        "MSDC_REACQUIRE_INTERVAL": "2",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
    "v2_reacquire_interval5_center240": {
        "MSDC_REACQUIRE_INTERVAL": "5",
        "MSDC_REACQUIRE_CENTER_DIST": "240",
        "MSDC_REACQUIRE_MAX_CENTER_DIST": "320",
        "MSDC_USE_TEMPLATE": "0",
        "MSDC_TEMPLATE_ENABLE": "0",
        "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "1",
    },
```

- [ ] **Step 4: Extend summary switch fields**

In `tools/evaluation/msdc_experiment_summary.py`, replace `ABLATION_FIELDS` with:

```python
ABLATION_FIELDS = [
    "MSDC_USE_LOW_DET",
    "MSDC_USE_MOTION",
    "MSDC_USE_TEMPLATE",
    "MSDC_TEMPLATE_ENABLE",
    "MSDC_USE_REACQUIRE",
    "MSDC_USE_ROI_REDETECT",
    "MSDC_REUSE_GUARD_ENABLE",
    "MSDC_EXPORT_SHARE_LOW_HIGH_DET",
    "MSDC_OUTPUT_MAX_REAL_DET_AGE",
    "MSDC_OUTPUT_MIN_BOX_SIZE",
    "MSDC_LOW_CONFIRM_MIN_HITS",
    "MSDC_LOW_CONFIRM_WINDOW",
    "MSDC_CONFIRM_MIN_REAL_DET_HITS",
    "MSDC_CANDIDATE_MAX_AGE",
    "MSDC_ROI_REDETECT_ACTIVE_INTERVAL",
    "MSDC_ROI_REDETECT_MAX_TRACKS",
    "MSDC_REACQUIRE_INTERVAL",
    "MSDC_REACQUIRE_CENTER_DIST",
]
```

Replace `ABLATION_SWITCH_DEFAULTS` with:

```python
ABLATION_SWITCH_DEFAULTS = {
    "MSDC_USE_LOW_DET": "True",
    "MSDC_USE_MOTION": "True",
    "MSDC_USE_TEMPLATE": "False",
    "MSDC_TEMPLATE_ENABLE": "False",
    "MSDC_USE_REACQUIRE": "True",
    "MSDC_USE_ROI_REDETECT": "True",
    "MSDC_REUSE_GUARD_ENABLE": "True",
    "MSDC_EXPORT_SHARE_LOW_HIGH_DET": "True",
    "MSDC_OUTPUT_MAX_REAL_DET_AGE": "3",
    "MSDC_OUTPUT_MIN_BOX_SIZE": "12",
    "MSDC_LOW_CONFIRM_MIN_HITS": "5",
    "MSDC_LOW_CONFIRM_WINDOW": "8",
    "MSDC_CONFIRM_MIN_REAL_DET_HITS": "4",
    "MSDC_CANDIDATE_MAX_AGE": "5",
    "MSDC_ROI_REDETECT_ACTIVE_INTERVAL": "8",
    "MSDC_ROI_REDETECT_MAX_TRACKS": "2",
    "MSDC_REACQUIRE_INTERVAL": "5",
    "MSDC_REACQUIRE_CENTER_DIST": "160",
}
```

Update `ABLATION_SWITCH_OVERRIDES` with exact overrides matching the v2 variant dictionary from Step 3, converting `"1"`/`"0"` for boolean fields to `"True"`/`"False"`.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py test/test_msdc_experiment_summary.py -q
```

Expected: PASS.

- [ ] **Step 6: Update README**

Add a v2 experiment matrix subsection containing:

```markdown
### MS-DC-ELT v2 experiment variants

- `v2_template_off`: TemplateLock off, dual high/low detection retained for template-only isolation.
- `v2_shared_det`: TemplateLock off, shared low-threshold inference split into high/low boxes.
- `v2_low_clean`: v2 default low-det behavior; low-det supports active tracks without refreshing active boxes.
- `v2_output_age5_size8` / `v2_output_age8_size8` / `v2_output_age12_size8`: output recall gate sweep.
- `v2_candidate_low3_window6` / `v2_candidate_low4_window8` / `v2_candidate_real2_age8`: candidate confirmation sweep.
- `v2_roi_budget_active8_max2` / `v2_roi_budget_active10_max2`: ROI frequency and track budget sweep.
- `v2_reacquire_interval1` / `v2_reacquire_interval2` / `v2_reacquire_interval5_center240`: reacquire frequency and gate sweep.
```

- [ ] **Step 7: Commit**

Run:

```bash
git add tools/experiments/run_msdc_ablation.py tools/evaluation/msdc_experiment_summary.py test/test_msdc_paper_experiment_runner.py test/test_msdc_experiment_summary.py README.md
git commit -m "exp: add msdc v2 formal variants"
```

Expected: commit succeeds.

---

### Task 5: Let Formal Runner Select V2 Variant Groups

**Files:**
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Modify: `test/test_msdc_paper_experiment_runner.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing runner test**

Append this test to `test/test_msdc_paper_experiment_runner.py`:

```python
def test_formal_runner_accepts_custom_ablation_variants(tmp_path):
    import tools.evaluation.run_msdc_paper_experiments as runner

    args = runner.parse_args([
        "--dataset-root",
        "/data/a",
        "/data/b",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "run",
        "--ablation-variants",
        "v2_template_off",
        "v2_shared_det",
    ])

    cmd = runner.build_ablation_command(
        dataset_roots=list(args.dataset_root),
        output_root=tmp_path / "run" / "ablation",
        run_id="ablation_full",
        commit_hash="abc123",
        progress_interval=500,
        run=False,
        variants=list(args.ablation_variants),
    )

    joined = " ".join(cmd)
    assert "v2_template_off" in joined
    assert "v2_shared_det" in joined
    assert "Ours-full" not in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py::test_formal_runner_accepts_custom_ablation_variants -q
```

Expected: FAIL because `--ablation-variants` and the `variants` parameter do not exist.

- [ ] **Step 3: Update runner API**

In `tools/evaluation/run_msdc_paper_experiments.py`, change `build_ablation_command` signature to:

```python
def build_ablation_command(
    dataset_roots: list[str],
    output_root: Path,
    run_id: str,
    commit_hash: str,
    progress_interval: int,
    run: bool,
    variants: list[str] | None = None,
) -> list[str]:
```

Inside the function, add:

```python
    selected_variants = list(variants or ABLATION_VARIANTS)
```

Replace:

```python
        *ABLATION_VARIANTS,
```

with:

```python
        *selected_variants,
```

In `parse_args`, add:

```python
    parser.add_argument("--ablation-variants", nargs="+", default=ABLATION_VARIANTS)
```

In `main`, pass the selected variants:

```python
        variants=list(args.ablation_variants),
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Update README**

Add formal v2 command example:

```powershell
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --run-formal --run-id msdc_v2_<timestamp> --ablation-variants v2_template_off v2_shared_det v2_low_clean v2_output_age5_size8 v2_output_age8_size8 v2_candidate_low3_window6 v2_candidate_real2_age8 v2_roi_budget_active8_max2 v2_reacquire_interval1
```

- [ ] **Step 6: Commit**

Run:

```bash
git add tools/evaluation/run_msdc_paper_experiments.py test/test_msdc_paper_experiment_runner.py README.md
git commit -m "feat: allow selecting msdc formal variants"
```

Expected: commit succeeds.

---

### Task 6: Add Formal Run Artifact Validator

**Files:**
- Create: `tools/evaluation/validate_msdc_formal_run.py`
- Create: `test/test_validate_msdc_formal_run.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing validator tests**

Create `test/test_validate_msdc_formal_run.py`:

```python
import csv
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from tools.evaluation.validate_msdc_formal_run import validate_run


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_valid_run(root):
    (root / "main/main_full/trackers/msdc_elt/data").mkdir(parents=True)
    (root / "main/main_full/trackers/msdc_elt/data/seq.txt").write_text("1,1,10,10,20,20,1,-1,-1,-1\n", encoding="utf-8")
    (root / "main/main_full/trackers/msdc_elt/diagnostics/seq").mkdir(parents=True)
    (root / "main/main_full/trackers/msdc_elt/diagnostics/seq/stage_observations.jsonl").write_text(
        json.dumps({"frame_idx": 0, "boxes": []}) + "\n",
        encoding="utf-8",
    )
    (root / "main/main_full/diagnostics/seq").mkdir(parents=True)
    (root / "main/main_full/diagnostics/seq/msdc_elt_per_gt_diagnostics.csv").write_text("gt_id,coverage_ratio\n1,1.0\n", encoding="utf-8")
    (root / "main/main_full/visualizations/seq").mkdir(parents=True)
    (root / "main/main_full/visualizations/seq/msdc_elt.mp4").write_bytes(b"mp4")
    (root / "summary").mkdir(parents=True)
    _write_csv(
        root / "summary/main_results.csv",
        [{
            "tracker": "msdc_elt",
            "HOTA": "67.0",
            "MOTA": "80.0",
            "IDF1": "66.0",
            "IDSW": "64",
            "FP": "3000",
            "FN": "21000",
            "IDTP": "1",
            "IDFP": "2",
            "IDFN": "3",
        }],
        ["tracker", "HOTA", "MOTA", "IDF1", "IDSW", "FP", "FN", "IDTP", "IDFP", "IDFN"],
    )
    _write_csv(
        root / "summary/speed_results.csv",
        [{
            "tracker": "msdc_elt",
            "processed_frames": "1000",
            "total_time_s": "200",
            "mean_latency_ms": "200",
            "mean_fps": "5",
            "detector_calls_total": "1100",
            "detector_calls_high_det": "0",
            "detector_calls_low_det": "1000",
            "detector_calls_tracker_update": "0",
            "detector_calls_roi_redetect": "100",
            "mean_read_ms": "1",
            "mean_high_det_ms": "0",
            "mean_low_det_ms": "80",
            "mean_roi_redetect_ms": "10",
            "mean_tracker_ms": "30",
        }],
        [
            "tracker",
            "processed_frames",
            "total_time_s",
            "mean_latency_ms",
            "mean_fps",
            "detector_calls_total",
            "detector_calls_high_det",
            "detector_calls_low_det",
            "detector_calls_tracker_update",
            "detector_calls_roi_redetect",
            "mean_read_ms",
            "mean_high_det_ms",
            "mean_low_det_ms",
            "mean_roi_redetect_ms",
            "mean_tracker_ms",
        ],
    )


def test_validate_run_accepts_complete_formal_outputs(tmp_path):
    _make_valid_run(tmp_path)

    result = validate_run(tmp_path)

    assert result["ok"] is True
    assert result["missing"] == []


def test_validate_run_rejects_missing_visualization(tmp_path):
    _make_valid_run(tmp_path)
    (tmp_path / "main/main_full/visualizations/seq/msdc_elt.mp4").unlink()

    result = validate_run(tmp_path)

    assert result["ok"] is False
    assert any("visualization" in item for item in result["missing"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_validate_msdc_formal_run.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `tools.evaluation.validate_msdc_formal_run`.

- [ ] **Step 3: Implement validator**

Create `tools/evaluation/validate_msdc_formal_run.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_METRICS = ["HOTA", "MOTA", "IDF1", "IDSW", "FP", "FN", "IDTP", "IDFP", "IDFN"]
REQUIRED_SPEED = [
    "processed_frames",
    "total_time_s",
    "mean_latency_ms",
    "mean_fps",
    "detector_calls_total",
    "detector_calls_high_det",
    "detector_calls_low_det",
    "detector_calls_tracker_update",
    "detector_calls_roi_redetect",
    "mean_read_ms",
    "mean_high_det_ms",
    "mean_low_det_ms",
    "mean_roi_redetect_ms",
    "mean_tracker_ms",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _has_nonempty_glob(root: Path, pattern: str) -> bool:
    return any(path.is_file() and path.stat().st_size > 0 for path in root.glob(pattern))


def _csv_has_fields(path: Path, fields: list[str]) -> bool:
    if not path.is_file():
        return False
    rows = _read_csv(path)
    if not rows:
        return False
    for field in fields:
        if field not in rows[0] or str(rows[0].get(field, "")).strip() == "":
            return False
    return True


def validate_run(run_root: str | Path) -> dict:
    root = Path(run_root)
    missing: list[str] = []
    if not root.is_dir():
        return {"ok": False, "missing": [f"run_root:{root}"], "run_root": str(root)}

    checks = [
        ("mot_result", "**/trackers/*/data/*.txt"),
        ("stage_observations", "**/trackers/*/diagnostics/**/stage_observations.jsonl"),
        ("diagnostic_csv", "**/diagnostics/**/*.csv"),
        ("visualization", "**/visualizations/**/*.mp4"),
    ]
    for label, pattern in checks:
        if not _has_nonempty_glob(root, pattern):
            missing.append(f"{label}:{pattern}")

    main_results = root / "summary" / "main_results.csv"
    if not _csv_has_fields(main_results, REQUIRED_METRICS):
        missing.append(f"metrics:{main_results}")

    speed_results = root / "summary" / "speed_results.csv"
    if not _csv_has_fields(speed_results, REQUIRED_SPEED):
        missing.append(f"speed:{speed_results}")

    return {"ok": not missing, "missing": missing, "run_root": str(root)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MS-DC-ELT formal run artifacts")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()
    result = validate_run(args.run_root)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run validator tests**

Run:

```bash
conda run -n ship_detect pytest test/test_validate_msdc_formal_run.py -q
```

Expected: PASS.

- [ ] **Step 5: Update README**

Add:

```powershell
conda run -n ship_detect python tools/evaluation/validate_msdc_formal_run.py --run-root results/msdc_paper_phase1/<run-id> --json-output results/msdc_paper_phase1/<run-id>/summary/formal_validation.json
```

State that formal completion cannot be claimed unless this command exits 0.

- [ ] **Step 6: Commit**

Run:

```bash
git add tools/evaluation/validate_msdc_formal_run.py test/test_validate_msdc_formal_run.py README.md
git commit -m "tool: validate msdc formal run artifacts"
```

Expected: commit succeeds.

---

### Task 7: Run Smoke Evaluation Before Formal Runs

**Files:**
- No code files modified unless a smoke failure exposes a bug.
- Runtime outputs under `results/msdc_paper_phase1/<timestamp>/smoke`.

- [ ] **Step 1: Confirm dataset and video references**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --check-latest
```

Expected: If no latest formal run exists, this may fail. That is not a blocker for a new run.

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --smoke --run-id msdc_v2_smoke_$(date +%Y%m%d_%H%M%S) --progress-interval 50
```

Expected: smoke writes a 100-frame run under `results/msdc_paper_phase1/<run-id>/smoke`, renders annotated videos, and exits 0.

- [ ] **Step 2: If smoke fails due missing external data**

Stop and report the exact missing path. Do not claim formal testing. Use the path printed in the exception from `resolve_single_sequence_dataset` or `read_video_info`.

- [ ] **Step 3: If smoke fails due code/test bug**

Use `superpowers:systematic-debugging` before changing code. Add a failing test for the failure, fix it, run:

```bash
conda run -n ship_detect pytest test -q
```

Expected: PASS before restarting smoke.

---

### Task 8: Run V2 Formal Main, Ablation, Speed, Summary, And Validator

**Files:**
- Runtime outputs under `results/msdc_paper_phase1/<timestamp>`.
- Modify: `docs/MSDC_EXPERIMENT_RESULT.md` through existing summary script.
- Modify: `README.md` only if formal command behavior changed during execution.

- [ ] **Step 1: Start formal run**

Run:

```bash
RUN_ID=msdc_v2_$(date +%Y%m%d_%H%M%S)
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py \
  --run-formal \
  --run-id "$RUN_ID" \
  --progress-interval 500 \
  --speed-frames 1000 \
  --ablation-variants \
    v2_template_off \
    v2_shared_det \
    v2_low_clean \
    v2_output_age5_size8 \
    v2_output_age8_size8 \
    v2_output_age12_size8 \
    v2_candidate_low3_window6 \
    v2_candidate_low4_window8 \
    v2_candidate_real2_age8 \
    v2_roi_budget_active8_max2 \
    v2_roi_budget_active10_max2 \
    v2_reacquire_interval1 \
    v2_reacquire_interval2 \
    v2_reacquire_interval5_center240
```

Expected: exits 0. The run root is `results/msdc_paper_phase1/$RUN_ID`.

- [ ] **Step 2: Validate formal artifacts**

Run:

```bash
conda run -n ship_detect python tools/evaluation/validate_msdc_formal_run.py \
  --run-root "results/msdc_paper_phase1/$RUN_ID" \
  --json-output "results/msdc_paper_phase1/$RUN_ID/summary/formal_validation.json"
```

Expected: exits 0 with `"ok": true`.

- [ ] **Step 3: Check latest run metadata**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --check-latest
```

Expected: exits 0 and prints `[OK] Latest run paths exist`.

- [ ] **Step 4: Inspect metric summary**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --print-latest
```

Then inspect:

```bash
conda run -n ship_detect python - <<'PY'
import csv
from pathlib import Path
root = Path("results/msdc_paper_phase1/latest_run.json")
print(root.read_text(encoding="utf-8"))
PY
```

Expected: latest metadata points to `main_results.csv`, `ablation_results.csv`, `speed_results.csv`, `MSDC_EXPERIMENT_REPORT.md`, and `docs/MSDC_EXPERIMENT_RESULT.md`.

- [ ] **Step 5: Do not summarize until required fields are present**

Open `results/msdc_paper_phase1/$RUN_ID/summary/main_results.csv` and verify these columns have non-empty values for `msdc_elt`: `HOTA`, `AssA`, `MOTA`, `IDF1`, `IDSW`, `FP`, `FN`, `IDTP`, `IDFP`, `IDFN`.

Open `results/msdc_paper_phase1/$RUN_ID/summary/speed_results.csv` and verify these columns have non-empty values for `msdc_elt`: `processed_frames`, `total_time_s`, `mean_latency_ms`, `mean_fps`, `detector_calls_total`, `detector_calls_low_det`, `detector_calls_roi_redetect`, `mean_tracker_ms`.

- [ ] **Step 6: Commit docs result if formal run completed**

Run:

```bash
git add docs/MSDC_EXPERIMENT_RESULT.md README.md
git commit -m "docs: record msdc v2 formal evaluation"
```

Expected: commit succeeds if docs changed. If README did not change, `git add README.md` is harmless.

---

### Task 9: Report Formal Results With Explicit Pass/Fail Against Targets

**Files:**
- No code files modified.
- User-facing report in final response.

- [ ] **Step 1: Extract main comparison metrics**

Run:

```bash
conda run -n ship_detect python - <<'PY'
import csv, json
from pathlib import Path
latest = json.loads(Path("results/msdc_paper_phase1/latest_run.json").read_text(encoding="utf-8"))
main_csv = Path(latest["main_results_csv"])
with main_csv.open("r", encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))
for row in rows:
    if row.get("tracker") in {"ocsort", "botsort", "msdc_elt", "Ours-full"}:
        print({k: row.get(k) for k in ["tracker", "HOTA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "IDTP", "IDFP", "IDFN", "mot_result_path"]})
PY
```

Expected: prints real metric values; no metric field should be empty or invented.

- [ ] **Step 2: Extract ablation winner metrics**

Run:

```bash
conda run -n ship_detect python - <<'PY'
import csv, json
from pathlib import Path
latest = json.loads(Path("results/msdc_paper_phase1/latest_run.json").read_text(encoding="utf-8"))
abl_csv = Path(latest["ablation_results_csv"])
with abl_csv.open("r", encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))
def f(row, key):
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")
rows = sorted(rows, key=lambda r: f(r, "HOTA"), reverse=True)
for row in rows[:5]:
    print({k: row.get(k) for k in ["variant", "HOTA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN"]})
PY
```

Expected: prints top 5 real v2 variants by HOTA.

- [ ] **Step 3: Extract speed values**

Run:

```bash
conda run -n ship_detect python - <<'PY'
import csv, json
from pathlib import Path
latest = json.loads(Path("results/msdc_paper_phase1/latest_run.json").read_text(encoding="utf-8"))
speed_csv = Path(latest["speed_results_csv"])
with speed_csv.open("r", encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))
for row in rows:
    print({k: row.get(k) for k in ["tracker", "processed_frames", "total_time_s", "mean_fps", "detector_calls_total", "detector_calls_high_det", "detector_calls_low_det", "detector_calls_roi_redetect", "detector_calls_tracker_update", "mean_read_ms", "mean_high_det_ms", "mean_low_det_ms", "mean_roi_redetect_ms", "mean_tracker_ms"]})
PY
```

Expected: prints real speed values; `msdc_elt detector_calls_total` should be near `processed_frames + detector_calls_roi_redetect` when shared high/low detection is active.

- [ ] **Step 4: Report against stated targets**

In the final response, include:

```markdown
Formal run: `results/msdc_paper_phase1/<RUN_ID>`

MS-DC-ELT v2 main:
- HOTA: <value> (target >= 67.31: PASS/FAIL)
- IDF1: <value> (target >= 66.11: PASS/FAIL)
- AssA: <value> (target >= 64.93: PASS/FAIL)
- IDSW: <value> (target <= 64: PASS/FAIL)
- FP/FN: <values>

Best v2 ablation: `<variant>`
- HOTA / IDF1 / AssA / IDSW / FP / FN: <values>

Speed:
- processed frames: <value>
- total time: <value>s
- FPS: <value>
- detector calls total/high/low/ROI/tracker_update: <values>

Artifacts:
- MOT txt: `<path>`
- TrackEval summary: `<path>`
- diagnostics: `<path>`
- visualization MP4: `<path>`
- speed CSV: `<path>`
- formal validation JSON: `<path>`
```

If any required artifact or metric is missing, state that formal evaluation is incomplete and include the failing validator output.

---

## Self-Review

Spec coverage:
- ROI budget tightening is covered by Tasks 1 and 2.
- Output gate and candidate/reacquire sweeps are covered by Task 4 variants and Task 8 formal run.
- Reacquire mechanism tuning and template-only exclusion are covered by Task 3 and Task 4 variants.
- Formal TrackEval, speed, visualization, MOT txt, diagnostics, and timing outputs are covered by Tasks 6, 8, and 9.
- README updates are included in every code-changing task that changes documented behavior.

Placeholder scan:
- All implementation steps are explicit and runnable.
- Formal run commands include exact tracker variants and required validation command.
- Tests and code snippets include concrete names and expected outcomes.

Type consistency:
- Variant names match across `run_msdc_ablation.py`, `run_msdc_paper_experiments.py`, and `msdc_experiment_summary.py`.
- Speed fields use `detector_calls_roi_redetect` and `mean_roi_redetect_ms` consistently.
- Reacquire config names use `MSDC_REACQUIRE_CENTER_SCALE_FACTOR` and `MSDC_REACQUIRE_MAX_CENTER_DIST` consistently.
