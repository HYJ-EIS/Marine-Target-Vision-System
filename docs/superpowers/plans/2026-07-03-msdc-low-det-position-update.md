# MS-DC Low-Det Position Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a controlled MS-DC option that lets matched `low_det` observations refresh ACTIVE and reacquired LOST track boxes, then run a full-length no-render 2x2 ablation.

**Architecture:** Keep historical behavior as the default by adding explicit config flags: ACTIVE low-det refresh remains disabled by default, while LOST reacquire low-det refresh remains enabled because the current code already does that. Route both ACTIVE matches and LOST reacquire matches through the existing `_apply_matched_observation()` refresh decision, then add explicit all-off/on variants for a clean 2x2 ablation.

**Tech Stack:** Python, pytest, MS-DC evidence-state tracker, MOT replay benchmark, existing cached detections.

---

### Task 1: Add Behavior Tests

**Files:**
- Modify: `test/test_msdc_evidence_state.py`

- [ ] **Step 1: Add a config fixture class**

Add a test config near the other config classes:

```python
class LowDetPositionUpdateConfig(ReacquireLifecycleConfig):
    MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE = True
    MSDC_LOW_UPDATE_LOST_BOX_ENABLE = True


class LowDetPositionUpdateDisabledConfig(ReacquireLifecycleConfig):
    MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE = False
    MSDC_LOW_UPDATE_LOST_BOX_ENABLE = False
```

- [ ] **Step 2: Write failing ACTIVE-track test**

Add a test next to `test_low_detection_supports_active_without_refreshing_primary_box()`:

```python
def test_low_detection_refreshes_active_primary_box_when_enabled():
    updater = EvidenceStateUpdater(LowDetPositionUpdateConfig)
    track = EvidenceTrack(
        gid=1,
        public_id=1,
        state=TrackState.ACTIVE,
        box=[10, 10, 30, 30],
        velocity=[0.0, 0.0],
        evidence_score=3.0,
        hits=4,
        misses=0,
        age=4,
        last_seen=4,
        last_real_det_frame=4,
        real_det_hits=4,
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [track],
        [_obs(5, source="low_det", score=0.8, box=[45, 10, 65, 30])],
        frame_idx=5,
    )

    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].box.tolist() == [45.0, 10.0, 65.0, 30.0]
    assert tracks[0].velocity == [35.0, 0.0]
    assert tracks[0].last_real_det_box.tolist() == [45.0, 10.0, 65.0, 30.0]
    assert events == []
```

- [ ] **Step 3: Write failing LOST-reacquire test**

Extend the low-det reacquire coverage with a disabled-config test that captures the all-off ablation behavior:

```python
def test_low_detection_does_not_refresh_lost_primary_box_when_disabled():
    updater = EvidenceStateUpdater(LowDetPositionUpdateDisabledConfig)
    tracks = [
        EvidenceTrack(
            gid=7,
            public_id=7,
            state=TrackState.LOST,
            box=[100, 100, 120, 120],
            velocity=[0.0, 0.0],
            evidence_score=1.0,
            hits=3,
            misses=2,
            age=6,
            last_seen=4,
            class_id=2,
            class_name="UAV",
        )
    ]

    tracks, events = updater.update_tracks(
        tracks,
        [_obs(10, source="low_det", score=1.0, box=[102, 101, 122, 121])],
        frame_idx=10,
    )

    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].box == [100, 100, 120, 120]
    assert tracks[0].velocity == [0.0, 0.0]
    assert tracks[0].last_real_det_box.tolist() == [102.0, 101.0, 122.0, 121.0]
    assert [event.event_type for event in events] == ["LOST_REACQUIRED"]
```

- [ ] **Step 4: Verify RED**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py::test_low_detection_refreshes_active_primary_box_when_enabled test/test_msdc_evidence_state.py::test_low_detection_does_not_refresh_lost_primary_box_when_disabled -q
```

Expected: tests fail because the new flags are not honored: ACTIVE low-only groups do not refresh primary boxes, while LOST low-only reacquire currently always refreshes.

### Task 2: Implement Config-Gated Refresh

**Files:**
- Modify: `target_module/image_detect_module/config.py`
- Modify: `target_module/image_detect_module/utils/evidence_state.py`

- [ ] **Step 1: Add config defaults**

Add these defaults next to the low/reacquire lifecycle settings:

```python
MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE = _env_bool("MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE", False)
MSDC_LOW_UPDATE_LOST_BOX_ENABLE = _env_bool("MSDC_LOW_UPDATE_LOST_BOX_ENABLE", True)
```

- [ ] **Step 2: Update the refresh predicate**

Change `_matched_group_refreshes_primary_box()` to accept the matched track state and return:

```python
sources = set(group.source_scores)
if sources & {"high_det", "reacquire"}:
    return True
if "low_det" not in sources:
    return _as_state(track.state) != TrackState.ACTIVE
state = _as_state(track.state)
if state == TrackState.ACTIVE:
    return bool(self._cfg("MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE", False))
if state == TrackState.LOST:
    return bool(self._cfg("MSDC_LOW_UPDATE_LOST_BOX_ENABLE", True))
return True
```

- [ ] **Step 3: Verify GREEN**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py::test_low_detection_supports_active_without_refreshing_primary_box test/test_msdc_evidence_state.py::test_low_detection_refreshes_active_primary_box_when_enabled test/test_msdc_evidence_state.py::test_lost_track_reacquires_only_on_interval_with_low_detection test/test_msdc_evidence_state.py::test_low_detection_does_not_refresh_lost_primary_box_when_disabled -q
```

Expected: pass. Existing default behavior still keeps ACTIVE low-only matches from refreshing the primary box and still refreshes LOST low-only reacquire matches.

### Task 3: Add 2x2 Ablation Variants

**Files:**
- Modify: `tools/experiments/run_msdc_ablation.py`
- Modify: `tools/evaluation/detection_replay_benchmark.py`
- Modify: `test/test_msdc_paper_experiment_runner.py`
- Modify: `test/test_detection_replay_benchmark.py`

- [ ] **Step 1: Freeze default variant environment**

Add to `FORMAL_V3_ENV`:

```python
"MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE": "0",
"MSDC_LOW_UPDATE_LOST_BOX_ENABLE": "1",
```

Add to `_MSDC_FORMAL_REPLAY_CONFIG_BASELINE`:

```python
"MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE": False,
"MSDC_LOW_UPDATE_LOST_BOX_ENABLE": True,
```

- [ ] **Step 2: Add ablation variants**

Add:

```python
"low_position_update_all_off": {
    **FORMAL_V3_ENV,
    "MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE": "0",
    "MSDC_LOW_UPDATE_LOST_BOX_ENABLE": "0",
},
"low_position_update_on": {
    **FORMAL_V3_ENV,
    "MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE": "1",
    "MSDC_LOW_UPDATE_LOST_BOX_ENABLE": "1",
},
"removed_recovery_off_low_position_update_all_off": {
    **FORMAL_V3_ENV,
    "MSDC_REMOVED_RECOVERY_ENABLE": "0",
    "MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE": "0",
    "MSDC_LOW_UPDATE_LOST_BOX_ENABLE": "0",
},
"removed_recovery_off_low_position_update_on": {
    **FORMAL_V3_ENV,
    "MSDC_REMOVED_RECOVERY_ENABLE": "0",
    "MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE": "1",
    "MSDC_LOW_UPDATE_LOST_BOX_ENABLE": "1",
},
```

- [ ] **Step 3: Update tests for required variants**

Extend expected variant sets to include:

```python
"low_position_update_on",
"low_position_update_all_off",
"removed_recovery_off_low_position_update_on",
"removed_recovery_off_low_position_update_all_off",
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py test/test_detection_replay_benchmark.py -q
```

Expected: pass.

### Task 4: Documentation and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the new knobs and ablation**

Add a concise note that `MSDC_LOW_UPDATE_ACTIVE_BOX_ENABLE` is disabled by default, `MSDC_LOW_UPDATE_LOST_BOX_ENABLE` is enabled by default to preserve historical reacquire behavior, and the all-off/on variants are used for the low-det position-update ablation.

- [ ] **Step 2: Run full tests**

Run:

```bash
conda run -n ship_detect pytest test -q
```

Expected: pass.

### Task 5: Full-Length No-Render Evaluation

**Files:**
- Output only under a fresh run directory in `results/msdc_low_update/`

- [ ] **Step 1: Run full no-render MOT replay**

Run both sequences for BoT-SORT and the four MS-DC 2x2 variants:

```bash
conda run -n ship_detect python tools/evaluation/detection_replay_benchmark.py \
  --dataset-root /home/hyj/Anti_Drone_Project/DroneVehicle/data/VisDrone2019-MOT-test-dev \
  --output-root results/msdc_low_update \
  --run-id 20260703_full_no_render_low_update \
  --formal-frame-limit 5400 \
  --trackers botsort msdc_elt \
  --variants msdc_v3 removed_recovery_off low_position_update_all_off low_position_update_on removed_recovery_off_low_position_update_all_off removed_recovery_off_low_position_update_on \
  --source-detection-cache-root results/msdc_paper_phase1/20260624_190710/main/main_full/detections \
  --progress-interval 500 \
  --render-class-source cache
```

This command intentionally omits `--render`; it does not satisfy the project’s visualization requirement for a formal evaluation report.

- [ ] **Step 2: Run speed replay for before/after main variants**

Run the no-render speed benchmark at least for `msdc_v3` low-update off and low-update on, writing separate timestamped output directories.

- [ ] **Step 3: Summarize outputs**

Report MOT metrics, speed metrics, timing breakdown files, and paths to MOT txt, TrackEval summary, diagnostic JSONL/CSV, and speed JSON/CSV. Mark visualization as intentionally skipped by user request.
