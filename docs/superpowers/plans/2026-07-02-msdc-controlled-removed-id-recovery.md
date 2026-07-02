# MS-DC Controlled Removed ID Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MS-DC's removed guard from a veto-only "allocate a new ID near a removed signature" rule with a gated recovery path that can reuse the removed track's public ID when geometry, class, and recency checks support it, then validate tracking and speed before and after the change.

**Architecture:** Keep the removed guard enabled, but split it into two outcomes: `recover_removed_id` when a recent removed track is a strong match, and `guard_new_id` when it is not. The recovery path reactivates the existing removed `EvidenceTrack` as a candidate with the same `public_id` and `gid`, while the legacy fallback still creates a new track and emits the old guard event for unsafe matches. Formal comparison is done by adding an explicit ablation variant with recovery disabled, so before/after metrics can be produced from the same code revision and detector cache.

**Tech Stack:** Python, NumPy, pytest, TrackEval-compatible MOT export, existing MS-DC replay/speed scripts, `conda run -n ship_detect`.

---

## File Structure

- Modify `target_module/image_detect_module/config.py`: add environment-configurable recovery gates next to the existing removed guard settings.
- Modify `target_module/image_detect_module/utils/evidence_state.py`: extend removed guard data with track indices, add a recovery decision helper, reactivate removed tracks in `spawn_candidates()`, and emit recovery diagnostics.
- Modify `tools/evaluation/msdc_diagnostic_metrics.py`: count removed recovery attempts, successes, and guard fallbacks in diagnostic CSVs.
- Modify `tools/experiments/run_msdc_ablation.py`: add `removed_recovery_off` as the legacy veto-only ablation.
- Modify `test/test_msdc_evidence_state.py`: replace the current veto-only expectation with recovery behavior and add unsafe fallback cases.
- Modify `test/test_msdc_diagnostic_metrics.py`: assert the new diagnostic fields.
- Modify `README.md`: document the new recovery knobs, event names, and formal evaluation requirement after implementation.

## Baseline Facts To Preserve

Use the existing full run as the known pre-change reference:

```text
results/msdc_paper_phase1/20260624_190710/summary/main_results.csv
BoT-SORT: HOTA 78.13808, MOTA 77.95047, IDF1 86.52260, IDSW 10, FP 1412, FN 5033
MS-DC v3: HOTA 72.51247, MOTA 77.67720, IDF1 76.72734, IDSW 16, FP 1809, FN 4710
```

Use the known failure case to verify the behavioral fix:

```text
Sequence: DJI_20250916100639_0001_V
Existing MS-DC split: GT 2 is matched by public IDs 2 and 19.
Lifecycle chain:
frame 2487: gid 2 ACTIVE_TO_LOST
frame 2525: gid 2 LOST_TO_removed
frame 2525: PREVENT_removed_ID_REUSE with blocked_gid 2, new_gid 48, guard_iou 0.3071, guard_center_distance 14.9164
frame 2528: gid 48 CONFIRM_ACTIVE, public_id 19
```

## Task 1: Write Failing Unit Tests For Controlled Recovery

**Files:**
- Modify: `test/test_msdc_evidence_state.py`

- [ ] **Step 1: Add a recovery-enabled config class near the existing guard config classes**

Add this test config in `test/test_msdc_evidence_state.py` close to `GuardLifecycleConfig`:

```python
class RemovedRecoveryConfig(GuardLifecycleConfig):
    MSDC_REUSE_GUARD_ENABLE = True
    MSDC_REMOVED_RECOVERY_ENABLE = True
    MSDC_REMOVED_RECOVERY_REQUIRE_CLASS_MATCH = True
    MSDC_REMOVED_RECOVERY_MAX_AGE = 80
    MSDC_REMOVED_RECOVERY_MIN_IOU = 0.20
    MSDC_REMOVED_RECOVERY_MAX_CENTER_DIST = 80.0
    MSDC_REMOVED_RECOVERY_MIN_SCORE = 0.35
```

- [ ] **Step 2: Replace the veto-only test with a recovery test**

Replace `test_removed_guard_prevents_old_gid_reuse_and_creates_new_id()` with:

```python
def test_removed_guard_recovers_old_public_id_when_match_is_safe():
    updater = EvidenceStateUpdater(RemovedRecoveryConfig)
    removed = EvidenceTrack(
        gid=9,
        public_id=4,
        state=TrackState.REMOVED,
        box=[40, 40, 60, 60],
        velocity=[1.0, 0.0],
        evidence_score=0.0,
        hits=4,
        misses=5,
        age=12,
        last_seen=5,
        retired_signature={
            "last_box": [40.0, 40.0, 60.0, 60.0],
            "last_seen": 5,
            "removed_frame_idx": 5,
            "last_velocity": [1.0, 0.0],
            "class_id": 2,
            "class_name": "UAV",
        },
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [removed],
        [_obs(6, source="low_det", score=1.0, box=[41, 40, 61, 60], class_id=2, class_name="UAV")],
        frame_idx=6,
    )

    event_types = [event.event_type for event in events]
    recovered = next(track for track in tracks if track.gid == 9)
    assert "REMOVED_ID_RECOVERY_CANDIDATE" in event_types
    assert "PREVENT_removed_ID_REUSE" not in event_types
    assert "NEW_ID_CREATED" not in event_types
    assert recovered.public_id == 4
    assert recovered.state == TrackState.LOW_CANDIDATE
    assert recovered.last_seen == 6
    assert recovered.retired_signature is None
    assert len(tracks) == 1
    assert updater.last_removed_guard_debug["num_recoveries"] == 1
```

- [ ] **Step 3: Add the unsafe class mismatch fallback test**

Append this test after the recovery test:

```python
def test_removed_guard_keeps_new_id_when_recovery_class_mismatches():
    updater = EvidenceStateUpdater(RemovedRecoveryConfig)
    removed = EvidenceTrack(
        gid=9,
        public_id=4,
        state=TrackState.REMOVED,
        box=[40, 40, 60, 60],
        retired_signature={
            "last_box": [40.0, 40.0, 60.0, 60.0],
            "last_seen": 5,
            "removed_frame_idx": 5,
            "class_id": 2,
            "class_name": "UAV",
        },
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [removed],
        [_obs(6, source="low_det", score=1.0, box=[41, 40, 61, 60], class_id=7, class_name="USV")],
        frame_idx=6,
    )

    event_types = [event.event_type for event in events]
    assert "REMOVED_ID_RECOVERY_CANDIDATE" not in event_types
    assert "PREVENT_removed_ID_REUSE" in event_types
    assert "NEW_ID_CREATED" in event_types
    assert tracks[0].gid == 9
    assert tracks[1].gid == 10
    assert tracks[1].public_id is None
    assert updater.last_removed_guard_debug["num_recoveries"] == 0
    assert updater.last_removed_guard_debug["num_vetoes"] == 1
```

- [ ] **Step 4: Add the stale signature fallback test**

Append:

```python
def test_removed_guard_keeps_new_id_when_recovery_signature_is_too_old():
    updater = EvidenceStateUpdater(RemovedRecoveryConfig)
    removed = EvidenceTrack(
        gid=9,
        public_id=4,
        state=TrackState.REMOVED,
        box=[40, 40, 60, 60],
        retired_signature={
            "last_box": [40.0, 40.0, 60.0, 60.0],
            "last_seen": 5,
            "removed_frame_idx": 5,
            "class_id": 2,
            "class_name": "UAV",
        },
        class_id=2,
        class_name="UAV",
    )

    tracks, events = updater.update_tracks(
        [removed],
        [_obs(90, source="low_det", score=1.0, box=[41, 40, 61, 60], class_id=2, class_name="UAV")],
        frame_idx=90,
    )

    event_types = [event.event_type for event in events]
    assert "REMOVED_ID_RECOVERY_CANDIDATE" not in event_types
    assert "NEW_LOW_CANDIDATE" in event_types
    assert tracks[0].gid == 10
    assert updater.last_removed_guard_debug["num_recent_signatures"] == 0
```

- [ ] **Step 5: Run the tests and verify they fail before implementation**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py -q
```

Expected before implementation:

```text
FAILED test/test_msdc_evidence_state.py::test_removed_guard_recovers_old_public_id_when_match_is_safe
```

The failure should show that `REMOVED_ID_RECOVERY_CANDIDATE` is missing and the current code still emits `PREVENT_removed_ID_REUSE` plus `NEW_ID_CREATED`.

## Task 2: Add Recovery Configuration Knobs

**Files:**
- Modify: `target_module/image_detect_module/config.py`

- [ ] **Step 1: Add recovery config next to the removed guard config**

In `Config`, replace the removed guard block at lines 154-159 with:

```python
    # removed guard keeps short-term signatures; safe matches can recover the old public ID.
    MSDC_REMOVED_GUARD_FRAMES = _env_int("MSDC_REMOVED_GUARD_FRAMES", 80)
    MSDC_removed_GUARD_FRAMES = MSDC_REMOVED_GUARD_FRAMES
    MSDC_REUSE_GUARD_ENABLE = _env_bool("MSDC_REUSE_GUARD_ENABLE", True)
    MSDC_REMOVED_GUARD_IOU_THRESH = _env_float("MSDC_REMOVED_GUARD_IOU_THRESH", 0.3)
    MSDC_REMOVED_GUARD_CENTER_DIST = _env_float("MSDC_REMOVED_GUARD_CENTER_DIST", 80.0)
    MSDC_REMOVED_RECOVERY_ENABLE = _env_bool("MSDC_REMOVED_RECOVERY_ENABLE", True)
    MSDC_REMOVED_RECOVERY_REQUIRE_CLASS_MATCH = _env_bool("MSDC_REMOVED_RECOVERY_REQUIRE_CLASS_MATCH", True)
    MSDC_REMOVED_RECOVERY_MAX_AGE = _env_int("MSDC_REMOVED_RECOVERY_MAX_AGE", 80)
    MSDC_REMOVED_RECOVERY_MIN_IOU = _env_float("MSDC_REMOVED_RECOVERY_MIN_IOU", 0.20)
    MSDC_REMOVED_RECOVERY_MAX_CENTER_DIST = _env_float("MSDC_REMOVED_RECOVERY_MAX_CENTER_DIST", 80.0)
    MSDC_REMOVED_RECOVERY_MIN_SCORE = _env_float("MSDC_REMOVED_RECOVERY_MIN_SCORE", 0.35)
```

- [ ] **Step 2: Run the focused config import check**

Run:

```bash
conda run -n ship_detect python - <<'PY'
from target_module.image_detect_module.config import Config
print(Config.MSDC_REMOVED_RECOVERY_ENABLE)
print(Config.MSDC_REMOVED_RECOVERY_MIN_IOU)
PY
```

Expected:

```text
True
0.2
```

## Task 3: Implement Controlled Recovery In EvidenceStateUpdater

**Files:**
- Modify: `target_module/image_detect_module/utils/evidence_state.py`

- [ ] **Step 1: Extend removed guard data so a conflict can locate the removed track**

In `_removed_guard_data()`, add `track_indices = []` before the loop, change the loop header to `for track_index, track in enumerate(existing_tracks):`, append `track_indices.append(int(track_index))` beside `gids.append(int(track.gid))`, and include it in `payload`:

```python
        payload = {
            "boxes": np.vstack(boxes) if boxes else np.zeros((0, 4), dtype=np.float64),
            "gids": gids,
            "track_indices": track_indices,
            "removed_frame_indices": removed_frame_indices,
            "ages": ages,
            "signatures": signatures,
        }
```

Update `_removed_guard_payload()` to return the track index:

```python
            "track_index": int(guard_data["track_indices"][index]),
```

- [ ] **Step 2: Add recovery gate helpers below `_removed_guard_payload()`**

Add:

```python
    def _removed_recovery_allowed(self, group: _ObservationGroup, conflict: dict) -> tuple[bool, str]:
        if not bool(self._cfg("MSDC_REMOVED_RECOVERY_ENABLE", True)):
            return False, "recovery_disabled"

        max_age = int(self._cfg("MSDC_REMOVED_RECOVERY_MAX_AGE", self._cfg("MSDC_REMOVED_GUARD_FRAMES", 80)))
        if int(conflict["signature_age"]) > max_age:
            return False, "signature_too_old"

        min_iou = float(self._cfg("MSDC_REMOVED_RECOVERY_MIN_IOU", 0.20))
        max_center = float(self._cfg("MSDC_REMOVED_RECOVERY_MAX_CENTER_DIST", 80.0))
        if float(conflict["iou"]) < min_iou and float(conflict["center_distance"]) > max_center:
            return False, "geometry_below_recovery_gate"

        min_score = float(self._cfg("MSDC_REMOVED_RECOVERY_MIN_SCORE", 0.35))
        if float(self._spawn_evidence_score(group)) < min_score:
            return False, "score_below_recovery_gate"

        if bool(self._cfg("MSDC_REMOVED_RECOVERY_REQUIRE_CLASS_MATCH", True)):
            signature = conflict.get("removed_signature") if isinstance(conflict.get("removed_signature"), dict) else {}
            removed_class_id = int(signature.get("class_id", -1))
            if removed_class_id >= 0 and int(group.class_id) >= 0 and removed_class_id != int(group.class_id):
                return False, "class_mismatch"

        return True, "safe_removed_signature_match"

    def _recover_removed_track(
        self,
        group: _ObservationGroup,
        existing_tracks: list[EvidenceTrack],
        frame_idx: int,
        conflict: dict,
    ) -> tuple[EvidenceTrack | None, LifecycleEvent | None]:
        ok, reason = self._removed_recovery_allowed(group, conflict)
        if not ok:
            conflict["recovery_block_reason"] = reason
            return None, None

        track_index = int(conflict["track_index"])
        if track_index < 0 or track_index >= len(existing_tracks):
            conflict["recovery_block_reason"] = "track_index_out_of_range"
            return None, None
        track = existing_tracks[track_index]
        if _as_state(track.state) != TrackState.REMOVED:
            conflict["recovery_block_reason"] = "track_not_removed"
            return None, None

        evidence_score = self._spawn_evidence_score(group)
        real_det_hits = 1 if self._group_has_real_detection(group) else 0
        state = TrackState.LOW_CANDIDATE if self._group_should_spawn_low_candidate(group) else TrackState.CANDIDATE
        track.state = state
        track.box = group.box.copy()
        track.velocity = [0.0, 0.0]
        track.evidence_score = evidence_score
        track.hits = 1
        track.misses = 0
        track.age = 1
        track.last_seen = int(frame_idx)
        track.last_real_det_frame = int(frame_idx) if real_det_hits else -1
        track.real_det_hits = real_det_hits
        track.last_real_det_box = group.box.copy() if real_det_hits else None
        track.low_det_history = self._low_history_from_group(group, frame_idx)
        track.source_history = list(group.source_history)
        track.class_id = int(group.class_id)
        track.class_name = str(group.class_name)
        track.retired_signature = None

        event = self._make_event(
            frame_idx=frame_idx,
            track=track,
            event_type="REMOVED_ID_RECOVERY_CANDIDATE",
            from_state=TrackState.REMOVED,
            to_state=state,
            reason=reason,
            extra={
                "recovered_gid": int(track.gid),
                "recovered_public_id": int(track.public_id) if track.public_id is not None else None,
                "guard_iou": float(conflict["iou"]),
                "guard_center_distance": float(conflict["center_distance"]),
                "signature_age": int(conflict["signature_age"]),
                "removed_signature": conflict["removed_signature"],
            },
        )
        return track, event
```

- [ ] **Step 3: Change `spawn_candidates()` to try recovery before allocating a new gid**

In `spawn_candidates()`, after `evidence_score = self._spawn_evidence_score(group)` and before `gid = self.next_gid`, insert:

```python
            removed_conflict = self._removed_guard_conflict(group, existing_tracks, frame_idx)
            if removed_conflict is not None:
                recovered_track, recovery_event = self._recover_removed_track(
                    group=group,
                    existing_tracks=existing_tracks,
                    frame_idx=frame_idx,
                    conflict=removed_conflict,
                )
                if recovered_track is not None and recovery_event is not None:
                    events.append(recovery_event)
                    continue
```

Remove the later duplicate line:

```python
            removed_conflict = self._removed_guard_conflict(group, existing_tracks, frame_idx)
```

Keep the existing legacy event block for `removed_conflict is not None`; it now only runs when recovery is disabled or blocked.

- [ ] **Step 4: Extend removed guard debug counters**

In `_removed_guard_conflict()`, change the debug object to include recoveries:

```python
        self.last_removed_guard_debug = {
            "enabled": True,
            "guard_frames": int(guard_frames),
            "num_recent_signatures": int(len(guard_data["gids"])),
            "num_vetoes": int(matched_indices.size),
            "num_recoveries": 0,
            "vetoes": vetoes,
        }
```

In `_recover_removed_track()`, after building the event and before returning, set:

```python
        if isinstance(self.last_removed_guard_debug, dict):
            self.last_removed_guard_debug["num_recoveries"] = int(self.last_removed_guard_debug.get("num_recoveries", 0)) + 1
            self.last_removed_guard_debug["num_vetoes"] = max(0, int(self.last_removed_guard_debug.get("num_vetoes", 0)) - 1)
```

- [ ] **Step 5: Run focused evidence-state tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py -q
```

Expected:

```text
passed
```

## Task 4: Add Diagnostics For Recovery And Guard Fallback

**Files:**
- Modify: `tools/evaluation/msdc_diagnostic_metrics.py`
- Modify: `test/test_msdc_diagnostic_metrics.py`

- [ ] **Step 1: Add diagnostic fields**

In `DIAGNOSTIC_FIELDS`, add these entries after `reacquire_success_rate`:

```python
    "removed_recovery_attempts",
    "removed_recovery_success",
    "removed_recovery_success_rate",
    "removed_guard_fallback_new_id",
```

- [ ] **Step 2: Count recovery and fallback events**

In `summarize_msdc_diagnostics()`, initialize counters after `reacquire_success = 0`:

```python
    removed_recovery_attempts = 0
    removed_recovery_success = 0
    removed_guard_fallback_new_id = 0
```

Inside the event loop after the reacquire counting block, add:

```python
        if event_type == "REMOVED_ID_RECOVERY_CANDIDATE":
            removed_recovery_attempts += 1
            recovered_public_id = extra.get("recovered_public_id")
            if recovered_public_id not in (None, ""):
                removed_recovery_success += 1

        if event_type == "PREVENT_REMOVED_ID_REUSE":
            removed_recovery_attempts += 1

        if event_type == "NEW_ID_CREATED":
            reason = str(event.get("reason", "")).strip()
            if reason == "removed_guard_conflict_new_gid":
                removed_guard_fallback_new_id += 1
```

In the returned dict, add:

```python
        "removed_recovery_attempts": int(removed_recovery_attempts),
        "removed_recovery_success": int(removed_recovery_success),
        "removed_recovery_success_rate": round(removed_recovery_success / removed_recovery_attempts, 4)
        if removed_recovery_attempts
        else "N/A",
        "removed_guard_fallback_new_id": int(removed_guard_fallback_new_id),
```

- [ ] **Step 3: Add the diagnostic unit test**

Append to `test/test_msdc_diagnostic_metrics.py`:

```python
def test_removed_recovery_diagnostics_count_success_and_fallback(tmp_path):
    events_path = tmp_path / "lifecycle_events.jsonl"
    stage_path = tmp_path / "stage_observations.jsonl"
    per_gt_path = tmp_path / "per_gt.csv"
    _write_jsonl(
        events_path,
        [
            {
                "frame_idx": 10,
                "gid": 2,
                "event_type": "REMOVED_ID_RECOVERY_CANDIDATE",
                "extra": {"recovered_public_id": 4},
            },
            {
                "frame_idx": 11,
                "gid": 3,
                "event_type": "REMOVED_ID_RECOVERY_CANDIDATE",
                "extra": {"recovered_public_id": None},
            },
            {"frame_idx": 20, "gid": 5, "event_type": "PREVENT_removed_ID_REUSE"},
            {
                "frame_idx": 20,
                "gid": 8,
                "event_type": "NEW_ID_CREATED",
                "reason": "removed_guard_conflict_new_gid",
            },
        ],
    )
    _write_jsonl(stage_path, [])
    per_gt_path.write_text("gt_id,predicted_id_count,matched_segments\n", encoding="utf-8")

    row = summarize_msdc_diagnostics(events_path, stage_path, per_gt_path)

    assert row["removed_recovery_attempts"] == 3
    assert row["removed_recovery_success"] == 1
    assert row["removed_recovery_success_rate"] == 0.3333
    assert row["removed_guard_fallback_new_id"] == 1
```

- [ ] **Step 4: Run diagnostic tests**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_diagnostic_metrics.py -q
```

Expected:

```text
passed
```

## Task 5: Add A Before/After Ablation Variant

**Files:**
- Modify: `tools/experiments/run_msdc_ablation.py`
- Modify: `tools/evaluation/detection_replay_benchmark.py` only if a variant-name assertion fails in tests.

- [ ] **Step 1: Enable recovery in the formal v3 environment**

Add these entries to `FORMAL_V3_ENV`:

```python
    "MSDC_REMOVED_RECOVERY_ENABLE": "1",
    "MSDC_REMOVED_RECOVERY_REQUIRE_CLASS_MATCH": "1",
    "MSDC_REMOVED_RECOVERY_MAX_AGE": "80",
    "MSDC_REMOVED_RECOVERY_MIN_IOU": "0.20",
    "MSDC_REMOVED_RECOVERY_MAX_CENTER_DIST": "80",
    "MSDC_REMOVED_RECOVERY_MIN_SCORE": "0.35",
```

- [ ] **Step 2: Add the legacy-veto ablation**

In `ABLATION_VARIANTS`, add:

```python
    "removed_recovery_off": {**FORMAL_V3_ENV, "MSDC_REMOVED_RECOVERY_ENABLE": "0"},
```

- [ ] **Step 3: Verify the variant is accepted by replay CLI**

Run:

```bash
conda run -n ship_detect python tools/evaluation/detection_replay_benchmark.py --help | rg "removed_recovery_off|--variants"
```

Expected:

```text
--variants
```

If `removed_recovery_off` does not appear in the argparse choices text due terminal wrapping, run:

```bash
conda run -n ship_detect python - <<'PY'
from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS
print("removed_recovery_off" in ABLATION_VARIANTS)
print(ABLATION_VARIANTS["removed_recovery_off"]["MSDC_REMOVED_RECOVERY_ENABLE"])
PY
```

Expected:

```text
True
0
```

## Task 6: Run Unit And Smoke Verification

**Files:**
- No planned source edits.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_evidence_state.py \
  test/test_msdc_lifecycle_tracker.py \
  test/test_msdc_diagnostic_metrics.py \
  test/test_detection_replay_benchmark.py \
  test/test_msdc_experiment_summary.py \
  test/test_msdc_speed_benchmark.py \
  -q
```

Expected:

```text
passed
```

- [ ] **Step 2: Run a 300-frame replay smoke test with both variants**

Run:

```bash
conda run -n ship_detect python tools/evaluation/detection_replay_benchmark.py \
  --dataset-root "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集" "/home/hyj/Anti_Drone_Project/USV_MOT标注数据集" \
  --output-root results/msdc_removed_recovery \
  --run-id 20260702_smoke_removed_recovery \
  --max-frames 300 \
  --trackers msdc_elt \
  --variants msdc_v3 removed_recovery_off \
  --source-detection-cache-root results/msdc_paper_phase1/20260624_190710/main/main_full/detections \
  --progress-interval 100 \
  --render \
  --render-class-source cache
```

Expected artifacts:

```text
results/msdc_removed_recovery/20260702_smoke_removed_recovery/visualizations/
results/msdc_removed_recovery/20260702_smoke_removed_recovery/trackers/
results/msdc_removed_recovery/20260702_smoke_removed_recovery/diagnostics/
```

- [ ] **Step 3: Verify the known sequence emits recovery instead of new ID**

Run:

```bash
conda run -n ship_detect python - <<'PY'
import json
from pathlib import Path

root = Path("results/msdc_removed_recovery/20260702_smoke_removed_recovery")
events = sorted(root.glob("**/msdc_v3*/diagnostics/DJI_20250916100639_0001_V/lifecycle_events.jsonl"))
if not events:
    events = sorted(root.glob("**/diagnostics/DJI_20250916100639_0001_V/lifecycle_events.jsonl"))
print(events[0] if events else "missing")
count = 0
for path in events[:1]:
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("event_type") == "REMOVED_ID_RECOVERY_CANDIDATE":
                count += 1
print(count)
PY
```

Expected:

```text
1
```

If the 300-frame smoke does not reach frame 2525 for this sequence, run the same command after Task 7 formal replay.

## Task 7: Formal Before/After Metric And Speed Validation

**Files:**
- No planned source edits unless validation exposes a defect.

- [ ] **Step 1: Run full formal replay with visualization and both variants**

Run:

```bash
conda run -n ship_detect python tools/evaluation/detection_replay_benchmark.py \
  --dataset-root "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集" "/home/hyj/Anti_Drone_Project/USV_MOT标注数据集" \
  --output-root results/msdc_removed_recovery \
  --run-id 20260702_formal_removed_recovery \
  --formal-frame-limit 5400 \
  --trackers botsort msdc_elt \
  --variants msdc_v3 removed_recovery_off \
  --source-detection-cache-root results/msdc_paper_phase1/20260624_190710/main/main_full/detections \
  --progress-interval 500 \
  --render \
  --render-class-source cache
```

Required formal artifacts:

```text
results/msdc_removed_recovery/20260702_formal_removed_recovery/visualizations/
results/msdc_removed_recovery/20260702_formal_removed_recovery/trackers/
results/msdc_removed_recovery/20260702_formal_removed_recovery/diagnostics/
results/msdc_removed_recovery/20260702_formal_removed_recovery/summary/
```

- [ ] **Step 2: Run speed benchmark on the same frame budget**

Run:

```bash
conda run -n ship_detect python tools/evaluation/msdc_speed_benchmark.py \
  --dataset-root "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集" \
  --output-root results/msdc_removed_recovery/20260702_formal_removed_recovery/speed \
  --run-id speed_5400 \
  --frames 5400 \
  --trackers botsort msdc_elt \
  --progress-interval 500 \
  --file-type visible
```

Required speed artifacts:

```text
results/msdc_removed_recovery/20260702_formal_removed_recovery/speed/speed_5400/speed_results.csv
results/msdc_removed_recovery/20260702_formal_removed_recovery/speed/speed_5400/stage_timing.csv
```

- [ ] **Step 3: Extract before/after tracking metrics**

Run:

```bash
conda run -n ship_detect python - <<'PY'
import csv
from pathlib import Path

summary_paths = sorted(Path("results/msdc_removed_recovery/20260702_formal_removed_recovery").glob("**/*summary*.csv"))
for path in summary_paths:
    print(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = [field for field in ["tracker", "variant", "HOTA", "MOTA", "IDF1", "IDSW", "FP", "FN", "IDTP", "IDFP", "IDFN"] if field in (reader.fieldnames or [])]
        if fields:
            print(",".join(fields))
            for row in reader:
                print(",".join(str(row.get(field, "")) for field in fields))
PY
```

Report at minimum:

```text
MOTA, IDF1, IDSW, FN, FP
```

Report when present:

```text
HOTA, IDTP, IDFP, IDFN
```

- [ ] **Step 4: Extract speed and stage timing metrics**

Run:

```bash
conda run -n ship_detect python - <<'PY'
import csv
from pathlib import Path

root = Path("results/msdc_removed_recovery/20260702_formal_removed_recovery/speed/speed_5400")
for name in ["speed_results.csv", "stage_timing.csv"]:
    path = root / name
    print(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        print(",".join(reader.fieldnames or []))
        for row in reader:
            print(",".join(str(row.get(field, "")) for field in (reader.fieldnames or [])))
PY
```

Report at minimum:

```text
total_frames, total_time_s, mean_frame_time_ms, mean_fps
video read/decode, low-threshold detection, high-threshold filtering, lifecycle tracker, visualization/render, write/export
```

- [ ] **Step 5: Check the known DJI sequence identity split**

Run:

```bash
conda run -n ship_detect python - <<'PY'
import json
from pathlib import Path

root = Path("results/msdc_removed_recovery/20260702_formal_removed_recovery")
for path in sorted(root.glob("**/DJI_20250916100639_0001_V/lifecycle_events.jsonl")):
    recovered = 0
    fallback = 0
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("event_type") == "REMOVED_ID_RECOVERY_CANDIDATE":
                recovered += 1
            if row.get("event_type") == "NEW_ID_CREATED" and row.get("reason") == "removed_guard_conflict_new_gid":
                fallback += 1
    print(path, "recovered=", recovered, "fallback=", fallback)
PY
```

Expected for improved `msdc_v3`:

```text
recovered >= 1
```

Expected for `removed_recovery_off`:

```text
fallback >= 1
```

## Task 8: Acceptance Criteria And Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Apply acceptance criteria**

Accept the algorithm change only if all of these are true on the formal run:

```text
1. Improved msdc_v3 IDF1 is higher than removed_recovery_off.
2. Improved msdc_v3 IDSW is not higher than removed_recovery_off by more than 1.
3. Improved msdc_v3 FP is not higher than removed_recovery_off by more than 3%.
4. Improved msdc_v3 mean_fps is at least 95% of removed_recovery_off or lifecycle tracker mean time increases by less than 0.20 ms/frame.
5. DJI_20250916100639_0001_V has at least one REMOVED_ID_RECOVERY_CANDIDATE event in improved msdc_v3.
6. Formal report includes visualization MP4, MOT summary, diagnostic CSV/JSONL, speed CSV, stage timing CSV, and all output paths.
```

- [ ] **Step 2: Update README**

Add a short subsection under the MS-DC evaluation/configuration area:

```markdown
### MS-DC Removed ID Recovery

`MSDC_REMOVED_RECOVERY_ENABLE=1` changes the removed guard from a veto-only rule to a gated recovery rule. A new observation near a recent removed signature can reuse the removed track's `public_id` only when signature age, IoU or center distance, evidence score, and class match gates pass. Unsafe matches keep the legacy behavior and emit `PREVENT_removed_ID_REUSE` plus `NEW_ID_CREATED`.

Relevant environment knobs:

- `MSDC_REMOVED_RECOVERY_ENABLE`
- `MSDC_REMOVED_RECOVERY_REQUIRE_CLASS_MATCH`
- `MSDC_REMOVED_RECOVERY_MAX_AGE`
- `MSDC_REMOVED_RECOVERY_MIN_IOU`
- `MSDC_REMOVED_RECOVERY_MAX_CENTER_DIST`
- `MSDC_REMOVED_RECOVERY_MIN_SCORE`

Formal removed-recovery comparisons must include `msdc_v3` and `removed_recovery_off`, plus the standard MOT summary, visualized MP4, diagnostic files, speed statistics, and stage timing CSV.
```

- [ ] **Step 3: Run final verification**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_evidence_state.py \
  test/test_msdc_lifecycle_tracker.py \
  test/test_msdc_diagnostic_metrics.py \
  test/test_detection_replay_benchmark.py \
  test/test_msdc_experiment_summary.py \
  test/test_msdc_speed_benchmark.py \
  -q
```

Expected:

```text
passed
```

- [ ] **Step 4: Commit**

Run:

```bash
git status --short
git add \
  target_module/image_detect_module/config.py \
  target_module/image_detect_module/utils/evidence_state.py \
  tools/evaluation/msdc_diagnostic_metrics.py \
  tools/experiments/run_msdc_ablation.py \
  test/test_msdc_evidence_state.py \
  test/test_msdc_diagnostic_metrics.py \
  README.md
git commit -m "feat: recover safe removed MS-DC IDs"
```

## Self-Review

Spec coverage:

```text
Controlled old-ID recovery: Tasks 1-3.
Unsafe fallback still prevents bad reuse: Tasks 1 and 3.
Before/after metrics: Tasks 5 and 7.
Speed metrics and stage timings: Task 7.
Formal output paths and visualization: Task 7.
README update after code changes: Task 8.
```

Placeholder scan:

```text
No TBD markers.
No open-ended "add tests" step without concrete test code.
No unspecified output paths.
```

Type consistency:

```text
Event names use existing LifecycleEvent string format.
Track state values use TrackState.REMOVED, TrackState.CANDIDATE, and TrackState.LOW_CANDIDATE.
The ablation variant name is removed_recovery_off in tests, CLI commands, and acceptance criteria.
```
