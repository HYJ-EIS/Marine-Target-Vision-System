# MS-DC-ELT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MS-DC-ELT as an optional lifecycle-aware tracker path while preserving the existing FFCA-YOLO + OC-SORT / BoT-SORT tracking-by-detection baseline.

**Architecture:** Keep OC-SORT / BoT-SORT untouched. Add a separate `MSDCLifecycleTracker` branch behind `--tracker msdc_elt`, reuse existing detector threshold override, GMC, IoU/distance helpers, and output the same tracked box schema expected by visualization, MQ, and MOT export. Add lifecycle diagnostics as separate JSONL under a non-overwriting run directory.

**Tech Stack:** Python, OpenCV, NumPy, existing ONNX Runtime detector wrapper, existing `Config`, existing tracker helpers, existing pytest suite via `conda run -n ship_detect ...`.

---

## 0. Source Documents Read

- Read: `/home/hyj/projects/open-set-id-redesign-logs/refine-logs/FINAL_PROPOSAL.md`.
- Read: `/home/hyj/projects/open-set-id-redesign-logs/refine-logs/STORY_9Q.md`.
- Core idea used: **MS-DC-ELT: Motion-Seeded Delayed-Confirmation Evidence Lifecycle Tracking with Lifecycle-Gated Template Locking**.
- Implementation constraint used: add modules on top of current code; do not rewrite OC-SORT / BoT-SORT internals.

This document is planning only. No code implementation or experiment has been performed in this planning round.

## 1. Code Audit Findings

### 1.1 `video_main.py` tracker initialization and call site

Found.

- `--tracker` is defined in `video_main.py:56-63`.
- Current choices are `""`, `bytetrack`, `ocsort`, `botsort`, `dist_tracker`, `official_ocsort`, `official_botsort` at `video_main.py:59`.
- Tracker is initialized at `video_main.py:124-126`:

```python
tracker_type = args.tracker if args.tracker else None
tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_type)
```

- Main tracking call is at `video_main.py:189-201`:

```python
result = detector.detect_from_image_file(tmp_frame_path, file_type=file_type)
boxes = result.get("data", {}).get("boxes", [])
tracked_boxes = tracker.update(boxes, frame.shape, frame=frame)
```

- Tracking output is written back at `video_main.py:214-220` and visualized with `visualize_detections(...)`.
- Local video output path defaults to `Config.VIDEO_OUTPUT_PATH` at `video_main.py:132-138`.
- RTSP output uses `create_video_output_manager(...)` at `video_main.py:147-154`.
- MQ dispatch is every 2 seconds at `video_main.py:235-241`, using `_mq_sender_worker` at `video_main.py:269-281`.

Planning conclusion: add `msdc_elt` as a new branch in `video_main.py` without changing current baseline branch.

### 1.2 Other tracker CLI choice locations

Found additional choice lists beyond `video_main.py`.

- `tools/evaluation/export_mot_results.py:32-39` defines `TRACKER_CHOICES`.
- `tools/evaluation/export_mot_results.py:116-123` exposes `--tracker`.
- `tools/validation/tracker_effect_test.py:41-48` defines `TRACKER_CHOICES`.
- `tools/validation/tracker_effect_test.py:147-153` exposes `--trackers`.
- `tools/validation/video_test_tracking.py:62-68` exposes `--tracker`.
- `tools/dataset/extract_tracking_frames.py:147` also has tracker choices, but dataset extraction is not required for the first MS-DC-ELT path and should not be changed until the video/evaluation path is stable.

Planning conclusion: minimum implementation changes `video_main.py` first, then `export_mot_results.py` and validation tools when Task 9 starts.

### 1.3 `MultiObjectTracker.update()` input/output format

Found in `target_module/image_detect_module/utils/tracker.py`.

- Module docstring defines input at `tracker.py:14-15`:

```python
[{"x": int, "y": int, "w": int, "h": int, "confidence": float, "class": str}, ...]
```

- Module docstring defines output at `tracker.py:17-19`:

```python
[{"track_id": int, "x": int, "y": int, "w": int, "h": int,
  "confidence": float, "class": str}, ...]
```

- `MultiObjectTracker.__init__` dispatches tracker types at `tracker.py:361-415`.
- `MultiObjectTracker.update(detections, frame_shape, frame=None)` is defined at `tracker.py:417-438`.
- OC-SORT / BoT-SORT branch converts project boxes to `[x1,y1,x2,y2]` at `tracker.py:536-569`.
- OC-SORT / BoT-SORT output dict is built at `tracker.py:571-581`.
- ByteTrack output dict is built at `tracker.py:521-531`.
- DistTracker keeps the same project contract in `dist_tracker.py:43-85` and output dict in `dist_tracker.py:207-230`.

Planning conclusion: MS-DC-ELT must return at least:

```python
{
    "track_id": int,
    "x": int,
    "y": int,
    "w": int,
    "h": int,
    "confidence": float,
    "class": str,
    "class_confidence": float,
}
```

Extra diagnostic fields such as `lifecycle_state`, `evidence_score`, and `observation_sources` are allowed for in-process diagnostics, but MOT export and MQ behavior must be checked explicitly.

### 1.4 `TargetDetector.detect_from_image_file()` input/output format

Found in `target_module/image_detect_module/target_detection.py`.

- Signature at `target_detection.py:98-100`:

```python
def detect_from_image_file(self, image_path: str, output_dir: Optional[str] = None,
                           file_type: Optional[str] = None,
                           enable_tracking: bool = False) -> Dict:
```

- It validates file existence and readability at `target_detection.py:115-126`.
- It calls `_perform_detection(...)` at `target_detection.py:123-126`.
- `_perform_detection(...)` calls `self.processor.process(image_path, file_type=file_type)` at `target_detection.py:173-174`.
- Success response format is built at `target_detection.py:202-213`:

```python
{
    "data": {
        "boxes": boxes,
        "count": len(boxes),
        "result_image_path": ""
    },
    "message": "Success",
    "success": True,
    "timestamp": int(time.time() * 1000),
    "type": image_type
}
```

- Error response format is built at `target_detection.py:260-272`.
- `detect_targets(...)` only supports file path strings at `target_detection.py:286-342`, matching `AGENTS.md` rule 4.

Not found: `detect_from_image_file(..., conf_override=...)` does not exist today.

Planning conclusion: Task 1 should avoid changing `TargetDetector` at first by using `detector.processor.process_frame(...)` directly in the MS-DC-ELT video branch. Adding `conf_override` passthrough to `TargetDetector` can be a later compatibility improvement.

### 1.5 `ImageProcessor.process_frame(frame, file_type, conf_override=None)`

Found.

- Signature exists at `image_processor.py:64`.
- It validates `frame is not None`, `np.ndarray`, and non-empty at `image_processor.py:66-74`.
- It resolves modality and calls `_process_image_array(..., conf_override=...)` at `image_processor.py:76-77`.
- `_process_image_array` calls `detector.detect(img, conf_override=conf_override)` at `image_processor.py:91-99`.
- Output stats are built at `image_processor.py:133-140`:

```python
{
    "boxes": boxes_formatted,
    "count": len(results),
    "total": len(results),
    "type": file_type,
    "processing_time": 0.0,
    "_frame": img,
}
```

Planning conclusion: this is the preferred low-threshold detection hook for video MS-DC-ELT.

### 1.6 `OnnxDetector.detect(img, conf_override=None)`

Found.

- Signature exists at `detectors.py:147`.
- It resolves `conf_thres = self.conf_thres if conf_override is None else float(conf_override)` at `detectors.py:148-149`.
- It filters scores with `mask = scores > conf_thres` at `detectors.py:192`.
- It uses `cv2.dnn.NMSBoxes(..., score_threshold=conf_thres, nms_threshold=self.iou_thres)` at `detectors.py:207-212`.
- It returns detections with `x1,y1,x2,y2,confidence,class,class_confidence` at `detectors.py:233-238`.

Planning conclusion: low-confidence candidate extraction is already supported by the detector layer.

### 1.7 `utils/gmc.py` reuse

Found and reusable.

- `GMC` class exists at `utils/gmc.py:17`.
- Constructor supports `method="sparse_flow"`, `orb`, or `none` at `utils/gmc.py:20-30`.
- `apply(frame)` returns a 2x3 affine matrix and maintains previous grayscale frame at `utils/gmc.py:37-74`.
- `reset()` exists at `utils/gmc.py:76-78`.
- Sparse optical flow method is implemented at `utils/gmc.py:82-115`.
- ORB matching method is implemented at `utils/gmc.py:119-147`.

Planning conclusion: `MotionSeedGenerator` can either own a `GMC` instance or accept a precomputed matrix. For minimal coupling, it should own its own `GMC` and expose `reset()`.

### 1.8 `tracker.py` helper reuse

Found and reusable.

- `_iou_batch` exists at `tracker.py:38-65`.
- `_center_distance_batch` exists at `tracker.py:68-81`.
- `KalmanBoxTracker` is imported in `tracker.py:29` and defined in `utils/kalman_bbox.py:18-198`.
- `KalmanBoxTracker.predict()` returns `[x1,y1,x2,y2]` at `kalman_bbox.py:80-95`.
- `KalmanBoxTracker.update(...)` is at `kalman_bbox.py:97-127`.
- `KalmanBoxTracker.apply_affine(...)` is at `kalman_bbox.py:132-142`.

Planning conclusion: reuse these helpers for association and prediction, but do not use `KalmanBoxTracker._count` as the lifecycle global ID source. MS-DC-ELT needs its own `next_gid` to avoid accidental ID reset semantics.

### 1.9 MOT export output fields

Found in `tools/evaluation/export_mot_results.py`.

- Output layout is documented at `export_mot_results.py:4-8`.
- `format_mot_result_line(frame_id, box)` at `export_mot_results.py:42-49` requires:

```python
box["track_id"]
box["x"]
box["y"]
box["w"]
box["h"]
box.get("confidence", 1.0)
```

- Export path is `<output-root>/<tracker>/data/<seq-name>.txt` at `export_mot_results.py:81`.
- Current export only uses `MultiObjectTracker` at `export_mot_results.py:77-106`.

Planning conclusion: MS-DC-ELT output is MOT-compatible if it supplies the required fields. `export_mot_results.py` needs a new `msdc_elt` branch because `MSDCLifecycleTracker.update(...)` has a different input contract than `MultiObjectTracker.update(...)`.

### 1.10 Visualization tolerance for `lifecycle_state`

Found: visualization can tolerate extra box fields.

- `visualize_detections(...)` iterates boxes at `visualization.py:61-64`.
- It only reads `h,w,x,y,confidence,class,track_id,id,class_confidence` via `box.get(...)` at `visualization.py:66-74`.
- It ignores unknown extra fields.

Planning conclusion: adding `lifecycle_state` to tracked boxes will not break visualization. The label currently does not display it; adding display is optional and should be gated to avoid clutter.

### 1.11 Result directories and logging mechanisms

Found.

- Global output root is `Config.OUTPUT_DIR = os.path.join(BASE_DIR, "results")` at `config.py:9-10`.
- Default video output is `Config.VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.mp4")` at `config.py:57-62`.
- `video_main.py` creates the output video directory at `video_main.py:132-138`.
- `video_main.py` prints progress every 50 frames at `video_main.py:243-250`.
- `video_main.py` prints final summary at `video_main.py:263-266`.
- `VideoOutputManager` uses Python `logging` at `video_output_manager.py:69-82` and creates video writer directories at `video_output_manager.py:115-140`.
- `tools/validation/tracker_effect_test.py` writes `tracks.csv`, `frames.jsonl`, `summary.json`, and optional `error.txt` under `results/tracker_effect_test/...` at `tracker_effect_test.py:213-227`, `tracker_effect_test.py:232-266`, and `tracker_effect_test.py:348-365`.
- `tools/validation/video_detect_only.py` writes detection-only JSONL/CSV/summary outputs under a user-provided output dir at `video_detect_only.py:245-249`, `video_detect_only.py:267-313`, and `video_detect_only.py:335-363`.
- `image_main.py` uses `logging.basicConfig(...)` at `image_main.py:47-56`, has upload/results folders at `image_main.py:66-73`, and saves visualized images at `image_main.py:188-204`.

Planning conclusion: MS-DC-ELT diagnostics should write to a new run directory such as `results/msdc_elt/<timestamp_or_video_stem>/lifecycle_events.jsonl`. Never write to existing `results/output.mp4` or existing tracker result directories unless explicitly requested.

### 1.12 Existing MS-DC-ELT code status

Not found.

- `target_module/image_detect_module/utils/msdc_types.py`: not found.
- `target_module/image_detect_module/utils/motion_seed.py`: not found.
- `target_module/image_detect_module/utils/evidence_state.py`: not found.
- `target_module/image_detect_module/utils/template_lock.py`: not found.
- `target_module/image_detect_module/utils/lifecycle_tracker.py`: not found.
- `lifecycle_state` field: not found in project code.
- candidate / active / lost / retired lifecycle implementation: not found, except unrelated third-party BoT-SORT internal `TrackState`.

## 2. Overall Integration Strategy

1. Do not modify OC-SORT / BoT-SORT internals in `target_module/image_detect_module/utils/tracker.py`.
2. Keep existing `MultiObjectTracker` behavior unchanged for `botsort`, `ocsort`, `dist_tracker`, `official_ocsort`, and `official_botsort`.
3. Add `--tracker msdc_elt` as an explicit opt-in branch.
4. Add `MSDCLifecycleTracker` under `target_module/image_detect_module/utils/lifecycle_tracker.py`.
5. MS-DC-ELT should return tracked boxes compatible with the current project schema:

```python
{
    "track_id": int,
    "x": int,
    "y": int,
    "w": int,
    "h": int,
    "confidence": float,
    "class": str,
    "class_confidence": float,
}
```

6. MS-DC-ELT may add optional fields:

```python
{
    "lifecycle_state": "active",
    "evidence_score": float,
    "observation_sources": ["high_det", "motion"]
}
```

7. Lifecycle diagnostics should be written separately:

```text
results/msdc_elt/<run_id>/lifecycle_events.jsonl
results/msdc_elt/<run_id>/low_conf_debug.jsonl
results/msdc_elt/<run_id>/motion_seed_debug.jsonl
results/msdc_elt/<run_id>/summary.json
```

8. Use a unique `run_id`, for example `<video_stem>_<YYYYMMDD_HHMMSS>`, to avoid overwriting existing results.
9. Add a config switch `Config.MSDC_ENABLE = False` and require either `--tracker msdc_elt` or explicit config enablement. Other trackers should never instantiate MS-DC-ELT.

## 3. New Module Plan

### 3.1 `target_module/image_detect_module/utils/msdc_types.py`

**Purpose:** Defines lightweight, JSON-serializable lifecycle data structures.

**Inputs:** Project boxes, frame index, modality, source names, evidence scores.

**Outputs:** Dataclass instances and dicts suitable for JSONL diagnostics.

**Core classes:**

- `TrackState`: enum-like string constants: `candidate`, `active`, `lost`, `retired`.
- `Observation`: one evidence item from high detector, low detector, motion seed, template, or reacquire.
- `EvidenceTrack`: lifecycle state for one target hypothesis.
- `LifecycleEvent`: serialized transition/debug record.

**Core functions:**

- `xywh_to_xyxy(box: dict) -> np.ndarray`
- `xyxy_to_project_box(xyxy, track_id, confidence, cls, class_confidence=None, **extras) -> dict`
- `observation_from_project_box(box, source, frame_idx, modality, reliability=1.0) -> Observation`
- `event_to_json_dict(event: LifecycleEvent) -> dict`

**Dependencies:** Standard library `dataclasses`, `enum` or string constants, `typing`; NumPy only if conversion helpers return arrays.

**Minimum unit tests:**

- `Observation.to_dict()` round-trips through `json.dumps/json.loads`.
- `EvidenceTrack.to_debug_dict()` contains `gid`, `state`, `evidence_score`, `box`.
- `xywh_to_xyxy` and `xyxy_to_project_box` preserve coordinates.

### 3.2 `target_module/image_detect_module/utils/motion_seed.py`

**Purpose:** Produces class-agnostic motion candidate boxes from consecutive frames.

**Inputs:** `prev_frame`, `current_frame`, optional frame index and modality.

**Outputs:** List of `Observation` or project-style motion boxes; optional debug mask.

**Core classes:**

- `MotionSeedGenerator`

**Core functions/methods:**

- `MotionSeedGenerator.__init__(method, downscale, min_area_ratio, max_area_ratio, diff_percentile)`
- `MotionSeedGenerator.reset()`
- `MotionSeedGenerator.update(frame, frame_idx, modality) -> list[Observation]`
- `_compute_compensated_difference(prev_frame, frame) -> np.ndarray`
- `_mask_to_boxes(mask, frame_shape) -> list[dict]`
- `write_debug_mask(path, mask)` only when debug is enabled

**Dependencies:** OpenCV, NumPy, existing `GMC` from `utils/gmc.py`.

**Minimum unit tests:**

- Two synthetic frames with one moving bright square produce one motion box near the square.
- Identical frames produce no boxes.
- First frame returns no boxes and initializes internal previous frame.
- `reset()` clears previous frame state.

### 3.3 `target_module/image_detect_module/utils/evidence_state.py`

**Purpose:** Merges observations, updates evidence scores, and handles candidate/active/lost/retired transitions.

**Inputs:** Existing `EvidenceTrack` list, high/low/motion/template observations, frame shape, frame index.

**Outputs:** Updated tracks, lifecycle events, active output tracks.

**Core classes:**

- `EvidenceStateUpdater`
- `EvidenceConfig` if keeping thresholds separate from `Config` is useful.

**Core functions/methods:**

- `merge_observations(high_obs, low_obs, motion_obs) -> list[Observation]`
- `associate_tracks_to_observations(tracks, observations, frame_shape) -> matches`
- `update_tracks(tracks, observations, frame_idx, frame_shape) -> tuple[list[EvidenceTrack], list[LifecycleEvent]]`
- `spawn_candidates(unmatched_observations)`
- `transition(track, reason) -> LifecycleEvent`
- `active_tracks_to_project_boxes(tracks) -> list[dict]`

**Dependencies:** NumPy, `scipy.optimize.linear_sum_assignment` if matching follows existing trackers, existing `_iou_batch`, `_center_distance_batch`, and optionally `KalmanBoxTracker`.

**Minimum unit tests:**

- A low-confidence observation repeated for `MSDC_CONFIRM_MIN_HITS` frames transitions candidate to active when `evidence_score >= MSDC_CONFIRM_SCORE`.
- A one-frame low observation is pruned before active.
- An active track with no observations transitions to lost after `MSDC_ACTIVE_MISSING_PATIENCE`.
- A lost track transitions to retired after `MSDC_LOST_MAX_AGE`.

### 3.4 `target_module/image_detect_module/utils/template_lock.py`

**Purpose:** Provides active-only instance matching using a bounded template bank.

**Inputs:** Current frame, active `EvidenceTrack` objects with templates, predicted boxes.

**Outputs:** Template observations and template scores.

**Core classes:**

- `TemplateLock`
- `TemplateRecord`

**Core functions/methods:**

- `initialize(track, frame)`
- `match(track, frame) -> Observation | None`
- `update_template(track, frame, matched_box, score)`
- `evict_if_over_budget(active_tracks)`
- `_crop_with_padding(frame, box, scale)`
- `_match_ncc(template, search_patch) -> tuple[box, score]`

**Dependencies:** OpenCV, NumPy. First implementation should use `cv2.matchTemplate` with normalized correlation; no large new dependency.

**Minimum unit tests:**

- A synthetic patch shifted by a small offset is matched with high NCC.
- Candidate tracks are ignored.
- Template update is skipped when score is below `MSDC_TEMPLATE_UPDATE_THRESH`.
- Number of active templates never exceeds `MSDC_MAX_ACTIVE_TEMPLATES`.

### 3.5 `target_module/image_detect_module/utils/lifecycle_tracker.py`

**Purpose:** Top-level MS-DC-ELT tracker that coordinates low/high detector observations, motion seeds, evidence update, optional template lock, and diagnostics.

**Inputs:** `frame`, `file_type`, `high_boxes`, `low_boxes`, `frame_idx`, optional `timestamp_sec`.

**Outputs:** Existing project-compatible `tracked_boxes`, plus lifecycle JSONL diagnostics.

**Core class:**

- `MSDCLifecycleTracker`

**Core functions/methods:**

- `__init__(frame_rate, output_dir=None, enable_template=False, debug=False)`
- `reset()`
- `update(frame, file_type, high_boxes, low_boxes=None, frame_idx=None, timestamp_sec=None) -> list[dict]`
- `_boxes_to_observations(boxes, source, frame_idx, modality)`
- `_low_only_boxes(high_boxes, low_boxes)`
- `_write_debug_event(event)`
- `_write_frame_summary(frame_idx, observations, tracks)`

**Dependencies:** OpenCV, NumPy, `MotionSeedGenerator`, `EvidenceStateUpdater`, `TemplateLock`, `msdc_types`, existing `Config`.

**Minimum unit tests:**

- With high boxes in consecutive frames, returns active project boxes with stable `track_id`.
- With empty high boxes but valid low/motion observations, candidate accumulates and becomes active only after confirmation threshold.
- Returned boxes pass the same schema assertion used by `test/test_dist_tracker.py`.
- Extra `lifecycle_state` does not break `visualize_detections`.

## 4. Existing File Change Plan

### 4.1 `target_module/image_detect_module/config.py`

**Must modify:** Yes.

**Modification points:**

- Add `MSDC_ENABLE = False`.
- Add low-confidence thresholds:
  - `MSDC_LOW_CONF_VISIBLE`
  - `MSDC_LOW_CONF_INFRARED`
- Add lifecycle thresholds:
  - `MSDC_CANDIDATE_MAX_AGE`
  - `MSDC_CONFIRM_MIN_HITS`
  - `MSDC_CONFIRM_SCORE`
  - `MSDC_PRUNE_SCORE`
  - `MSDC_ACTIVE_MISSING_PATIENCE`
  - `MSDC_LOST_MAX_AGE`
  - `MSDC_REACQUIRE_INTERVAL`
  - `MSDC_RETIRED_GUARD_FRAMES`
- Add budgets:
  - `MSDC_MAX_CANDIDATES`
  - `MSDC_MAX_ACTIVE_TEMPLATES`
- Add motion seed thresholds:
  - `MSDC_MOTION_MIN_AREA_RATIO`
  - `MSDC_MOTION_MAX_AREA_RATIO`
  - `MSDC_MOTION_DIFF_PERCENTILE`
- Add output settings:
  - `MSDC_OUTPUT_ROOT = os.path.join(OUTPUT_DIR, "msdc_elt")`
  - `MSDC_DEBUG = False`

**Backward compatibility:** Defaults leave current trackers unchanged. `TRACKER_TYPE` remains `"botsort"` unless user explicitly passes `--tracker msdc_elt`.

**Close switch:** `MSDC_ENABLE = False` and not selecting `--tracker msdc_elt`.

### 4.2 `video_main.py`

**Must modify:** Yes for user-facing video path.

**Modification points:**

- Add `"msdc_elt"` to `--tracker` choices.
- Import `MSDCLifecycleTracker` only inside the `msdc_elt` branch to avoid affecting baseline imports.
- Instantiate:

```python
if selected_tracker == "msdc_elt":
    tracker = MSDCLifecycleTracker(frame_rate=fps, output_dir=msdc_run_dir)
else:
    tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_type)
```

- For `msdc_elt`, obtain high and low boxes via in-memory detection:

```python
high_stats = detector.processor.process_frame(frame, file_type)
low_stats = detector.processor.process_frame(frame, file_type, conf_override=low_threshold)
tracked_boxes = tracker.update(frame, file_type, high_boxes, low_boxes, frame_idx=frame_count)
```

- For all other trackers, preserve the current `detect_from_image_file(...) -> MultiObjectTracker.update(...)` branch exactly.
- Use `imwrite_unicode` instead of direct `cv2.imwrite` if touching the temp frame path, because `AGENTS.md` requires unicode-safe disk image writing. If the baseline branch is left untouched in Task 6, do not change this in the same patch.

**Backward compatibility:** All existing tracker choices continue to use the old code path and output schema.

**Close switch:** Select any tracker other than `msdc_elt`.

### 4.3 `target_module/image_detect_module/target_detection.py`

**Must modify:** Not required for initial video MS-DC-ELT. Optional later.

**Optional modification:** Add `conf_override` passthrough to:

- `TargetDetector.detect_from_image_file(...)`
- `TargetDetector._perform_detection(...)`

Then call `self.processor.process(image_path, file_type=file_type, conf_override=conf_override)`.

**Backward compatibility:** Default `conf_override=None` preserves existing behavior.

**Close switch:** Do not use the new argument.

### 4.4 `visualization.py`

**Must modify:** No for compatibility.

**Audit result:** It ignores extra fields like `lifecycle_state`, because it only reads known keys through `box.get(...)`.

**Optional modification:** Add a debug display mode to append lifecycle state in the label:

```text
UAV:0.82 ID:3 active
```

This should be gated by a config flag or CLI flag to avoid clutter.

**Backward compatibility:** No change required for default visualization.

### 4.5 `tools/evaluation/export_mot_results.py`

**Must modify:** Yes for formal MS-DC-ELT export.

**Modification points:**

- Add `"msdc_elt"` to `TRACKER_CHOICES`.
- If `tracker_type != "msdc_elt"`, preserve current `MultiObjectTracker` path.
- If `tracker_type == "msdc_elt"`, instantiate `MSDCLifecycleTracker`.
- Use `detector.processor.process_frame(...)` for high/low detections.
- Call `format_mot_result_line(frame_id, box)` unchanged, because MS-DC-ELT emits compatible boxes.
- Write lifecycle diagnostics under `output_root / "msdc_elt" / "diagnostics" / seq_name`.

**Backward compatibility:** Existing exports for botsort/ocsort/etc. remain unchanged.

**Close switch:** Run export with any existing tracker choice.

### 4.6 Additional existing files to update later

- `tools/validation/tracker_effect_test.py`: add `msdc_elt` for side-by-side visual comparison after Task 6.
- `tools/validation/video_test_tracking.py`: add `msdc_elt` only after `video_main.py` path works.
- `README.md`: required by `AGENTS.md` after code changes; document new tracker option and diagnostics.
- `messaging/mq_publisher.py`: not required initially. Current MQ converter strips unknown fields, so lifecycle diagnostics should stay out-of-band JSONL unless MQ consumers request them.

## 5. Task-by-Task Implementation Plan

### Task 0: Freeze Baseline

**Goal:** Ensure existing trackers still run before any MS-DC-ELT changes.

**Modify files:** None.

**Create files:** None.

**Core interface:** Existing `video_main.py --tracker botsort|ocsort`.

**Minimum commands:**

```bash
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker botsort --output "results/baseline_check/botsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker ocsort --output "results/baseline_check/ocsort.mp4" --no-display
conda run -n ship_detect pytest test/test_dist_tracker.py test/test_mot_result_export.py test/test_tracker_effect_export.py -q
```

**Acceptance criteria:**

- Existing commands do not raise tracker-choice or schema errors.
- Existing output boxes still contain `track_id,x,y,w,h,confidence,class`.
- No output format change in `format_mot_result_line`.

**Failure triage:**

- If model path fails, check `Config.BASE_DIR`, ONNX paths, and current working directory.
- If video open fails, check sample path and codecs.
- If tracker choice fails, inspect `video_main.py:59` and `MultiObjectTracker.__init__`.

**Rollback:** None; no changes.

### Task 1: Low-Threshold Detection Debug Path

**Goal:** Verify high/low detector boxes can be obtained without connecting a new tracker.

**Modify files:**

- `target_module/image_detect_module/config.py`
- Optional small debug script: `tools/validation/msdc_low_conf_debug.py`

**Create files:**

- Prefer create `tools/validation/msdc_low_conf_debug.py` rather than changing `video_main.py` for this task.

**Core interface:**

```python
stats_high = detector.processor.process_frame(frame, file_type)
stats_low = detector.processor.process_frame(frame, file_type, conf_override=low_threshold)
```

**Minimum command:**

```bash
conda run -n ship_detect python tools/validation/msdc_low_conf_debug.py --input "<sample.mp4>" --file-type visible --max-frames 100 --output-dir "results/msdc_elt/low_conf_debug"
```

**Expected output:**

```text
results/msdc_elt/low_conf_debug/<video_stem>_<run_id>/low_conf_debug.jsonl
results/msdc_elt/low_conf_debug/<video_stem>_<run_id>/summary.json
```

**Acceptance criteria:**

- JSONL has one row per processed frame.
- Each row includes `frame_index`, `high_count`, `low_count`, `low_only_count`, `low_high_overlap_ratio`.
- Baseline `video_main.py --tracker botsort` remains unchanged.

**Failure triage:**

- If `detector.processor` is `None`, inspect ONNX model loading in `ImageProcessor.__init__`.
- If `low_count < high_count`, inspect overlap filtering; raw low threshold should usually include at least high detections unless NMS changes behavior.
- If performance is too slow, note it in summary; do not optimize before confirming correctness.

**Rollback:**

- Remove `tools/validation/msdc_low_conf_debug.py`.
- Remove new `MSDC_LOW_CONF_*` config constants if no later task uses them.

### Task 2: Implement `MotionSeedGenerator`

**Goal:** Generate motion candidate boxes from consecutive frames, independent of tracker.

**Modify files:**

- `target_module/image_detect_module/config.py`

**Create files:**

- `target_module/image_detect_module/utils/motion_seed.py`
- `test/test_msdc_motion_seed.py`

**Core interface:**

```python
generator = MotionSeedGenerator(
    method=Config.GMC_METHOD,
    downscale=Config.GMC_DOWNSCALE,
)
motion_observations = generator.update(frame, frame_idx=frame_idx, modality=file_type)
```

**Minimum command:**

```bash
conda run -n ship_detect pytest test/test_msdc_motion_seed.py -q
conda run -n ship_detect python tools/validation/msdc_low_conf_debug.py --input "<sample.mp4>" --file-type visible --max-frames 100 --enable-motion-debug --output-dir "results/msdc_elt/motion_debug"
```

**Acceptance criteria:**

- First frame initializes state and returns no motion boxes.
- Synthetic moving object test returns a box overlapping the moved square.
- Identical frames return no boxes or only below-threshold boxes.
- Debug mask is written only when explicitly enabled and to a unique run directory.

**Failure triage:**

- If every frame returns huge boxes, inspect GMC matrix and morphology thresholds.
- If no boxes appear on synthetic motion, inspect grayscale conversion, threshold percentile, and area filters.
- If debug image writing fails on non-ASCII paths, use `imwrite_unicode`.

**Rollback:**

- Delete `motion_seed.py` and `test/test_msdc_motion_seed.py`.
- Remove motion-related `MSDC_*` config constants if unused.

### Task 3: Implement `msdc_types.py`

**Goal:** Define serializable lifecycle types.

**Modify files:** None initially.

**Create files:**

- `target_module/image_detect_module/utils/msdc_types.py`
- `test/test_msdc_types.py`

**Core interface:**

```python
obs = Observation.from_project_box(box, source="low_det", frame_idx=12, modality="visible")
event = LifecycleEvent(frame_idx=12, gid=3, old_state="candidate", new_state="active", reason="confirm")
json.dumps(event.to_dict(), ensure_ascii=False)
```

**Minimum command:**

```bash
conda run -n ship_detect pytest test/test_msdc_types.py -q
```

**Acceptance criteria:**

- All dataclasses are JSON serializable through `.to_dict()`.
- Box conversion helpers pass deterministic coordinate tests.
- Track states are constrained to `candidate`, `active`, `lost`, `retired`.

**Failure triage:**

- If NumPy scalar types break JSON serialization, cast to Python `int`/`float` in `.to_dict()`.
- If boxes have negative width/height, normalize and clip at conversion boundaries.

**Rollback:**

- Delete `msdc_types.py` and `test/test_msdc_types.py`.

### Task 4: Implement `EvidenceStateUpdater`

**Goal:** Merge high/low/motion observations and maintain basic lifecycle transitions without template lock.

**Modify files:**

- `target_module/image_detect_module/config.py`

**Create files:**

- `target_module/image_detect_module/utils/evidence_state.py`
- `test/test_msdc_evidence_state.py`

**Core interface:**

```python
updater = EvidenceStateUpdater()
tracks, events = updater.update(
    tracks=tracks,
    observations=observations,
    frame_idx=frame_idx,
    frame_shape=frame.shape,
)
active_boxes = updater.active_tracks_to_project_boxes(tracks)
```

**Minimum command:**

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py -q
```

**Acceptance criteria:**

- Candidate confirmation requires repeated evidence and threshold crossing.
- Candidate false positives are pruned.
- Active tracks with missing observations move to lost, then retired.
- Returned active boxes match existing tracker schema.

**Failure triage:**

- If candidate never confirms, inspect `MSDC_CONFIRM_SCORE`, score decay, and source weights.
- If false candidates confirm too easily, inspect `MSDC_CONFIRM_MIN_HITS`, `MSDC_PRUNE_SCORE`, and negative evidence penalties.
- If track IDs reset unintentionally, inspect lifecycle `next_gid` handling.

**Rollback:**

- Delete `evidence_state.py` and `test/test_msdc_evidence_state.py`.
- Remove evidence-related `MSDC_*` config constants if no later task uses them.

### Task 5: Implement `MSDCLifecycleTracker-lite`

**Goal:** Create the first working MS-DC-ELT tracker without template lock.

**Modify files:**

- `target_module/image_detect_module/config.py`

**Create files:**

- `target_module/image_detect_module/utils/lifecycle_tracker.py`
- `test/test_msdc_lifecycle_tracker.py`

**Core interface:**

```python
tracker = MSDCLifecycleTracker(frame_rate=fps, output_dir=debug_dir, enable_template=False)
tracked_boxes = tracker.update(
    frame=frame,
    file_type=file_type,
    high_boxes=high_boxes,
    low_boxes=low_boxes,
    frame_idx=frame_idx,
    timestamp_sec=timestamp_sec,
)
```

**Minimum command:**

```bash
conda run -n ship_detect pytest test/test_msdc_types.py test/test_msdc_motion_seed.py test/test_msdc_evidence_state.py test/test_msdc_lifecycle_tracker.py -q
```

**Acceptance criteria:**

- Output boxes include existing tracker fields.
- Optional `lifecycle_state` is `"active"` for emitted boxes.
- Diagnostics JSONL includes frame events and state transitions.
- Empty detections do not crash; active tracks can enter lost rather than disappearing immediately.

**Failure triage:**

- If visualization fails, inspect output schema and `track_id`.
- If JSONL fails, inspect output directory creation and Python scalar conversion.
- If runtime is slow, log per-stage timings; do not optimize before correctness is stable.

**Rollback:**

- Delete `lifecycle_tracker.py` and `test/test_msdc_lifecycle_tracker.py`.
- Leave earlier modules only if still used by validation tools.

### Task 6: Connect `video_main.py`

**Goal:** Expose `--tracker msdc_elt` while preserving all baseline tracker behavior.

**Modify files:**

- `video_main.py`
- `README.md` after code change, per `AGENTS.md`

**Create files:** None, unless adding a small helper for run ID generation.

**Core interface:**

```bash
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker msdc_elt --output "results/msdc_elt_runs/<run_id>/annotated.mp4" --no-display
```

**Minimum commands:**

```bash
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker botsort --output "results/regression/botsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker ocsort --output "results/regression/ocsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker msdc_elt --output "results/msdc_elt_runs/check/annotated.mp4" --no-display
```

**Acceptance criteria:**

- `botsort` and `ocsort` branches still run through `MultiObjectTracker`.
- `msdc_elt` branch uses `MSDCLifecycleTracker`.
- `result["data"]["boxes"]` remains a list of project boxes before visualization/MQ.
- Diagnostics are created only for `msdc_elt`.

**Failure triage:**

- If `argparse` rejects `msdc_elt`, inspect `video_main.py` choices.
- If baseline trackers break, compare branch conditions and restore old path.
- If `detector.processor.process_frame` is unavailable, inspect detector initialization and import path.

**Rollback:**

- Remove `"msdc_elt"` from `video_main.py` choices.
- Remove MS-DC-ELT branch and import.
- Revert README MS-DC-ELT documentation.

### Task 7: Implement `TemplateLock`

**Goal:** Add active-only template matching with bounded compute.

**Modify files:**

- `target_module/image_detect_module/config.py`
- `target_module/image_detect_module/utils/lifecycle_tracker.py`
- `target_module/image_detect_module/utils/evidence_state.py` if template observations feed evidence scoring.

**Create files:**

- `target_module/image_detect_module/utils/template_lock.py`
- `test/test_msdc_template_lock.py`

**Core interface:**

```python
template_lock = TemplateLock(max_templates=Config.MSDC_MAX_ACTIVE_TEMPLATES)
template_obs = template_lock.match_active_tracks(active_tracks, frame, frame_idx)
```

**Minimum command:**

```bash
conda run -n ship_detect pytest test/test_msdc_template_lock.py test/test_msdc_lifecycle_tracker.py -q
```

**Acceptance criteria:**

- Candidate tracks do not get templates.
- Active tracks can get template observations.
- Template update is gated by score and track state.
- Template count stays within `MSDC_MAX_ACTIVE_TEMPLATES`.
- Turning template lock off returns the same behavior as Task 5 lite mode.

**Failure triage:**

- If template drifts in synthetic test, inspect search window scale and update threshold.
- If compute spikes, inspect template budget and crop size.
- If active tracks disappear when template fails, ensure template evidence is optional, not mandatory.

**Rollback:**

- Set `MSDC_TEMPLATE_ENABLE = False`.
- Remove `TemplateLock` calls from `lifecycle_tracker.py`.
- Keep `template_lock.py` if tests remain isolated, or delete it with its tests.

### Task 8: Implement Lost Reacquire and Retired Guard

**Goal:** Close the lifecycle loop with low-frequency reacquire and ID reuse suppression.

**Modify files:**

- `target_module/image_detect_module/utils/evidence_state.py`
- `target_module/image_detect_module/utils/lifecycle_tracker.py`
- `target_module/image_detect_module/config.py`

**Create files:**

- Additional tests in `test/test_msdc_evidence_state.py`
- Additional tests in `test/test_msdc_lifecycle_tracker.py`

**Core interface:**

```python
reacquire_matches = updater.reacquire_lost_tracks(lost_tracks, observations, frame_idx)
reuse_allowed = updater.retired_guard_allows(candidate, retired_tracks, frame_idx)
```

**Minimum command:**

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py test/test_msdc_lifecycle_tracker.py -q
```

**Acceptance criteria:**

- Lost tracks are only reacquired on `MSDC_REACQUIRE_INTERVAL`.
- Retired tracks are not emitted as normal outputs.
- A new candidate cannot inherit a retired `gid` unless guard conditions pass.
- Diagnostics include `lost_to_active`, `lost_to_retired`, and `reuse_veto` events.

**Failure triage:**

- If old IDs revive too easily, inspect retired signature checks.
- If no lost tracks reacquire, inspect interval scheduling and search gates.
- If retired state grows without bound, inspect `MSDC_RETIRED_GUARD_FRAMES` cleanup.

**Rollback:**

- Disable reacquire with `MSDC_REACQUIRE_INTERVAL = 0`.
- Disable retired guard by setting `MSDC_RETIRED_GUARD_FRAMES = 0`.
- Revert only the Task 8 changes if Task 5/7 behavior was stable.

### Task 9: Connect Evaluation and Diagnostics

**Goal:** Make MS-DC-ELT evaluable without changing MOT result format.

**Modify files:**

- `tools/evaluation/export_mot_results.py`
- `tools/validation/tracker_effect_test.py`
- `tools/validation/video_test_tracking.py`
- `README.md`

**Create files:**

- Optional `tools/evaluation/msdc_lifecycle_metrics.py`
- Optional tests:
  - `test/test_msdc_mot_export.py`
  - `test/test_msdc_lifecycle_metrics.py`

**Core interface:**

```bash
conda run -n ship_detect python tools/evaluation/export_mot_results.py --input "<sample.mp4>" --tracker msdc_elt --seq-name "seq01" --output-root "results/motchallenge_trackers"
conda run -n ship_detect python tools/validation/tracker_effect_test.py --trackers botsort msdc_elt --modalities RGB --max-frames 100
```

**Acceptance criteria:**

- MOT txt rows remain `frame,id,x,y,w,h,conf,-1,-1,-1`.
- `format_mot_result_line` remains unchanged.
- MS-DC-ELT diagnostics are written beside, not inside, MOT result txt.
- Existing tracker effect outputs still write `annotated.mp4`, `tracks.csv`, `frames.jsonl`, `summary.json`.

**Failure triage:**

- If `export_mot_results.py` rejects `msdc_elt`, inspect `TRACKER_CHOICES`.
- If export crashes due to tracker interface mismatch, ensure `msdc_elt` branch does not instantiate `MultiObjectTracker`.
- If lifecycle metrics disagree with MOT frame count, check frame indexing and skipped frames.

**Rollback:**

- Remove `msdc_elt` from evaluation/validation `TRACKER_CHOICES`.
- Remove MS-DC-ELT branch from export and validation tools.
- Keep core tracker modules if `video_main.py --tracker msdc_elt` still works.

## 6. Test Plan by Layer

### Unit tests

```bash
conda run -n ship_detect pytest \
  test/test_msdc_types.py \
  test/test_msdc_motion_seed.py \
  test/test_msdc_evidence_state.py \
  test/test_msdc_template_lock.py \
  test/test_msdc_lifecycle_tracker.py \
  -q
```

### Baseline regression tests

```bash
conda run -n ship_detect pytest \
  test/test_dist_tracker.py \
  test/test_official_tracker_adapter.py \
  test/test_mot_result_export.py \
  test/test_tracker_effect_export.py \
  -q
```

### Smoke commands

```bash
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker botsort --output "results/regression/botsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker ocsort --output "results/regression/ocsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker msdc_elt --output "results/msdc_elt_runs/<run_id>/annotated.mp4" --no-display
```

## 7. Non-Goals and Prohibitions

- Do not delete or replace baseline trackers.
- Do not rewrite `MultiObjectTracker` into a lifecycle tracker.
- Do not modify dataset source files.
- Do not introduce large new dependencies.
- Do not claim experiments are completed until they are actually run.
- Do not write diagnostics into existing result directories unless a unique run subdirectory is created.
- Do not implement all modules in one patch; complete Task 0 through Task 9 incrementally.
- Do not use `detect_targets(...)` with in-memory frames; `AGENTS.md` states it only supports file path strings.

## 8. Open Risks Before Implementation

1. `video_main.py` currently writes temp frames using `cv2.imwrite`; `AGENTS.md` prefers `imwrite_unicode` for disk image writes. Do not mix this cleanup into the MS-DC-ELT branch unless it is necessary for the touched path.
2. `TargetDetector.detect_from_image_file` lacks `conf_override`; first implementation should use `detector.processor.process_frame` directly.
3. Current MQ conversion strips class and extra fields in `messaging/mq_publisher.py`, so lifecycle diagnostics should be file-based until MQ schema changes are requested.
4. `KalmanBoxTracker.reset_count()` resets local tracker IDs; MS-DC-ELT needs a separate lifecycle ID allocator.
5. Current `visualization.py` tolerates extra fields but does not show lifecycle state; visual lifecycle debugging should be an optional later change.
6. Several repository files are already modified/untracked before this plan was created. Future implementation workers must preserve unrelated user changes.

## 9. Completion Checklist for Future Implementation

- [ ] Baseline `botsort` and `ocsort` still run.
- [ ] `--tracker msdc_elt` is opt-in only.
- [ ] MS-DC-ELT output boxes match existing tracker schema.
- [ ] Lifecycle diagnostics are separate JSONL files.
- [ ] MOT export still emits the standard 10 columns.
- [ ] Unit tests cover each new module.
- [ ] README documents new tracker option and diagnostics.
- [ ] No dataset files are modified.
- [ ] No existing result directory is overwritten by default.
