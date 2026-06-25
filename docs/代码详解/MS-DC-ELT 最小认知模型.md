```
[1. Frame Input]
video_main.py
  cap.read()
  frame_idx = frame_count - 1
        |
        v

[2. Detection Layer]
msdc_detection.py
  resolve_msdc_high_low_boxes()
        |
        |-- low_boxes
        |     run_msdc_low_threshold_detection()
        |
        |-- high_boxes / boxes
              split_msdc_high_from_low_boxes()
        |
        v

[3. ELT Input Adapter]
tracking_update.py
  update_tracking_for_frame()
        |
        |-- if tracker_type == "msdc_elt"
        |
        v

lifecycle_tracker.py
  MSDCLifecycleTracker.update(
    frame,
    frame_idx,
    file_type,
    high_boxes,
    low_boxes
  )
        |
        v

[4. Detection → Observation]
lifecycle_tracker.py
  _filter_low_only_boxes()
  _budget_low_only_boxes()
  _boxes_to_observations()
        |
        v

msdc_types.py
  Observation(
    box=xyxy,
    source=high_det | low_det,
    score,
    frame_idx,
    class_name
  )
        |
        v

[5. ELT State Machine]
evidence_state.py
  EvidenceStateUpdater.update_tracks()
        |
        |-- merge_observations()
        |-- associate_tracks_to_observations()
        |-- _reacquire_lost_track_indices()
        |-- _apply_matched_observation()
        |-- _apply_negative_evidence()
        |-- _resolve_low_candidate_inheritance()
        |-- _transition_track()
        |-- spawn_candidates()
        |
        v

[6. Track States]
msdc_types.py
  LOW_CANDIDATE
  CANDIDATE
  ACTIVE
  LOST
  REMOVED
        |
        v

[7. State Transitions]
evidence_state.py
  None          → LOW_CANDIDATE
  None          → CANDIDATE

  LOW_CANDIDATE → ACTIVE
  LOW_CANDIDATE → REMOVED

  CANDIDATE     → ACTIVE
  CANDIDATE     → REMOVED

  ACTIVE        → LOST
  ACTIVE        → REMOVED

  LOST          → ACTIVE
  LOST          → REMOVED

  REMOVED       → pruned from track list
        |
        v

[8. Output]
lifecycle_tracker.py
  _tracks_to_output_boxes()
        |
        |-- keep ACTIVE tracks
        |-- _track_to_output_box()
        |-- _nms_output_boxes()
        |
        v

msdc_types.py
  xyxy_to_project_box()
        |
        v

tracked_boxes:
  track_id
  x
  y
  w
  h
  confidence
  class
  class_confidence
  gid
  public_id
  lifecycle_state
  real_det_age
```

```
核心压缩版

frame
  → low/high detection
  → Observation(high_det | low_det)
  → EvidenceTrack update
  → state transition
  → ACTIVE tracks only
  → tracked_boxes
```