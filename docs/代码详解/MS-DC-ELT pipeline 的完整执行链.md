```
1. frame extraction
   file: video_main.py
   function: main()
   line: cap.read()
   output: frame

2. frame index
   file: video_main.py
   function: main()
   line: frame_count += 1
   output: frame_idx = frame_count - 1

3. MS-DC-ELT branch
   file: video_main.py
   function: main()
   line: if use_msdc_elt:
   output: msdc_elt execution path

4. detection
   file: video_main.py
   function: main()
   line: boxes, low_boxes = resolve_msdc_high_low_boxes(detector, frame, file_type)
   output: boxes, low_boxes

5. low-threshold detection
   file: target_module/image_detect_module/utils/msdc_detection.py
   function: run_msdc_low_threshold_detection()
   line: processor.process_frame(frame, file_type, conf_override=low_conf)
   output: low_boxes

6. high-threshold split
   file: target_module/image_detect_module/utils/msdc_detection.py
   function: split_msdc_high_from_low_boxes()
   line: confidence >= threshold
   output: boxes

7. adapter
   file: target_module/image_detect_module/utils/official_adapter.py
   function: OfficialTrackerAdapter
   status: not used by msdc_elt
   used_by: official_ocsort, official_botsort

8. ELT dispatch
   file: video_main.py
   function: _update_result_tracking_for_frame()
   line: tracked_boxes = update_tracking_for_frame(...)
   input: frame, frame_idx, file_type, boxes, low_boxes
   output: tracked_boxes

9. tracker type gate
   file: target_module/image_detect_module/utils/tracking_update.py
   function: is_msdc_tracker()
   line: return tracker_type == "msdc_elt"
   output: True

10. ELT input interface
    file: target_module/image_detect_module/utils/tracking_update.py
    function: update_tracking_for_frame()
    line: lifecycle_tracker.update(...)
    input:
      frame
      frame_idx
      file_type
      high_boxes=boxes
      low_boxes=low_boxes
    output: lifecycle_tracker output

11. ELT tracker update
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: MSDCLifecycleTracker.update()
    input:
      frame
      frame_idx
      file_type
      high_boxes
      low_boxes

12. low-only filtering
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: _filter_low_only_boxes()
    input: high_boxes, low_boxes
    output: raw_low_only_boxes

13. low observation budget
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: _budget_low_only_boxes()
    input: raw_low_only_boxes, self.tracks
    output: low_only_boxes

14. high detection format conversion
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: _boxes_to_observations()
    line: source="high_det"
    input: high_boxes
    output: Observation[]

15. low detection format conversion
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: _boxes_to_observations()
    line: source="low_det"
    input: low_only_boxes
    output: Observation[]

16. project box to xyxy
    file: target_module/image_detect_module/utils/msdc_types.py
    function: xywh_to_xyxy()
    input: x, y, w, h
    output: x1, y1, x2, y2

17. project box to Observation
    file: target_module/image_detect_module/utils/msdc_types.py
    function: Observation.from_project_box()
    input:
      box
      source
      frame_idx
      modality
      reliability
      class_id
    output:
      Observation.box
      Observation.source
      Observation.score
      Observation.reliability
      Observation.modality
      Observation.frame_idx
      Observation.class_id
      Observation.class_name
      Observation.raw

18. evidence update entry
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: MSDCLifecycleTracker.update()
    line: self.evidence_updater.update_tracks(self.tracks, observations, idx)
    input: self.tracks, observations, frame_idx
    output: self.tracks, events

19. observation merge
    file: target_module/image_detect_module/utils/evidence_state.py
    function: EvidenceStateUpdater.update_tracks()
    line: groups = self.merge_observations(observations)
    output: observation groups

20. track-observation association
    file: target_module/image_detect_module/utils/evidence_state.py
    function: associate_tracks_to_observations()
    input: tracks, observation_groups
    output: matches, unmatched_track_indices, unmatched_group_indices

21. lost-track reacquire
    file: target_module/image_detect_module/utils/evidence_state.py
    function: _reacquire_lost_track_indices()
    input: LOST tracks, unmatched groups
    output: reacquire_matches

22. matched track update
    file: target_module/image_detect_module/utils/evidence_state.py
    function: _apply_matched_observation()
    input: matched track, observation group, positive_score
    output: updated track evidence

23. unmatched track update
    file: target_module/image_detect_module/utils/evidence_state.py
    function: _apply_negative_evidence()
    input: unmatched track
    output: misses++, evidence_score decay

24. low-candidate inheritance
    file: target_module/image_detect_module/utils/evidence_state.py
    function: _resolve_low_candidate_inheritance()
    input: LOW_CANDIDATE tracks, LOST tracks
    output: inheritance events

25. state transition
    file: target_module/image_detect_module/utils/evidence_state.py
    function: _transition_track()
    input: track, frame_idx, matched_real, reacquire_score
    output: LifecycleEvent[]

26. state write
    file: target_module/image_detect_module/utils/evidence_state.py
    function: _set_state()
    line: track.state = to_state
    output: updated track.state

27. new candidate spawn
    file: target_module/image_detect_module/utils/evidence_state.py
    function: spawn_candidates()
    input: unmatched observation groups
    output: new EvidenceTrack[]

28. removed track pruning
    file: target_module/image_detect_module/utils/evidence_state.py
    function: _prune_stale_removed_tracks()
    input: tracks
    output: pruned tracks

29. track cap enforcement
    file: target_module/image_detect_module/utils/evidence_state.py
    function: _enforce_track_caps()
    input: tracks
    output: capped tracks

30. output conversion
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: _tracks_to_output_boxes()
    input: self.tracks
    output: active_boxes

31. active track to output box
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: _track_to_output_box()
    input: ACTIVE EvidenceTrack
    output: project output box

32. xyxy to project output
    file: target_module/image_detect_module/utils/msdc_types.py
    function: xyxy_to_project_box()
    output:
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

33. output NMS
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: _nms_output_boxes()
    input: active_boxes
    output: deduplicated active_boxes

34. tracker output return
    file: target_module/image_detect_module/utils/lifecycle_tracker.py
    function: MSDCLifecycleTracker.update()
    line: return output_boxes
    output: tracked_boxes

35. video_main receives output
    file: video_main.py
    function: _update_result_tracking_for_frame()
    line: result["data"]["boxes"] = tracked_boxes
    output: result.data.boxes

36. visualization / video output
    file: video_main.py
    function: main()
    input: tracked_boxes
    output: annotated frame / output video
```


1. 检测输出 (Detection Output)
   ↳ resolve_msdc_high_low_boxes()
   ↳ 生成: high_boxes 与 low_boxes

2. ELT 输入路由 (ELT Input Routing)
   ↳ _update_result_tracking_for_frame()
   ↳ MSDCLifecycleTracker.update()

3. 内部格式转换 (Internal Format Conversion)
   ↳ _filter_low_only_boxes()
   ↳ _boxes_to_observations()
   ↳ 生成: Observation(source='high_det' | 'low_det')

4. 证据与状态更新 (Evidence & State Update)
   ↳ EvidenceStateUpdater.update_tracks()
   ↳ 匹配轨迹 → 更新证据 → 状态转换 → 生成新候选
   ↳ 返回: tracked_boxes