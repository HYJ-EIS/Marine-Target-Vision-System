本文档梳理了 MS-DC-ELT (Multi-Stage Dual-Confidence Evidence Lifecycle Tracker) 系统在连续两帧（Frame t 与 Frame t+1）内的数据流转、状态更新及代码执行链路。

## 🕒 Frame t：初始帧/当前帧

### 1. Input (输入阶段)

**逻辑概述**：读取当前帧，更新帧计数器，并将 `frame_idx` 作为 lifecycle 的时间轴。

- **关联文件**：`video_main.py`

```
# video_main.py:250
ret, frame = cap.read()
# 作用：读取第 t 帧原始图像，后续所有 detection 和 tracking 都基于这个 frame。

# video_main.py:259
frame_count += 1
# 作用：推进帧计数；MS-DC-ELT 后面使用 frame_idx = frame_count - 1 作为 lifecycle 时间轴。
```

### 2. Detection (目标检测)

**逻辑概述**：执行高低双阈值检测，生成 `low_boxes` 与 `high_boxes`。

- **关联文件**：`video_main.py`, `msdc_detection.py`

```
# video_main.py:262
if use_msdc_elt:
	boxes, low_boxes = resolve_msdc_high_low_boxes(detector, frame, file_type)
# 作用：进入 msdc_elt 专用路径。对当前帧执行检测，low_boxes 来自低阈值检测；boxes（即 high_boxes）是从中按常规阈值筛出的高阈值框。

# msdc_detection.py:27
def run_msdc_low_threshold_detection(detector, frame, file_type: str) -> list[dict]:
# 作用：执行低阈值检测，生成当前帧的 low_boxes。

# msdc_detection.py:36
def split_msdc_high_from_low_boxes(low_boxes: list[dict], file_type: str) -> list[dict]:
# 作用：从 low_boxes 中筛出高阈值框，作为当前帧的 high_boxes。
```

### 3. ELT Update (生命周期更新外层)

**逻辑概述**：判断 Tracker 类型，调用 `lifecycle_tracker.update(...)` 将当前帧的检测结果传入跟踪器。

- **关联文件**：`tracking_update.py`

```
# video_main.py:290
tracked_boxes = _update_result_tracking_for_frame(...)
# 作用：调用当前帧 tracking update 封装函数。将当前帧 frame、frame_idx、high_boxes、low_boxes 传入 lifecycle tracker，最终返回带 track_id 的 tracked_boxes。

# tracking_update.py:23 & 28
if is_msdc_tracker(tracker_type):
return lifecycle_tracker.update(...)
# 作用：判断是否为 msdc_elt 并真正进入 MS-DC-ELT lifecycle tracker 的单帧 update cycle。
```

### 4. ELT Update Internal (内部更新逻辑)

**逻辑概述**：过滤并预算 `low_only_boxes`，构造 `Observation` 并交由 Evidence Updater 处理。

- **关联文件**：`lifecycle_tracker.py`

```
# lifecycle_tracker.py:55
def update(...)
# 作用：MS-DC-ELT 每帧 lifecycle update 的外层入口。

# lifecycle_tracker.py:77-78
raw_low_only_boxes = self._filter_low_only_boxes(high_boxes, low_boxes)
low_only_boxes = self._budget_low_only_boxes(raw_low_only_boxes, self.tracks)
# 作用：去除已与高阈值框重叠的部分，并对剩余的 low-only boxes 做置信度、Top-K、轨迹邻近等预算过滤，限制进入 lifecycle 的低阈值候选数量。

# lifecycle_tracker.py:82-84
observations = []
observations.extend(self._boxes_to_observations(high_boxes, "high_det", idx, modality))
observations.extend(self._boxes_to_observations(low_only_boxes, "low_det", idx, modality))
# 作用：构造证据观察（Observations），区分来源为 high_det 和 low_det。

# lifecycle_tracker.py:88
self.tracks, events = self.evidence_updater.update_tracks(self.tracks, observations, idx)
# 作用：将上一帧遗留的 tracks 和当前帧 observations 送入 evidence/state updater，执行状态更新。
```

### 5. State Transition (证据与状态流转)

**逻辑概述**：在 `EvidenceStateUpdater.update_tracks()` 内按顺序执行关联、证据施加与状态转移。

- **关联文件**：`evidence_state.py`

**核心执行顺序**：

1. **Merge**：合并 observations (`merge_observations`)。
2. **Associate**：关联现有的 `LOW_CANDIDATE | CANDIDATE | ACTIVE` tracks。
3. **Reacquire**：尝试重新捕获 `LOST` tracks。
4. **Positive Evidence**：对 matched tracks 加正证据。
5. **Negative Evidence**：对 unmatched / lost / aux-only tracks 加负证据。
6. **Inheritance**：处理 low-candidate 的公有 ID 继承。
7. **Transition**：执行核心状态转移 `_transition_track(...)`。
8. **Spawn**：由 unmatched observation 生成新轨迹。
9. **Prune/Cap**：清理移除过期的轨迹，强制执行容量上限。

**核心代码映射**：

```
# evidence_state.py:187-198 (Merge & Associate)
groups = self.merge_observations(observations)
matches, unmatched_track_indices, unmatched_group_indices = self.associate_tracks_to_observations(...)
reacquire_matches = self._reacquire_lost_track_indices(...)

# evidence_state.py:222 (Positive Evidence)
for track_idx, group_idx in matches: # 更新 box, velocity, hits 等

# evidence_state.py:239-244 (Negative Evidence)
negative_track_indices = list(dict.fromkeys(list(unmatched_track_indices) + lost_indices + aux_only_track_indices))
self._apply_negative_evidence(track) # 增加 misses，降低 score

# evidence_state.py:246 (Inheritance)
events.extend(self._resolve_low_candidate_inheritance(tracks, frame_idx))

# evidence_state.py:254 (Transition - 核心状态转移点)
events.extend(self._transition_track(track, frame_idx, matched_real, reacquire_score))

# evidence_state.py:258 (Spawn)
new_tracks, new_events = self.spawn_candidates([groups[idx] for idx in spawn_group_indices], frame_idx, tracks)

# evidence_state.py:261-262 (Prune & Cap)
tracks = self._prune_stale_removed_tracks(tracks, frame_idx)
tracks = self._enforce_track_caps(tracks, frame_idx)
```

#### 📌 状态转移规则 (State Transition Rules)

|**当前状态**|**触发条件简述**|**下一状态**|**代码位置**|
|---|---|---|---|
|**LOW_CANDIDATE**|满足 low temporal gate 确认 / 满足清理条件 (miss/window/score)|ACTIVE / REMOVED|`evidence_state.py:1428`|
|**CANDIDATE**|满足 evidence 确认 / 满足清理条件 (age/score)|ACTIVE / REMOVED|`evidence_state.py:1441`|
|**ACTIVE**|超过 missing patience (连续丢失忍耐度)|LOST|`evidence_state.py:1466`|
|**LOST**|被 reacquire 成功匹配 / 达到 lost timeout (超时)|ACTIVE / REMOVED|`evidence_state.py:1480`|
|**None (New)**|Unmatched observation 首次生成|LOW_CANDIDATE / CANDIDATE|`evidence_state.py:258`|

### 6. Output (结果输出)

**逻辑概述**：提取 `ACTIVE` 状态的轨迹转换为输出格式，经过 NMS 后返回给主程序。

- **关联文件**：`lifecycle_tracker.py`, `msdc_types.py`

```
# lifecycle_tracker.py:103
output_boxes = self._tracks_to_output_boxes(self.tracks, frame_idx=idx)
# 作用：把本帧 update 后的 tracks 转换为输出框。

# lifecycle_tracker.py:168
if state == TrackState.ACTIVE:
# 作用：默认只把 ACTIVE tracks 输出成带 track_id 的检测框（可选输出 CANDIDATE）。输出字段来自 xyxy_to_project_box(...)，包含：track_id, x, y, w, h, confidence, class, class_confidence, gid, public_id, lifecycle_state, real_det_age。

# lifecycle_tracker.py:176 & 135
active_boxes = self._nms_output_boxes(active_boxes)
return output_boxes
# 作用：对输出做 NMS 去重，并返回当前帧最终的 tracked_boxes 写入 result["data"]["boxes"]。
```

## 🕒 Frame t+1：下一帧处理逻辑

当时间推移至 `t+1` 帧，流程将复用主循环，但基于上一帧累积的 **lifecycle state memory** 继续演进。

### 1. Memory & Input

```
# video_main.py:250
ret, frame = cap.read()
# 作用：读取下一帧。此时 MSDCLifecycleTracker.self.tracks 已在内存中保存了 Frame t 结束后的完整生命周期状态。
```

### 2. Detection (重生成)

```
# video_main.py:264
boxes, low_boxes = resolve_msdc_high_low_boxes(detector, frame, file_type)
# 作用：对 Frame t+1 重新生成当前帧视角的 high_boxes 和 low_boxes。
```

### 3. Update & State Transition (跨帧关联与状态演进)

```
# video_main.py:290 -> lifecycle_tracker.py:88
self.tracks, events = self.evidence_updater.update_tracks(self.tracks, observations, idx)
# 作用：Frame t+1 的 observations 驱动 Frame t 遗留 tracks 发生 lifecycle 变化：
# 1. Matched (evidence_state.py:222)：若 t 帧轨迹被匹配，更新正证据，可能保持原状态或满足条件跃迁至 ACTIVE。
# 2. Unmatched (evidence_state.py:244)：若未匹配，施加负证据，可能导致 ACTIVE → LOST、CANDIDATE → REMOVED、LOW_CANDIDATE → REMOVED。
# 3. Transition Engine (evidence_state.py:254)：跨帧生命周期变化在此统一落地。
```

### 4. Output (输出持续跟踪结果)

```
# lifecycle_tracker.py:103
output_boxes = self._tracks_to_output_boxes(self.tracks, frame_idx=idx)
# 作用：输出 Frame t+1 更新判定后确认为 ACTIVE 的轨迹框，完成连续跟踪循环。
```