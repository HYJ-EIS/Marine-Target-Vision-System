# MS-DC-ELT 模块级实施计划（中文版）

> **给后续执行 agent 的要求：** 实施本计划时必须使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`，并按任务逐项执行。本文档只做代码审计和实施规划，不代表已经实现、测试或验证任何实验结果。

**目标：** 在保留现有 FFCA-YOLO + OC-SORT / BoT-SORT tracking-by-detection baseline 的前提下，新增一个可选的 MS-DC-ELT 生命周期跟踪分支。

**总体架构：** 不改 OC-SORT / BoT-SORT 主体。新增独立的 `MSDCLifecycleTracker`，通过 `--tracker msdc_elt` 显式启用；复用现有 detector 阈值覆盖能力、GMC、IoU/中心距离工具和现有输出格式；额外输出 lifecycle 诊断 JSONL，且写入不覆盖既有结果的独立 run 目录。

**技术栈：** Python、OpenCV、NumPy、现有 ONNX Runtime detector wrapper、现有 `Config`、现有 tracker 工具函数、现有 pytest 测试体系。所有 `python` / `pytest` 命令按项目 `AGENTS.md` 要求使用 `conda run -n ship_detect ...`。

---

## 0. 已读取的研究文档

- 已读取：`/home/hyj/projects/open-set-id-redesign-logs/refine-logs/FINAL_PROPOSAL.md`。
- 已读取：`/home/hyj/projects/open-set-id-redesign-logs/refine-logs/STORY_9Q.md`。
- 使用的核心方案：**MS-DC-ELT: Motion-Seeded Delayed-Confirmation Evidence Lifecycle Tracking with Lifecycle-Gated Template Locking**。
- 使用的实施约束：在现有代码基础上新增模块，不重写 OC-SORT / BoT-SORT 内部。

本文档只是实施规划。本轮没有修改业务代码，没有运行实验，也没有声称任何实验结果已经完成。

## 1. 代码审计结论

### 1.1 `video_main.py` 中 tracker 初始化和调用位置

已找到。

- `--tracker` 定义在 `video_main.py:56-63`。
- 当前可选值在 `video_main.py:59`，包括：`""`、`bytetrack`、`ocsort`、`botsort`、`dist_tracker`、`official_ocsort`、`official_botsort`。
- tracker 初始化位置为 `video_main.py:124-126`：

```python
tracker_type = args.tracker if args.tracker else None
tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_type)
```

- 主跟踪调用位置为 `video_main.py:189-201`：

```python
result = detector.detect_from_image_file(tmp_frame_path, file_type=file_type)
boxes = result.get("data", {}).get("boxes", [])
tracked_boxes = tracker.update(boxes, frame.shape, frame=frame)
```

- 跟踪结果写回位置为 `video_main.py:214-220`，随后调用 `visualize_detections(...)` 可视化。
- 本地视频输出默认使用 `Config.VIDEO_OUTPUT_PATH`，对应 `video_main.py:132-138`。
- RTSP 输出使用 `create_video_output_manager(...)`，对应 `video_main.py:147-154`。
- MQ 每 2 秒发送一次，位置在 `video_main.py:235-241`；具体发送线程为 `video_main.py:269-281` 的 `_mq_sender_worker`。

实施结论：应在 `video_main.py` 中新增 `msdc_elt` 分支，但不要改变当前 baseline 分支。

### 1.2 其他 tracker CLI 选择位置

除 `video_main.py` 外，还找到以下 tracker choices：

- `tools/evaluation/export_mot_results.py:32-39` 定义 `TRACKER_CHOICES`。
- `tools/evaluation/export_mot_results.py:116-123` 暴露 `--tracker`。
- `tools/validation/tracker_effect_test.py:41-48` 定义 `TRACKER_CHOICES`。
- `tools/validation/tracker_effect_test.py:147-153` 暴露 `--trackers`。
- `tools/validation/video_test_tracking.py:62-68` 暴露 `--tracker`。
- 抽帧 tracker choices 已迁移到 `/home/hyj/Anti_Drone_Project/Marine-Frame-Extraction`，不属于本仓库 MS-DC-ELT 视频/评测链路。

实施结论：最小实施先改 `video_main.py`；正式评测阶段再改 `export_mot_results.py` 和 validation 工具。

### 1.3 `MultiObjectTracker.update()` 的输入输出格式

已在 `target_module/image_detect_module/utils/tracker.py` 中找到。

- 模块注释在 `tracker.py:14-15` 写明输入格式：

```python
[{"x": int, "y": int, "w": int, "h": int, "confidence": float, "class": str}, ...]
```

- 模块注释在 `tracker.py:17-19` 写明输出格式：

```python
[{"track_id": int, "x": int, "y": int, "w": int, "h": int,
  "confidence": float, "class": str}, ...]
```

- `MultiObjectTracker.__init__` 在 `tracker.py:361-415` 根据 tracker type 分发。
- `MultiObjectTracker.update(detections, frame_shape, frame=None)` 定义在 `tracker.py:417-438`。
- OC-SORT / BoT-SORT 分支在 `tracker.py:536-569` 将 project boxes 转成 `[x1,y1,x2,y2]`。
- OC-SORT / BoT-SORT 输出 dict 在 `tracker.py:571-581` 构造。
- ByteTrack 输出 dict 在 `tracker.py:521-531` 构造。
- DistTracker 在 `dist_tracker.py:43-85` 保持相同项目契约，输出 dict 在 `dist_tracker.py:207-230` 构造。

实施结论：MS-DC-ELT 至少必须返回如下字段：

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

允许增加诊断字段，例如：

```python
{
    "lifecycle_state": "active",
    "evidence_score": 2.7,
    "observation_sources": ["high_det", "motion"]
}
```

但必须单独检查 MOT 导出和 MQ 行为，不能默认所有下游都需要这些额外字段。

### 1.4 `TargetDetector.detect_from_image_file()` 的输入输出格式

已在 `target_module/image_detect_module/target_detection.py` 中找到。

- 函数签名在 `target_detection.py:98-100`：

```python
def detect_from_image_file(self, image_path: str, output_dir: Optional[str] = None,
                           file_type: Optional[str] = None,
                           enable_tracking: bool = False) -> Dict:
```

- 文件存在性和可读性检查在 `target_detection.py:115-126`。
- `_perform_detection(...)` 调用位置在 `target_detection.py:123-126`。
- `_perform_detection(...)` 在 `target_detection.py:173-174` 调用：

```python
detection_stats = self.processor.process(image_path, file_type=file_type)
```

- 成功响应在 `target_detection.py:202-213` 构造：

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

- 错误响应在 `target_detection.py:260-272` 构造。
- `detect_targets(...)` 只支持文件路径字符串，位置为 `target_detection.py:286-342`，这也符合 `AGENTS.md` 中“不要把内存帧或 np.ndarray 直接传给 detect_targets”的要求。

未找到：当前 `detect_from_image_file(..., conf_override=...)` 不存在。

实施结论：Task 1 和第一版视频接入不应先改 `TargetDetector`，而应直接使用 `detector.processor.process_frame(...)` 获取高/低阈值检测结果。后续可选再给 `TargetDetector` 增加 `conf_override` 透传。

### 1.5 `ImageProcessor.process_frame(frame, file_type, conf_override=None)`

已找到。

- 函数签名在 `image_processor.py:64`。
- 该函数在 `image_processor.py:66-74` 检查 `frame is not None`、类型为 `np.ndarray` 且非空。
- 在 `image_processor.py:76-77` 解析模态后调用 `_process_image_array(..., conf_override=...)`。
- `_process_image_array` 在 `image_processor.py:91-99` 调用：

```python
results = detector.detect(img, conf_override=conf_override)
```

- 输出 stats 在 `image_processor.py:133-140` 构造：

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

实施结论：这是 MS-DC-ELT 视频链路获取低阈值检测结果的首选入口。

### 1.6 `OnnxDetector.detect(img, conf_override=None)`

已找到。

- 函数签名在 `detectors.py:147`。
- 阈值覆盖逻辑在 `detectors.py:148-149`：

```python
conf_thres = self.conf_thres if conf_override is None else float(conf_override)
```

- 使用 `mask = scores > conf_thres` 过滤，位置为 `detectors.py:192`。
- `cv2.dnn.NMSBoxes(...)` 使用同一个 `conf_thres` 作为 score threshold，位置为 `detectors.py:207-212`。
- 返回检测字段 `x1,y1,x2,y2,confidence,class,class_confidence`，位置为 `detectors.py:233-238`。

实施结论：低置信候选提取已经在 detector 层具备，不需要新增 detector。

### 1.7 `target_module/image_detect_module/utils/gmc.py` 是否可复用

已找到，可复用。

- `GMC` 类在 `utils/gmc.py:17`。
- 构造函数支持 `method="sparse_flow"`、`orb` 或 `none`，位置为 `utils/gmc.py:20-30`。
- `apply(frame)` 在 `utils/gmc.py:37-74` 返回 2x3 仿射矩阵，并维护上一帧灰度图。
- `reset()` 在 `utils/gmc.py:76-78`。
- 稀疏光流方法在 `utils/gmc.py:82-115`。
- ORB 匹配方法在 `utils/gmc.py:119-147`。

实施结论：`MotionSeedGenerator` 可以内部持有自己的 `GMC`，并暴露 `reset()`；也可以接收外部预计算矩阵。第一版建议内部持有，降低耦合。

### 1.8 `tracker.py` 中 `_iou_batch`、`_center_distance_batch`、`KalmanBoxTracker` 是否可复用

已找到，可复用。

- `_iou_batch` 在 `tracker.py:38-65`。
- `_center_distance_batch` 在 `tracker.py:68-81`。
- `KalmanBoxTracker` 在 `tracker.py:29` 被导入，定义在 `utils/kalman_bbox.py:18-198`。
- `KalmanBoxTracker.predict()` 在 `kalman_bbox.py:80-95`，返回 `[x1,y1,x2,y2]`。
- `KalmanBoxTracker.update(...)` 在 `kalman_bbox.py:97-127`。
- `KalmanBoxTracker.apply_affine(...)` 在 `kalman_bbox.py:132-142`。

实施结论：MS-DC-ELT 可以复用这些几何和运动预测工具。但不应使用 `KalmanBoxTracker._count` 作为 lifecycle global ID 分配器，因为 MS-DC-ELT 需要自己的 `next_gid`，避免受现有 tracker reset 语义影响。

### 1.9 `tools/evaluation/export_mot_results.py` 的输出字段要求

已找到。

- 输出布局在 `export_mot_results.py:4-8` 注释中说明。
- `format_mot_result_line(frame_id, box)` 在 `export_mot_results.py:42-49`，要求如下字段：

```python
box["track_id"]
box["x"]
box["y"]
box["w"]
box["h"]
box.get("confidence", 1.0)
```

- 输出路径为 `<output-root>/<tracker>/data/<seq-name>.txt`，位置为 `export_mot_results.py:81`。
- 当前导出只使用 `MultiObjectTracker`，位置为 `export_mot_results.py:77-106`。

实施结论：只要 MS-DC-ELT 输出上述字段，就能与 MOT 行格式兼容。但 `export_mot_results.py` 仍需新增 `msdc_elt` 分支，因为 `MSDCLifecycleTracker.update(...)` 的输入契约不同于 `MultiObjectTracker.update(...)`。

### 1.10 可视化是否能容忍额外字段 `lifecycle_state`

已确认可以容忍。

- `visualize_detections(...)` 在 `visualization.py:61-64` 遍历 boxes。
- 它只通过 `box.get(...)` 读取 `h,w,x,y,confidence,class,track_id,id,class_confidence`，位置为 `visualization.py:66-74`。
- 未知额外字段会被忽略。

实施结论：给 tracked box 增加 `lifecycle_state` 不会破坏现有可视化。当前 label 不显示该字段；是否显示应作为可选 debug 功能，不建议默认开启。

### 1.11 结果保存目录和日志输出机制

已找到。

- 全局输出根目录为 `Config.OUTPUT_DIR = os.path.join(BASE_DIR, "results")`，位置为 `config.py:9-10`。
- 默认视频输出为 `Config.VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.mp4")`，位置为 `config.py:57-62`。
- `video_main.py` 在 `video_main.py:132-138` 创建输出视频目录。
- `video_main.py` 在 `video_main.py:243-250` 每 50 帧打印进度。
- `video_main.py` 在 `video_main.py:263-266` 打印最终统计。
- `VideoOutputManager` 使用 Python logging，位置为 `video_output_manager.py:69-82`，并在 `video_output_manager.py:115-140` 创建视频写入目录。
- `tools/validation/tracker_effect_test.py` 在 `tracker_effect_test.py:213-227`、`232-266`、`348-365` 写出 `tracks.csv`、`frames.jsonl`、`summary.json` 和可选 `error.txt`。
- `tools/validation/video_detect_only.py` 在 `video_detect_only.py:245-249`、`267-313`、`335-363` 写 detection-only JSONL/CSV/summary。
- `image_main.py` 在 `image_main.py:47-56` 配置 logging，在 `image_main.py:66-73` 配置 upload/results 目录，在 `image_main.py:188-204` 保存可视化结果图。

实施结论：MS-DC-ELT 诊断输出应写到新 run 目录，例如：

```text
results/msdc_elt/<video_stem>_<YYYYMMDD_HHMMSS>/lifecycle_events.jsonl
```

不要默认写入或覆盖已有 `results/output.mp4` 或其他 tracker 结果目录。

### 1.12 当前是否已有 MS-DC-ELT 代码

未找到。

- 未找到 `target_module/image_detect_module/utils/msdc_types.py`。
- 未找到 `target_module/image_detect_module/utils/motion_seed.py`。
- 未找到 `target_module/image_detect_module/utils/evidence_state.py`。
- 未找到 `target_module/image_detect_module/utils/template_lock.py`。
- 未找到 `target_module/image_detect_module/utils/lifecycle_tracker.py`。
- 未找到 `lifecycle_state` 字段在项目代码中的使用。
- 未找到 candidate / active / lost / retired 生命周期实现。第三方 BoT-SORT 内部的 `TrackState` 与本文 MS-DC-ELT lifecycle 不是同一层能力。

## 2. 总体接入策略

1. 不修改 `target_module/image_detect_module/utils/tracker.py` 中 OC-SORT / BoT-SORT 主体。
2. 保持 `botsort`、`ocsort`、`dist_tracker`、`official_ocsort`、`official_botsort` 的现有行为不变。
3. 新增 `--tracker msdc_elt`，作为显式 opt-in 分支。
4. 新增 `MSDCLifecycleTracker`，路径为 `target_module/image_detect_module/utils/lifecycle_tracker.py`。
5. MS-DC-ELT 输出与现有 tracker 兼容的 boxes 格式：

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

6. MS-DC-ELT 可以额外输出可选诊断字段：

```python
{
    "lifecycle_state": "active",
    "evidence_score": float,
    "observation_sources": ["high_det", "motion"]
}
```

7. Lifecycle 诊断单独输出 JSONL：

```text
results/msdc_elt/<run_id>/lifecycle_events.jsonl
results/msdc_elt/<run_id>/low_conf_debug.jsonl
results/msdc_elt/<run_id>/motion_seed_debug.jsonl
results/msdc_elt/<run_id>/summary.json
```

8. `run_id` 应包含视频名和时间戳，例如 `<video_stem>_<YYYYMMDD_HHMMSS>`，避免覆盖已有结果。
9. 新增配置开关 `Config.MSDC_ENABLE = False`。只有显式选择 `--tracker msdc_elt` 或显式打开配置时才初始化 MS-DC-ELT。

## 3. 新增模块规划

### 3.1 `target_module/image_detect_module/utils/msdc_types.py`

**作用：** 定义轻量、可 JSON 序列化的生命周期数据结构。

**输入：** project boxes、frame index、modality、source name、evidence score。

**输出：** dataclass 实例，以及可写入 JSONL 的 dict。

**核心类：**

- `TrackState`：字符串常量或 enum，包含 `candidate`、`active`、`lost`、`retired`。
- `Observation`：来自 high detector、low detector、motion seed、template 或 reacquire 的单个证据。
- `EvidenceTrack`：单个目标假设的生命周期状态。
- `LifecycleEvent`：状态转移和调试事件。

**核心函数：**

- `xywh_to_xyxy(box: dict) -> np.ndarray`
- `xyxy_to_project_box(xyxy, track_id, confidence, cls, class_confidence=None, **extras) -> dict`
- `observation_from_project_box(box, source, frame_idx, modality, reliability=1.0) -> Observation`
- `event_to_json_dict(event: LifecycleEvent) -> dict`

**依赖：** 标准库 `dataclasses`、`enum` 或字符串常量、`typing`；如果坐标转换返回数组，则使用 NumPy。

**最小单元测试：**

- `Observation.to_dict()` 可以 `json.dumps/json.loads`。
- `EvidenceTrack.to_debug_dict()` 包含 `gid`、`state`、`evidence_score`、`box`。
- `xywh_to_xyxy` 与 `xyxy_to_project_box` 坐标往返正确。

### 3.2 `target_module/image_detect_module/utils/motion_seed.py`

**作用：** 从连续帧生成类别无关的运动候选框。

**输入：** `prev_frame`、`current_frame`、可选 frame index 和 modality。

**输出：** `Observation` 列表或 project-style motion boxes；可选 debug mask。

**核心类：**

- `MotionSeedGenerator`

**核心函数/方法：**

- `MotionSeedGenerator.__init__(method, downscale, min_area_ratio, max_area_ratio, diff_percentile)`
- `MotionSeedGenerator.reset()`
- `MotionSeedGenerator.update(frame, frame_idx, modality) -> list[Observation]`
- `_compute_compensated_difference(prev_frame, frame) -> np.ndarray`
- `_mask_to_boxes(mask, frame_shape) -> list[dict]`
- `write_debug_mask(path, mask)`，仅在 debug 开启时写出。

**依赖：** OpenCV、NumPy、现有 `utils/gmc.py` 中的 `GMC`。

**最小单元测试：**

- 两张合成帧中，一个亮色方块移动后，应产生一个接近该方块的 motion box。
- 两张完全相同的帧不应产生 motion boxes。
- 第一帧只初始化内部状态，不产生 motion boxes。
- `reset()` 后 previous frame 状态被清空。

### 3.3 `target_module/image_detect_module/utils/evidence_state.py`

**作用：** 合并观测、更新 evidence score，并处理 candidate / active / lost / retired 状态转移。

**输入：** 现有 `EvidenceTrack` 列表、high/low/motion/template observations、frame shape、frame index。

**输出：** 更新后的 tracks、lifecycle events、active 输出轨迹。

**核心类：**

- `EvidenceStateUpdater`
- 可选 `EvidenceConfig`，如果希望把阈值从 `Config` 中整理成局部配置。

**核心函数/方法：**

- `merge_observations(high_obs, low_obs, motion_obs) -> list[Observation]`
- `associate_tracks_to_observations(tracks, observations, frame_shape) -> matches`
- `update_tracks(tracks, observations, frame_idx, frame_shape) -> tuple[list[EvidenceTrack], list[LifecycleEvent]]`
- `spawn_candidates(unmatched_observations)`
- `transition(track, reason) -> LifecycleEvent`
- `active_tracks_to_project_boxes(tracks) -> list[dict]`

**依赖：** NumPy；若沿用现有 tracker 匈牙利匹配，可使用 `scipy.optimize.linear_sum_assignment`；复用 `_iou_batch`、`_center_distance_batch`，可选复用 `KalmanBoxTracker`。

**最小单元测试：**

- 低置信观测连续出现 `MSDC_CONFIRM_MIN_HITS` 帧，且 `evidence_score >= MSDC_CONFIRM_SCORE` 后，从 candidate 转为 active。
- 单帧低置信假候选应在 active 前被剪枝。
- active track 连续无观测后转为 lost。
- lost track 超过 `MSDC_LOST_MAX_AGE` 后转为 retired。

### 3.4 `target_module/image_detect_module/utils/template_lock.py`

**作用：** 仅对 active track 提供有预算上限的实例级模板匹配。

**输入：** 当前帧、带模板的 active `EvidenceTrack`、预测框。

**输出：** template observations 和 template scores。

**核心类：**

- `TemplateLock`
- `TemplateRecord`

**核心函数/方法：**

- `initialize(track, frame)`
- `match(track, frame) -> Observation | None`
- `update_template(track, frame, matched_box, score)`
- `evict_if_over_budget(active_tracks)`
- `_crop_with_padding(frame, box, scale)`
- `_match_ncc(template, search_patch) -> tuple[box, score]`

**依赖：** OpenCV、NumPy。第一版使用 `cv2.matchTemplate` 的 normalized correlation，不引入大型新依赖。

**最小单元测试：**

- 合成 patch 平移后，NCC 能找到高相似位置。
- candidate track 不初始化模板。
- 分数低于 `MSDC_TEMPLATE_UPDATE_THRESH` 时不更新模板。
- active template 数量不超过 `MSDC_MAX_ACTIVE_TEMPLATES`。

### 3.5 `target_module/image_detect_module/utils/lifecycle_tracker.py`

**作用：** MS-DC-ELT 顶层 tracker，协调 high/low detections、motion seeds、evidence update、可选 template lock 和 diagnostics。

**输入：** `frame`、`file_type`、`high_boxes`、`low_boxes`、`frame_idx`、可选 `timestamp_sec`。

**输出：** 与现有项目兼容的 `tracked_boxes`，以及 lifecycle JSONL 诊断。

**核心类：**

- `MSDCLifecycleTracker`

**核心函数/方法：**

- `__init__(frame_rate, output_dir=None, enable_template=False, debug=False)`
- `reset()`
- `update(frame, file_type, high_boxes, low_boxes=None, frame_idx=None, timestamp_sec=None) -> list[dict]`
- `_boxes_to_observations(boxes, source, frame_idx, modality)`
- `_low_only_boxes(high_boxes, low_boxes)`
- `_write_debug_event(event)`
- `_write_frame_summary(frame_idx, observations, tracks)`

**依赖：** OpenCV、NumPy、`MotionSeedGenerator`、`EvidenceStateUpdater`、`TemplateLock`、`msdc_types`、现有 `Config`。

**最小单元测试：**

- 连续 high boxes 输入后，输出 active project boxes，且 `track_id` 稳定。
- high boxes 为空但存在 low/motion 观测时，candidate 先积累，达到确认条件后才 active。
- 输出 boxes 通过 `test/test_dist_tracker.py` 中类似 schema 检查。
- 额外 `lifecycle_state` 不破坏 `visualize_detections`。

## 4. 现有文件修改规划

### 4.1 `target_module/image_detect_module/config.py`

**是否必须修改：** 必须。

**修改点：**

- 增加 `MSDC_ENABLE = False`。
- 增加低阈值检测配置：
  - `MSDC_LOW_CONF_VISIBLE`
  - `MSDC_LOW_CONF_INFRARED`
- 增加生命周期阈值：
  - `MSDC_CANDIDATE_MAX_AGE`
  - `MSDC_CONFIRM_MIN_HITS`
  - `MSDC_CONFIRM_SCORE`
  - `MSDC_PRUNE_SCORE`
  - `MSDC_ACTIVE_MISSING_PATIENCE`
  - `MSDC_LOST_MAX_AGE`
  - `MSDC_REACQUIRE_INTERVAL`
  - `MSDC_RETIRED_GUARD_FRAMES`
- 增加预算配置：
  - `MSDC_MAX_CANDIDATES`
  - `MSDC_MAX_ACTIVE_TEMPLATES`
- 增加 motion seed 阈值：
  - `MSDC_MOTION_MIN_AREA_RATIO`
  - `MSDC_MOTION_MAX_AREA_RATIO`
  - `MSDC_MOTION_DIFF_PERCENTILE`
- 增加输出配置：
  - `MSDC_OUTPUT_ROOT = os.path.join(OUTPUT_DIR, "msdc_elt")`
  - `MSDC_DEBUG = False`

**向后兼容：** 默认值不改变现有 tracker 行为。`TRACKER_TYPE` 保持 `"botsort"`，除非用户显式传入 `--tracker msdc_elt`。

**关闭方式：** `MSDC_ENABLE = False` 且不选择 `--tracker msdc_elt`。

### 4.2 `video_main.py`

**是否必须修改：** 对用户可直接运行的视频主链路来说必须。

**修改点：**

- 给 `--tracker` choices 增加 `"msdc_elt"`。
- 只在 `msdc_elt` 分支内部导入 `MSDCLifecycleTracker`，避免影响 baseline import。
- 初始化逻辑：

```python
if selected_tracker == "msdc_elt":
    tracker = MSDCLifecycleTracker(frame_rate=fps, output_dir=msdc_run_dir)
else:
    tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_type)
```

- `msdc_elt` 分支中通过内存帧检测获取 high / low boxes：

```python
high_stats = detector.processor.process_frame(frame, file_type)
low_stats = detector.processor.process_frame(frame, file_type, conf_override=low_threshold)
tracked_boxes = tracker.update(frame, file_type, high_boxes, low_boxes, frame_idx=frame_count)
```

- 其他 tracker 保持当前 `detect_from_image_file(...) -> MultiObjectTracker.update(...)` 路径不变。
- 如果改到临时图片写入路径，应按 `AGENTS.md` 要求使用 `imwrite_unicode`，不要继续直接使用 `cv2.imwrite`。但不要把这个清理和 MS-DC-ELT 主逻辑混在一个 patch 里。

**向后兼容：** 所有现有 tracker choices 继续走旧路径，输出格式不变。

**关闭方式：** 使用任意非 `msdc_elt` tracker。

### 4.3 `target_module/image_detect_module/target_detection.py`

**是否必须修改：** 初始视频版 MS-DC-ELT 不必须。

**可选修改：** 给以下函数增加 `conf_override` 透传：

- `TargetDetector.detect_from_image_file(...)`
- `TargetDetector._perform_detection(...)`

然后调用：

```python
self.processor.process(image_path, file_type=file_type, conf_override=conf_override)
```

**向后兼容：** 默认 `conf_override=None`，保持现有行为。

**关闭方式：** 不使用新增参数。

### 4.4 `visualization.py`

**是否必须修改：** 不必须。

**审计结论：** 当前代码会忽略 `lifecycle_state` 等额外字段，因此兼容。

**可选修改：** 增加 debug 显示模式，在 label 后附加 lifecycle state，例如：

```text
UAV:0.82 ID:3 active
```

该功能应由配置或 CLI flag 控制，避免默认画面过于拥挤。

**向后兼容：** 默认无需修改。

### 4.5 `tools/evaluation/export_mot_results.py`

**是否必须修改：** 正式导出 MS-DC-ELT MOT 结果时必须。

**修改点：**

- `TRACKER_CHOICES` 增加 `"msdc_elt"`。
- `tracker_type != "msdc_elt"` 时保留当前 `MultiObjectTracker` 路径。
- `tracker_type == "msdc_elt"` 时实例化 `MSDCLifecycleTracker`。
- 使用 `detector.processor.process_frame(...)` 获取 high / low detections。
- 保持 `format_mot_result_line(frame_id, box)` 不变，因为 MS-DC-ELT 输出兼容 fields。
- lifecycle diagnostics 写入 `output_root / "msdc_elt" / "diagnostics" / seq_name`。

**向后兼容：** 现有 botsort/ocsort/dist_tracker/official trackers 导出路径不变。

**关闭方式：** 使用任意现有 tracker choice。

### 4.6 后续可能需要更新的文件

- `tools/validation/tracker_effect_test.py`：Task 6 后增加 `msdc_elt`，用于并排可视化对比。
- `tools/validation/video_test_tracking.py`：`video_main.py` 路径稳定后再加 `msdc_elt`。
- `README.md`：按 `AGENTS.md`，代码修改后需要同步说明新 tracker option 和 diagnostics。
- `messaging/mq_publisher.py`：第一阶段不改。当前 MQ 转换会剥离未知字段，lifecycle diagnostics 先使用文件 JSONL。

## 5. 分任务实施计划

### Task 0：冻结 baseline

**目标：** 在任何 MS-DC-ELT 修改前，确认现有 tracker 仍可运行。

**修改文件：** 无。

**新增文件：** 无。

**核心接口：** 现有 `video_main.py --tracker botsort|ocsort`。

**最小命令：**

```bash
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker botsort --output "results/baseline_check/botsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker ocsort --output "results/baseline_check/ocsort.mp4" --no-display
conda run -n ship_detect pytest test/test_dist_tracker.py test/test_mot_result_export.py test/test_tracker_effect_export.py -q
```

**验收标准：**

- 现有命令不出现 tracker choice 或 schema 错误。
- 现有输出 boxes 仍包含 `track_id,x,y,w,h,confidence,class`。
- `format_mot_result_line` 输出格式不变。

**失败排查：**

- 模型路径失败：检查 `Config.BASE_DIR`、ONNX 路径和当前工作目录。
- 视频打开失败：检查样本路径和编码。
- tracker choice 失败：检查 `video_main.py:59` 和 `MultiObjectTracker.__init__`。

**回滚方式：** 无修改，无需回滚。

### Task 1：打通低阈值检测

**目标：** 在不接入新 tracker 的情况下，验证可以拿到 high_boxes 和 low_boxes。

**修改文件：**

- `target_module/image_detect_module/config.py`
- 可选 debug 脚本：`tools/validation/msdc_low_conf_debug.py`

**新增文件：**

- 建议新增 `tools/validation/msdc_low_conf_debug.py`，避免直接改 `video_main.py`。

**核心接口：**

```python
stats_high = detector.processor.process_frame(frame, file_type)
stats_low = detector.processor.process_frame(frame, file_type, conf_override=low_threshold)
```

**最小命令：**

```bash
conda run -n ship_detect python tools/validation/msdc_low_conf_debug.py --input "<sample.mp4>" --file-type visible --max-frames 100 --output-dir "results/msdc_elt/low_conf_debug"
```

**预期输出：**

```text
results/msdc_elt/low_conf_debug/<video_stem>_<run_id>/low_conf_debug.jsonl
results/msdc_elt/low_conf_debug/<video_stem>_<run_id>/summary.json
```

**验收标准：**

- JSONL 每处理一帧写一行。
- 每行包含 `frame_index`、`high_count`、`low_count`、`low_only_count`、`low_high_overlap_ratio`。
- baseline `video_main.py --tracker botsort` 行为不变。

**失败排查：**

- `detector.processor` 为 `None`：检查 `ImageProcessor.__init__` 中 ONNX 模型加载。
- `low_count < high_count`：检查 low/high 重叠过滤逻辑；注意不同阈值 NMS 可能导致框集合不完全包含。
- 性能很慢：记录到 summary，不要在 correctness 前做优化。

**回滚方式：**

- 删除 `tools/validation/msdc_low_conf_debug.py`。
- 如果后续任务不用，删除新增 `MSDC_LOW_CONF_*` 配置。

### Task 2：实现 `MotionSeedGenerator`

**目标：** 从连续帧生成 motion_boxes，暂不接 tracker。

**修改文件：**

- `target_module/image_detect_module/config.py`

**新增文件：**

- `target_module/image_detect_module/utils/motion_seed.py`
- `test/test_msdc_motion_seed.py`

**核心接口：**

```python
generator = MotionSeedGenerator(
    method=Config.GMC_METHOD,
    downscale=Config.GMC_DOWNSCALE,
)
motion_observations = generator.update(frame, frame_idx=frame_idx, modality=file_type)
```

**最小命令：**

```bash
conda run -n ship_detect pytest test/test_msdc_motion_seed.py -q
conda run -n ship_detect python tools/validation/msdc_low_conf_debug.py --input "<sample.mp4>" --file-type visible --max-frames 100 --enable-motion-debug --output-dir "results/msdc_elt/motion_debug"
```

**验收标准：**

- 第一帧只初始化状态，不输出 motion boxes。
- 合成移动目标测试能产生与移动方块重叠的框。
- 完全相同帧不产生框，或只产生低于阈值的候选。
- debug mask 只在显式开启时写入唯一 run 目录。

**失败排查：**

- 每帧都输出巨大框：检查 GMC 矩阵和形态学阈值。
- 合成运动无框：检查灰度转换、diff percentile 和面积过滤。
- debug 图写入失败：涉及磁盘图片写入时用 `imwrite_unicode`。

**回滚方式：**

- 删除 `motion_seed.py` 和 `test/test_msdc_motion_seed.py`。
- 若未被后续任务使用，删除 motion 相关 `MSDC_*` 配置。

### Task 3：实现 `msdc_types.py`

**目标：** 定义生命周期状态、观测、轨迹和事件数据结构。

**修改文件：** 无。

**新增文件：**

- `target_module/image_detect_module/utils/msdc_types.py`
- `test/test_msdc_types.py`

**核心接口：**

```python
obs = Observation.from_project_box(box, source="low_det", frame_idx=12, modality="visible")
event = LifecycleEvent(frame_idx=12, gid=3, old_state="candidate", new_state="active", reason="confirm")
json.dumps(event.to_dict(), ensure_ascii=False)
```

**最小命令：**

```bash
conda run -n ship_detect pytest test/test_msdc_types.py -q
```

**验收标准：**

- 所有 dataclass 可通过 `.to_dict()` JSON 序列化。
- box 转换函数通过确定性坐标测试。
- 状态只允许 `candidate`、`active`、`lost`、`retired`。

**失败排查：**

- NumPy scalar 不能 JSON 序列化：在 `.to_dict()` 中转为 Python `int` / `float`。
- box 出现负宽高：在坐标转换边界做 normalize 和 clip。

**回滚方式：**

- 删除 `msdc_types.py` 和 `test/test_msdc_types.py`。

### Task 4：实现 `EvidenceStateUpdater`

**目标：** 合并 high_det、low_det、motion observation，维护 evidence score，实现基础状态转移；暂不实现模板锁定。

**修改文件：**

- `target_module/image_detect_module/config.py`

**新增文件：**

- `target_module/image_detect_module/utils/evidence_state.py`
- `test/test_msdc_evidence_state.py`

**核心接口：**

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

**最小命令：**

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py -q
```

**验收标准：**

- candidate 确认需要重复证据和分数过阈值。
- 单帧 false candidate 会被剪枝。
- active 无观测后转为 lost，再转 retired。
- active 输出 boxes 与现有 tracker schema 兼容。

**失败排查：**

- candidate 从不确认：检查 `MSDC_CONFIRM_SCORE`、分数衰减和 source 权重。
- false candidate 太容易确认：检查 `MSDC_CONFIRM_MIN_HITS`、`MSDC_PRUNE_SCORE` 和负证据惩罚。
- ID 意外 reset：检查 lifecycle `next_gid`。

**回滚方式：**

- 删除 `evidence_state.py` 和 `test/test_msdc_evidence_state.py`。
- 若后续任务未使用，删除 evidence 相关配置。

### Task 5：实现 `MSDCLifecycleTracker-lite`

**目标：** 新增无模板锁定的第一版 MS-DC-ELT tracker。

**修改文件：**

- `target_module/image_detect_module/config.py`

**新增文件：**

- `target_module/image_detect_module/utils/lifecycle_tracker.py`
- `test/test_msdc_lifecycle_tracker.py`

**核心接口：**

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

**最小命令：**

```bash
conda run -n ship_detect pytest test/test_msdc_types.py test/test_msdc_motion_seed.py test/test_msdc_evidence_state.py test/test_msdc_lifecycle_tracker.py -q
```

**验收标准：**

- 输出 boxes 包含现有 tracker 字段。
- 对外输出的 active box 可带 `lifecycle_state="active"`。
- diagnostics JSONL 包含 frame event 和 state transition。
- high/low detection 为空时不崩溃；active track 可进入 lost，而不是立即消失。

**失败排查：**

- 可视化失败：检查输出 schema 和 `track_id`。
- JSONL 写入失败：检查输出目录创建和 Python scalar 转换。
- 运行慢：记录分阶段 timing；正确性稳定前不要优化。

**回滚方式：**

- 删除 `lifecycle_tracker.py` 和 `test/test_msdc_lifecycle_tracker.py`。
- 如果仍需保留验证工具，可保留 earlier modules。

### Task 6：接入 `video_main.py`

**目标：** 暴露 `--tracker msdc_elt`，同时确保其他 tracker 行为完全不变。

**修改文件：**

- `video_main.py`
- `README.md`，因为 `AGENTS.md` 要求代码修改后同步文档。

**新增文件：** 无，除非新增 run id 小工具。

**核心接口：**

```bash
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker msdc_elt --output "results/msdc_elt_runs/<run_id>/annotated.mp4" --no-display
```

**最小命令：**

```bash
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker botsort --output "results/regression/botsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker ocsort --output "results/regression/ocsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker msdc_elt --output "results/msdc_elt_runs/check/annotated.mp4" --no-display
```

**验收标准：**

- `botsort` 和 `ocsort` 分支仍走 `MultiObjectTracker`。
- `msdc_elt` 分支使用 `MSDCLifecycleTracker`。
- 可视化/MQ 前的 `result["data"]["boxes"]` 仍是 project boxes。
- diagnostics 只在 `msdc_elt` 分支创建。

**失败排查：**

- `argparse` 不接受 `msdc_elt`：检查 `video_main.py` choices。
- baseline tracker 被破坏：检查分支条件，恢复旧路径。
- `detector.processor.process_frame` 不可用：检查 detector 初始化和 import path。

**回滚方式：**

- 从 `video_main.py` choices 删除 `"msdc_elt"`。
- 删除 MS-DC-ELT branch 和 import。
- 回退 README 中 MS-DC-ELT 说明。

### Task 7：实现 `TemplateLock`

**目标：** 只对 active track 启用有预算上限的模板匹配。

**修改文件：**

- `target_module/image_detect_module/config.py`
- `target_module/image_detect_module/utils/lifecycle_tracker.py`
- 如果 template observation 进入 evidence score，则修改 `target_module/image_detect_module/utils/evidence_state.py`

**新增文件：**

- `target_module/image_detect_module/utils/template_lock.py`
- `test/test_msdc_template_lock.py`

**核心接口：**

```python
template_lock = TemplateLock(max_templates=Config.MSDC_MAX_ACTIVE_TEMPLATES)
template_obs = template_lock.match_active_tracks(active_tracks, frame, frame_idx)
```

**最小命令：**

```bash
conda run -n ship_detect pytest test/test_msdc_template_lock.py test/test_msdc_lifecycle_tracker.py -q
```

**验收标准：**

- candidate track 不初始化模板。
- active track 可产生 template observation。
- 模板更新受 score 和 state 控制。
- template 数量不超过 `MSDC_MAX_ACTIVE_TEMPLATES`。
- 关闭 template lock 时，行为回到 Task 5 lite mode。

**失败排查：**

- 合成测试漂移：检查 search window scale 和 update threshold。
- 计算量尖峰：检查 template budget 和 crop size。
- template 失败导致 active 消失：确保 template evidence 是辅助证据，不是必需条件。

**回滚方式：**

- 设置 `MSDC_TEMPLATE_ENABLE = False`。
- 从 `lifecycle_tracker.py` 删除 `TemplateLock` 调用。
- 若保留独立测试无影响，可保留 `template_lock.py`；否则连同测试一起删除。

### Task 8：实现 lost reacquire 和 retired guard

**目标：** 实现 lost 低频重捕和 retired ID 复用抑制，闭合生命周期。

**修改文件：**

- `target_module/image_detect_module/utils/evidence_state.py`
- `target_module/image_detect_module/utils/lifecycle_tracker.py`
- `target_module/image_detect_module/config.py`

**新增文件：**

- 在 `test/test_msdc_evidence_state.py` 增加测试。
- 在 `test/test_msdc_lifecycle_tracker.py` 增加测试。

**核心接口：**

```python
reacquire_matches = updater.reacquire_lost_tracks(lost_tracks, observations, frame_idx)
reuse_allowed = updater.retired_guard_allows(candidate, retired_tracks, frame_idx)
```

**最小命令：**

```bash
conda run -n ship_detect pytest test/test_msdc_evidence_state.py test/test_msdc_lifecycle_tracker.py -q
```

**验收标准：**

- lost tracks 只在 `MSDC_REACQUIRE_INTERVAL` 触发重捕。
- retired tracks 不作为正常输出。
- 新 candidate 不可无条件继承 retired `gid`。
- diagnostics 包含 `lost_to_active`、`lost_to_retired`、`reuse_veto` 事件。

**失败排查：**

- 旧 ID 太容易复活：检查 retired signature gate。
- lost 从不重捕：检查 interval 和 search gate。
- retired 状态无限增长：检查 `MSDC_RETIRED_GUARD_FRAMES` 清理逻辑。

**回滚方式：**

- 设置 `MSDC_REACQUIRE_INTERVAL = 0` 关闭重捕。
- 设置 `MSDC_RETIRED_GUARD_FRAMES = 0` 关闭 retired guard。
- 如果 Task 5/7 稳定，只回退 Task 8 改动。

### Task 9：接入评测与诊断

**目标：** 让 MS-DC-ELT 可评测，同时保持 MOT 结果格式不变。

**修改文件：**

- `tools/evaluation/export_mot_results.py`
- `tools/validation/tracker_effect_test.py`
- `tools/validation/video_test_tracking.py`
- `README.md`

**新增文件：**

- 可选：`tools/evaluation/msdc_lifecycle_metrics.py`
- 可选测试：
  - `test/test_msdc_mot_export.py`
  - `test/test_msdc_lifecycle_metrics.py`

**核心接口：**

```bash
conda run -n ship_detect python tools/evaluation/export_mot_results.py --input "<sample.mp4>" --tracker msdc_elt --seq-name "seq01" --output-root "results/motchallenge_trackers"
conda run -n ship_detect python tools/validation/tracker_effect_test.py --trackers botsort msdc_elt --modalities RGB --max-frames 100
```

**验收标准：**

- MOT txt 行仍为 `frame,id,x,y,w,h,conf,-1,-1,-1`。
- `format_mot_result_line` 不变。
- MS-DC-ELT diagnostics 写在 MOT txt 旁边，而不是写进 MOT txt。
- 现有 tracker effect 输出仍包含 `annotated.mp4`、`tracks.csv`、`frames.jsonl`、`summary.json`。

**失败排查：**

- `export_mot_results.py` 不接受 `msdc_elt`：检查 `TRACKER_CHOICES`。
- export 因 tracker 接口不匹配崩溃：确保 `msdc_elt` 分支没有实例化 `MultiObjectTracker`。
- lifecycle metrics 与 MOT frame count 不一致：检查 frame indexing 和 skipped frames。

**回滚方式：**

- 从 evaluation/validation 的 `TRACKER_CHOICES` 删除 `msdc_elt`。
- 删除 export 和 validation 工具里的 MS-DC-ELT 分支。
- 如果 `video_main.py --tracker msdc_elt` 仍稳定，可保留核心 tracker modules。

## 6. 测试计划

### 6.1 单元测试

```bash
conda run -n ship_detect pytest \
  test/test_msdc_types.py \
  test/test_msdc_motion_seed.py \
  test/test_msdc_evidence_state.py \
  test/test_msdc_template_lock.py \
  test/test_msdc_lifecycle_tracker.py \
  -q
```

### 6.2 baseline 回归测试

```bash
conda run -n ship_detect pytest \
  test/test_dist_tracker.py \
  test/test_official_tracker_adapter.py \
  test/test_mot_result_export.py \
  test/test_tracker_effect_export.py \
  -q
```

### 6.3 smoke commands

```bash
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker botsort --output "results/regression/botsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker ocsort --output "results/regression/ocsort.mp4" --no-display
conda run -n ship_detect python video_main.py --input "<sample.mp4>" --tracker msdc_elt --output "results/msdc_elt_runs/<run_id>/annotated.mp4" --no-display
```

## 7. 非目标与禁止事项

- 不删除、不替换 baseline trackers。
- 不把 `MultiObjectTracker` 重写成 lifecycle tracker。
- 不修改数据集原文件。
- 不引入大型新依赖。
- 没有真实运行实验前，不声称实验已完成。
- 不默认写入已有结果目录；必须创建唯一 run 子目录。
- 不一次性实现所有模块；必须按 Task 0 到 Task 9 分步完成。
- 不把内存帧传给 `detect_targets(...)`；`AGENTS.md` 明确说明它只支持文件路径字符串。

## 8. 实施前风险

1. `video_main.py` 当前用 `cv2.imwrite` 写临时帧；`AGENTS.md` 要求磁盘图片读写优先使用 `imwrite_unicode`。不要把这个清理和 MS-DC-ELT 主逻辑混在一个 patch 里，除非触及同一路径。
2. `TargetDetector.detect_from_image_file` 目前没有 `conf_override`；第一版应直接用 `detector.processor.process_frame`。
3. 当前 `messaging/mq_publisher.py` 的 MQ 转换会剥离 class 和额外字段，因此 lifecycle diagnostics 第一阶段应使用文件 JSONL。
4. `KalmanBoxTracker.reset_count()` 会重置 local tracker ID；MS-DC-ELT 需要独立 lifecycle ID 分配器。
5. 当前 `visualization.py` 能容忍额外字段，但不显示 lifecycle state；视觉调试应作为可选功能。
6. 创建本文档前，仓库已有多个 modified/untracked 文件。后续实施者必须保留与本任务无关的用户改动。

## 9. 后续实施完成检查表

- [ ] baseline `botsort` 和 `ocsort` 仍可运行。
- [ ] `--tracker msdc_elt` 只作为显式 opt-in 启用。
- [ ] MS-DC-ELT 输出 boxes 与现有 tracker schema 兼容。
- [ ] lifecycle diagnostics 单独写入 JSONL。
- [ ] MOT export 仍输出标准 10 列。
- [ ] 每个新增模块都有单元测试。
- [ ] README 说明新 tracker option 和 diagnostics。
- [ ] 不修改数据集文件。
- [ ] 默认不覆盖任何已有结果目录。
