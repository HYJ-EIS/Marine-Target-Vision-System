# 抽帧原理、处理流程与参数说明

本文是旧批量抽帧脚本的历史说明。对应的 `tools/dataset/extract_tracking_frames.py` 当前不在仓库中，以下流程只作为旧链路参考，重点回答以下问题：

- 抽帧入口在哪里，处理什么数据
- 视频是如何被扫描、分组和逐帧处理的
- 哪些帧会被保留，哪些帧会被过滤
- RGB 与 IR 是如何对齐的
- 最终会输出哪些文件
- 相关参数从哪里来，默认值是什么，调大调小会有什么影响

当前仓库的正式链路以 `video_main.py` 和 `tools/evaluation/` 为准。

## 1. 功能定位与入口脚本

历史抽帧主入口是 `tools/dataset/extract_tracking_frames.py`。

它处理的是离线批量视频抽帧，目标是从 `_V` / `_T` 视频中筛出更适合训练或标注的关键帧，并同步导出：

- 图像文件
- YOLO 标签文件
- `classes.txt`
- 汇总清单 `frames_manifest.csv`

这条链路和 `video_main.py` 不同：

- `tools/dataset/extract_tracking_frames.py` 面向离线批处理和数据集构建
- `video_main.py` 面向实时/准实时视频检测与跟踪输出

因此，本文只记录旧抽帧链路的设计与参数含义。

## 2. 输入扫描与视频分组规则

### 2.1 扫描范围

脚本通过 `iter_target_videos()` 递归扫描 `--input-root` 下的文件，只处理同时满足以下条件的视频：

- 扩展名是 `.mp4`、`.avi`、`.mov`
- 文件名去扩展名前以 `_V` 或 `_T` 结尾

约定上：

- `_V` 表示可见光视频
- `_T` 表示红外视频

### 2.2 分组规则

每个视频会先转换为一个 `VideoJob`，其中最重要的几个字段是：

- `session_id`
- `video_id`
- `modality`
- `file_type`
- `group_key`

分组逻辑由 `build_video_job()` 和 `discover_video_groups()` 完成：

- `video_id`：由视频文件名去掉 `_V` / `_T` 后得到
- `session_id`：由日期目录和上一级目录组合成 `date_folder__parent_dir`
- `modality`：可见光映射为 `rgb`，红外映射为 `ir`
- `group_key`：`session_id__video_id`

也就是说，同一组 RGB / IR 配对视频必须：

- 来自同一个会话目录
- 去掉模态后拥有同一个 `video_id`

最终每个分组会形成如下处理单元：

- 同组下的 `rgb` 视频
- 同组下的 `ir` 视频

如果某组只有单模态，也会继续处理，只是不会进行双模态对齐。

## 3. 抽帧主流程时序说明

单个视频的主处理函数是 `process_video()`。整体时序可概括为：

```text
扫描输入目录
-> 构建 VideoJob 并按 group_key 分组
-> 打开视频并读取 FPS / 总帧数
-> 逐帧读取
-> 调用 ImageProcessor.process_frame() 做目标检测
-> 调用 MultiObjectTracker.update() 做目标跟踪
-> 合并检测框与跟踪框
-> 根据目标状态变化和 pHash 规则决定是否保留
-> 必要时补充空场景帧 / 丢失恢复帧
-> 同组 RGB / IR 互相对齐
-> 代表帧剪枝
-> 写出图像、标签和 manifest
```

### 3.1 逐帧读取

脚本使用 `cv2.VideoCapture` 顺序读取整段视频：

- `fps = capture.get(cv2.CAP_PROP_FPS) or 25.0`
- `frame_count = CAP_PROP_FRAME_COUNT`
- `timestamp_sec = frame_idx / fps`

如果通过 `--max-frames-per-video` 限制了最大处理帧数，则到达阈值后提前停止。

### 3.2 目标检测

每一帧会交给：

- `ImageProcessor.process_frame(frame, job.file_type, conf_override=...)`

检测置信度由模态决定：

- 可见光历史默认值为 `0.45`
- 红外历史默认值为 `0.45`

### 3.3 目标跟踪

检测框出来后，会交给 `MultiObjectTracker`：

- 跟踪器类型由 `--tracker` 指定
- 当前可选 `bytetrack`、`ocsort`、`botsort`、`dist_tracker`、`official_ocsort`、`official_botsort`
- 历史默认值为 `botsort`

跟踪器初始化时会使用原始视频 FPS，历史最小确认次数默认值为 `1`：

- `MultiObjectTracker(frame_rate=fps, tracker_type=args.tracker, min_hits=1)`

### 3.4 检测框与跟踪框合并

脚本不是简单地只保留跟踪结果，而是调用 `merge_raw_and_tracked_boxes()` 合并两类结果：

- 优先保留原始检测框的几何位置
- 如果某个检测框与某个跟踪框 IoU >= 0.2，则把对应 `track_id` 合并到检测框上

这意味着最终导出的框坐标更接近检测器输出，而 `track_id` 来自跟踪器。

### 3.5 跟踪补帧

如果当前帧没有检测框，脚本会尝试从跟踪器最近仍有效的轨迹中补出 `recent_tracks`：

- 容忍丢失时间历史默认值为 `2` 帧

如果补到了框，本帧会被标记为：

- `tracker_fill`

但这些框在导出标签时是否保留，还要看 `allow_tracker_fill_labels` 配置。

## 4. 保留帧判定规则

脚本不是按固定时间间隔抽帧，而是“事件驱动 + 相似度过滤”的抽帧方式。

### 4.1 keep_reason 的含义

当前代码里保留原因按以下顺序归一化写入 manifest：

- `first_detection`：第一次检测到目标
- `new_track`：出现新的轨迹 ID
- `motion_change`：目标中心位置变化达到阈值
- `pose_angle_change`：目标姿态角变化达到阈值
- `scale_change`：目标面积变化达到阈值
- `tracker_fill`：当前无检测框，但用跟踪器的近期轨迹补出了框
- `detection_lost`：上一时刻还有目标，这一帧目标消失
- `detection_recovered`：前一段空场景结束，恢复前的最后一帧空场景被补记
- `empty_scene_distinct`：空场景且和上一次保留的空场景差异足够大
- `paired_alignment`：为 RGB / IR 对齐而额外保留或补出的帧

### 4.2 检测帧的事件判断

当 `effective_boxes` 非空时，脚本会用 `evaluate_detection_keep_reasons()` 比较“当前跟踪状态”和“上一保留状态”。

每个目标会抽取一个 `TrackSnapshot`，其中包含：

- 中心点 `center`
- 尺寸 `size`
- 面积 `area`
- 姿态角 `pose_angle_deg`

随后按如下规则生成原因：

- 如果某个 `track_id` 以前不存在：
  - 还没见过检测帧时记为 `first_detection`
  - 否则记为 `new_track`
- 如果目标中心位移超过 `motion_threshold * scale_basis`：
  - 记为 `motion_change`
- 如果目标面积变化比例超过 `area_threshold`：
  - 记为 `scale_change`
- 如果姿态角变化超过 `pose_angle_threshold_deg`：
  - 记为 `pose_angle_change`

其中 `scale_basis` 取上一状态目标宽高中的较大值，用于把位移归一化成“相对目标尺寸”的变化。

### 4.3 姿态角估计方式

姿态角由 `estimate_pose_angle()` 估计，流程大致为：

- 从目标框中裁出 ROI
- 转灰度并高斯模糊
- 优先尝试阈值分割后的轮廓
- 如果轮廓不可靠，再退化到 Canny 边缘点
- 先试 `minAreaRect`
- 如果形状不够细长，再用 PCA 主方向估计

为避免噪声误触发，姿态角估计还受以下历史默认约束：

- ROI 最小边长 `12`
- 最少有效点数 `20`
- 目标最小长宽比 `1.15`
- 轮廓或边缘点最小面积占比 `0.01`

如果目标太小、太接近方形或有效点太少，则本帧姿态角返回 `None`，不会触发 `pose_angle_change`。

### 4.4 强新颖性判定

即便已经产生了 `motion_change` / `scale_change` / `pose_angle_change` 等原因，也不一定马上保留。

脚本还会调用 `has_strong_detection_novelty()` 做第二层事件过滤：

- `first_detection` 或 `new_track` 直接认为足够新颖
- 否则要求目标变化达到更强的“新颖性阈值”

这里使用的历史默认阈值更严格：

- 位移变化阈值 `0.30`
- 面积变化阈值 `0.20`
- 姿态角变化阈值 `18.0` 度

这层逻辑的作用是：避免一些轻微运动虽然跨过了基础阈值，但仍频繁触发抽帧。

### 4.5 空场景与丢失恢复

当 `effective_boxes` 为空时，分两种情况：

1. 上一帧还有目标

- 当前帧会尝试记为 `detection_lost`
- 只有在 `include_empty_frames=True` 时才能真正导出，因为空框记录默认不写出

2. 已经处于空场景

- 只有满足最小间隔后，才考虑保留 `empty_scene_distinct`
- 最小间隔按帧数计算：
  - `fps * 1.0`
- 同时还要通过空场景 pHash 去重

此外，脚本会把最近一帧空场景保存到 `last_empty_candidate`。如果后续重新检测到目标，且当前不是纯 `tracker_fill`，则会把这张“恢复前最后的空场景”补记为：

- `detection_recovered`

注意：`detection_lost`、`detection_recovered`、`empty_scene_distinct` 这三类帧都属于“空框记录”，默认只有在 `--include-empty-frames` 打开后才真正导出。

## 5. pHash 去重与代表帧剪枝

当前代码有两层“相似帧压缩”机制。

## 5.1 第一层：抽帧阶段的 pHash 判定

### 全图 pHash

`compute_frame_phash()` 的实现流程是：

- 转灰度
- 缩放到 `32 x 32`
- 做 DCT
- 取左上角 `8 x 8` 低频区域
- 以中位数为阈值转成 0/1 哈希

两个 pHash 的差异通过汉明距离 `phash_hamming_distance()` 计算。

### 目标 ROI pHash

`compute_boxes_roi_phash()` 会取所有框的最小包围区域，并在四周做少量 padding，再对这个 ROI 计算 pHash。

因此脚本同时观察两类变化：

- 整张图是否变化明显
- 目标局部区域是否变化明显

### 保留规则

有目标的检测帧使用 `should_keep_detection_frame()`，逻辑如下：

- 如果保留原因属于豁免类型，直接保留
  - `first_detection`
  - `new_track`
  - `detection_lost`
  - `detection_recovered`
  - `paired_alignment`
- 如果没有上一张已保留检测帧，也直接保留
- 否则先比全图 pHash：
  - 大于历史阈值 `10` 则保留
- 如果全图变化不够大，再比目标 ROI pHash：
  - 大于历史阈值 `8` 才保留
- 如果两者都不够大，则丢弃

空场景使用 `should_keep_by_phash()`：

- 如果与上一次保留空场景的 pHash 汉明距离小于等于 `phash_hamming_threshold`，则不保留
- 否则保留

因此，第一层过滤的核心目标是：

- 目标事件要发生
- 且画面或目标区域要与上一张保留帧足够不同

## 5.2 第二层：代表帧剪枝

当一段视频已经筛出一批候选帧后，还可能继续进入 `prune_representative_records()` 做代表帧剪枝。该功能默认开启。

### 聚类条件

脚本按时间排序后，把“足够相似”的相邻候选帧归为一个 cluster。判断相似时会同时看：

- 类别签名是否一致
- 时间跨度是否不超过 `representative_max_cluster_span_sec`
- 全图 pHash 差异是否不超过 `representative_global_phash_threshold`
- 目标 ROI pHash 差异是否不超过 `representative_target_phash_threshold`
- 目标中心位移比例是否不超过 `representative_motion_threshold`
- 目标总面积占比变化是否不超过 `representative_area_threshold`

### 每个 cluster 如何选代表帧

同一个 cluster 中不会全部保留，而是按 `representative_score()` 选分数最高的一张。分数由以下因素组成：

- 保留原因权重
  - `new_track` 权重大于 `motion_change`
  - `motion_change` 大于 `scale_change`
  - `paired_alignment` 权重较低
- 目标面积占比越大分越高
- 清晰度越高分越高
- 目标数量越多略有加分

### 最小时间间隔约束

在 cluster 选完后，脚本还会再次应用：

- `representative_min_time_gap_sec`

如果两张代表帧间隔过短，就只保留得分更高的一张。

这层机制的目标是把“连续但高度相似”的候选帧压缩成更少、更代表性的样本。

## 6. RGB/IR 对齐机制

如果同一组里同时存在 RGB 和 IR，脚本会做两步对齐。

### 6.1 双向补齐时间点

主循环中会先执行两次 `align_results()`：

- `align_results(rgb -> ir)`
- `align_results(ir -> rgb)`

它的作用是：

- 遍历源模态已经保留的时间戳
- 在目标模态中找到对应时间点的最近帧
- 如果该时间点的记录尚不存在，则重新读回那一帧并再做一次检测
- 如果通过相似度判断，就把它以 `paired_alignment` 原因补进目标模态记录集

定位目标帧时使用：

- `nearest_frame_index(timestamp_sec, fps, frame_count)`

并要求时间差不超过：

- `pair_tolerance_sec`

### 6.2 默认以 RGB 为最终参考

如果同组中既有 RGB 又有 IR，并且 `paired_rgb_reference=True`，则在 RGB 完成代表帧剪枝后，IR 不再独立保留自己的全集，而是通过 `select_records_by_reference()` 只保留与 RGB 锚点对应的一组记录。

它的实际策略是：

- 以 RGB 的 `anchor_timestamps` 为参考序列
- 在 IR 已有记录中优先找时间差最小且未被使用的记录
- 若找不到，再直接回读 IR 对应帧重新检测并补记录
- 只保留与 RGB 参考时间点成功匹配的 IR 帧

这意味着默认情况下：

- RGB 决定最终“关键时间点”
- IR 跟随 RGB 对齐输出

### 6.3 空 IR 目标的保留

如果 `paired_reference_include_empty_target=True`，则在 RGB 参考时间点上，即使 IR 对应帧没有检测框，也允许把这张空帧作为 `paired_alignment` 保留。

这个配置的意义是：

- 保证双模态训练样本的时间点一一对应
- 即使某一模态在该时刻没有检测出目标，也可保留配对图像

## 7. 输出目录、文件命名与 manifest 字段

### 7.1 目录结构

输出根目录由 `--output-root` 指定。单个 `VideoJob` 的输出路径由 `build_output_paths()` 和 `build_classes_path()` 决定。

目录结构如下：

```text
output-root/
  images/
    rgb/{session_id}/
    ir/{session_id}/
  labels/
    rgb/{session_id}/
      classes.txt
    ir/{session_id}/
      classes.txt
  manifests/
    frames_manifest.csv
```

### 7.2 frame_id 规则

帧 ID 由 `build_frame_id()` 生成：

```text
{session_id}__{video_id}__{timestamp_ms}
```

其中：

- `timestamp_ms` 是秒数乘以 1000 后四舍五入得到的 6 位字符串
- 如果模态是 IR，会在末尾追加 `_ir`

例如：

- RGB：`0711__DJI_xxx__045600`
- IR：`0711__DJI_xxx__045600_ir`

### 7.3 图像与标签文件

每条导出记录对应：

- 一张 `.jpg` 图像
- 一个 `.txt` 标签文件

标签格式由 `build_yolo_lines()` 生成，符合 YOLO 常见格式：

```text
class_id x_center y_center width height
```

坐标均已归一化到 `[0, 1]`。

类别映射来自：

- `Config.CLASSES = ["USV", "fishship", "UAV"]`

因此：

- `USV -> 0`
- `fishship -> 1`
- `UAV -> 2`

### 7.4 tracker_fill 标签导出行为

导出前会调用 `sanitize_export_boxes()`：

- 默认会丢弃 `is_tracker_prediction=True` 的框
- 只有 `--allow-tracker-fill-labels` 打开时，才允许这些纯跟踪补帧框进入标签文件

因此，“保留一张帧”和“该帧标签里是否真的写入补帧框”是两个不同层次的决策。

### 7.5 manifest 字段

每条导出记录都会写入 `manifests/frames_manifest.csv`。字段包括：

- `frame_id`
- `frame_path`
- `session_id`
- `modality`
- `source_video`
- `timestamp_sec`
- `label_path`
- `split`
- `is_enriched`
- `is_augmented`
- `source_type`
- `qc_status`
- `annotation_status`
- `keep_reason`
- `class_usv`
- `class_fishship`
- `class_uav`

其中比较关键的字段含义如下：

- `modality`：`rgb` 或 `ir`
- `source_video`：来源视频绝对路径
- `timestamp_sec`：视频中的秒级时间戳
- `keep_reason`：该帧被保留的原因，多个原因用 `|`
- `class_usv` / `class_fishship` / `class_uav`：该帧是否包含该类目标，1 表示包含，0 表示不包含

`annotation_status` 当前固定写为 `labeled`，`qc_status` 固定写为 `pending`，属于数据管理字段，不影响抽帧逻辑。

## 8. 参数说明与默认值表

## 8.1 参数来源

当前脚本有两类参数来源：

1. 命令行参数

- 定义在 `parse_args()`
- 适合按任务临时覆盖

2. 脚本内默认值

旧抽帧脚本不在当前仓库中，对应默认参数也已从 `target_module/image_detect_module/config.py` 删除。

## 8.2 当前命令行参数

| 参数名 | 默认值 | 作用 | 调大 / 调小影响 |
| --- | --- | --- | --- |
| `--input-root` | 无当前仓库默认值 | 输入视频根目录 | 只改变扫描范围，不改变抽帧逻辑 |
| `--output-root` | 无当前仓库默认值 | 输出目录 | 只影响输出位置 |
| `--tracker` | `botsort` | 跟踪器类型 | 更换跟踪器可能改变 `track_id` 稳定性和 `new_track` 触发频率 |
| `--motion-threshold` | `0.15` | 目标位移触发阈值 | 调大后更不容易触发 `motion_change`；调小后更容易保留运动帧 |
| `--pose-angle-threshold-deg` | `12.0` | 姿态角变化触发阈值 | 调大后姿态变化更不敏感；调小后更容易保留姿态变化帧 |
| `--area-threshold` | `0.12` | 面积变化触发阈值 | 调大后缩放变化更难触发；调小后更容易保留尺度变化帧 |
| `--pair-tolerance-sec` | `0.25` | RGB / IR 时间对齐容忍误差 | 调大更容易匹配到另一模态；调小对齐更严格 |
| `--phash-hamming-threshold` | `4` | 空场景 pHash 去重阈值 | 调大后更容易把相似空场景视为重复；调小后会保留更多空场景帧 |
| `--include-empty-frames` | `False` | 是否导出空框记录 | 打开后会保留 `detection_lost` / `detection_recovered` / `empty_scene_distinct` 等空场景帧 |
| `--allow-tracker-fill-labels` | `False` | 是否允许纯跟踪预测框写入标签 | 打开后标签更完整但风险是引入非检测器原生框；关闭时标签更干净 |
| `--disable-representative-pruning` | 默认不传，即开启剪枝 | 是否关闭代表帧剪枝 | 关闭后导出帧更多；开启后样本更稀疏、更代表性 |
| `--disable-paired-rgb-reference` | 默认不传，即开启 RGB 参考 | 是否关闭“RGB 作为 IR 参考” | 关闭后 IR 可独立保留更多帧；开启后 IR 更贴合 RGB 时间点 |
| `--disable-paired-reference-include-empty-target` | 默认不传，即允许配对空目标 | 是否关闭“配对时允许目标模态为空” | 关闭后双模态更干净但可能丢失配对；开启后配对更完整 |
| `--resume` | `False` | 从已有 manifest 续跑 | 打开后已导出 `frame_id` 不重复写入 |
| `--max-videos` | `0` | 最多处理多少组视频 | 0 表示不限制；调小便于局部试跑 |
| `--max-frames-per-video` | `0` | 每段视频最多处理多少帧 | 0 表示不限制；调小便于快速验证参数 |

## 8.3 历史默认配置说明

旧批量抽帧参数没有当前可执行入口，默认值不再作为 `Config` 属性维护。需要恢复这条链路时，应在新脚本参数或独立配置中重新定义，并同步更新本文。

## 9. 常见组合配置建议

### 9.1 追求干净训练集

建议：

- 保持 `include_empty_frames=False`
- 保持 `allow_tracker_fill_labels=False`
- 保持代表帧剪枝开启

结果：

- 输出更少
- 标签更干净
- 更适合作为标准检测训练集

### 9.2 追求双模态严格配对

建议：

- 保持 `paired_rgb_reference=True`
- 保持 `paired_reference_include_empty_target=True`
- 适度调大 `pair_tolerance_sec`

结果：

- RGB / IR 时间点更容易一一对应
- 即使某一模态未检出目标，也能尽量保留配对样本

### 9.3 追求更多事件帧

建议：

- 适当调小 `motion-threshold`
- 适当调小 `area-threshold`
- 适当调小活跃场景的 pHash 阈值
- 必要时关闭代表帧剪枝

结果：

- 输出帧数明显增加
- 更容易覆盖细粒度动作变化
- 但也更容易引入冗余样本

### 9.4 先试跑再正式全量抽帧

建议：

- 用 `--max-videos`
- 配合 `--max-frames-per-video`
- 确认输出目录、配对结果、标签质量后再跑全量

结果：

- 能更快验证参数是否合适
- 可减少全量跑错后的返工成本

## 10. 小结

当前抽帧脚本的核心思想不是“固定间隔采样”，而是：

- 用检测 + 跟踪确定目标状态
- 用事件变化决定候选关键帧
- 用 pHash 控制重复
- 用代表帧剪枝压缩冗余
- 在双模态场景下以 RGB 为主线对齐 IR

因此，最终输出结果同时受以下几类因素影响：

- 检测器置信度阈值
- 跟踪器稳定性
- 事件触发阈值
- pHash 去重阈值
- 代表帧剪枝阈值
- RGB / IR 配对策略

如果后续要调整抽帧数量或样本风格，优先从这些参数入手，而不是把它理解为一个简单的“每隔多少秒抽一帧”的脚本。
