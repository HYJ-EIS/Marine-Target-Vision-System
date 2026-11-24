# Marine Target Vision System

海上目标检测与多目标跟踪项目。当前主链路由 ONNX Runtime 检测器、图片检测 HTTP 服务、视频检测/跟踪入口和 MS-DC-ELT 正式评测工具组成。

单目标视频的 CVAT 预标注可使用 `conda run --no-capture-output -n ship_detect python tools/evaluation/export_cvat_single_target.py --input <视频> --output <新的时间戳目录> --label UAV`。该入口要求事先确认全片只有同一个目标，使用 `msdc_v3` 配置，在关联前仅保留指定类别。原始 ID 保存在 `raw_msdc.txt`；导出按单目标先验统一为 ID 1，同帧多个输出优先保留上一原始 ID，其次取最高置信度框，逐帧选择记录在 `frames.jsonl`。未输出的帧不补框，需在 CVAT 检查；这不是原始算法的身份一致性评测。

输出 `cvat_mot_1_1.zip` 包含 `gt/gt.txt`、`gt/labels.txt` 和 `seqinfo.ini`；按 [CVAT MOT 格式](https://docs.cvat.ai/docs/dataset_management/formats/format-mot/)使用 1 起始帧号、1 起始类别号及九列标注。`visibility=1` 是未测量占位值。先用原视频创建 CVAT 任务，标签与 `--label` 一致，起始帧 0、步长 1，再以 MOT 1.1 导入该 ZIP。包内不包含视频或逐帧图片。可视化位于本次 run 的 `visualizations/`，由 `render_mot_video.py` 从导出 MOT 渲染；另存六阶段计时、配置快照、缺失帧列表及诊断。没有人工 GT 时不计算 MOT 指标，不作为正式评测。

KM2 单 UAV 预标注结果位于 `results/cvat_km2_uav/20260917_104500/`，共解码 10,597 帧，输出 8,299 帧标注，2,298 帧未输出框（344 个区间，见 `missing_ranges.csv`）。源视频实际为带重复时间戳的 MPEG 容器，报告帧数 10,601 与 OpenCV/FFmpeg 一致得到的解码帧数不符；该次运行经独立解码核对后打包。导入 CVAT 请使用本次提供的 `KM2_UAV_CVAT_25fps.mp4`，保留原帧顺序并规范时间戳，详见 run 内 `IMPORT_README.md`。该次跟踪计时因帧数检查异常未落盘，渲染计时独立保存；后续入口已调整为先保存计时再检查帧数。不宣称正式评测完成。

## 当前状态

- 检测类别、模型路径、置信度阈值、NMS 阈值、图片输入输出目录、视频默认输入输出和 MS-DC-ELT 参数集中在 `target_module/image_detect_module/config.py`。
- 跟踪器选择、正式 MS-DC 变体名、论文方法展示名和速度统计字段集中在 `target_module/image_detect_module/constants.py`。
- 当前正式 MS-DC-ELT 主变体名为 `msdc_v3`。论文主对比使用 vendored 官方适配层 `official_ocsort`、`official_botsort`、`bytetrack` 和 `msdc_v3`。项目内自研的轻量 OC-SORT、BoT-SORT 和 Dist-Tracker 已删除。
- 视频运行和 MOT 导出的逐帧 tracking 分发共用 `target_module/image_detect_module/utils/tracking_update.py`。
- MS-DC-ELT 当前为单次低阈值检测后拆分高阈值框，并通过 low observation 预算控制候选规模。
- 正式可视化以 MOT txt 为输入，使用 `tools/evaluation/render_mot_video.py`。

## 环境

所有 Python 和 pytest 命令使用 `ship_detect` 环境：

```bash
conda run -n ship_detect python <script.py>
conda run -n ship_detect pytest test -q
```

依赖文件为 `files/requirements.txt`。当前入口脚本还直接依赖 `Flask`、`Flask-CORS`、`onnxruntime`，运行前不要假设 requirements 已覆盖完整运行时依赖。

## 入口脚本

| 场景 | 入口 | 说明 |
| --- | --- | --- |
| 图片检测 HTTP 服务 | `image_main.py` | 正式图片检测接口 |
| 图片检测调试副本 | `image_main copy.py` | 请求处理逻辑应与 `image_main.py` 同步，可额外保留 `TEST_MODE` |
| 视频检测与跟踪 | `video_main.py` | 支持 RTSP 和本地视频 |
| 统一图片检测入口 | `target_module/image_detect_module/target_detection.py` | `detect_targets(image_input, output_dir=None, enable_tracking=False)` |

`detect_targets()` 当前实际只接受图片路径字符串。不要把内存帧或 `np.ndarray` 直接传给它；视频链路需要帧级检测时走 detector/processor 内部接口。

## 图片检测 API

启动服务：

```bash
conda run -n ship_detect python image_main.py
```

默认监听 `0.0.0.0:8080`，可用环境变量 `PORT` 覆盖。当前路由：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/detect/image` | 图片检测主接口 |
| `POST` | `/api/detect/image` | 兼容路由 |
| `GET` | `/api/v1/results/<result_id>` | 读取 `results/result_<result_id>.jpg` |

`POST` 支持两类输入：

- `multipart/form-data`，字段名为 `file`。
- JSON，字段 `image_path` 指向本机已有图片路径。

当前代码没有实现 base64 `image_data` 解码。HTTP 成功分支调用 `APIResponse.success()`，但该函数目前没有返回标准成功体，因此成功响应体为 JSON `null`，不要把 HTTP 200 当作检测结果 JSON。检测结果会尝试发送到 RabbitMQ；可视化结果图会保存到 `results/result_<uuid>.jpg`，但 `POST` 响应不会返回 `result_id`。错误响应仍使用包含 `success=false`、`message`、`error_code`、`details`、`timestamp` 的 JSON 对象。

涉及磁盘图片路径读写时使用 `cv_utils.imread_unicode()` / `cv_utils.imwrite_unicode()`，不要直接使用 `cv2.imread()` / `cv2.imwrite()`。

## 视频检测与跟踪

本地视频示例：

```bash
conda run -n ship_detect python video_main.py \
  --input /path/to/video.mp4 \
  --tracker msdc_elt \
  --output outputs/demo/annotated.mp4 \
  --no-display
```

RTSP 模式：

```bash
conda run -n ship_detect python video_main.py --no-display
```

`video_main.py` 参数：

| 参数 | 说明 |
| --- | --- |
| `--input` | 本地视频路径；留空时使用 `Config.VIDEO_RTSP_INPUT` |
| `--tracker` | 跟踪器；留空时使用 `Config.TRACKER_TYPE` |
| `--output` | 输出视频路径；留空时 baseline 使用 `Config.VIDEO_OUTPUT_PATH`，`msdc_elt` 使用独立 run 目录 |
| `--no-display` | 无 GUI 环境下不弹窗 |

RTSP 默认输入、RTSP 默认输出和本地默认输出统一修改：

```text
Config.VIDEO_RTSP_INPUT
Config.VIDEO_RTSP_OUTPUT
Config.VIDEO_OUTPUT_PATH
```

不要在 `video_main.py` 中重新写死这些路径。

## 跟踪器

当前可选跟踪器定义在 `target_module/image_detect_module/constants.py`：

```text
bytetrack
official_ocsort
official_botsort
msdc_elt
```

`Config.TRACKER_TYPE` 默认使用 `official_botsort`。

ByteTrack 的对应参数已对齐 `msdc_elt + msdc_v3`：高分检测和新轨迹阈值共用 `VISIBLE_CONF_THRESH=0.50` / `INFRARED_CONF_THRESH=0.65`，由每帧的 `file_type` 选择；`BYTETRACK_LOST_BUFFER=40` 按实际帧数保留，不随 FPS 缩放；`BYTETRACK_MIN_HITS=4` 包含创建帧（supervision 从创建后的 update 计数，因此传入 `minimum_consecutive_frames=3`）；`BYTETRACK_MATCH_COST_THRESH=0.8`。这些设置集中在 `Config`，其他 baseline 继续使用原有 `TRACKER_*`。图片入口显式传入的 `IMAGE_TRACKER_MIN_HITS=1` 仍优先；reset 会重建相同配置。

这里的匹配代价为 `1 - IoU × score`，上限 `0.8` 表示 `IoU × score ≥ 0.2`，与 MSDC 的 IoU/中心距离组合门控不完全等价。ByteTrack 保留原库的两阶段关联、严格大于高分阈值的分组条件、内部低分下限 `0.1`、第二轮代价上限 `0.5` 和未确认轨迹代价上限 `0.7`；实际可用的低分框还取决于调用方输入。MSDC 的证据分数、重获、removed-ID recovery、Top-K、输出 NMS 等没有可直接映射的 ByteTrack 参数。该调整不代表两种算法具有相同的确认条件或状态转移规则。

当前论文实验入口的主对比 tracker 集合为：

```text
official_ocsort
official_botsort
bytetrack
msdc_elt + msdc_v3
```

baseline 跟踪器通过 `MultiObjectTracker.update()` 更新；`msdc_elt` 通过 `MSDCLifecycleTracker.update()` 更新。统一分发逻辑在：

```text
target_module/image_detect_module/utils/tracking_update.py
```

MS-DC-ELT 主要文件：

| 文件 | 作用 |
| --- | --- |
| `utils/lifecycle_tracker.py` | 生命周期状态机、ID 分配、输出过滤 |
| `utils/msdc_detection.py` | 单次低阈值检测、高阈值框拆分和 low observation 预算 |
| `utils/evidence_state.py` | evidence score、candidate 确认和轨迹状态更新 |
| `utils/msdc_types.py` | MS-DC 内部数据结构 |

`msdc_v3` 默认关闭 low candidate 的新轨迹生成，但保留 low observation 对已有轨迹的关联与重获；direct reacquire、removed-ID recovery、output NMS 和 low observation Top-K 预算仍默认启用。`low_candidate_on` 用于重新开启 low candidate；`no_low_candidate` 仅作为历史命令/报告的兼容别名。`pending_recovery_candidate`、`removed_recovery_off`、`reacquire_every_frame_low_score` 等是消融/对比变体，不是默认主变体。

正式 v3 的参数统一在 `target_module/image_detect_module/config.py` 修改（2026-10-12 迁移）：

- 公共参数修改 `Config.MSDC_*`，例如 `MSDC_OUTPUT_MIN_BOX_SIZE=5`；v3 自动继承，不再由 replay 脚本中的重复参数表覆盖。
- 仅供 v3 使用的差异修改文件末尾 `MSDC_V3_OVERRIDES`：目前为 `MSDC_LOW_CANDIDATE_ENABLE=False`、`MSDC_DEBUG_EVENTS=True`，以及 ACTIVE/LOST/CANDIDATE/LOW_CANDIDATE/TOTAL 容量 `128/64/64/48/256`。也可在此加入已有 `MSDC_*` 字段，使其仅在 v3 中采用指定值；未知字段会报错。
- 启动时由基础 Config 加上 v3 overrides 生成完整 `MSDC_V3_CONFIG` 和环境变量格式的 `FORMAL_V3_ENV`。消融入口保留兼容导出，replay、速度测试、敏感性分析和单目标预标注均读取同一来源。修改配置文件后需重启进程。
- 优先级为：基础 Config（含启动时环境变量）→ `MSDC_V3_OVERRIDES` → 消融/显式实验覆盖。启动后临时修改 Config 或环境变量不会污染已生成的正式配置快照。纯速度测试仍按既有计时约定关闭诊断日志。
- 导出子进程会接收完整配置；`Config` 加载时也读取此前采用字面值定义的 `MSDC_*` 环境变量，确保公共参数和 v3 专用覆盖在子进程中一致。普通入口未选择 v3 时继续使用基础 Config。

迁移前后已核对全部 15 个现有变体的有效配置，数值一致；本次配置迁移未重新执行正式评测。

普通关联门控诊断可运行 `conda run --no-capture-output -n ship_detect python tools/evaluation/association_gate_analysis.py --run-root <含检测缓存及历史轨迹日志的统一评测目录> --output <新的分析目录>`。脚本复用高低阈值检测缓存及当前 v3 配置，调用统一 tracking update，不运行检测、视频解码或可视化，不作为正式 benchmark。当前 lifecycle 不读取图像像素，因此回放使用相同尺寸的空白数组；逐帧将内部轨迹快照与历史日志比较。统计单位是普通关联中的“预测轨迹框—合并观测框”候选对，包含 CANDIDATE/LOW_CANDIDATE/ACTIVE，不包含 LOST 重获、removed-ID recovery 等独立门控。另统计每帧去重后的观测组及最终选中的匹配，避免把候选对数量误认为原始检测框数量。

2026-10-12 对四个剧烈摇晃数据集的诊断结果位于 `results/association_gate_analysis/20261012_163100/`。完整回放 24,594 帧，内部轨迹快照与 2026-09-30 日志全部一致（本次最小输出尺寸为 5，历史为 12，不影响内部关联状态）。普通关联测试 17,345 对，16,725 对通过：同时满足 IoU≥0.2 与中心距离≤80 像素的 15,495 对（92.65%），仅中心距离满足的 1,230 对（7.35%），仅 IoU 满足的 0 对；另有 620 对均不满足。四个序列的“同时满足/仅中心距离满足”分别为 08：277/9，09：398/0，KM2：9349/576，on_sea：5471/645。满足 IoU 的候选最大中心距离为 20.30 像素。

本次最终普通关联匹配为 16,719 对，其中同时满足两门限 15,495 对、仅中心距离满足 1,224 对。逐帧固定原有状态后，仅删除 IoU 门控分支（保留原评分），以及仅删除评分中的 IoU 项（保留原门控），均未改变该帧匹配；全数据仅有 6 帧存在候选竞争。结论限定于本次缓存、参数和普通关联：IoU 门控没有增加候选，不能由此判断合并观测、重获及输出 NMS 中的 IoU 也无作用。`summary.csv`、`analysis.json` 保存计数、口径、配置和源文件哈希，`*_pairs.csv` 保存所有通过门控的候选对、IoU、中心距离、是否选中及评分；此诊断不提供修改算法后的 MOT 精度结论。

核心关联由 `MSDC_ASSOC_METHOD` 控制：`msdc_v3` 固定使用原有 `greedy` 匹配；`gated_hungarian` 复用相同的 IoU/中心距离门控和组合分数，并在门控后的代价矩阵上执行匈牙利匹配。可通过 replay 入口使用 `--variants msdc_v3 gated_hungarian` 在同一检测缓存上比较两者。

速度评测的 `--msdc-variant` 会应用该变体的 `FORMAL_V3_ENV`；除计时阶段关闭 `MSDC_DEBUG_EVENTS` 以排除诊断写盘外，low candidate、重获、removed-ID 恢复及关联方式等配置与 replay 指标评测一致。
`speed_results.csv` 和逐帧 `speed_timings.jsonl` 同时记录 `variant` 与 `association_method`，用于区分同为 `msdc_elt` 的速度结果。

统一全视频评测可通过 `tools/evaluation/run_unified_comparison.py --dataset-root <数据集目录> --run-root <新的时间戳目录> --msdc-variant msdc_v3` 明确加载当前正式变体；其他方法保持当前 Config。四方法共享一次独立推理生成的低阈值缓存和原样复制的 GT，全部帧参与评测，MOT 可视化写入本次 run 的 `visualizations/`。`run_metadata.json` 记录 tracker 名称映射与配置。replay 计时将 MSDC 的实测 `debug_s`（诊断生成/写盘）从 update 墙钟时间扣除并归入 `write_export`；逐帧 `.frames.jsonl` 保留原始 update、诊断与纯跟踪时间。总体时间仍包含诊断、检测、读取、跟踪、渲染和导出。汇总器按实际序列数和 tracker 名称输出，不复用历史 run 的参数或标注来源结论。

## 正式评测

正式评测总入口：

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py
```

不带 `--run-formal` 时只打印计划命令。smoke：

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --smoke
```

正式 run：

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py \
  --dataset-root \
  /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 \
  /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --formal-frame-limit 5400 \
  --speed-frames 5400 \
  --progress-interval 500 \
  --run-formal
```

默认输出根目录为 `results/msdc_paper_phase1/<timestamp>/`。正式 run 会执行：

```text
main
ablation
slice
slice_metrics
speed
sensitivity_matrix
sensitivity_metrics
summary
```

main、ablation 和 sensitivity metrics 阶段使用 replay detections。每个序列先生成或复用同一份 high/low detection cache，再让 ByteTrack、OC-SORT、BoT-SORT 和 MS-DC-ELT 读取同一检测流。直接运行 `tools/evaluation/detection_replay_benchmark.py` 时，可用 `--source-detection-cache-root <detections_dir>` 复用 `<seq>_high_low_detections.jsonl`；源缓存不完整时会报错，不会静默重新生成。

正式 run 完成后必须通过：

```bash
conda run -n ship_detect python tools/evaluation/validate_msdc_formal_run.py --run-root <run_root>
```

正式结果不能和 smoke、失败中断或 partial 输出混在一起；每次正式测试必须写入新的时间戳 run 目录。

## 正式评测交付项

不能只凭终端日志宣称正式测试完成。一次正式评测至少需要同时交付：

- 带标注可视化视频，包含目标框、类别和 track ID，写入本次 run 独立 `visualizations/` 目录。
- MOT/TrackEval 指标汇总，至少包含 `MOTA`、`IDF1`、`IDSW`、`FN`、`FP`；若已有 `HOTA`、`IDTP`、`IDFP`、`IDFN` 也一并汇报。
- 速度统计，至少包含总处理帧数、总耗时、平均单帧耗时和平均 FPS。
- 分阶段耗时，至少拆分读取/解码、低阈值检测、高阈值框筛选、lifecycle tracker、可视化渲染、结果写盘/导出；不适用阶段写 `0` 或 `N/A`。
- 输出路径清单，包括 MOT txt、TrackEval summary、诊断 JSONL/CSV、可视化 MP4、速度/耗时统计文件。

`validate_msdc_formal_run.py` 会检查 `summary/main_results.csv`、`summary/speed_results.csv`、`summary/diagnostic_results.csv`、`summary/path_manifest.csv`、`slice/slice_manifest.csv`、`slice/slice_metrics/slice_metrics.csv`、`sensitivity/sensitivity_full/sensitivity_matrix.csv`、`sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv`、`**/effective_config/*.json`、MOT txt、诊断文件和可视化视频。

如果只运行 full-video no-render replay，例如为了快速比较 `official_ocsort`、`official_botsort`、`bytetrack` 和 `msdc_v3`，可以直接使用 `tools/evaluation/detection_replay_benchmark.py` 且不传 `--render`。这类 run 会生成 MOT txt、TrackEval summary、diagnostics 和 detection cache，但不会生成可视化 MP4，也不会运行 slice/qualitative 分析；因此不能按本节“正式评测交付项”宣称为完整正式评测。

检测回放评测与 `render_mot_video.py` 在检测缓存、MOT txt、MP4 旁生成独立 `.timing.json`，记录处理帧数、实际耗时、平均延迟/FPS，以及读取解码、低阈值检测、高阈值筛选、tracker、渲染、写盘六阶段累计秒数；当前阶段未执行的项为 0。跟踪计时包含跟踪器内部诊断写盘，渲染写盘包含视频编码。检测缓存仅计算一次；报告每种方法的组合处理耗时时须注明复用了检测耗时，不能把回放 FPS 当成端到端 FPS。回放更新使用共享 `tracking_update.py`。

`eval/motchallenge_summary.csv` 补充 `LocA`、`MOTP`、`IDP`、`IDR`、`TP`、`Frag`，覆盖 17 项指标。百分比指标采用 0–100 标度，MOTP 为匹配框平均 IoU（越高越好），TP/FP/FN 为 CLEAR 在 IoU=0.5 下的计数。评测前须核实未标注帧是否为空目标帧以及 GT ID 是否代表真实跨帧身份；原始标注不作自动身份修复。

`tools/evaluation/summarize_replay_comparison.py --run-root <run> --seq-name <seq> --frames <N>` 用于本次 UAV 数据集的 BoT-SORT/msdc_v3 双方法回放交付：读取 `run_metadata.json`（包含 Config 快照和标注清理记录），校验检测缓存连续性、GT 计数及两个 MP4 的完整解码，生成 17 项指标对比、速度与六阶段计时 CSV、文件清单和 `REPORT.md`。`validation.json` 仅在校验通过后标记完成。

本次结果位于 `results/on_sea_botsort_msdc/20260915_222448/`：全视频 13,304 帧；按要求从原始 GT 删除 16 个 USV 框，保留 5,744 个 UAV 框，原文件备份在 run 的 `annotation_backup/`。`summary/gt_geometry_audit.json` 记录 GT 与检测框的坐标重合统计，`REPORT.md` 将该情况与定位指标一并说明。

四方法扩展对比位于 `results/on_sea_four_trackers/20260916_103507/`，复用上次检测缓存并核对 SHA256，新增 supervision ByteTrack 和 vendored 官方 OC-SORT；BoT-SORT/MSDC-ELT 的 MOT、可视化及耗时从原 run 复制并注明来源，四者统一重新计算 TrackEval 指标。三个 baseline 均接收缓存高阈值框，MSDC 接收高、低阈值框；该输入约定限制了 baseline 内部低分框关联的可用输入，结果代表当前项目配置。汇总器通过 `run_metadata.json` 的 `report_trackers` 映射支持四列指标、速度及阶段耗时，逐个验证视频完整性。

上述历史 run 的 ByteTrack 适配直接使用 `TRACKER_MATCH_THRESH=0.2` 作为 supervision 首轮关联的代价上限；安装版本默认值为 0.8。融合检测分数后代价为 `1 - IoU × score`，当时设置要求 `IoU × score ≥ 0.8`，明显更严格；历史报告保留该配置事实。当前代码已采用前述独立的 `BYTETRACK_*` 配置，历史指标不代表修改后的结果。

检测输入对照支持 `detection_replay_benchmark.py --baseline-input all`：向 baseline 传入完整 `low_boxes`（其中已包含高阈值框，不重复拼接）；默认 `high` 保持历史行为。跟踪器内部阈值保持 Config 原值，MSDC 的高低双输入不变。策略保存在 run 的 `baseline_input_policy.json` 和跟踪计时文件，切换策略必须使用新 run 目录。

全视频低分输入对照结果：`results/on_sea_low_input/20260916_105249/INPUT_COMPARISON.md`。输入从 5,899 框增至 6,872 框，三个 baseline 的 FN 分别减少 40/1/33（BoT-SORT/ByteTrack/OC-SORT），FP 增加 433/13/164；其他参数未变，综合指标未提升。完整 17 项指标、视频和六阶段耗时见同目录 `REPORT.md`，历史 run 保留。

## 常用评测工具

| 脚本 | 用途 |
| --- | --- |
| `tools/evaluation/export_mot_results.py` | 导出 MOTChallenge tracker result txt |
| `tools/evaluation/motchallenge_eval.py` | 运行 MOTChallenge/TrackEval 指标 |
| `tools/evaluation/detection_replay_benchmark.py` | 回放 high/low detection cache 并生成 tracker 指标 |
| `tools/evaluation/msdc_dataset_benchmark.py` | 数据集级 benchmark 编排，主要用于 smoke/辅助流程 |
| `tools/evaluation/msdc_speed_benchmark.py` | 无渲染速度与分阶段耗时评测 |
| `tools/evaluation/msdc_diagnostic_metrics.py` | 汇总 low candidate、inherit、reacquire、removed recovery 和 fragmentation 诊断指标 |
| `tools/evaluation/msdc_slice_eval.py` | 从 per-GT diagnostics 生成短漏检 slice manifest |
| `tools/evaluation/msdc_slice_metrics.py` | 基于 slice manifest 裁剪 MOT txt 并计算 slice 指标 |
| `tools/evaluation/msdc_sensitivity_matrix.py` | 生成 MS-DC-ELT 动态超参数敏感性矩阵 |
| `tools/evaluation/run_msdc_sensitivity_benchmark.py` | 复用 detection cache 执行敏感性 replay 指标 |
| `tools/evaluation/msdc_experiment_summary.py` | 汇总 main、ablation、speed、slice、sensitivity 和 diagnostics |
| `tools/evaluation/render_mot_video.py` | 基于 MOT txt 渲染正式标注视频 |
| `tools/evaluation/render_msdc_diagnostics_video.py` | 基于 MS-DC diagnostics JSONL 渲染诊断视频 |
| `tools/evaluation/validate_msdc_formal_run.py` | 检查正式 run 产物完整性 |
| `tools/experiments/run_msdc_ablation.py` | 单视频 MS-DC 消融命令生成或执行 |

## 输出目录

| 路径 | 内容 |
| --- | --- |
| `uploads/` | 图片 API 上传文件临时保存目录 |
| `results/` | 图片结果图、benchmark、MOT、summary 和速度评测产物 |
| `outputs/` | 视频入口输出 |
| `outputs/msdc_elt/<run_id>/` | `video_main.py --tracker msdc_elt` 的独立视频输出和诊断 |
| `results/msdc_paper_phase1/` | phase-1 论文实验默认输出根目录 |

## 测试

完整测试：

```bash
conda run -n ship_detect pytest test -q
```

常用聚焦测试：

```bash
conda run -n ship_detect pytest \
  test/test_msdc_constants.py \
  test/test_tracking_update_helper.py \
  test/test_msdc_speed_benchmark.py \
  test/test_msdc_experiment_summary.py \
  test/test_validate_msdc_formal_run.py \
  -q
```

视频端到端、RTSP、RabbitMQ、模型文件和本地数据依赖较重；运行完整流程前先确认外部服务和样本可用。

## 目录速览

```text
.
├── image_main.py
├── image_main copy.py
├── video_main.py
├── cv_utils.py
├── messaging/
├── output_rtsp_video/
├── target_module/
│   ├── image_detect_module/
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── target_detection.py
│   │   └── utils/
│   └── models/
├── tools/
│   ├── evaluation/
│   ├── experiments/
│   └── validation/
├── test/
├── outputs/
└── results/
```

### 全视频四方法统一评测（当前 Config）

`conda run -n ship_detect python tools/evaluation/run_unified_comparison.py --dataset-root <数据集目录> --run-root <新的时间戳目录>` 会递归枚举视频、解码到 EOF 核实帧数、原样复制各序列 GT，重新生成共享低阈值检测缓存，并顺序运行 BoT-SORT、MSDC-ELT、ByteTrack、OC-SORT。三个 baseline 接收完整 low_boxes，MSDC 接收同一缓存拆出的高低阈值框；`preserve_config=True` 禁止变体覆盖并保留当前诊断开关。每个视频需在同目录具有 `gt/gt.txt` 或 `gt.txt`。

输出包含 17 项百分比/计数指标 CSV、两类速度 CSV、六阶段和其他开销 CSV、GT/检测重合审计、原始 TrackEval 输出、逐 GT 诊断及从 MOT txt 渲染的完整视频。总体速度为实测检测＋回放＋渲染离线阶段总和，各方法计入同一份检测成本一次；纯 tracker 速度只计 update。诊断开关保持 Config 不变。只有全部缓存、帧数、视频和 GT 计数校验通过后，`status.json` 才标记 complete。原始视频时间戳/容器帧数误报不作为裁剪依据。

TrackEval 原生 detailed CSV 的比例为 0–1。`conda run -n ship_detect python tools/evaluation/normalize_comparison_csv.py <run>` 保留原始文件，在 `summary/*_detailed_percent.csv` 中提供全部比例指标（包括各 HOTA 阈值）的 0–100 版本；计数和非百分比量不缩放，转换列清单写入 `summary/csv_scales.json`。
统一入口结束时自动运行 detailed CSV 标度转换，并保留原始 TrackEval CSV；负 MOTA 按公式保留，不强行截断至 0。

KM2 全量四方法实测结果：`results/km2_four_trackers/20260920_151508/REPORT.md`。1 段视频，每方法 10,597 帧，四个完整 MOT 可视化视频；GT 10,227 框、2 ID。展示表及百分比明细均为 0–100；原始 TrackEval CSV 保留原生比例用于追溯。报告明确列出 GT 与检测框以及历史 MSDC 预标注的重合、低 FP/高 FN 的取舍，以及离线整体速度和纯 update 速度的不同口径。

### 每类一个目标的 CVAT 多类别预标注

`conda run --no-capture-output -n ship_detect python tools/evaluation/export_cvat_known_targets.py --input <视频> --output <新目录> --labels UAV fishship` 使用当前 Config 的 MSDC-ELT，对同一帧全部指定类别联合跟踪。按用户“每类一个真实目标”先验，每类导出固定 ID，原始 ID、选择及检测记录另存。仅对前后有框、内部缺失 1–2 帧的区间逐坐标线性插值，不补长缺口或外推首尾。输出 CVAT MOT 1.1 ZIP（九列 GT、1 起始帧号/类别号、labels.txt）、标注视频、插值记录、剩余缺口和计时；导入说明在输出目录 IMPORT_README.md。08 视频用 `--labels USV UAV`，09 用 `--labels UAV fishship`。这是预标注，不是正式评测。

2026-09-22 首轮双目标预标注在 `results/cvat_two_targets/20260922_211505/`。当前模型未形成船轨迹；09 的 UAV 为 396/399 帧，08 的 UAV 为 8/287 帧（多数小框低于该次运行的 12 像素输出门槛）。未出现可补的内部 1–2 帧缺口。两个 ZIP 格式有效，但标注不完整，状态为 `needs_annotation_review`，不能当作完整双目标 GT。

2026-09-22 按用户要求将 `Config.MSDC_OUTPUT_MIN_BOX_SIZE` 默认值从 12 改为 5：输出框宽、高均须 ≥5 像素，环境变量仍可覆盖。USV、fishship、UAV 共用检测阈值；可见光普通/高阈值为 0.50，MSDC 低阈值检测为 0.18，红外对应为 0.65、0.22。此次仅降低输出尺寸门槛，检测置信度保持不变；历史预标注包不随配置修改自动更新。

2026-09-23 视觉辅助船标注位于 `results/cvat_assisted/20260923_111339/`，最终交付子目录为 `09_KM2_drone_boat_sway/` 和 `08_drone_boat_continuous_sway/`。UAV 使用 5 像素门槛重跑 MSDC；船目标使用视觉指定框、模板传播/逐帧局部定位及全部船区域联系表核验，来源逐框记录于 `annotation_sources.csv`。09 的出画/可见部分不足区间不补框；首尾及长缺口不外推。各目录包含 CVAT MOT 1.1 ZIP、标注视频、来源 CSV、缺失区间与导入说明。`*_msdc/` 保留算法原始输出，其余 review 目录属于定位诊断材料，不是最终 GT。新增 `tools/evaluation/assist_boat_annotation.py` 可从指定视觉框传播并生成全部帧的核验图；自动传播仍须视觉检查，不能直接当作算法输出或人工 GT。

### 四个剧烈摇晃数据集统一全量评测（2026-09-23）

`tools/evaluation/run_unified_comparison.py --dataset-root <数据集父目录> --run-root <新的时间戳目录>` 对所有视频全帧生成同一检测缓存，保持当前 Config 跟踪参数，依次运行四方法并用 MOT txt 渲染完整视频。`unified_dataset_report.py` 补充逐数据集 17 项指标、速度与分阶段 CSV、四数据集等权平均表，以及与 TrackEval 全序列聚合不同的统计说明。等权平均计数为每数据集平均次数，聚合计数为总和；整体 FPS 使用总帧数/总耗时。纯 tracker update 计时排除外围检测、解码、渲染、导出。原始 TrackEval CSV 保留，展示比例统一乘 100，负 MOTA 不截断。检测与 GT 重合按每个数据集单独审计，不能用定位重合代替关联能力证据。

统一报告同时列出逐数据集 FN 比例、低 FP/高 FN 的取舍、MSDC 相对三个 baseline 的差值，以及一次计时的解释限制。KM2 容器标称 10,601 帧，OpenCV 与 FFmpeg 完整解码均为 10,597 帧；本次按实际解码帧数评测并保存 FFmpeg 核验记录，不补造帧。
统一报告附加 `summary/detection_gt_coverage.csv`，按 GT 类别统计同帧是否存在同类且 IoU≥0.5 的缓存检测，帮助区分输入检测覆盖不足与跟踪输出漏检。这是每个 GT 的重合诊断，允许一个检测覆盖多个 GT，不冒充一对一检测召回指标。
海上视频容器标称 13,304 帧，而此次 OpenCV 全解码为 13,311 帧；统一入口以读到 EOF 为准，保留额外的 7 帧。四数据集实际合计 24,594 帧/方法（287、399、10,597、13,311），报告逐项列出容器标称值和实际处理值。

本次全量交付：`results/shaking_four_datasets/20260923_155957/REPORT.md`。四数据集/四方法共 98,376 次逐帧 update、16 个完整 MOT 标注视频，完整性检查见 `validation.json`。`summary/per_dataset_metrics.csv` 为逐数据集全部指标，`performance_macro_mean.csv` 为四数据集等权平均，`performance.csv` 为 TrackEval 总体聚合；速度和分阶段耗时亦分别提供逐数据集、等权平均及总体口径。等权平均 HOTA 为 BoT-SORT 35.09、MSDC-ELT 57.07、ByteTrack 1.63、OC-SORT 27.94；MSDC 累计 FN 2,000 多于 BoT-SORT 的 1,303，ByteTrack 的 98 FP 伴随 13,221 FN。报告同时说明 GT/检测框重合、短视频船类检测覆盖不足，以及单次离线速度测量的限制。

2026-09-30 按当前参数并显式加载 `msdc_v3` 的四数据集全量交付：`results/shaking_four_msdc_v3/20260930_143800/REPORT.md`。每方法 24,594 帧，16 个完整标注 MP4；`validation.json` 与 `delivery_validation.json` 均通过。总体聚合 HOTA 为 BoT-SORT 20.02、MSDC 47.42、ByteTrack 11.50、OC-SORT 16.04；MSDC 的 32 IDSW、283 FP 同时伴随 4,325 FN，ByteTrack 的 72 FP 伴随 10,075 FN。四数据集等权平均下，MSDC 的 AssA、MOTA 低于 BoT-SORT，不能将总体聚合优势外推为所有场景都更好。

本次 `08` 序列的 `msdc_v3` 只有 8 TP、566 FN。原始日志核对表明，284 条满足证据和检测年龄要求的 ACTIVE 记录中有 276 条被正式变体的 12 像素输出门槛过滤；该参数与基础 Config 的 5 像素不同，本次遵照要求保持不变。审计明细见 run 内 `summary/msdc_08_output_filter_audit.csv`。`summary/performance.csv`、`speed.csv`、`stages.csv` 保存完整总体表，逐数据集与等权平均表一并保存；`files.csv` 和 `summary/artifacts.csv` 列出原始结果、诊断、计时、配置、源码和视频路径。首次仅针对 KM2 的运行已因用户扩大数据集范围而停止，保留为 `results/km2_msdc_v3_comparison/20260930_143200_partial_scope_changed/`，不属于正式结果。

### 同 ID 最长连续跟踪时长分析（2026-10-12）

2026-10-12：正式 `msdc_v3` 的 `MSDC_OUTPUT_MIN_BOX_SIZE` 改为读取基础 `Config`，默认 **5 像素**，要求输出框宽、高均 ≥5；正式 replay 的基准配置同步读取该值。进程启动前设置同名环境变量仍可覆盖，继承正式 v3 配置的消融变体同步生效。此前 2026-09-30 正式结果及下述离线分析使用的是 **12 像素**门槛，保留为历史结果；本次参数修改未重新运行正式评测。

`tools/evaluation/stable_tracking_analysis.py --run-root <已有统一评测目录> --output <新的分析目录>` 对已有 MOT 和 GT 做离线统计，不重新执行检测/跟踪，也不构成新的正式 benchmark。使用现有 TrackEval CLEAR 的单类匹配约定（IoU≥0.5、优先保持上次匹配 ID），并核对逐序列 TP、FN、FP、IDSW。每个 GT ID 的连续匹配段遇到漏匹配、预测 ID 改变或 GT 缺帧时结束；最长段帧数除以该目标有效 GT 帧数，得到“同 ID 最长连续跟踪占比”。该口径不累计恢复后的多个片段，不设置最短稳定时长，不能未经口径确认就等同于通用海试稳定跟踪率。历史无协议结果按原几何匹配分析；新评价目录含 evaluation_protocol.json 时，分析同时要求预测类别与 GT 类别相同。

本次分析复用 `results/shaking_four_msdc_v3/20260930_143800/`，核对源代码清单和四份 GT 均未变化。结果位于 `results/stable_tracking_analysis/20261012_115900/`：`datasets.csv` 为逐数据集结果，`targets.csv` 为逐 GT ID 结果，`segments.csv` 保存全部连续匹配段及帧号，`analysis.json` 保存定义、源文件哈希和校验结果。08、09、KM2、on_sea 的上述占比分别为 0.70%、54.70%、9.06%、26.13%；按目标时长汇总为 113.12 / 690.76 = 16.38%。四个序列的帧率均按已有 seqinfo 的 25 FPS 换算。分母只统计有效标注目标帧，不代表已核实的理论可观测时长；缺失标注、GT 身份及预标注来源限制仍适用。复现验证：`conda run --no-capture-output -n ship_detect python -m pytest -q test/test_stable_tracking_analysis.py`。

### 历史视频有效跟踪时长统计（2026-10-13）

`conda run --no-capture-output -n ship_detect python tools/evaluation/tracking_duration_report.py --output <新的分析目录>` 统计现存完整 MOT、GT、seqinfo 和 TrackEval detailed；排除显式 test、smoke、partial 记录，并逐组核对 TP、FN、FP、IDSW、GT 框数及预测框数。这是历史结果后处理，不重新执行跟踪或作为一次新的正式评测。有效时长为 CLEAR 正确匹配目标帧数/FPS，可累计中断前后的片段；另外统计每目标同 ID 最长连续段、漏跟时长、漏跟后恢复次数、IDSW、原 TrackEval Frag、IDTP 对应时长及 MOTA/IDF1/HOTA。GT 缺帧打断连续段但不计漏跟；多目标分别累加为目标秒。

本次输出 `results/tracking_duration_analysis/20261013_165000/`：7 段不同视频、48 组历史结果；其中 6 段有 V3，主表每视频只取最新可用 V3。`REPORT.md` 说明口径，`latest_v3.csv` / `latest_v3_targets.csv` 为主表，`all_runs.csv` / `targets.csv` / `segments.csv` 保留全部记录、目标和片段，`comparisons.csv` 仅比较同 run 同视频的方法，`analysis.json` 保存哈希与排除条目，`source_artifacts.csv` 索引历史计时与可视化。安装可选报告依赖 `openpyxl` 时同步生成 `跟踪时长统计.xlsx`；统计本身不依赖该包。补充连续片段门槛默认 1 秒，可用 `--minimum-segment-seconds` 设置，不能直接称为通用海试稳定跟踪率。

主表历史 V3 仍使用 12 像素最小输出尺寸；不能当作当前默认 5 像素的重测结果。四段摇晃视频累计有效时长 517.76/690.76 目标秒（74.96%），每目标最长连续段之和仍为 113.12 目标秒（16.38%）。当前 vendored CLEAR 对整帧无预测的时段跳过部分状态更新，原 Frag 与新增“漏跟后恢复次数”并不相同，两列分别保留。验证：`conda run --no-capture-output -n ship_detect python -m pytest -q test/test_tracking_duration_report.py test/test_stable_tracking_analysis.py`。

### V3 最小输出尺寸 5 的重新评测

`tools/evaluation/rerun_msdc_size5.py --run-root <新目录>` 使用当前正式 V3 配置及三种 baseline，对此前六个 V3 序列的同一观测区间重新运行：四段摇晃视频全帧，旧 USV 序列 4,772 帧，旧 UAV_USV 序列前 5,400 帧。后两段失效的缓存可从原逐帧日志中的原始 `high_det` / `low_det` 输入恢复，严禁采用 `low_only`、`output` 或 GT；每个序列先以尺寸 12 复现旧 MOT 并核验字节一致，复现结果仅作输入/实现一致性诊断，不计入正式速度。正式输出在独立 `trackers/`、`visualizations/`、`eval/`、`summary/` 和 `duration/` 下保存。

本流程检测输入共享，三种 baseline 接收完整低阈值框，V3 接收高、低阈值框。速度计入本次解码、跟踪、诊断/导出及可视化；未执行的低阈值检测和高阈值筛选阶段记 0，不能将其 FPS 描述为含检测的在线速度。原图坐标参与指标计算，4K 源视频可视化缩放到 1920 宽，其余不放大。`status.json` 为运行状态，`validation.json` 通过且完整交付后才能称为正式完成。测试覆盖日志恢复、配置、时间统计和 TrackEval。当前重测目录：`results/msdc_size5_formal/20261013_172900/`。

完成后运行 `conda run --no-capture-output -n ship_detect python tools/evaluation/finalize_size5_report.py <run目录>`，使用可选 `openpyxl` 生成 `V3尺寸5_跟踪时长与MOT指标.xlsx` 和 `REPORT.md`。同时保存尺寸12/5受控比较、带GT类别的逐目标时长、六序列/四摇晃序列分别聚合的TrackEval结果，补齐MT/PT/ML及比例、Precision/Recall。最终交付检查 `delivery_validation.json` 会复核缓存/GT哈希及分阶段时间总和。

2026-10-13 重测已完成：每方法 34,766 帧，四方法 24 个完整标注视频；六段尺寸12复现与历史MOT均逐字节一致，尺寸5实际配置已核验。V3六序列总体 HOTA 68.93、MOTA 79.67、IDF1 71.28，IDSW 49、FP 2,818、FN 6,596。08累计正确跟踪时长由0.32增至11.32目标秒；KM2/on_sea分别增至366.04/217.36目标秒，但IDF1分别由55.84/32.72降至55.13/28.68，不能据覆盖率提升宣称身份连续性全面改善。V3缓存回放及渲染共490.23秒，14.10 ms/帧、70.92 FPS，不含检测；逐阶段及四方法速度见本次run的`summary/`。运行前相关测试16项通过。

### 四个摇晃数据集含检测的当前配置全量评测（2026-10-14）

本次独立目录为 `results/shaking_four_current/20261014_161000/`，使用 `run_unified_comparison.py --msdc-variant msdc_v3`，保持当前正式 v3（最小输出框尺寸 5）与三个 baseline 的内部参数。重新逐帧检测生成共享缓存，原样复制 GT，顺序执行四方法及 MOT 视频渲染；整体耗时包含本次检测，纯跟踪耗时扣除实测诊断导出。`summary/` 分别提供四数据集等权平均、各数据集、总体聚合的指标和速度，比例展示为 0–100。原生 TrackEval CSV 保留在 `eval/`，百分比版另存，不覆盖原始数据。

`validate_delivery.py` 独立核验指标公式、等权平均、逐帧时间、阶段时间总和、输入哈希、源码快照和文件清单；以 `status.json`、`validation.json`、`delivery_validation.json` 的完成/通过状态为准。启动前评测相关测试 44 项通过。`20261014_160500_partial_cpu_fallback/` 为沙箱内 CUDA 初始化失败后中断的运行，不纳入正式比较。

本次已完成：每方法 **24,594 帧**、共 **98,376 次 update**、**16 个完整标注视频**，两份 validation 均通过。四数据集等权平均 HOTA 为 BoT-SORT **35.09**、MSDC-ELT_v3 **57.46**、ByteTrack **19.86**、OC-SORT **27.94**；总体聚合 HOTA 分别为 **20.02 / 49.13 / 11.50 / 16.04**。MSDC 总 IDSW 为 39、FP 为 1,100，但 FN 为 2,005，多于 BoT-SORT 的 1,303；ByteTrack 的 72 FP 同时伴随 10,075 FN。含本次检测的 Overall FPS 分别为 **7.16 / 7.47 / 7.47 / 7.51**，纯 Tracker FPS 为 **170.64 / 2767.54 / 2108.31 / 1254.51**。这些是同设备单次离线多阶段计时，检测成本按每方法计入一次，不等同于在线单流水线速度。

完整分析、逐数据集 17 指标和两类速度见该目录 `REPORT.md`；`summary/key_metrics_by_dataset.csv`、`summary/per_dataset_metrics.csv`、`summary/performance_macro_mean.csv` 分别提供重点逐数据集结果、完整逐数据集结果和等权平均。`summary/metric_definitions.csv` 明确指标方向和单位，`summary/artifacts.csv` 及 `files.csv` 索引全部原始产物。报告记录四数据集 GT/检测几何重合、短视频船类检测覆盖不足及现有 Frag 实现的解释限制。未更改任何 tracker 内部参数或评测算法代码。

## 2026-10-30 类别一致的 MOT 评测

当前 `run_motchallenge_eval()` 默认要求预测类别与 GT 类别相同才允许匹配。模型及跟踪内部索引按 `Config.CLASSES` 为 0=USV、1=fishship、2=UAV；预测 MOT 和 GT 的第 8 列都使用 1=USV、2=fishship、3=UAV。导出器从实际预测的类名／0 基 class_id 转换，缺失、未知或相互冲突的类别字段会报错。

新预测格式为 `frame,id,x,y,w,h,confidence,class_id,-1,-1`。`class_aware_mot.py` 在 TrackEval 分配前将不同类别配对的 IoU 相似度置零，HOTA、CLEAR（含 MOTA／IDSW／FP／FN）和 Identity（含 IDF1）共用这一限制；错类框仍保留在预测与 GT 数量中，未匹配时分别计 FP 和 FN。轨迹 ID 不按类别拆分；同一 ID 在某帧切错类别，该帧不能匹配。

GT 存在 labels.txt 时，其顺序必须与 Config.CLASSES 完全相同；GT 复制入口同时验证显式发现的类别表，并在评测副本中保存统一 labels.txt。无类别的旧预测（第 8 列为 -1 或缺列）会被拒绝，需要用新导出器重新生成。使用新的 run／eval 目录，避免与旧类别合并结果混用。

每次成功评价保存 `evaluation_protocol.json`，记录模型索引、MOT 类别编号及 same_class_and_iou 匹配规则。输出 CSV 仍使用兼容既有汇总脚本的 `pedestrian` 槽名，实际含义是全部模型类别在类别一致条件下的总体指标，不是行人指标。`render_mot_video.py` 优先显示 MOT 中的实际预测类别；已带类别的结果不会被检测缓存重新赋类。`class_source="mot"` 可强制要求结果自带类别。

`unified_dataset_report.py` 的检测覆盖要求缓存框与 GT 同类且 IoU>=0.5，仍是按 GT 框的覆盖诊断，不是一对一检测召回。`msdc_dataset_benchmark.py` 的 per-GT 和 stage coverage 也要求同类，沿用原宽松 IoU／中心距离阈值作为诊断，不等于正式指标的一对一 TP。

本次只运行人工数据和单元测试，未启动九序列正式实验。关键验证包括三类完美匹配、同框错类计 FP／FN、重叠异类目标在分配前门控、逐帧切类、全局身份不按类拆分、空输入和旧无类别结果拒绝。


## 九序列固定实验与完整统计（2026-11-01）

可见光实验固定每方法一套参数、九份共享检测缓存；108核心＋72单参数结果，共180个配置—序列结果。Full和两条曲线中心值复用。

`PAPER_EXPERIMENT_SPEC`指向配置JSON，导入跟踪器前实际加载Full、检测器与基线固定值，回放再应用指定变体的差异。新增中心距准入、候选确认模式、候选专用θ、独立Lost邻域新生抑制四字段，16个paper变体注册在`run_msdc_ablation.py`。计数式−E保留四命中和当前命中，取消候选分数门限、分数剪枝及高分历史，保留共享分数递推。实验模式只初始化可见光模型。

完整日志位于`trackers/<method>/paper_logs/<sequence>/`，五类JSONL每类每帧写`{seq,method,frame,records}`，包含空帧、删除前状态和NMS过滤，不受原摘要上限影响。日志成本归write_export并从tracker扣除。基线实际构造参数保存于`paper_metadata/<sequence>.json`。

统计入口为`tools/evaluation/export_paper_statistics.py --run-root RUN --spec SPEC --phase core|all`，使用WSL的`/home/hyj/anaconda3/bin/conda run --no-capture-output -n ship_detect python -B`执行。输出位于`RUN/paper_statistics`，含标准与宏平均、主表、消融、S0、交互、确认／恢复、额外ID、重复输出、bootstrap和参数曲线。GT级首次公开与首次正确公开都要求当帧同类CLEAR支持；R中的b(p)仍为公开ID全序列首次实际输出。H40仅限制窗口成功和惩罚延迟，绝对首次时刻保留全生命期值；GT缺段与截尾单列。

`tools/experiments/paper_queue.py`生成固定21作业manifest，`paper_queue_manager.py`为ARIS experiment-queue的本机WSL适配。串行运行，前置作业全部成功且产物验收完成才推进；退出码0和成功marker共同判定完成。OOM最多3次总尝试，等待120秒；其他错误修复后最多一次定向重试。失败、blocked及attempt日志持久化；主比较只执行一次。

正式全序列可视化宽度最高1920像素，所有方法统一设置；原始视频、检测、GT和MOT指标坐标保持原分辨率，缩放只计入render。render timing记录原始／输出尺寸及scale。论文案例保留原尺寸。

案例入口为`export_paper_cases.py --selection RUN/paper_statistics/case_selection.csv --output-dir RUN/paper_statistics/cases`，从完整MOT裁剪40帧，保留原帧号、类别及ID，不重新跟踪。

代码测试统一使用WSL ship_detect。人工数据测试和独立最小视频sanity不纳入九序列论文结果；正式结果以本次RUN、队列状态和统计manifest为准。

### MSDC缓存回放执行优化

本执行副本保留`replay_tracker_from_cache(frame_source="decode")`默认行为。核心消融与参数扫描的队列包装显式传入`--frame-source cache_only`，只接受固定17种MSDC配置、显式正整数N和恰好按序覆盖1..N的缓存；动态配置及三个基线均被拒绝。原视频元数据读取自已完成main的`mot_gt/<sequence>/seqinfo.ini`。缓存模式不打开或解码输入视频，以非空1×1数组满足现有MSDC输入检查，每个缓存帧仍调用同一状态更新、MOT格式化和五类日志；空检测帧照常推进。渲染继续读取原视频。

回放计时新增`frame_source`和`pixel_frames_decoded`，缓存模式后者及自身`read_decode`均为0。统计计时行追加这两字段、`replay_read_decode_s`及`speed_comparison_scope`。主比较保持解码默认路径和原速度计算；消融计时标记为`ablation_execution`。算法参数、固定180结果矩阵和指标／事件计算均保持原值。


### GT候选修订与固定轨迹重评

使用`tools/experiments/revise_gt_and_reevaluate.py`分两步处理经核验的GT修订：

```bash
conda run --no-capture-output -n ship_detect python -B tools/experiments/revise_gt_and_reevaluate.py prepare --run-root RUN --changes changes.csv --candidate-dir CANDIDATE
conda run --no-capture-output -n ship_detect python -B tools/experiments/revise_gt_and_reevaluate.py promote --run-root RUN --candidate-dir CANDIDATE
```

候选目录放在RUN之外。CSV共同字段为`sequence,frame,gt_id,action,reason,evidence_path`；`drop/replace`通过`old_x,old_y,old_w,old_h`精确定位原框，`replace/add`显式填写`new_x,new_y,new_w,new_h`。`add`的原框字段留空且必须提供证据路径；已有ID继承原GT唯一类别，新ID须显式提供`class_id`（1/2/3）。候选manifest记录原行ID/类别保持不变，以及各序列新增、消失的ID集合。

`prepare`只生成原副本、修订副本与逐行变更记录。`promote`要求main退出码0、原验收通过、scheduler明确held且实际为T/t、后续作业均pending；作废候选会被拒绝。工具归档旧GT和报告，只更新RUN内两份GT副本，并对原四方法、九序列的现有MOT统一重评；不重新检测、跟踪或渲染。原数据集、轨迹、缓存、视频和原计时保留，失败自动恢复旧GT与报告。每个序列须保留至少一个有效GT框。工具不会恢复scheduler。

合成回归入口：`conda run --no-capture-output -n ship_detect python -B -m pytest -q -p no:cacheprovider test/test_revise_gt_and_reevaluate.py`。


### 显式H264编码器

后续核心消融与参数扫描使用`paper_queue.py run ... --ffmpeg-executable ENCODER`显式选择H264；通用默认仍为MP4V，论文案例保持原设置。当前冻结编码器为`results/paper_visible9_20261101_141936/queue/encoder_resources/ffmpeg-7.0.2-static`，来自隔离安装的`imageio-ffmpeg==0.6.0`所附Linux FFmpeg 7.0.2-static，SHA256为`e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99`。二进制保留在RUN资源目录，不纳入Git源码。

编码采用`libx264 / CRF20 / veryfast / yuv420p`，输入标注仍来自原MOT。编码器来源、摘要与编码耗时写入render timing；不同编码器的render耗时和整体FPS不作为算法速度比较，tracker耗时按原口径单独报告。已审阅的5个源码／测试文件与冻结H264快照逐字节同步；42项独立测试的原始范围和结果见该RUN的`queue/encoding_review/fresh_review.json`。
