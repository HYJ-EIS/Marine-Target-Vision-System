# Marine Target Vision System

海上目标检测与多目标跟踪项目。当前主链路由 ONNX Runtime 检测器、图片检测 HTTP 服务、视频检测/跟踪入口和 MS-DC-ELT 正式评测工具组成。

## 当前状态

- 检测类别、模型路径、置信度阈值、NMS 阈值、图片输入输出目录、视频默认输入输出和 MS-DC-ELT 参数集中在 `target_module/image_detect_module/config.py`。
- 跟踪器选择、正式 MS-DC 变体名、论文方法展示名和速度统计字段集中在 `target_module/image_detect_module/constants.py`。
- 当前正式 MS-DC-ELT 主变体名为 `msdc_v3`。
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
ocsort
botsort
dist_tracker
official_ocsort
official_botsort
msdc_elt
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

`msdc_v3` 默认启用 low candidate、direct reacquire、low inheritance、removed-ID recovery、output NMS 和 low observation Top-K 预算。`pending_recovery_candidate`、`removed_recovery_off`、`reacquire_every_frame_low_score` 等是消融/对比变体，不是默认主变体。

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
