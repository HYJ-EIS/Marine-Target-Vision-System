# Marine Target Vision System

海上目标检测与多目标跟踪项目。当前主链路基于 ONNX Runtime 检测器，图片检测统一入口为 `target_module/image_detect_module/target_detection.py`，视频检测与跟踪入口为 `video_main.py`，正式评测工具集中在 `tools/evaluation/`。

## 当前状态

- 默认检测模型、阈值、输入输出路径集中在 `target_module/image_detect_module/config.py`。
- 跟踪器选择、正式 MS-DC 变体名、方法展示名、速度字段集中在 `target_module/image_detect_module/constants.py`。
- 视频运行与 MOT 导出的逐帧 tracking 更新逻辑共用 `target_module/image_detect_module/utils/tracking_update.py`。
- 当前正式 MS-DC-ELT 变体为 `v3_candidate_topk_no_roi_no_motion`。
- 已删除早期一次性验证脚本；`tools/validation/` 当前仅保留包入口。

## 环境

所有 Python 和 pytest 命令都应通过 `ship_detect` 环境执行：

```bash
conda run -n ship_detect python <script.py>
conda run -n ship_detect pytest test -q
```

依赖文件：

```text
files/requirements.txt
```

运行入口还直接依赖 `Flask`、`Flask-CORS`、`onnxruntime`。执行前不要假设 `files/requirements.txt` 已完整覆盖当前本地运行环境。

## 入口脚本

| 场景 | 入口 | 说明 |
| --- | --- | --- |
| 图片检测 HTTP 服务 | `image_main.py` | 正式图片检测接口 |
| 图片检测调试副本 | `image_main copy.py` | 与 `image_main.py` 请求处理逻辑保持同步，可保留 `TEST_MODE` |
| 视频检测与跟踪 | `video_main.py` | 支持 RTSP 和本地视频 |
| 统一检测模块 | `target_module/image_detect_module/target_detection.py` | `detect_targets(image_input, output_dir=None, enable_tracking=False)` |

注意：`detect_targets()` 当前实际只支持图片路径字符串，不要把内存帧或 `np.ndarray` 直接传入。

## 快速运行

启动图片检测服务：

```bash
conda run -n ship_detect python image_main.py
```

常用接口：

- `POST /api/v1/detect/image`
- `GET /api/v1/results/<result_id>`

启动本地视频检测与跟踪：

```bash
conda run -n ship_detect python video_main.py \
  --input /path/to/video.mp4 \
  --tracker msdc_elt \
  --output outputs/demo/annotated.mp4 \
  --no-display
```

`video_main.py` 参数：

| 参数 | 说明 |
| --- | --- |
| `--input` | 本地视频路径；留空时使用 `Config.VIDEO_RTSP_INPUT` |
| `--tracker` | 跟踪器；留空时使用 `Config.TRACKER_TYPE` |
| `--output` | 输出视频路径；留空时使用配置默认值或 MS-DC 独立 run 目录 |
| `--no-display` | 无 GUI 环境下不弹窗 |

RTSP 默认输入、RTSP 默认输出、本地默认输出统一在 `Config.VIDEO_RTSP_INPUT`、`Config.VIDEO_RTSP_OUTPUT`、`Config.VIDEO_OUTPUT_PATH` 修改，不要在 `video_main.py` 重新写死。

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

基线跟踪统一走 `MultiObjectTracker.update()`；`msdc_elt` 走 `MSDCLifecycleTracker.update()`。这层分发逻辑在：

```text
target_module/image_detect_module/utils/tracking_update.py
```

MS-DC-ELT 主要实现文件：

| 文件 | 作用 |
| --- | --- |
| `utils/lifecycle_tracker.py` | 生命周期状态机和最终输出 |
| `utils/msdc_detection.py` | 单次低阈值检测与高阈值框筛选 |
| `utils/evidence_state.py` | 证据累计与候选确认 |
| `utils/msdc_types.py` | MS-DC 内部数据结构 |

正式 v3 变体名为 `v3_candidate_topk_no_roi_no_motion`。该变体用于当前论文主实验标签，MS-DC 路径只运行一次低阈值检测，并从低阈值结果中筛出高阈值框；低分候选 Top-K 和轨迹近邻预算用于控制候选规模。旧 ROI、motion、template 辅助模块已删除。

## 评测工具

### MOT 与正式评测

| 脚本 | 用途 |
| --- | --- |
| `tools/evaluation/export_mot_results.py` | 导出 MOTChallenge tracker result txt |
| `tools/evaluation/motchallenge_eval.py` | 运行 MOTChallenge/TrackEval 指标 |
| `tools/evaluation/msdc_dataset_benchmark.py` | 数据集级 benchmark 编排 |
| `tools/evaluation/msdc_speed_benchmark.py` | 无渲染速度与分阶段耗时评测 |
| `tools/evaluation/msdc_diagnostic_metrics.py` | 汇总低分候选、继承、重捕获和碎片化诊断指标 |
| `tools/evaluation/msdc_sensitivity_matrix.py` | 写出 MS-DC-ELT v3 超参数敏感性矩阵 |
| `tools/evaluation/msdc_slice_eval.py` | 从 per-GT diagnostics 生成短漏检 slice manifest |
| `tools/evaluation/msdc_experiment_summary.py` | 汇总主实验、消融和速度结果 |
| `tools/evaluation/run_msdc_paper_experiments.py` | 论文 phase-1 实验总入口 |
| `tools/evaluation/validate_msdc_formal_run.py` | 检查正式 run 产物完整性 |

### 可视化与回放

| 脚本 | 用途 |
| --- | --- |
| `tools/evaluation/render_mot_video.py` | 基于 MOT txt 渲染标注视频 |
| `tools/evaluation/render_msdc_diagnostics_video.py` | 基于 MS-DC 诊断 JSONL 渲染视频 |
| `tools/evaluation/detection_replay_benchmark.py` | 回放缓存检测框进行 tracker 对比 |

### 实验

| 脚本 | 用途 |
| --- | --- |
| `tools/experiments/run_msdc_ablation.py` | 单视频 MS-DC 消融命令生成或执行 |

## 正式评测

推荐从总入口生成计划或执行：

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py
```

执行 smoke：

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --smoke
```

执行正式 run：

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --run-formal
```

正式 run 会依次规划/执行 main、ablation、slice、speed、sensitivity 和 summary 阶段；`slice/` 阶段产物为 `slice/slice_manifest.csv`，`sensitivity/` 阶段产物为 `sensitivity/sensitivity_full/sensitivity_matrix.csv`，`latest_run.json` 会记录 `slice_manifest_csv` 和 `sensitivity_matrix_csv`。

常用默认数据集根目录：

```text
/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集
/home/hyj/Anti_Drone_Project/USV_MOT标注数据集
```

正式测试完成时必须同时交付：

- 带标注可视化视频，至少包含目标框、类别和 track ID，并写入本次 run 独立 `visualizations/` 目录。
- MOT 指标汇总，至少包含 `MOTA`、`IDF1`、`IDSW`、`FN`、`FP`；若已有 `HOTA`、`IDTP`、`IDFP`、`IDFN` 也一并汇报。
- 速度统计，至少包含总处理帧数、总耗时、平均单帧耗时和平均 FPS。
- 分阶段耗时，至少拆分为读取/解码、低阈值检测、高阈值框筛选、lifecycle tracker、可视化渲染、结果写盘/导出；不适用阶段写 `0` 或 `N/A`。
- 输出路径清单，包括 MOT txt、TrackEval summary、诊断 JSONL/CSV、可视化 MP4 和速度/耗时统计文件。
- MS-DC 正式和 replay benchmark 会在 per-GT diagnostics 可用时额外写出 `diagnostics/msdc_diagnostic_summary.csv`，汇总低分候选确认、继承、重捕获、碎片化和轨迹断裂诊断。

正式测试结果必须写入新的时间戳 run 目录。临时 smoke、失败中断、partial 输出要标清楚或清理，避免和正式结果混在一起。

## 输出目录

常用输出位置：

| 路径 | 内容 |
| --- | --- |
| `outputs/` | 服务或视频入口生成的运行输出 |
| `outputs/msdc_elt/<run_id>/` | `video_main.py --tracker msdc_elt` 的独立视频输出和诊断 |
| `results/` | benchmark、MOT、summary、速度评测产物 |
| `results/msdc_paper_phase1/` | phase-1 论文实验默认输出根目录 |

生成正式结果时不要覆盖历史 run。

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

视频端到端、RTSP、RabbitMQ、模型文件和本地数据依赖较重；运行完整流程前先确认对应外部资源存在。

## 目录速览

```text
.
├── image_main.py
├── image_main copy.py
├── video_main.py
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

## 最近维护

- 2026-06-24：`msdc_slice_eval.py` 在 diagnostics 根目录缺失或没有 per-GT diagnostics CSV 时会失败退出，避免正式 slice 阶段静默生成空清单。
- 2026-06-22：集中 tracker choices、formal MS-DC 变体名、method labels、speed fields。
- 2026-06-22：抽出 tracking update helper，统一 `video_main.py` 与 MOT 导出路径的 tracking 分发逻辑。
- 2026-06-22：删除默认关闭且评测退化的 MS-DC motion seed、ROI redetect、template lock 辅助模块。
- 2026-06-22：删除 MS-DC 高阈值和低阈值分别检测的 two-pass 路径，只保留单次低阈值检测后筛高阈值框。
- 2026-06-22：删除离线直跑检测/跟踪/渲染的 `render_tracking_video.py`，正式可视化保留 MOT txt 驱动的 `render_mot_video.py`。
