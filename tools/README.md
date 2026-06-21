# Tools

本目录只放离线工具、人工验证脚本和实验编排脚本。正式服务入口仍在仓库根目录：

- `image_main.py`：图片检测 HTTP 服务
- `video_main.py`：视频检测与跟踪入口

所有 Python 命令默认使用项目 Conda 环境：

```powershell
conda run -n ship_detect python <script> ...
```

## dataset

离线抽帧、数据集质检、YOLO 标签可视化和代表帧过滤工具已迁移到独立项目：

```text
/home/hyj/Anti_Drone_Project/Marine-Frame-Extraction
```

本仓库的 `tools/dataset/` 只保留迁移说明，不再维护可执行 Python 入口。

## validation

`tools/validation/` 放人工验证和调试脚本。它们通常依赖本地视频、模型文件或输出目录，适合手动运行，不作为 pytest 单元测试。

- `video_test_tracking.py`：本地视频检测与跟踪验证
- `tracker_effect_test.py`：多 tracker 同帧输入效果对比
- `video_detect_only.py`：本地视频纯检测可视化与统计
- `msdc_low_conf_debug.py`：MS-DC-ELT 低阈值检测诊断
- `msdc_motion_seed_debug.py`：MS-DC-ELT 运动种子诊断

## experiments

`tools/experiments/` 放实验编排和对比脚本，产物通常写入 `results/`。

- `run_msdc_ablation.py`：MS-DC-ELT 消融变体定义与命令编排

## evaluation

`tools/evaluation/` 放正式 MOTChallenge 风格评测工具。正式 IDF1/HOTA/MOTA 必须使用带真实跨帧 identity 的 GT，不从 tracker 输出自行估计。

- `export_mot_results.py`：把项目 tracker 输出导出为 `<tracker>/data/<seq>.txt`
- `motchallenge_eval.py`：调用 vendored TrackEval 计算 HOTA、MOTA、IDF1
- `msdc_dataset_benchmark.py`：解析双单序列标注数据集，编排 MOT 导出、TrackEval、诊断和可选可视化
- `msdc_speed_benchmark.py`：在不渲染、不发 MQ、不跑 TrackEval 的条件下，对 OC-SORT、BoT-SORT、MS-DC-ELT 做固定帧数速度统计
- `msdc_experiment_summary.py`：把 TrackEval summary、速度结果和输出路径汇总为论文实验 CSV 与 Markdown 报告
- `run_msdc_paper_experiments.py`：编排 MS-DC-ELT 论文第一阶段主结果、消融、速度与报告生成；默认只打印计划命令，正式运行需显式传 `--run-formal`

## tests

自动化单元测试保留在仓库根目录的 `test/` 下。当前主要是：

- `test/test_tracking_proxy_metrics_removed.py`

运行：

```powershell
conda run -n ship_detect pytest -q
```

`tools/validation/` 和 `tools/experiments/` 中的脚本即使文件名里带 `test`，也按人工验证或实验脚本管理。
