# Tools

本目录只放离线工具、人工验证脚本和实验编排脚本。正式服务入口仍在仓库根目录：

- `image_main.py`：图片检测 HTTP 服务
- `video_main.py`：视频检测与跟踪入口

所有 Python 命令默认使用项目 Conda 环境：

```powershell
conda run -n ship_detect python <script> ...
```

## dataset

旧 `tools/dataset/` 离线数据集整理脚本当前不在仓库中。新增数据整理工具时再恢复该目录，并在脚本内维护对应默认参数。

## validation

`tools/validation/` 放人工验证和调试脚本。当前过时的手工验证入口已移除；新增脚本应优先放入正式 `tools/evaluation/` 链路或配套 pytest。

## experiments

`tools/experiments/` 放实验编排和对比脚本，产物通常写入 `results/`。

## evaluation

`tools/evaluation/` 放正式 MOTChallenge 风格评测工具。正式 IDF1/HOTA/MOTA 必须使用带真实跨帧 identity 的 GT，不从 tracker 输出自行估计。

- `export_mot_results.py`：把项目 tracker 输出导出为 `<tracker>/data/<seq>.txt`
- `motchallenge_eval.py`：调用 vendored TrackEval 计算 HOTA、MOTA、IDF1
- `msdc_dataset_benchmark.py`：解析双单序列标注数据集，编排 MOT 导出、TrackEval、诊断和可选可视化
- `detection_replay_benchmark.py`：先缓存 high/low detector 输出，再回放给 baseline 与 MS-DC-ELT，保证主实验和消融使用同一 detector 输入；replay 可视化默认从缓存取类别并可跳过已完整产物以恢复中断 run。
- `msdc_speed_benchmark.py`：在不渲染、不发 MQ、不跑 TrackEval 的条件下，对 OC-SORT、BoT-SORT、MS-DC-ELT 做固定帧数速度统计
- `msdc_diagnostic_metrics.py`：汇总 low_candidate、inherit、reacquire、fragmentation 和 track break 诊断指标。
- `msdc_experiment_summary.py`：把 TrackEval summary、速度、敏感性、slice、诊断结果和输出路径汇总为论文实验 CSV 与 Markdown 报告；缺失的可选源文件只写 `status=missing`/`failure=<path>`，不补造指标。
- `msdc_slice_eval.py`：从 per-GT diagnostics 生成短时漏检/低置信片段切片清单。
- `msdc_sensitivity_matrix.py`：输出 v3 超参数敏感性扫描矩阵。
- `validate_msdc_formal_run.py`：校验正式 run 的 MOT、TrackEval、诊断、slice、速度、敏感性和 summary/manifest 产物
- `run_msdc_paper_experiments.py`：编排 MS-DC-ELT 论文第一阶段 main、ablation、slice、speed、sensitivity 与 summary/validation 流程；默认只打印计划命令，正式运行需显式传 `--run-formal`

formal v3 完成标准：`summary/` 必须物化 `main_results.csv`、`ablation_results.csv`、`speed_results.csv`、`sensitivity_results.csv`、`slice_metrics.csv`、`diagnostic_results.csv` 和 `path_manifest.csv`。`ablation_results.csv` 必须包含 `variant_key`，run 内必须存在 `**/effective_config/*.json`；`sensitivity_results.csv` 来自 `sensitivity/sensitivity_metrics/eval/motchallenge_summary.csv` 且保留每个 sensitivity point 的真实指标；`slice_metrics.csv` 来自 `slice/slice_metrics/slice_metrics.csv`；`diagnostic_results.csv` 来自 `msdc_diagnostic_summary.csv`，并包含 low_candidate、reacquire、inherit 的 denominators/opportunities 字段及对应成功/错误/歧义计数。

## tests

自动化单元测试保留在仓库根目录的 `test/` 下，当前主要覆盖正式评测、MS-DC 变体常量和 benchmark 编排逻辑。

运行：

```powershell
conda run -n ship_detect pytest -q
```

`tools/validation/` 和 `tools/experiments/` 中的脚本即使文件名里带 `test`，也按人工验证或实验脚本管理。
