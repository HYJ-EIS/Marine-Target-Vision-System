# Dataset Tools

本目录集中放置离线数据集整理、抽帧、质检和标注可视化脚本。它们不是 HTTP 服务入口；正式图片/视频服务仍分别是仓库根目录下的 `image_main.py` 和 `video_main.py`。

所有命令建议在仓库根目录执行，并使用项目约定的 Conda 环境：

```powershell
conda run -n ship_detect python tools/dataset/<script>.py ...
```

## extract_tracking_frames.py

用途：从 `_V` / `_T` 原始视频中抽取有代表性的训练帧，并同步导出 YOLO 标签和清单。

输入：`--input-root` 指向原始视频目录；默认值来自 `Config.EXTERNAL_FRAMES_INPUT_ROOT`。

输出：默认写入 `Config.EXTERNAL_FRAMES_OUTPUT_ROOT`，包含 `images/`、`labels/` 和 `manifests/frames_manifest.csv`。

主要逻辑：递归扫描 `_V` / `_T` 视频，按 RGB/IR 分组，调用 `ImageProcessor` 检测、`MultiObjectTracker` 跟踪，再按新目标、运动变化、姿态变化、尺度变化、目标丢失/恢复、双模态对齐和 pHash 去重保留代表帧。

示例：

```powershell
conda run -n ship_detect python tools/dataset/extract_tracking_frames.py
conda run -n ship_detect python tools/dataset/extract_tracking_frames.py --input-root "D:\Desktop\烟台项目数据\原始数据集\视频" --output-root "D:\Desktop\烟台项目数据\原始数据集\external_frames" --resume
```

## video_dataset_classify.py

用途：对原始视频目录做离线批量分析，判断数据质量、场景覆盖、目标分布和 RGB/IR 配对缺口。

输入：`--input-root` 指向原始视频目录；默认值来自 `Config.DATASET_INPUT_ROOT`。

输出：默认写入 `results/video_dataset_analysis/<run_id>/`，包含：

- `precheck_by_date.csv`
- `window_results.csv`
- `video_results.csv`
- `report_all_units.csv`
- `report_paired_only.csv`
- `gap_report.csv`
- `run_summary.json`

示例：

```powershell
conda run -n ship_detect python tools/dataset/video_dataset_classify.py --input-root "D:\Desktop\烟台项目数据\原始数据集\视频" --resume --workers 2
```
