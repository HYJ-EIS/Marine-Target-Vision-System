# Tools

本目录只放离线工具、人工验证脚本和实验编排脚本。正式服务入口仍在仓库根目录：

- `image_main.py`：图片检测 HTTP 服务
- `video_main.py`：视频检测与跟踪入口

所有 Python 命令默认使用项目 Conda 环境：

```powershell
conda run -n ship_detect python <script> ...
```

## dataset

`tools/dataset/` 放数据集整理工具，面向“生成、清理、检查训练数据”的离线流程。

- `ensure_classes_txt.py`：给 YOLO 标签目录补齐 `classes.txt`
- `extract_tracking_frames.py`：从 `_V` / `_T` 视频抽代表帧并导出 YOLO 标签
- `video_dataset_classify.py`：按时间窗分析原始视频数据质量、场景和目标分布
- `visualize_yolo_labels.py`：将 YOLO 标签画回图片供人工检查
- `filter_existing_frames.py`：对已抽帧 LabelMe 数据做代表帧压缩

详细用法见 `tools/dataset/README.md`。

## validation

`tools/validation/` 放人工验证和调试脚本。它们通常依赖本地视频、模型文件或输出目录，适合手动运行，不作为 pytest 单元测试。

- `video_test_tracking.py`：本地视频检测与跟踪验证
- `test_image_tracking_api.py`：`detect_targets(..., enable_tracking=True)` 行为验证
- `debug_bbox_offset.py`：检测框偏移诊断

## experiments

`tools/experiments/` 放实验编排和对比脚本，产物通常写入 `results/`。

- `run_all_tests.py`：固定测试视频批量跑追踪验证
- `run_compare_ir_models.py`：两版红外 ONNX 模型对比

## evaluation

`tools/evaluation/` 放正式 MOTChallenge 风格评测工具。正式 IDF1/HOTA/MOTA 必须使用带真实跨帧 identity 的 GT，不从 tracker 输出自行估计。

- `export_mot_results.py`：把项目 tracker 输出导出为 `<tracker>/data/<seq>.txt`
- `motchallenge_eval.py`：调用 vendored TrackEval 计算 HOTA、MOTA、IDF1

## tests

自动化单元测试保留在仓库根目录的 `test/` 下。当前主要是：

- `test/test_extract_tracking_frames.py`
- `test/test_video_dataset_classify.py`

运行：

```powershell
conda run -n ship_detect pytest -q
```

`tools/validation/` 和 `tools/experiments/` 中的脚本即使文件名里带 `test`，也按人工验证或实验脚本管理。
