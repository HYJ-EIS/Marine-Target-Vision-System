# 抽帧功能迁移说明

批量抽帧、数据集质检、YOLO 标签可视化和代表帧过滤工具已从本检测跟踪仓库剥离，迁移到独立项目：

```text
/home/hyj/Anti_Drone_Project/Marine-Frame-Extraction
```

新项目不 import 本仓库代码；需要复用的检测封装、轻量 tracker、工具函数和 ONNX 模型都已复制到新项目内。

常用入口：

```bash
cd /home/hyj/Anti_Drone_Project/Marine-Frame-Extraction
conda run -n ship_detect python -m frame_extraction.dataset.extract_tracking_frames --help
conda run -n ship_detect python -m frame_extraction.dataset.video_dataset_classify --help
```
