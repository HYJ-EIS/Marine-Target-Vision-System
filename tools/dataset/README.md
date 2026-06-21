# Dataset Tools

离线抽帧、数据集质检、YOLO 标签可视化和代表帧过滤工具已迁移到独立项目：

```text
/home/hyj/Anti_Drone_Project/Marine-Frame-Extraction
```

本仓库只保留检测、跟踪、评测和人工验证链路，不再维护 `tools/dataset/*.py` 入口。需要运行批量抽帧或数据集分析时，请在新项目中执行：

```bash
cd /home/hyj/Anti_Drone_Project/Marine-Frame-Extraction
conda run -n ship_detect python -m frame_extraction.dataset.extract_tracking_frames --help
conda run -n ship_detect python -m frame_extraction.dataset.video_dataset_classify --help
```
