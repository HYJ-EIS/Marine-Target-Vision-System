# AGENTS.md

## Environment

- 所有 `python` / `pytest` 命令使用 `conda run -n ship_detect ...`
- 依赖文件：`files/requirements.txt`
- 当前入口脚本还直接依赖 `Flask`、`Flask-CORS`、`onnxruntime`；执行前不要假设 `files/requirements.txt` 已完整覆盖运行时依赖

## Entry Points

- `image_main.py`：图片检测 HTTP 服务
- `image_main copy.py`：调试副本；除手动测试分支外，和 `image_main.py` 的请求处理逻辑应保持同步
- `video_main.py`：视频检测与跟踪入口
- 统一检测入口：`target_module/image_detect_module/target_detection.py` → `detect_targets(image_input, output_dir=None, enable_tracking=False)`

## Execution Rules

1. 涉及磁盘图片路径读写时，不要直接使用 `cv2.imread` / `cv2.imwrite`；使用 `cv_utils.imread_unicode()` / `imwrite_unicode()`。
2. 不要在调用处重复写死检测类别、模型路径、置信度阈值、NMS 阈值或图片输入输出目录；以 `target_module/image_detect_module/config.py` 的 `Config` 为准。
3. 修改 `image_main.py` 的请求解析、检测调用、结果图保存或返回结构时，同步检查 `image_main copy.py` 的对应逻辑；`image_main copy.py` 允许额外保留 `TEST_MODE` 调试分支。
4. 当前 `detect_targets()` 实际只支持图片路径字符串；不要把内存帧或 `np.ndarray` 直接传给它。
5. 不要假设 HTTP 成功响应等于检测结果；当前 `image_main.py` / `image_main copy.py` 中的 `APIResponse.success()` 未返回标准成功体。若修改图片 API 返回契约，需同步更新 `README.md`。
6. `video_main.py` 和端到端测试依赖外部服务或本地数据时，先确认 RTSP、RabbitMQ、模型文件和输入样本可用，再运行完整流程。
7. 修改视频默认输入源、RTSP 输出地址或默认输出文件时，统一改 `target_module/image_detect_module/config.py` 中的 `Config.VIDEO_RTSP_INPUT`、`Config.VIDEO_RTSP_OUTPUT`、`Config.VIDEO_OUTPUT_PATH`；不要在 `video_main.py` 重新写死。
8. 每次完成代码修改后都要更新文档README.md。

## Formal Evaluation Reporting

- 每次“正式测试 / 正式评测 / full run / benchmark”完成后，必须同时交付以下结果；缺任一项不能宣称正式测试完成：
  1. 带标注的可视化视频，至少包含目标框、类别和 track ID；视频必须写入本次 run 的独立 `visualizations/` 目录，不覆盖历史结果。
  2. MOT 指标汇总，至少包含 `MOTA`、`IDF1`、`IDSW`、`FN`、`FP`；若已有 `HOTA`、`IDTP`、`IDFP`、`IDFN` 也一并汇报。
  3. 速度统计，至少包含总处理帧数、总耗时、平均单帧耗时和平均 FPS。
  4. 分阶段耗时，至少拆分为视频读取/解码、检测、低阈值检测或 ROI 重检、motion/template/lifecycle tracker、可视化渲染、结果写盘/导出；不适用的阶段写 `0` 或 `N/A`，不要省略。
  5. 输出路径清单，包括 MOT txt、TrackEval summary、诊断 JSONL/CSV、可视化 MP4 和速度/耗时统计文件。
- 如果当前评测脚本没有生成速度或分阶段耗时，先补充计时输出或额外日志，再运行正式测试；不要只凭终端主观估算。
- 正式测试结果必须写入新的时间戳 run 目录；临时 smoke、失败中断、partial 输出要标清楚或清理，避免和正式结果混在一起。

## Reference

- 需要接口细节、目录说明或变更记录时，查 `README.md`
