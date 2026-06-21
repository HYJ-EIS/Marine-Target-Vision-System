目标：实现一个最小可验证的 RGB/IR 双模态检测支撑实验，只用 BoT-SORT 验证“红外支撑”是否能提升检测与跟踪效果。

ir_rgb_match相关工作目录：
/home/hyj/Anti_Drone_Project/IR_RGB_match

相关输入：
- 配置文件：
  /home/hyj/Anti_Drone_Project/IR_RGB_match/config/alignment_config.yaml

- 映射报告：
  /home/hyj/Anti_Drone_Project/IR_RGB_match/outputs/minimal_alignment/alignment_report.json

- 可见光视频目录：
  /home/hyj/Anti_Drone_Project/MOT_DJI_20250711140455_0002_W

- 红外视频目录：
  /home/hyj/Anti_Drone_Project/MOT_DJI_20250711140455_0002_T

当前逐帧仿射映射平均误差约 7.36 px。 本轮实验必须使用已有 RGB/IR 标定或 alignment_report 中的仿射映射结果，不允许做无标定近似估计。如果 calib/alignment_report 缺失、字段不完整、仿射矩阵不可用，必须直接报错退出，不要自动估计或静默降级。

请先阅读并理解以下文件，不要跳过：
- video_main.py
- target_module/image_detect_module/image_processor.py
- target_module/image_detect_module/utils/tracker.py
- tools/evaluation/export_mot_results.py
- tools/evaluation/motchallenge_eval.py

一、总体任务

新增脚本：

tools/evaluation/export_dual_modal_mot_results.py

功能：
- 对同一序列的 RGB 视频和 IR 视频逐帧读取。
- 使用已有 RGB/IR calib 或 alignment_report 中的仿射参数，将 IR 检测框映射到 RGB 坐标系。
- 只支持 tracker=botsort。
- 支持两种 fusion 模式：
  1. rgb_only
  2. rgb_ir_support
- 输出 MOTChallenge tracker result 格式：
  <output-root>/<run-name>/data/<seq-name>.txt

MOT 每行格式：
frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1

二、实验目的

本实验只验证一个问题：

在相同 BoT-SORT 跟踪器下，相比 RGB-only，高置信 RGB 检测 + IR 支撑低置信 RGB 检测，是否能改善 HOTA、IDF1、IDSW、FP、FN 和 FPS。

只做两组实验：

1. botsort_rgb
- fusion=rgb_only
- tracker=botsort
- 目的：BoT-SORT 的 RGB-only 主基线。

2. botsort_rgb_ir_support
- fusion=rgb_ir_support
- tracker=botsort
- 目的：验证 IR support 是否能帮助 BoT-SORT 处理低置信 RGB 与短时漏检。

不要实现 ByteTrack、OC-SORT、Dist-Tracker、official_botsort、official_ocsort 的对比。
不要声称验证了所有 tracker 的通用性。

三、读取配置与标定要求

脚本需要读取：
- /home/hyj/Anti_Drone_Project/IR_RGB_match/config/alignment_config.yaml
- --calib 指定的 alignment_report.json 或 calib json

要求：
- rgb_only 模式下可以不读取 IR 检测，不要求 IR 视频和 calib 必须参与实际融合。
- rgb_ir_support 模式下必须读取 IR 视频和 calib/alignment_report。
- rgb_ir_support 模式下如果 calib 缺失、仿射矩阵缺失、字段不完整、IR/RGB 同步信息不可用，直接 raise error 退出。
- 不允许做无标定估计。
- 不允许使用 MOT GT 标注、ID 对应关系参与推理阶段的检测融合或跟踪。
- MOT 标注路径和 ID 对应关系只能作为 alignment 阶段已有结果的来源背景，不能在 export_dual_modal_mot_results.py 中用于生成 tracker 输出。

四、新增脚本参数

tools/evaluation/export_dual_modal_mot_results.py 至少支持：

--visible-input
--infrared-input
--tracker {botsort}
--fusion {rgb_only,rgb_ir_support}
--seq-name
--run-name
--output-root
--calib
--config default /home/hyj/Anti_Drone_Project/IR_RGB_match/config/alignment_config.yaml
--rgb-high-conf default 0.50
--rgb-low-conf default 0.30
--ir-conf default 0.45
--alpha default 0.7
--max-frames default 0
--progress-interval default 100
--track-iou-thresh default 0.20
--track-dist-factor default 1.5
--ir-rgb-match-iou default 0.10
--ir-rgb-match-dist-factor default 2.0

说明：
- --tracker 只允许 botsort。
- --max-frames=0 表示全帧。
- --track-iou-thresh 和 --track-dist-factor 主要用于判断低置信 RGB 框是否有已有轨迹预测支持，不要求改动 BoT-SORT 内部匹配阈值。
- --ir-rgb-match-iou 和 --ir-rgb-match-dist-factor 用于判断映射后的 IR 框是否支持 RGB 低置信框。

五、检测规则

RGB 检测：
- 对 RGB frame 使用 ImageProcessor.process_frame(frame, "visible", conf_override=rgb_low_conf) 获取所有 RGB 候选。
- 按 confidence 切分：
  - RGB high：confidence >= rgb_high_conf
  - RGB low：rgb_low_conf <= confidence < rgb_high_conf
- RGB high 直接保留，allow_new_track=True。
- RGB low 进入 IR support / track support 判断。

IR 检测：
- 仅在 fusion=rgb_ir_support 时执行。
- 对 IR frame 使用 ImageProcessor.process_frame(frame, "infrared", conf_override=ir_conf)。
- 将 IR 检测框通过 calib/alignment_report 中的仿射矩阵映射到 RGB 坐标系。
- 映射后需要裁剪到 RGB 图像范围。
- 无效框直接丢弃。
- IR 检测不能直接输出到 tracker。
- IR-only 框不能创建新轨迹。
- IR 不能覆盖 RGB 类别。
- IR 只能支撑 RGB 低置信框，并增强 RGB confidence。

六、RGB-only 融合规则

fusion=rgb_only 时：
- 只使用 RGB high boxes。
- 不使用 RGB low boxes。
- 不使用 IR boxes。
- 所有送入 tracker 的检测 allow_new_track=True。
- 这是主基线。

七、RGB + IR support 融合规则

fusion=rgb_ir_support 时：

1. RGB high：
- 直接送入 tracker。
- allow_new_track=True。

2. RGB low + IR support：
- 如果 RGB low 与任意映射到 RGB 坐标系后的 IR box 匹配，则保留。
- allow_new_track=True。
- 不改变 RGB class。
- 使用 IR confidence 增强 RGB confidence：

conf_fused = 1 - (1 - conf_rgb) * (1 - alpha * conf_ir)

- conf_fused 需要限制在 [conf_rgb, 1.0] 范围内。
- 如果一个 RGB low 匹配多个 IR box，使用匹配质量最高的 IR box；优先 IoU 高，其次中心距离近。

3. RGB low + track support only：
- 如果 RGB low 没有 IR support，但有已有轨迹预测支持，则保留。
- allow_new_track=False。
- 该检测只能参与匹配已有轨迹，不能初始化新 KalmanBoxTracker。
- confidence 保持 RGB 原始 confidence，不做 IR 融合。

4. RGB low suppressed：
- 如果 RGB low 没有 IR support，也没有 track support，则不送入 tracker。

5. IR-only：
- 本轮实验不要启用 IR-only 补框。
- IR-only box 不送入 tracker。
- IR-only box 不允许创建新轨迹。

八、IR support 匹配判据

给 RGB low box 和映射后的 IR box 判断是否匹配：

满足以下任一条件即可认为有 IR support：
- IoU >= --ir-rgb-match-iou
- 中心距离 <= --ir-rgb-match-dist-factor * diag(rgb_low_box)

其中 diag(rgb_low_box) 为 RGB low box 的对角线长度。

如果多个 IR box 匹配同一个 RGB low box：
- 优先选择 IoU 最大的。
- 如果 IoU 都很小，则选择中心距离最小的。

九、track support 判据

对 RGB low box 判断是否有已有轨迹预测支持：

需要从当前 BoT-SORT/OCSortTracker 内部已有 trackers 获取预测框或 last_observation/get_state。
不要新建轨迹来判断 support。

满足以下任一条件即可认为有 track support：
- RGB low box 与已有轨迹预测框 IoU >= --track-iou-thresh
- 中心距离 <= --track-dist-factor * diag(track_box)

只考虑已经确认或至少 hits >= min_hits 的轨迹。
不要用刚初始化但不稳定的轨迹支撑低置信框。
如果无法安全获取已有轨迹，则保守返回 no track support。

十、tracker.py 最小改动

只修改 BoT-SORT/OC-SORT 共享的本地 OCSortTracker 通道，不影响：
- bytetrack
- dist_tracker
- official_ocsort
- official_botsort

需要改动：

1. MultiObjectTracker._update_ocsort()
- 从 detection dict 中读取 allow_new_track，默认 True。
- 在过滤无效 box 时同步维护 allow_new_track_list，确保索引与 dets/confs/cls_ids 完全一致。
- 调用 OCSortTracker.update(dets, confs, cls_ids, allow_new_track=allow_new_track_array)。

2. OCSortTracker.update()
- 增加参数：
  allow_new_track: np.ndarray | None = None
- 如果 allow_new_track is None，则默认全部 True。
- 长度必须与 dets 一致，否则 raise ValueError。

3. 无现有轨迹时的初始化逻辑也必须检查 allow_new_track：
- 只有 allow_new_track[i] is True 的 detection 才能初始化 KalmanBoxTracker。
- allow_new_track[i] is False 的 detection 不能初始化新轨迹。

4. 未匹配检测创建新轨迹的位置也必须检查 allow_new_track：
- 如果 allow_new_track[d_idx] is False，continue。
- 否则才创建新的 KalmanBoxTracker。

目标：
- 高置信 RGB 和 IR-supported RGB 可以正常创建新轨迹。
- 低置信 RGB + track support 可以更新已有轨迹。
- 低置信 RGB + track support 如果未匹配成功，不能创建新 ID。
- 低置信 RGB 无 IR support、无 track support 时不进入 tracker。

十一、新脚本实现细节

新增脚本需要包含以下核心函数，名称可根据代码风格微调，但功能必须清晰：

1. load_alignment_config(config_path)
- 读取 alignment_config.yaml。
- 返回配置字典。
- 文件不存在或 YAML 解析失败要报错。

2. load_calib(calib_path)
- 读取 alignment_report.json 或 calib json。
- 提取 IR -> RGB 的仿射矩阵。
- 检查矩阵形状必须为 2x3 或可转换为 2x3。
- 缺失则报错。
- 不允许估计。

3. map_ir_box_to_rgb(box, affine_matrix, rgb_shape)
- 将 IR xywh box 四个角点映射到 RGB 坐标系。
- 取映射后外接矩形。
- 裁剪到 RGB 图像范围。
- 无效框返回 None。

4. compute_iou(box_a, box_b)
- xywh 格式。

5. center_distance(box_a, box_b)
- xywh 格式。

6. has_ir_support(rgb_low_box, mapped_ir_boxes, args)
- 返回：
  - supported: bool
  - best_ir_box: dict | None

7. has_track_support(rgb_low_box, tracker, args)
- 从 tracker 当前已有内部轨迹中读取预测框/状态框。
- 返回 bool。
- 不改变 tracker 状态。

8. fuse_rgb_ir_detections(rgb_boxes, ir_boxes_mapped, tracker, args)
- 输出送入 tracker 的 detection list。
- 每个 detection dict 必须包含：
  - x
  - y
  - w
  - h
  - confidence
  - class
  - allow_new_track
  - support_type，可选字段，用于日志统计，取值 high / ir_support / track_support

9. format_mot_result_line(frame_id, box)
- 与 export_mot_results.py 保持一致。

10. export_dual_modal_video_to_mot_results(...)
- 主导出逻辑。

十二、同步与帧读取

本轮做最小可验证实验：
- 默认按逐帧同步处理 RGB/IR。
- 如果 alignment_config.yaml 中定义了同步模式，需要读取并遵守。
- 如果同步模式不是逐帧同步，且脚本暂不支持，需要明确报错，不要默默按逐帧处理。
- RGB 与 IR 帧数不一致时，以可同时读取的最短长度为准，并在日志中打印 warning。
- --max-frames > 0 时只处理指定帧数。
- --max-frames=0 时处理完整可用帧数。

十三、输出目录结构

输出必须为：

results/rgb_ir_support_ablation/
├── botsort_rgb/
│   └── data/
│       └── <seq-name>.txt
└── botsort_rgb_ir_support/
    └── data/
        └── <seq-name>.txt

注意：
- 输出路径由 --output-root、--run-name、--seq-name 控制。
- 不要使用 tracker 名作为默认 run-name 覆盖 fusion 名称。
- run-name 必须能区分 botsort_rgb 和 botsort_rgb_ir_support。

十四、日志统计

每个 run 结束后必须打印并记录：

- processed frames
- RGB high boxes count
- RGB low boxes count
- RGB low promoted by IR count
- RGB low kept by track support count
- RGB low suppressed count
- IR boxes count
- mapped valid IR boxes count
- output MOT txt path
- total runtime seconds
- runtime FPS

其中：
- rgb_only 模式下，IR 相关统计可以为 0 或 N/A，但不要读取 IR 检测。
- rgb_ir_support 模式下，必须统计 IR boxes 和 mapped valid IR boxes。

十五、运行命令模板

完成代码后，在总结中给出以下两条导出命令模板。

1. BoT-SORT RGB-only：

python tools/evaluation/export_dual_modal_mot_results.py \
  --visible-input /home/hyj/Anti_Drone_Project/MOT_DJI_20250711140455_0002_W/<visible_video> \
  --tracker botsort \
  --fusion rgb_only \
  --run-name botsort_rgb \
  --seq-name <seq> \
  --output-root results/rgb_ir_support_ablation \
  --rgb-high-conf 0.50 \
  --rgb-low-conf 0.30 \
  --max-frames 0 \
  --progress-interval 100

2. BoT-SORT RGB + IR support：

python tools/evaluation/export_dual_modal_mot_results.py \
  --visible-input /home/hyj/Anti_Drone_Project/MOT_DJI_20250711140455_0002_W/<visible_video> \
  --infrared-input /home/hyj/Anti_Drone_Project/MOT_DJI_20250711140455_0002_T/<infrared_video> \
  --tracker botsort \
  --fusion rgb_ir_support \
  --run-name botsort_rgb_ir_support \
  --seq-name <seq> \
  --output-root results/rgb_ir_support_ablation \
  --calib /home/hyj/Anti_Drone_Project/IR_RGB_match/outputs/minimal_alignment/alignment_report.json \
  --config /home/hyj/Anti_Drone_Project/IR_RGB_match/config/alignment_config.yaml \
  --rgb-high-conf 0.50 \
  --rgb-low-conf 0.30 \
  --ir-conf 0.45 \
  --alpha 0.7 \
  --max-frames 0 \
  --progress-interval 100

十六、评估命令

给出 TrackEval 评估命令：

python tools/evaluation/motchallenge_eval.py \
  --gt-root <gt_root> \
  --trackers-root results/rgb_ir_support_ablation \
  --output-root results/rgb_ir_support_eval \
  --trackers botsort_rgb botsort_rgb_ir_support \
  --sequences <seq>

说明：
- 必须加 --sequences <seq>，避免 gt-root 下存在多个序列时要求所有 tracker 都有对应结果。
- 如果没有 GT 或没有实际导出的 MOT txt，不要伪造评估结果。

十七、验证要求

完成后运行：

python -m py_compile \
  target_module/image_detect_module/utils/tracker.py \
  tools/evaluation/export_dual_modal_mot_results.py \
  tools/evaluation/motchallenge_eval.py

如果环境中缺少实际视频、calib 文件、模型权重或 GT：
- 不要伪造实验结果。
- 只确认语法、导入、参数解析、核心函数可以正常运行。
- 明确说明哪些运行未执行，原因是什么。

如果实际视频、calib、模型、GT 都存在，则继续执行：
1. 导出 botsort_rgb 结果。
2. 导出 botsort_rgb_ir_support 结果。
3. 执行 motchallenge_eval.py。
4. 汇总两组结果。

十八、最终输出格式

最终总结必须包含：

1. 修改了哪些文件
- 列出新增和修改文件。
- 简要说明每个文件改了什么。

2. 实验如何运行
- 给出两条导出命令。
- 给出一条 TrackEval 评估命令。

3. 日志统计
- processed frames
- RGB high boxes count
- RGB low boxes count
- RGB low promoted by IR count
- RGB low kept by track support count
- RGB low suppressed count
- IR boxes count
- mapped valid IR boxes count
- output MOT txt path
- runtime FPS

4. 指标对比
如果评估成功，输出表格：

| run | HOTA | IDF1 | IDSW | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|
| botsort_rgb | | | | | | |
| botsort_rgb_ir_support | | | | | | |

并给出结论：
- IR-support 是否提升 HOTA / IDF1。
- 是否降低 IDSW / FN。
- FP 是否增加。
- FPS 下降是否明显。
- 如果提升不明显，分析原因，但不要编造不存在的数据。

如果评估没有成功：
- 不输出伪造指标。
- 明确写出：本次只完成代码修改与静态验证，尚未产生 HOTA / IDF1 / IDSW / FP / FN / FPS 对比结果。
- 说明缺少什么文件或哪一步未执行。

十九、禁止事项

- 不要做无标定近似估计。
- 不要用 MOT GT 或 ID 对应关系参与推理融合。
- 不要启用 IR-only 补框。
- 不要让 IR-only box 创建新轨迹。
- 不要让只有 track support 的低置信 RGB box 创建新轨迹。
- 不要修改 bytetrack、dist_tracker、official_ocsort、official_botsort 的行为。
- 不要把本实验结论扩展为所有 tracker 通用。
- 不要伪造实验结果。