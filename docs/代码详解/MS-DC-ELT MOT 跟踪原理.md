现在 MS-DC-ELT MOT 跟踪不是 OC-SORT/BoT-SORT 那种 tracker adapter，而是一个独立的 evidence lifecycle tracker。核心思想是：把检测框、低阈值候选、可选 motion/template/ROI 等统一转成 `Observation`，按几何关系关联到已有轨迹，然后用证据分数和状态机维护 `candidate -> active -> lost -> removed` 生命周期，最后只把合格的 `active` 轨迹写成 MOT 结果。

**MOT 调用链**
- MOT 导出入口：[tools/evaluation/export_mot_results.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/tools/evaluation/export_mot_results.py:92)
- 每帧 MS-DC-ELT 分支先算 high/low boxes：[export_mot_results.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/tools/evaluation/export_mot_results.py:153)
- 然后调用 lifecycle tracker：[export_mot_results.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/tools/evaluation/export_mot_results.py:63)
- 写 MOTChallenge 行格式的位置：[export_mot_results.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/tools/evaluation/export_mot_results.py:49)

**检测输入**
- high/low 检测 helper 在 [msdc_detection.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/msdc_detection.py:27)
- 当前默认 `MSDC_EXPORT_SHARE_LOW_HIGH_DET=True`，所以正式 MS-DC-ELT 通常只跑一次低阈值检测，再用默认 conf 阈值从 low boxes 里切出 high boxes：[msdc_detection.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/msdc_detection.py:55)

**主 tracker**
- 主类是 [MSDCLifecycleTracker](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/lifecycle_tracker.py:37)
- 每帧 `update()` 做这些事：
  - high/low 去重，只保留 low-only：[lifecycle_tracker.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/lifecycle_tracker.py:85)
  - low-only 做 topK/min_conf/track proximity 预算：[lifecycle_tracker.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/lifecycle_tracker.py:435)
  - 可选 ROI 重检、motion seed、template 观测：[lifecycle_tracker.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/lifecycle_tracker.py:91)
  - 转成 `Observation` 后交给 evidence updater：[lifecycle_tracker.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/lifecycle_tracker.py:111)

**状态和数据结构**
- 状态定义在 [msdc_types.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/msdc_types.py:17)：`low_candidate`、`candidate`、`active`、`lost`、`removed`
- `Observation` 和 `EvidenceTrack` 也在 [msdc_types.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/msdc_types.py:98)

**证据状态机**
- 核心更新器：[EvidenceStateUpdater](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:146)
- 状态判定函数：[assign_state()](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:94)
- 关联逻辑：用预测框和 observation group 做 IoU/中心距离门控，满足 `IoU >= MSDC_ASSOC_IOU_THRESH` 或中心距离小于阈值后，按 `IoU + center_bonus` 贪心匹配：[evidence_state.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:288)
- 命中更新证据：`score = alpha * old_score + weighted_observation_score`，未匹配则扣负证据：[evidence_state.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1405)
- 不同来源权重：`high_det`、`low_det`、`roi_low_det`、`motion`、`template`、`reacquire`：[evidence_state.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1649)

**确认/丢失/重捕**
- 普通 candidate 要证据分、hits、真实检测次数、高阈值检测要求都满足后才能变 active：[evidence_state.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1503)
- low candidate 需要在时间窗口内多次低阈值命中、均分足够、面积和运动稳定：[evidence_state.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1692)
- active 漏检超过 patience 后转 lost；lost 若被足够强的 observation 重捕则回 active，否则超时 removed：[evidence_state.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1528)

**输出规则**
- 默认只输出 `active`，不输出 candidate：[lifecycle_tracker.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/lifecycle_tracker.py:225)
- 输出还要求有近期真实检测，`real_det_age <= MSDC_OUTPUT_MAX_REAL_DET_AGE`，框尺寸不小于 `MSDC_OUTPUT_MIN_BOX_SIZE`：[lifecycle_tracker.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/lifecycle_tracker.py:240)
- 输出前做轻量 NMS/碎片去重：[lifecycle_tracker.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/lifecycle_tracker.py:285)

**当前默认配置**
主要在 [config.py](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/config.py:113)。当前 formal v3 方向是：
- low det 开
- motion 关
- template 关
- ROI redetect 关
- shared high/low 开
- low observation `topK=32`、`min_conf=0.25`
- debug events 关

所以现在正式 MOT 逻辑可以概括为：一次低阈值检测产生 low boxes，切出 high boxes；high boxes 负责强确认，low-only boxes 经过预算和轨迹邻近过滤后补充漏检；evidence state 通过几何关联和证据积分维护 ID 生命周期；最终只输出近期有真实检测支撑的 active track。