## ProbEn-lite RGBIR Late-Fusion 实验计划

### 0. 实验定位

在 `msdc-rgb-ir-support` 分支基础上新增一个受控融合模式：

```text
fusion=rgb_ir_proben
```

该实验不是完整复现 ProbEn，而是实现适配当前工程的 ProbEn-lite late-fusion：

```text
RGB 检测结果 + 仿射映射后的 IR 检测结果
→ 几何门控匹配
→ ProbEn-lite 分数融合
→ RGB-biased box fusion  score-only
→ 融合检测结果
→ BoT-SORT
```

保留现有 `rgb_only` 和 `rgb_ir_support`，将 `rgb_ir_proben` 作为新增对照组。

---

## 1. 总体原则

### 1.1 不做无标定近似

必须读取现有 `alignment_config.yaml` 和 `alignment_report.json`。

如果缺少以下内容，直接报错退出：

```text
RGBIR 视频路径
IR 黑边参数
同步模式
ID 对应关系
仿射矩阵
alignment_report
calib  affine 字段
```

禁止在 calib 缺失时用手工估计、图像尺寸比例缩放或默认单位矩阵代替。

### 1.2 第一版不允许 IR-only 直接起轨

第一版只验证：

```text
IR 是否能增强 RGB 低置信检测
```

暂时不验证：

```text
IR 是否能单独补回 RGB 完全漏检目标
```

因此：

```text
RGB-only      → 可进入 tracker
RGB + IR      → 融合后进入 tracker
IR-only       → 不进入 tracker，只写 diagnostics
```

IR-only 补漏放到第二阶段，通过 candidate buffer 和 lost-track 重激活实现。

### 1.3 先做 detection-level 对照，再做 tracking-level 对照

不能只看 BoT-SORT 结果。必须先导出融合后的 det.txt，单独评估检测层变化，再评估跟踪层变化。

---

## 2. 代码修改计划

### 2.1 新增融合模块

新增文件：

```text
toolsevaluationproben_fusion.py
```

包含以下纯函数：

```python
binary_proben_score(p_rgb, p_ir, eps=1e-4)

score_weighted_box_fusion(
    rgb_box,
    ir_box_mapped,
    p_rgb,
    p_ir,
    ir_box_weight=0.5,
)

compute_center_distance(box_a, box_b)

compute_scale_ratio(box_a, box_b)

match_rgb_ir_boxes(
    rgb_dets,
    mapped_ir_dets,
    match_iou,
    match_dist_factor,
    scale_ratio_min,
    scale_ratio_max,
)

fuse_rgb_ir_proben(
    rgb_dets,
    mapped_ir_dets,
    args,
)
```

核心分数融合公式：

```text
p_fused = (p_rgb  p_ir)  ((p_rgb  p_ir) + ((1 - p_rgb)  (1 - p_ir)))
```

需要对输入分数做 clamp：

```text
p_rgb = clamp(p_rgb, eps, 1 - eps)
p_ir  = clamp(p_ir, eps, 1 - eps)
```

第一版框融合使用 `s-avg` 近似：

```text
w_rgb = p_rgb
w_ir  = proben_ir_box_weight  p_ir

box_fused = (w_rgb  box_rgb + w_ir  box_ir_mapped)  (w_rgb + w_ir)
```

默认：

```text
proben_ir_box_weight = 0.5
```

原因：当前 IR→RGB 平均对齐误差约 7.36 px，小目标场景下不应让 IR 框过强地拖动 RGB 框。

---

## 3. 导出脚本接入

修改：

```text
toolsevaluationexport_dual_modal_mot_results.py
```

### 3.1 新增 fusion 模式

```text
--fusion rgb_ir_proben
```

保留现有模式：

```text
--fusion rgb_only
--fusion rgb_ir_support
```

### 3.2 新增参数

```text
--proben-match-iou 0.15
--proben-match-dist-factor 1.5
--proben-keep-conf 0.50
--proben-ir-box-weight 0.5
--proben-scale-ratio-min 0.4
--proben-scale-ratio-max 2.5
--proben-low-allow-new-track false
--proben-box-mode score_only  savg
```

默认建议：

```text
proben-box-mode = score_only
```

先验证分数融合是否有效，再验证框融合是否进一步提升。

### 3.3 输出规则

#### RGB 高置信框

```text
若无匹配 IR：
    保留原 RGB detection
若匹配 IR：
    做 ProbEn-lite 分数融合
    box 根据 proben-box-mode 决定
    输出 fused detection
    allow_new_track = True
```

#### RGB 低置信框

```text
若无匹配 IR：
    不输出或维持原低置信逻辑
若匹配 IR：
    做 ProbEn-lite 分数融合
    若 p_fused = proben_keep_conf：
        输出 fused detection
    否则：
        写入 rejected diagnostics
```

第一版中，低置信 RGB + IR 融合框是否能起新轨要谨慎处理：

```text
如果当前 BoT-SORT 封装不支持 per-detection allow_new_track：
    不实现 proben-low-allow-new-track
    只输出到普通 detection 流
    在实验记录中说明该限制
```

如果支持或可以改造 tracker：

```text
proben-low-allow-new-track = False 时：
    低置信 RGB + IR 只允许匹配 existinglost track
    不允许初始化新轨迹
```

#### IR-only 框

```text
不进入 tracker
不写入 det.txt
只写入 diagnostics
```

记录字段：

```text
frame_id
ir_box_mapped
ir_conf
reason = ir_only_not_emitted
nearest_rgb_dist
nearest_rgb_iou
```

---

## 4. 匹配策略

RGBIR 匹配必须同时满足几何门控和尺度门控。

### 4.1 基础门控

候选匹配条件：

```text
IoU = proben_match_iou
OR
center_distance = proben_match_dist_factor  sqrt(rgb_box_area)
```

### 4.2 尺度一致性

```text
scale_ratio = area_ir_mapped  area_rgb
```

必须满足：

```text
proben_scale_ratio_min = scale_ratio = proben_scale_ratio_max
```

默认：

```text
0.4 = scale_ratio = 2.5
```

### 4.3 匹配优先级

匹配代价建议：

```text
cost = center_distance_norm - lambda_iou  IoU + lambda_scale  scale_penalty
```

第一版可简化为：

```text
优先选择中心距离最小的 IR 框
若中心距离接近，再选择 IoU 更高者
若尺度差过大，直接拒绝
```

### 4.4 重复框处理

匹配成功后只能输出一个 fused box。

禁止同时输出：

```text
RGB 原框 + fused 框
```

融合完成后建议再做一次同类 NMS，避免重复检测进入 BoT-SORT。

---

## 5. 诊断统计

新增 run-level 统计字段：

```text
rgb_ir_proben_matched_count
rgb_ir_proben_promoted_count
rgb_ir_proben_rejected_by_geometry_count
rgb_ir_proben_rejected_by_score_count
rgb_ir_proben_ir_only_count
rgb_ir_proben_avg_score_gain
rgb_ir_proben_avg_box_shift
rgb_ir_proben_max_box_shift
```

新增 frame-level JSONL，可选开启：

```text
--save-proben-diagnostics
```

每帧记录：

```text
frame_id
rgb_box
ir_box_mapped
fused_box
p_rgb
p_ir
p_fused
score_gain
box_shift
match_iou
center_distance
scale_ratio
decision
reject_reason
```

重点关注：

```text
score_gain 是否过大
box_shift 是否超过 alignment 误差范围
promoted 目标是否真实
IR-only 数量是否异常
```

---

## 6. 单元测试计划

新增：

```text
testtest_proben_fusion.py
```

覆盖以下测试：

### 6.1 ProbEn 分数融合

验证两个模态一致时分数上升：

```text
p_rgb = 0.40
p_ir  = 0.70
p_fused ≈ 0.61
```

### 6.2 clamp 生效

验证输入为 0 或 1 时不会数值溢出。

### 6.3 单模态缺失

验证：

```text
RGB-only 保留原 RGB confidence
IR-only 不输出 tracker detection
```

### 6.4 低置信 RGB 被 IR 支撑提升

验证：

```text
RGB low + matched IR
→ p_fused = proben_keep_conf
→ promoted_count + 1
```

### 6.5 尺度差过大拒绝融合

验证：

```text
scale_ratio  0.4 或  2.5
→ rejected_by_geometry_count + 1
```

### 6.6 框融合权重可控

验证：

```text
proben_ir_box_weight 越小
fused_box 越接近 RGB box
```

### 6.7 重复框抑制

验证匹配成功后只输出一个 fused detection，不重复输出 RGB 原框。

---

## 7. 实验矩阵

### 7.1 检测层实验

先不接 BoT-SORT，只比较 det.txt。

 实验名                             目的                               
 ------------------------------  -------------------------------- 
 `rgb_det`                       RGB 检测基线                         
 `rgb_ir_support_det`            当前 IR 支撑策略检测输出                   
 `rgb_ir_proben_score_only_det`  只融合分数，不改框                        
 `rgb_ir_proben_savg_det`        分数融合 + score-weighted box fusion 
 `rgb_ir_proben_rgb_box_det`     ProbEn 分数 + RGB 原框，排除框偏移影响       

检测层指标：

```text
Precision
Recall
FP
FN
FPframe
small-object recall
mean confidence gain
mean box shift
```

### 7.2 跟踪层实验

再接 BoT-SORT。

 实验名                                 目的                                
 ----------------------------------  --------------------------------- 
 `botsort_rgb`                       主基线                               
 `botsort_rgb_ir_support`            当前 IR 支撑对照                        
 `botsort_rgb_ir_proben_score_only`  判断 ProbEn 分数融合是否提升跟踪              
 `botsort_rgb_ir_proben_savg`        判断框融合是否进一步提升                      
 `botsort_rgb_ir_proben_candidate`   第二阶段：IR-only candidate 补漏，不在第一轮实现 

跟踪层指标：

```text
HOTA
IDF1
AssA
MOTA
IDSW
Frag
FP
FN
FPS
latency
```

### 7.3 阈值实验

至少做两种口径：

#### 固定阈值

```text
所有方法使用相同 BoT-SORT 阈值
```

目的：评估直接替换输入 detection 后的工程收益。

#### 单独调参

```text
每种 fusion 单独扫描 track_high_thresh  track_low_thresh  new_track_thresh
```

目的：评估该融合策略的最佳潜力。

---

## 8. 速度实验

速度必须拆分，不只看总 FPS。

记录：

```text
RGB detector time
IR detector time
alignment  mapping time
fusion time
tracker time
total time
```

建议分两种模式：

### 8.1 Detection replay 模式

先缓存 RGBIR 每帧检测结果，再 replay 融合和跟踪。

目的：

```text
排除 detector 推理开销
公平比较 rgb_ir_support 与 rgb_ir_proben 的融合和 tracker 开销
```

### 8.2 End-to-end 模式

完整运行：

```text
RGB detector + IR detector + alignment + fusion + BoT-SORT
```

目的：

```text
评估实际部署代价
```

---

## 9. 验收标准

`rgb_ir_proben` 只有满足以下条件，才认为优于当前 `rgb_ir_support`：

### 9.1 主指标

```text
HOTA 不低于 botsort_rgb_ir_support
IDF1 不低于 botsort_rgb_ir_support
IDSW 不高于当前结果
```

### 9.2 检测指标

```text
FN 低于 botsort_rgb
FP 不高于 botsort_rgb_ir_support
small-object recall 高于 botsort_rgb
```

### 9.3 稳定性指标

```text
avg_box_shift 不应明显超过 alignment 平均误差
promoted detections 中真实目标比例应高
IR-only 数量异常时不得进入 tracker
```

### 9.4 速度指标

```text
fusion time 应远小于 detector time
replay 模式下 FPS 不应明显低于 rgb_ir_support
end-to-end 模式下需单独报告双 detector 推理成本
```

---

## 10. 第一阶段只实现以下内容

第一阶段实现范围：

```text
fusion=rgb_ir_proben
binary ProbEn score fusion
score_only  savg 两种 box mode
严格 RGBIR 几何匹配
IR-only diagnostics
检测层 det.txt 导出
BoT-SORT tracking 接入
单元测试
diagnostics JSONL
```

第一阶段不实现：

```text
完整 class probability vector
bbox variance head
v-avg box fusion
IR-only 直接起轨
IR-only candidate buffer
lost-track IR 重激活
端到端双模态网络训练
```

---

## 11. 推荐第一轮默认配置

```text
--fusion rgb_ir_proben
--proben-box-mode score_only
--proben-match-iou 0.15
--proben-match-dist-factor 1.5
--proben-keep-conf 0.50
--proben-ir-box-weight 0.5
--proben-scale-ratio-min 0.4
--proben-scale-ratio-max 2.5
--proben-low-allow-new-track false
--save-proben-diagnostics
```

第一轮优先跑：

```text
botsort_rgb
botsort_rgb_ir_support
botsort_rgb_ir_proben_score_only
botsort_rgb_ir_proben_savg
```

最终输出目录建议：

```text
resultsrgb_ir_proben_phase1
```

每个 run 必须包含：

```text
config.yaml
alignment_report.json copy
det.txt
mot_result.txt
metrics_summary.json
speed_profile.json
proben_diagnostics.jsonl
run_manifest.json
```

---

## 12. 最终判断逻辑

如果结果是：

```text
FN 下降
FP 接近 rgb_only 或低于 rgb_ir_support
IDSW 不升
HOTA  IDF1 上升
```

说明 ProbEn-lite 比当前 `rgb_ir_support` 更适合。

如果结果是：

```text
FN 下降
但 FP  IDSW 明显上升
```

说明融合分数过激或匹配门控过宽，需要提高 `proben_keep_conf`、收紧中心距离尺度门控，或禁止低置信融合框起轨。

如果结果是：

```text
检测层 Recall 上升
但跟踪层 IDF1  HOTA 下降
```

说明检测融合有效，但 BoT-SORT 接入策略有问题，应重点检查新轨初始化、重复框、低置信框匹配策略。

如果结果是：

```text
检测层也没有改善
```

说明当前 IR detector 或 RGBIR 对齐质量不足，暂时不应继续做 IR-only candidate，而应先检查 IR 检测质量和映射误差。
