# MS-DC-ELT 方法叙事独立审查：问题表、Claim-Evidence Map 与修正计划

审查时间：2026-06-24

审查对象：

- `paper-prepare/9 个科研问题 V5.md`
- `docs/代码详解/`
- `README.md`
- 局部核查：`target_module/image_detect_module/config.py`、`constants.py`、`utils/evidence_state.py`
- 参考旧结果：`results/msdc_gap_formal/20260615_*`

Gemini reviewer：

- jobId：`0ea55fe5a5f840abbbefecc1aca5682e`
- threadId：`6ef10bf6e2f9417a8b98b122d6b2d583`
- backend/model：`api` / `gemini-3.1-pro-preview`
- 状态：completed

## 总体判断

当前 MS-DC-ELT 的真实贡献不是“解决空海 MOT 中的平台运动、外观相似和长序列全局关联”，而是一个面向**不稳定检测流**的可解释轨迹生命周期管理器。它的核心价值在于：把高/低置信检测转为 observation，用多帧证据累计、低置信候选确认、lost 重捕获和 low-candidate 继承来决定何时确认、保留、恢复或删除轨迹。

最大风险是叙事前半段提出的问题过大：平台运动几何误差、目标相似性歧义、长序列全局身份保持；但当前 formal v3 实现是 `v3_candidate_topk_no_roi_no_motion`，没有 motion/template/ROI，没有 ReID/appearance embedding，关联主要依赖 IoU/中心距离贪心匹配。因此如果按当前写法投稿，审稿人会认为“提出问题 A，解决问题 B”，这是最危险的逻辑断裂。

建议立刻把论文主线收缩为：

> 面向远距离空海小目标检测闪烁、漏检和置信度波动的 dual-confidence evidence lifecycle tracker。

不要再把平台运动补偿误差或目标相似性作为本文已解决的核心挑战；这些只能作为适用边界或 future work。

## 问题表

| ID | 严重性 | 问题 | 证据 | 影响 | 修正建议 |
| --- | --- | --- | --- | --- | --- |
| P1 | Critical | 动机与方法错位 | 叙事强调平台运动误差和目标相似性；formal v3 是 `no_motion`，无 ReID，仅几何匹配 | 顶会审稿会直接质疑主问题没有被方法解决 | 重写 Abstract/Intro，把核心问题收窄到检测稀疏、置信度波动和短时漏检 |
| P2 | Critical | 对 SORT/OC-SORT/BoT-SORT 的批评不公平 | 叙事批评运动模型难处理非线性平台运动；MS-DC-ELT 自身没有显式运动补偿，弱于有 Kalman/GMC 的方法 | 容易被认为 strawman baseline | 不再攻击“运动补偿能力”，改为比较“低置信观测如何进入生命周期状态管理” |
| P3 | Major | 与 ByteTrack 差异不足 | 当前 low/high split 与低分框利用思路和 ByteTrack 很接近 | 审稿人会问是否只是复杂 heuristic 版 ByteTrack | 必须加入 ByteTrack baseline，并强调差异：ByteTrack 是两阶段匹配，MS-DC-ELT 是证据累计 + 状态机 |
| P4 | Major | 证据公式与代码不完全一致 | 叙事写整数 `round(...)`；代码 `_rounded_score()` 是 `round(value, 4)`；未匹配还有 `- negative_weight` | 数学严谨性被削弱 | 改为浮点递推：匹配 `p_t=alpha p_{t-1}+s_t`，未匹配 `p_t=max(0, alpha p_{t-1}-beta)` |
| P5 | Major | `p_t/v_t/m_t` 叙事过度理论化 | 代码中 `v_t` 多为 confirm_ready and matched 的二值门控，`m_t` 多为 0 或 theta_m | 审稿人会觉得包装大于实质 | 把它们写成状态机判定变量，不要宣称为独立估计的可靠统计量 |
| P6 | Major | 超参数工程感强 | 默认阈值包括 score 2.5、min_hits 4/5、window 8、center 160/240、inherit 0.40 等 | 容易被质疑过拟合数据集 | 加超参数敏感性分析，至少对 `alpha`、confirm score、low confirm hits/window 做 ±20% 扫描 |
| P7 | Major | formal v3 与可见旧结果版本不一致 | README/常量为 `v3_candidate_topk_no_roi_no_motion`；旧 summary 里是 `v2_candidate_topk_no_roi` | 不能把旧结果直接支撑 v3 论文 claim | 重跑/汇总 v3 正式结果，并在结果表中写清 variant、commit、配置 |
| P8 | Major | 已见旧结果不支持强性能主张 | UAV 旧结果中 v2 的 HOTA/IDF1 低于 BoT-SORT，IDSW 更高；速度明显慢于 baseline | 若 v3 无改善，不能写“全面优于” | 主张改为“特定检测不稳定片段收益”，并用切片指标证明 |
| P9 | Major | 缺少机制级诊断 | 叙事主张 low candidate、reacquire、inherit 有效，但没有给出确认 precision、继承正确率、恢复成功率 | 无法证明模块真的减少断轨而不是引入错误继承 | 增加 event-level diagnostics：low confirm precision/recall、inherit correctness、reacquire success |
| P10 | Moderate | 输出规则影响指标但叙事未解释 | 只输出 active，且 real_det_age<=3、box size>=12、输出 NMS | 可能减少 FP，也可能增加 FN/延迟 | 在方法和实验中解释输出 gate 的收益与代价，并消融 output gate/NMS |
| P11 | Moderate | 平台运动和目标相似性只能作为 limitation | 当前无外观特征、无全局运动补偿 | 若继续作为核心问题，会形成反证 | 放到 limitation：剧烈相机运动和相似目标交汇需结合 GMC/ReID/全局优化 |
| P12 | Moderate | “可复现实验流程”是工程贡献，不等于算法贡献 | README 中工具链完整，但不是方法本身的核心创新 | 论文贡献点可能显得散 | 把评测流程作为 secondary contribution 或 reproducibility appendix |

## Claim-Evidence Map

| Claim | 当前支持度 | 证据 | 允许保留的写法 | 必须补的证据 |
| --- | --- | --- | --- | --- |
| C1：本文解决空海 UAV/USV MOT 的跨帧身份保持 | Partial | 方法确实输出 track ID；但无运动/外观，无法覆盖全部空海 MOT 难点 | “面向检测不稳定空海小目标场景，改善 tracking-by-detection 的轨迹生命周期稳定性” | v3 全量 MOT 指标 + 场景切片 |
| C2：方法能处理平台运动导致的几何对齐误差 | Not supported | formal v3 是 `no_motion`，关联仍是 IoU/中心距离 | 只能写为 limitation，不写成贡献 | 若要保留，需加入 GMC/registration 或运动补偿实验 |
| C3：方法能缓解目标相似性导致的关联歧义 | Weak / mostly unsupported | 无 ReID、appearance embedding；inherit 仍靠几何/速度/类别 | “不针对外观相似性，目标交汇时可能失败” | 目标交汇切片、错误继承案例分析 |
| C4：双置信度输入能提高低置信小目标进入跟踪流程的可控性 | Plausible but not proven | low_det、low_candidate、Top-K/proximity budget 已实现 | “提出可控低置信候选入口” | low_candidate precision/recall、FP/FN trade-off、关闭 low candidate 消融 |
| C5：证据累计降低单帧置信度波动敏感性 | Supported by mechanism, needs results | `evidence_score = alpha*old + positive_score`，未匹配衰减/扣分 | “通过多帧证据累计缓解单帧波动” | 与 hits-only、ByteTrack-like 两阶段匹配对比 |
| C6：分级确认减少低置信噪声误轨迹 | Plausible but unproven | low confirm hits/window/avg score/area/center gate 已实现 | “通过时序门控约束低置信候选确认” | 低置信候选误确认率、确认延迟、噪声场景 FP |
| C7：reacquire 与 inherit 增强短时丢失目标身份延续 | Plausible but high-risk | direct reacquire 和 low inherit 已实现 | “为短时漏检后的 ID 延续提供两条恢复路径” | 重捕获成功率、继承正确率、IDSW 是否上升 |
| C8：方法优于 OC-SORT/BoT-SORT | Not supported by current provided evidence | 已见旧 v2 结果低于 BoT-SORT，且非 v3 | 不能写，除非 v3 结果支持 | v3 多序列正式结果，paired significance |
| C9：方法速度可接受或高效 | Not supported by旧结果 | 旧 speed 中 MS-DC-ELT FPS 明显低于 OC-SORT/BoT-SORT | “增加了生命周期管理开销，通过 Top-K 控制规模” | v3 speed breakdown、低阈值检测/更新耗时归因 |
| C10：形成可复现实验流程 | Supported | README 中 formal run、summary、speed、visualization、validation 工具链清楚 | 可作为工程/复现贡献 | 结果产物清单与 run manifest |
| C11：状态转移可解释、可追溯 | Supported by implementation | LifecycleEvent、state machine、debug events 机制存在 | “提供可追溯 lifecycle events” | 在论文中展示 1-2 个事件轨迹案例 |
| C12：适合顶会主方法投稿 | Currently weak | 当前创新偏规则状态机，强依赖实验说服力 | “应用型/系统型贡献更稳” | ByteTrack baseline、机制诊断、跨数据集收益 |

## 修正计划

### Phase 0：冻结对象与证据边界

1. 冻结论文方法为 `v3_candidate_topk_no_roi_no_motion`。
2. 记录 commit hash、配置覆盖、数据集、视频列表、检测模型、阈值。
3. 旧 `v2_candidate_topk_no_roi` 结果只能作为历史参考，不能支撑 v3 claim。

### Phase 1：叙事重写

1. 重写标题/摘要/引言，主线改为：检测闪烁、低置信小目标、短时漏检后的 ID 延续。
2. 删除或弱化平台运动补偿误差、目标相似性作为本文核心已解决问题。
3. 现有方法相关工作重写为：
   - ByteTrack：低分检测利用强，但缺少显式证据生命周期记忆。
   - OC-SORT/BoT-SORT：运动/外观能力强，但在空海小目标低置信闪烁下仍会断轨。
   - 本文：不替代运动/外观特征，而是研究检测不稳定时的状态管理层。
4. Method 中诚实写明：当前 MS-DC-ELT 不使用 ReID、appearance embedding、GMC、ROI redetect、template lock。

### Phase 2：数学和方法表述对齐代码

1. 删除整数 `round` 公式，改成连续分数：

   `s_t = sum_k w_k * score_k * reliability_k`

   匹配：

   `p_t = round4(alpha * p_{t-1} + s_t)`

   未匹配：

   `p_t = round4(max(0, alpha * p_{t-1} - beta))`

2. 把 `v_t/m_t` 写成状态机判定门控，不写成独立概率或理论指标。
3. 增加状态机图：`low_candidate -> active/removed`、`candidate -> active/removed`、`active -> lost`、`lost -> active/removed`、`low_candidate + lost -> inherit`。
4. 明确 low-only budget：Top-K、min_conf、track proximity 是正式方法的一部分。

### Phase 3：最小实验包

1. 主实验：
   - Baseline：ByteTrack、OC-SORT、BoT-SORT。
   - 方法：MS-DC-ELT v3。
   - 保持同一 detector 和同一检测输入策略，说明是否 replay detections。

2. 必要消融：
   - full v3
   - no low_candidate
   - no direct reacquire
   - no low inheritance
   - no evidence score，用 hits-only 替代
   - no output NMS / no output real_det_age gate
   - low observation budget 关闭或不同 Top-K

3. 诊断指标：
   - low_candidate confirmed precision / recall
   - low_candidate 平均确认延迟
   - inherit correct / wrong / ambiguous
   - reacquire success rate
   - fragmentation / track break count
   - IDSW before/after reacquire/inherit
   - 短时漏检片段 IDF1/HOTA/AssA

4. 速度和复杂度：
   - 总 FPS
   - low detection
   - high split
   - low filter/budget
   - observation build
   - evidence update
   - output/NMS
   - render/write

5. 超参数敏感性：
   - `MSDC_EVIDENCE_ALPHA`
   - `MSDC_CONFIRM_SCORE`
   - `MSDC_LOW_CONFIRM_MIN_HITS`
   - `MSDC_LOW_INHERIT_SCORE`
   - `MSDC_REACQUIRE_SCORE`
   - 至少做 ±20% 或离散 3-5 点扫描。

### Phase 4：结果到主张矩阵

| 实验结果情况 | 允许 claim | 禁止 claim |
| --- | --- | --- |
| v3 在 IDF1/AssA/IDSW 上稳定优于 baseline，MOTA/HOTA 不退化 | “改善检测不稳定下的身份延续” | “解决平台运动/外观相似” |
| v3 只在短时漏检片段优于 baseline，全局指标持平 | “在短时漏检和低置信闪烁片段有效” | “整体优于主流 MOT” |
| v3 降低 FN 但 FP/IDSW 上升 | “提高召回，但需要权衡误确认和错误继承” | “更稳定身份保持” |
| v3 速度明显慢 | “以额外生命周期管理开销换取诊断性和特定场景收益” | “高效实时” |
| v3 不优于 ByteTrack | “工程诊断工具/状态机分析框架” | “算法贡献强于 ByteTrack” |

## 建议重写贡献点

1. 提出一种面向空海小目标检测不稳定性的 dual-confidence evidence lifecycle tracker，将高/低置信检测统一为 observation，并通过多帧证据累计驱动轨迹确认、丢失和移除。
2. 设计低置信候选的时序确认机制，结合命中次数、平均置信度、尺度稳定性和中心位移约束，控制弱目标召回与噪声误确认之间的权衡。
3. 设计短时漏检后的 direct reacquire 与 low-candidate inheritance 机制，为弱检测重新出现时的 ID 延续提供可解释恢复路径。
4. 建立事件级诊断与正式评测流程，使每个轨迹的确认、丢失、重捕获和继承都可追溯。

## 不建议保留的表述

- “解决平台运动导致的几何对齐误差”
- “解决目标相似性导致的关联歧义”
- “能够同时容纳运动误差累积、观测稀疏与关联歧义”
- “相比主流方法全面提升”
- “端到端/全局建模能力”

这些表述都需要当前代码没有提供的运动补偿、外观特征、全局优化或强实验结果。

## 当前审稿式结论

按当前叙事，强审稿意见会是 reject，主要原因是动机和方法不匹配、对基线批评不公平、实验还没有支撑核心 claim。

按修正后叙事，MS-DC-ELT 可以成为一篇更稳的应用/系统型 MOT 方法：它不是通用最强 tracker，而是一个针对检测不稳定流的、可解释的生命周期管理层。能否上更高层级会议，取决于 v3 是否在 ByteTrack/OC-SORT/BoT-SORT 对比和机制级诊断上拿出清楚收益。

## 立即行动清单

1. 冻结 v3 配置并重跑正式主实验。
2. 补 low_candidate、reacquire、inherit、evidence score 的消融。
3. 做短时漏检/低置信片段切片评测。
4. 重写 `9 个科研问题 V5.md` 的第 1-4 问，把问题域从“大而全空海 MOT”收窄到“检测不稳定驱动的身份断裂”。
5. 在 Method 中按代码重写证据公式和状态机，不再使用整数 `round`。
