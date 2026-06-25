# 国内外研究现状初稿：面向检测不稳定小目标的在线多目标跟踪

## 1. 基于检测关联范式的在线多目标跟踪

多目标跟踪通常需要在连续视频中为目标维持一致的 track ID。Tracking-by-detection 是当前常用范式之一，其基本流程是在每帧先检测目标，再将检测框与已有轨迹关联。SORT 使用卡尔曼滤波预测轨迹位置，并以 IoU 和 Hungarian 算法完成检测框到轨迹的匹配。该方法结构简单、速度快，也说明检测器质量会显著影响跟踪结果。DeepSORT 在 SORT 基础上引入外观特征，通过深度关联度量减少遮挡后的 ID switch。此后，tracking-by-detection 方法围绕运动、几何、外观和相机运动补偿等线索持续改进。

近年来，OC-SORT、BoT-SORT、Deep OC-SORT 和 Hybrid-SORT 进一步强化了在线关联能力。OC-SORT 指出，当目标短时不可见时，卡尔曼滤波的线性预测误差会积累，因此使用 observation-centric re-update 和 observation-centric momentum 来修正恢复后的状态。BoT-SORT 结合运动、外观和 camera-motion compensation，并使用改进的卡尔曼状态向量提高关联稳定性。Deep OC-SORT 在 OC-SORT 上引入自适应 ReID。Hybrid-SORT 则将 velocity direction、confidence 和 height state 等弱线索加入关联。这些方法表明，MOT 关联性能可以通过更好的运动建模、外观建模和匹配代价设计获得提升。

这类方法主要解决检测框与已有轨迹之间的在线匹配问题。它们通常通过预测框与检测框的几何重叠、中心距离、运动方向、Mahalanobis distance、外观相似度或多线索融合构造代价矩阵，再用 Hungarian matching 或级联匹配完成关联。对于短时漏检，一些方法会保留 lost tracks 并尝试后续恢复；OC-SORT 还显式处理 lost period 后的运动状态修正。其不足在于，多数方法的核心仍是关联代价设计。对于空海小目标中常见的检测置信度波动和低置信目标重现，仅依赖当前帧匹配代价难以回答低置信观测何时应确认、何时应保留、何时应删除的问题。

## 2. 低置信检测与检测置信度利用

低置信检测利用是近年来 tracking-by-detection 的重要进展。ByteTrack 指出，传统方法通常只保留高置信检测，低分检测被直接丢弃会带来目标漏检和轨迹碎片。ByteTrack 采用两阶段关联：先用高置信检测匹配轨迹，再用低置信检测匹配第一阶段未匹配的轨迹，从而恢复被遮挡、模糊或弱检测导致的目标。该思想说明低置信检测并不等同于背景噪声，其中一部分仍可能是真实目标。

后续工作进一步研究检测置信度在关联中的作用。ConfTrack 面向 Kalman filter-based tracking-by-detection，针对低置信检测带来的噪声问题设计低置信惩罚和级联匹配，并区分 tentative tracks 与 confirmed tracks。BoostTrack 和 BoostTrack++ 使用 detection-tracklet confidence 调整相似度，并通过 soft confidence boost、shape、Mahalanobis distance 和 soft BIoU 等线索提高真阳性检测的选择能力。Hybrid-SORT 也把 confidence 作为弱线索之一，配合速度方向和高度状态改善在线关联。UTrack 和 UncertaintyTrack 则进一步使用检测不确定性或定位不确定性，而不仅是单一置信度分数。

这类研究已经较好地回答了“是否应利用低置信检测”的问题：低置信检测可以在经过匹配、惩罚、重加权或不确定性建模后参与跟踪。但多数方法将低置信检测用于当前帧或短期关联，重点是如何将低分框匹配到已有轨迹。相比之下，低置信检测在新轨迹生成阶段的延迟确认、在多帧证据中的累积、以及与 lost track ID 继承的关系，仍不是上述方法的主要对象。

## 3. 轨迹延续、lost track 和生命周期管理

轨迹生命周期管理是在线 MOT 的基础组成部分。SORT 系列方法通常包含轨迹初始化、确认、丢失保留和删除规则。DeepSORT 使用 tentative 和 confirmed track 处理新轨迹确认，并在未匹配帧数超过阈值后删除轨迹。更早的 MDP Tracking 将每个目标的生命周期建模为 Markov Decision Process，显式设置 active、tracked、lost 和 inactive 等状态；在 lost 状态中，系统需要决定目标是继续保持 lost、重新关联为 tracked，还是转为 inactive。该工作说明轨迹生命周期可以作为在线 MOT 的核心决策对象。

针对轨迹碎片和漏检，StrongSORT++ 提出 AFLink 和 GSI。AFLink 处理 missing association，将短 tracklets 链接成更完整的轨迹；GSI 使用 Gaussian-smoothed interpolation 缓解 missing detection。这类方法可以减少 fragmentation，但其链路更接近后处理或插值。OC-SORT 的 recovery 和 re-update 机制则属于在线处理，它关注 lost period 后卡尔曼状态误差的修正。上述研究共同表明，短时漏检后的轨迹保留和恢复是已有 MOT 研究关注的问题。

现有生命周期管理方法仍有一个与本文问题相关的空白：很多方法用固定 hit 数确认新轨迹，用固定 buffer 保留 lost tracks，用固定 age 删除轨迹。低置信检测通常停留在关联输入层面，较少进一步成为需要多帧确认的候选轨迹状态。对于空海小目标，弱检测可能连续出现数帧但置信度不足，也可能在短时漏检后以低分形式重现。此时，跟踪器需要同时考虑低分观测的历史数量、平均分数、尺度和中心位置稳定性、连续 miss 情况，以及是否与某条 lost track 的历史位置相符。现有代表方法只覆盖其中一部分。

## 4. 现有研究不足与本文切入点

综合上述研究，现有方法已经较好解决了三类问题。第一，在线检测关联的基本框架已经成熟，SORT、DeepSORT、OC-SORT 和 BoT-SORT 等方法提供了运动、几何、外观和相机运动补偿线索。第二，低置信检测已经被证明可以用于 MOT，ByteTrack、ConfTrack 和 BoostTrack 系列分别从二阶段关联、低置信惩罚和置信度重加权角度进行了研究。第三，track confirmation、lost buffer、reactivation 和 termination 是 MOT 系统中的常见组成部分，MDP Tracking、DeepSORT、OC-SORT 和 StrongSORT++ 都提供了相关机制。

不足主要体现在问题组合上。检测不稳定条件下，低置信检测既可能是真目标，也可能是背景噪声；短时漏检后，目标可能以高置信或低置信形式重新出现。现有方法通常分别处理低分检测关联、运动恢复、外观重识别或轨迹后处理，较少把高/低置信检测、低置信候选确认、历史证据衰减、lost track 保留和 ID 继承放在同一个在线生命周期层中建模。特别是在 UAV/USV 空海小目标场景中，检索未找到足够强相关文献同时覆盖低置信检测、多帧延迟确认、短时漏检恢复和 lost ID 继承。

因此，MS-DC-ELT 更适合定位为一种面向检测不稳定小目标 MOT 的在线轨迹生命周期管理方法。其研究对象限定为：在相同 detector 输出下，研究高/低置信检测如何参与 candidate、active、lost 和 removed 状态之间的转移。该定位能够与 ByteTrack 的低分框关联、OC-SORT 的 observation-centric recovery、BoT-SORT 的 ReID/GMC 关联和 StrongSORT++ 的 post-linking/interpolation 区分开。

本文方法的边界也应明确。当前 MS-DC-ELT 不解决平台全局运动补偿、外观 ReID、ROI redetection、template lock、RGB/IR fusion 或离线全局 tracklet linking。它主要讨论检测置信度波动、短时漏检和低置信目标重现条件下，轨迹状态确认、保留、重激活和删除的规则设计及其收益与代价。实验结论也应按现有证据陈述：MS-DC-ELT v3 在 replay detections 下相对 ByteTrack 获得主要指标优势，并相对 OC-SORT/BoT-SORT 降低 FN；同时它带来更高 FP，且 HOTA、IDF1、AssA 和 IDSW 不支持整体优于强基线的结论。
