# 检测不稳定条件下小目标 MOT 的证据引导轨迹生命周期管理

**方法名称：** ELTTrack: Evidence-Guided Lifecycle Tracking

## 1. Introduction

无人机和无人船平台获取的海空视频为远距离目标监测提供了重要数据来源，但也给在线多目标跟踪（MOT）带来了持续的检测不稳定问题。远距离船只或空中小目标在图像中通常只占据少量像素，目标边界容易受到海天线、浪花、反光、云层和平台运动带来的背景变化影响。对于 tracking-by-detection 系统而言，这类目标并不总是形成连续、稳定的高置信检测；同一目标可能在相邻帧中表现为高置信检测、低置信检测或短期漏检。此时，跟踪器不仅需要完成当前帧检测框与轨迹之间的匹配，还需要在不完整观测下维持可解释的轨迹生命周期。

Tracking-by-detection 是在线 MOT 中广泛使用的范式。SORT 使用卡尔曼滤波和 IoU 匹配建立了高效的在线跟踪流程 [Bewley et al., 2016]；DeepSORT 在此基础上加入外观度量以改善遮挡后的身份保持 [Wojke et al., 2017]；OC-SORT 通过 observation-centric 的运动更新缓解短期缺测后的状态误差 [Cao et al., 2023]；BoT-SORT 进一步结合运动、外观和相机运动补偿线索增强关联 [Aharon et al., 2022]。同时，ByteTrack 证明低置信检测并不应被简单丢弃，其二阶段关联能够利用低分框恢复部分真实目标 [Zhang et al., 2022]；ConfTrack、BoostTrack 和 Hybrid-SORT 等方法也从置信度惩罚、相似度调整和多线索关联角度研究了检测分数的作用 [Jung et al., 2024; Stanojevic and Todorovic, 2024; Yang et al., 2024]。

上述方法推动了在线关联和低置信检测利用的发展，但检测不稳定小目标场景仍暴露出一个更具体的生命周期问题。低置信框可能是真实目标，也可能是背景杂波或高置信框附近的重复响应；已确认轨迹可能在短期漏检后重新以高置信或低置信形式出现；未确认轨迹如果过早输出会增加虚警，如果过早删除又会增加漏检和轨迹碎片。因此，跟踪器需要统一决定弱观测是否可以生成候选轨迹、候选轨迹何时具备足够证据成为公开轨迹、已确认轨迹在连续漏检后应保留多久、以及重新出现的观测是否可以延续已有 track ID。现有方法分别处理了低分检测关联、运动恢复、固定 lost buffer 或离线轨迹段连接等问题，但对高置信检测、低置信检测和连续漏检交替出现时的在线生命周期决策仍缺少集中建模。

本文提出 ELTTrack，一种面向检测不稳定小目标 MOT 的证据引导生命周期跟踪方法。ELTTrack 将检测器输出划分为高置信检测和低置信检测，并通过去重、邻近门控和预算控制把低置信检测转化为受控观测输入；随后，跟踪器维护 `LOW_CANDIDATE`、`CANDIDATE`、`ACTIVE`、`LOST` 和 `REMOVED` 五类生命周期状态，通过证据得分、命中次数、漏检次数、真实检测命中次数和观测来源历史驱动状态转换；最后，几何门控重关联用于约束短期丢失轨迹的恢复，避免弱观测直接造成不受限制的 ID 继承。该设计将低置信检测的使用从单帧关联扩展为跨帧候选确认、丢失保留和重关联决策。

本文的贡献如下：

1. 针对小目标 MOT，提出将检测不稳定性显式表述为轨迹生命周期决策问题。该建模把高置信检测、低置信检测和连续漏检定义为影响轨迹状态的观测证据，为后续算法设计提供统一的问题边界。
2. 提出证据引导的生命周期跟踪方法 ELTTrack，包含双阈值检测筛选和证据驱动的状态转换，使弱观测在进入公开轨迹前必须经过多帧证据约束。
3. 设计几何门控的重关联机制，通过 IoU、中心距离、尺度一致性和时间约束限制丢失轨迹恢复，降低弱观测条件下错误 ID 延续的风险。

## 2. Related Work

### 2.1 Online MOT under Tracking-by-Detection

Tracking-by-detection 将多目标跟踪分解为逐帧检测和跨帧关联两个步骤，是在线 MOT 中常用且易部署的范式。SORT 以卡尔曼滤波预测轨迹位置，并使用 IoU 代价和 Hungarian 匹配完成检测框到轨迹的分配 [Bewley et al., 2016]。DeepSORT 在 SORT 基础上加入深度外观特征和匹配级联，以降低遮挡和相似运动条件下的 ID switch [Wojke et al., 2017]。这类方法说明，运动预测、几何重叠和轨迹生命周期规则构成了在线 MOT 的基础流程。

近年来的 SORT 系列方法进一步增强了在线关联能力。OC-SORT 关注缺测或遮挡后卡尔曼预测误差累积的问题，通过 observation-centric re-update 和 observation-centric momentum 改善运动状态恢复 [Cao et al., 2023]。BoT-SORT 结合改进的卡尔曼状态、外观特征和 camera-motion compensation，提升了多行人场景中的关联稳定性 [Aharon et al., 2022]。Hybrid-SORT 将速度方向、检测置信度和高度状态等线索纳入关联代价 [Yang et al., 2024]。这些方法主要通过运动、几何、外观或相机补偿改善检测框与已有轨迹之间的匹配，但其核心问题仍是短期关联代价设计；对于检测置信度波动和连续漏检共同影响下的确认、保留、重关联和移除决策，通常仍依赖固定阈值或局部规则。

### 2.2 Low-Confidence Detections in MOT

低置信检测处理是 tracking-by-detection 研究中的重要方向。ByteTrack 指出，低分检测中仍包含真实目标，如果在高置信检测关联后继续使用低置信检测匹配未关联轨迹，可以减少因直接丢弃低分框造成的漏检和轨迹碎片 [Zhang et al., 2022]。这一思想改变了低置信检测只作为噪声处理的做法，使检测分数成为关联流程中的显式输入。

后续方法进一步研究检测置信度在关联中的使用方式。ConfTrack 面向 Kalman filter-based online MOT，利用检测置信度设计低置信惩罚和级联匹配，并区分 tentative tracks 与 confirmed tracks [Jung et al., 2024]。BoostTrack 和 BoostTrack++ 使用 detection-tracklet confidence 调整相似度，并结合 shape、Mahalanobis distance 和 soft BIoU 等线索提高真阳性检测的选择能力 [Stanojevic and Todorovic, 2024; TODO: BoostTrack++ citation]。Hybrid-SORT 也将 confidence 作为弱线索之一参与在线关联 [Yang et al., 2024]。

这些工作说明低置信检测可以在匹配、惩罚或重加权后参与 MOT，但多数方法仍将其作为当前帧或短时关联输入。对于小目标视频，低置信检测还需要被判断为背景噪声、重复响应、候选轨迹证据或丢失轨迹恢复证据。也就是说，低分框不仅影响一次匹配，还影响候选轨迹是否创建、是否确认、是否继承旧 ID 以及是否应被拒绝。ELTTrack 将低置信检测视为需要跨帧过滤、累积、确认或拒绝的生命周期证据，而不是仅用于一次二阶段关联。

### 2.3 Track Lifecycle Management and Reassociation

轨迹生命周期管理是在线 MOT 系统的基础组成部分。SORT 系列方法通常包含新轨迹生成、未确认轨迹确认、丢失轨迹保留和超时删除等规则。DeepSORT 使用 tentative 和 confirmed 状态管理新轨迹，并在未匹配帧数超过阈值后删除轨迹 [Wojke et al., 2017]。MDP Tracking 更早地将目标生命周期建模为 Markov Decision Process，显式考虑 active、tracked、lost 和 inactive 等状态之间的决策 [Xiang et al., 2015]。这些方法表明，轨迹是否应被公开输出、保留或移除本身就是在线 MOT 的核心问题。

轨迹重关联和缺测修复也已有大量研究。OC-SORT 的恢复关联关注 lost period 后的运动状态修正 [Cao et al., 2023]。StrongSORT++ 提出 AFLink 和 GSI，分别处理轨迹段连接和 Gaussian-smoothed interpolation，以缓解 missing association 与 missing detection [Du et al., 2023]。这些机制可以减少轨迹碎片，但 AFLink 和 GSI 更接近离线链接或后处理插值；固定 hit 数、固定 age 阈值和固定 lost buffer 也难以表达低置信观测在多帧内逐步积累或被拒绝的过程。

因此，现有生命周期方法虽已包含试探、确认、丢失和移除状态，但较少把双阈值检测输入、低置信候选延迟确认、证据累积、漏检衰减、丢失保留和在线几何重关联放入同一个生命周期层。ELTTrack 的目标不是替代外观关联或全局补偿，而是在检测器输出之后、公开轨迹输出之前，为检测不稳定小目标建立可解释的在线状态转换规则。

### 2.4 Positioning of ELTTrack

ELTTrack 与上述方法的区别在于研究对象从单次检测关联转向生命周期级决策。不同于 ByteTrack 主要将低分框用于未匹配轨迹的二阶段关联，ELTTrack 对低置信检测先进行去重、邻近筛选和预算控制，再允许其作为候选确认或丢失恢复的证据。不同于 OC-SORT 重点修正缺测后的运动状态，ELTTrack 关注高置信、低置信和漏检序列如何共同改变轨迹状态。不同于 BoT-SORT 通过外观和相机补偿增强关联，ELTTrack 的生命周期层仅依赖检测分数、边界框几何和状态历史。不同于 StrongSORT++ 的离线轨迹段链接或插值，ELTTrack 在在线过程中完成候选确认、丢失保留、重关联和移除。

## 3. Method

本文提出 ELTTrack，一个基于 tracking-by-detection 范式的在线多目标跟踪方法。其核心设计是在检测器输出之后增加一个证据引导的轨迹生命周期层。该层将高置信检测、低置信检测和短期漏检统一转化为轨迹状态转换依据，用于决定候选生成、候选确认、活跃轨迹保留、丢失轨迹恢复和轨迹移除。

### 3.1 Overview of ELTTrack

给定第 $t$ 帧检测结果，ELTTrack 首先采用双阈值策略将检测划分为高置信检测和低置信检测。高置信检测直接作为可靠观测进入跟踪器；低置信检测仅在满足去重、空间邻近和数量预算约束后进入候选池。随后，有效观测与已有轨迹进行几何关联，并根据匹配结果更新轨迹证据。轨迹是否输出不由单帧检测分数直接决定，而由生命周期状态决定。

ELTTrack 维护五类轨迹状态：

$$
z_i^t \in \{\text{LOW\_CANDIDATE},\text{CANDIDATE},\text{ACTIVE},\text{LOST},\text{REMOVED}\}.
$$

- `LOW_CANDIDATE`：由低置信观测生成、尚未确认的候选轨迹。
- `CANDIDATE`：由可靠观测生成的普通候选轨迹。
- `ACTIVE`：已确认并参与输出的轨迹。
- `LOST`：短时漏检后仍保留重关联机会的轨迹。
- `REMOVED`：已终止维护的轨迹。

最终输出仅包含 `ACTIVE` 轨迹。除生命周期状态外，每条轨迹还维护证据得分、命中次数、漏检次数、真实检测命中次数、观测来源历史和低置信观测历史。这些变量用于支撑状态转换，但不需要在正文中逐项展开为完整状态向量。

### 3.2 Dual-Threshold Detection Screening

小目标场景中，低置信检测往往同时包含真实目标和背景杂波。若直接丢弃低置信检测，容易造成断轨；若全部保留，则会引入大量虚假候选。因此，ELTTrack 采用受控观测入口机制：高置信检测作为可靠输入，低置信检测必须经过去重、邻近筛选和预算控制。

给定第 $t$ 帧检测集合 $D_t$，每个检测 $d$ 包含边界框 $b(d)$、置信度 $s(d)$ 和类别标签。给定低阈值 $\tau_L$ 和高阈值 $\tau_H$，检测集合被划分为：

$$
D_t^L=\{d\in D_t\mid s(d)\geq \tau_L\},\qquad
D_t^H=\{d\in D_t^L\mid s(d)\geq \tau_H\}.
$$

其中，$D_t^H$ 是高置信检测集合。低置信独有检测从 $D_t^L\setminus D_t^H$ 中获得，并去除与高置信检测高度重叠的框：

$$
D_t^{lo}= \left\{ d\in D_t^L\setminus D_t^H \mid
\max_{d'\in D_t^H}\operatorname{IoU}(b(d),b(d')) < \eta_{HL}
\right\}.
$$

随后，ELTTrack 对低置信独有检测执行三类筛选：

- **最低分数筛选**：去除分数过低的弱响应。
- **轨迹邻近筛选**：仅保留靠近已有 `ACTIVE` 或 `LOST` 轨迹的低置信框。
- **预算控制**：保留全局 Top-$K$ 低置信框，并限制每帧最大低置信观测数。

通过筛选的检测被合并为有效观测集合 $G_t=\{g_j^t\}$。检测阶段的观测来源为 `high_det` 和 `low_det`；若某个观测在后续重关联阶段恢复了 `LOST` 轨迹，则额外记录 `reacquire` 来源。有效观测与 `LOW_CANDIDATE`、`CANDIDATE` 和 `ACTIVE` 轨迹通过 IoU 或中心距离进行关联。未匹配的高置信观测生成 `CANDIDATE`，未匹配且满足生成条件的低置信观测生成 `LOW_CANDIDATE`，但不会立即输出为公开轨迹。

### 3.3 Evidence-Driven State Transition

ELTTrack 的状态转换按帧执行。每一帧结束时，跟踪器先判断轨迹是否获得了实际观测，再根据证据、命中次数和漏检次数更新轨迹状态。这里的实际观测包括高置信检测、低置信检测，以及重关联成功后的 `reacquire` 观测。

对一条轨迹而言，当前帧只有两类更新结果：

- **匹配成功**：轨迹获得一个观测，证据得分增加，命中次数增加，漏检次数清零，并记录观测来源。
- **匹配失败**：轨迹没有观测，证据得分衰减，漏检次数增加。

证据得分的更新写作：

$$
e_i^t=
\begin{cases}
\alpha e_i^{t-1}+\Delta(g_j^t), & \text{matched},\\
\max(0,\alpha e_i^{t-1}-w_{neg}), & \text{unmatched},
\end{cases}
$$

其中 $e_i^t$ 为轨迹证据得分，$\alpha$ 为历史衰减系数，$w_{neg}$ 为漏检惩罚项。$\Delta(g_j^t)$ 表示当前观测带来的正证据；高置信观测的权重更高，低置信观测只提供弱证据。因此，单个低分框不会直接决定轨迹状态，它只能在多帧内逐步累积影响。

在此基础上，五类状态按以下规则转换：

- `CANDIDATE -> ACTIVE`：普通候选需要同时满足证据得分、命中次数、真实检测命中次数和高置信来源约束。该规则保证公开输出的轨迹至少得到过可靠检测支持。
- `LOW_CANDIDATE -> ACTIVE`：低置信候选需要先在短时间窗口内反复出现，并满足平均分数、漏检次数、面积变化和中心位移约束。达标后，它先尝试继承某条 `LOST` 轨迹；继承成功则沿用旧 ID，继承失败且不与现有 `ACTIVE` 轨迹冲突时，才作为新 ID 输出。
- `ACTIVE -> LOST`：已确认轨迹连续漏检超过容忍窗口后，不立即删除，而是进入 `LOST`，等待后续重关联。
- `LOST -> ACTIVE`：丢失轨迹若在保留窗口内重新匹配到可信观测，则恢复为 `ACTIVE` 并沿用原 ID。
- `CANDIDATE/LOW_CANDIDATE/LOST -> REMOVED`：候选长期无法确认，或丢失轨迹超过保留窗口，都会转为 `REMOVED`。

上述顺序使低置信候选在生成新 ID 之前先接受旧轨迹继承检查。只有无法继承 `LOST` 轨迹时，它才可能成为新的 `ACTIVE` 轨迹，从而减少短期漏检造成的 ID 断裂。

### 3.4 Geometry-Gated Track Reassociation

重关联用于判断一条 `LOST` 轨迹是否应恢复为原来的 ID。ELTTrack 先用几何关系筛除位置明显不合理的候选，再评估观测证据。

对于 `LOST` 轨迹的预测框 $\hat{b}_i^t$ 和当前未匹配观测 $g_j^t$，只有满足下列条件之一时，二者才会进入重关联候选集合：

$$
\operatorname{IoU}(\hat{b}_i^t,g_j^t)\geq \eta_{re}
\quad \text{or} \quad
\operatorname{dist}(\hat{b}_i^t,g_j^t)\leq \delta_{re}.
$$

通过几何门控后，ELTTrack 按观测证据对候选进行排序。几何关系决定候选是否可被接受，证据得分决定多个候选之间的优先级，从而避免仅凭一个低分检测恢复旧 ID。

重关联包含两条路径：

- **直接恢复**：当前帧的未匹配观测与某条 `LOST` 轨迹通过几何门控，并且证据足够，则该 `LOST` 轨迹恢复为 `ACTIVE`，沿用原 ID。
- **低置信候选继承**：一个已经积累多帧弱证据的 `LOW_CANDIDATE`，如果与某条 `LOST` 轨迹在时间、类别、尺度、位置和运动方向上都一致，则该候选并入旧轨迹，旧轨迹恢复为 `ACTIVE`，候选本身转为 `REMOVED`。

若一条轨迹已经进入 `REMOVED`，ELTTrack 不再允许它被普通重关联恢复。系统只保留其短期空间签名，用于阻止附近的弱响应马上复用旧 ID。这一 removed-track guard 将 ID 继承限制在 `LOST` 保留窗口内，避免已终止轨迹被反复激活。
