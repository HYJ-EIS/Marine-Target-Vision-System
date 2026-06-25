# 检测不稳定条件下小目标 MOT 的证据引导轨迹生命周期管理

**方法名称：** ELTTrack: Evidence-Guided Lifecycle Tracking

## 1. Introduction

无人机和无人船平台获取的海空视频为远距离目标监测提供了重要数据来源，但也给在线多目标跟踪（MOT）带来了持续的检测不稳定问题。远距离船只、无人机或其他小目标在图像中通常只占据少量像素，目标边界容易受到海天线、浪花、反光、云层和平台运动带来的背景变化影响。对于 tracking-by-detection 系统而言，这类目标并不总是形成连续、稳定的高置信检测；同一目标可能在相邻帧中表现为高置信检测、低置信检测或短期漏检。此时，跟踪器不仅需要完成当前帧检测框与轨迹之间的匹配，还需要在不完整观测下维持可解释的轨迹生命周期。

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

### 3.1 Overview of ELTTrack

ELTTrack 是一个用于 tracking-by-detection 的在线轨迹生命周期层。给定第 \(t\) 帧检测器输出，ELTTrack 首先通过双阈值筛选得到高置信检测集合和低置信检测集合，并对低置信独有检测执行去重、邻近门控和预算控制。随后，保留的检测被转换为带来源标签的 observation group，用于更新已有轨迹或生成候选轨迹。最后，证据驱动的状态转换决定每条轨迹处于 `LOW_CANDIDATE`、`CANDIDATE`、`ACTIVE`、`LOST` 或 `REMOVED` 状态；只有 `ACTIVE` 轨迹作为公开 MOT 输出。

ELTTrack 的轨迹状态由以下变量描述：

$$
\mathcal{T}_i^t=
\left(
z_i^t,\ b_i^t,\ v_i^t,\ \hat{b}_i^t,\ e_i^t,\ h_i^t,\ m_i^t,\ a_i^t,\ t_i^{last},\ r_i^t,\ \mathcal{S}_i^t,\ \mathcal{L}_i^t,\ \rho_i^t
\right),
$$

其中 \(z_i^t\) 是生命周期状态，\(b_i^t\) 是当前轨迹框，\(v_i^t\) 是由相邻匹配框中心位移得到的短期速度，\(\hat{b}_i^t\) 是由 \(b_i^{t-1}\) 和 \(v_i^{t-1}\) 得到的一步预测框，\(e_i^t\) 是证据得分，\(h_i^t\) 是命中次数，\(m_i^t\) 是漏检次数，\(a_i^t\) 是轨迹年龄，\(t_i^{last}\) 是最近一次实际检测命中的帧索引，\(r_i^t\) 是真实检测命中次数，\(\mathcal{S}_i^t\) 是观测来源历史，\(\mathcal{L}_i^t\) 是近期低置信检测历史，\(\rho_i^t\) 是移除态轨迹的短期签名。预测框由边界框和短期位移直接维护。生命周期状态集合为：

$$
z_i^t \in
\{\text{LOW\_CANDIDATE},\text{CANDIDATE},\text{ACTIVE},\text{LOST},\text{REMOVED}\}.
$$

`LOW_CANDIDATE` 表示由低置信观测生成、尚未公开输出的候选轨迹；`CANDIDATE` 表示普通候选轨迹；`ACTIVE` 表示已确认并可输出的轨迹；`LOST` 表示短期缺测后仍可恢复的已确认轨迹；`REMOVED` 表示已退出公开生命周期的轨迹。`REMOVED` 轨迹的短期签名仍用于 removed-track guard，以避免新候选在几何上接近刚移除轨迹时立即复用其 ID 语义。

### 3.2 Dual-Threshold Detection Screening

**设计动机。** 小目标检测流中的低置信框既可能包含真实目标，也可能包含背景杂波和重复检测。直接丢弃低置信框会增加漏检，直接使用所有低置信框又会增加候选噪声。因此，ELTTrack 将双阈值检测作为受控观测入口：高置信检测作为可靠输入，低置信检测必须经过去重、邻近筛选和预算控制后才能进入生命周期层。

**输入输出。** 输入为第 \(t\) 帧检测集合 \(D_t\)，每个检测 \(d\) 包含边界框 \(b(d)\)、置信度 \(s(d)\) 和类别标签。输出为带来源标签的 observation groups。检测筛选阶段的来源集合为 \(\mathcal{Q}_{det}=\{\text{high\_det}, \text{low\_det}\}\)，其中高置信观测主要用于普通候选生成和可靠状态更新，低置信观测用于弱证据累积、低置信候选生成和丢失轨迹重关联。3.4 节的重关联阶段会在恢复成功时追加 \(\text{reacquire}\) 来源标记；因此完整生命周期来源集合记为 \(\mathcal{Q}=\{\text{high\_det},\text{low\_det},\text{reacquire}\}\)。

**机制描述。** 给定低阈值 \(\tau_L\) 和高阈值 \(\tau_H\)，ELTTrack 首先保留低阈值以上检测，并从中划分高置信集合：

$$
D_t^L=\{d\in D_t\mid s(d)\geq \tau_L\},\qquad
D_t^H=\{d\in D_t^L\mid s(d)\geq \tau_H\}.
$$

低置信独有集合排除与高置信检测高度重叠的框，以减少同一目标的重复弱观测：

$$
D_t^{lo}=
\left\{
d\in D_t^L\setminus D_t^H
\mid
\max_{d'\in D_t^H}\operatorname{IoU}(b(d),b(d')) < \eta_{HL}
\right\}.
$$

随后，ELTTrack 对 \(D_t^{lo}\) 执行预算控制。低置信检测需满足最低低观测分数，并在已有 `ACTIVE` 或 `LOST` 轨迹附近；邻近关系由 IoU 或中心距离判定：

$$
\max_i \operatorname{IoU}(b(d),\hat{b}_i^t)\geq \eta_{prox}
\quad \text{or} \quad
\min_i \operatorname{dist}(b(d),\hat{b}_i^t)\leq \delta_{prox},
$$

其中 \(\hat{b}_i^t\) 是轨迹 \(i\) 在当前帧的一步预测框。通过筛选的低置信框按置信度排序，保留全局 Top-\(K\) 以及每条近邻轨迹附近的少量候选，并限制每帧最大低置信观测数。该预算避免密集背景弱响应主导候选池。

为形式化该预算，记通过最低分数和邻近门控的低置信集合为 \(\widetilde{D}_t^{lo}\)，轨迹 \(i\) 附近的低置信检测子集为 \(\mathcal{N}_t(i)\)。最终进入生命周期层的低置信输入为：

$$
B_t^{lo}=
\operatorname{Cap}_{K_{frame}}
\left(
\operatorname{TopK}_{K_g}(\widetilde{D}_t^{lo})
\cup
\bigcup_{i:z_i^t\in\{\text{ACTIVE},\text{LOST}\}}
\operatorname{TopK}_{K_i}(\mathcal{N}_t(i))
\right),
$$

其中 \(K_g\) 是全局预算，\(K_i\) 是每条近邻轨迹的预算，\(K_{frame}\) 是每帧低置信观测上限。

保留的高、低置信检测会被合并为 observation groups。若两个观测框的 IoU 超过合并阈值，则归入同一 group，并记录其来源历史和来源分数：

$$
G_t=\{g_j^t\},\qquad
Q_{det}(g_j^t)\subseteq \mathcal{Q}_{det}.
$$

Observation group 与 `LOW_CANDIDATE`、`CANDIDATE` 和 `ACTIVE` 轨迹进行几何匹配。有效配对需满足：

$$
\operatorname{IoU}(\hat{b}_i^t,g_j^t)\geq \eta_{assoc}
\quad \text{or} \quad
\operatorname{dist}(\hat{b}_i^t,g_j^t)\leq \delta_{assoc}.
$$

候选配对按几何分数排序并贪心匹配：

$$
A(i,j)=\operatorname{IoU}(\hat{b}_i^t,g_j^t)
+\lambda_c
\max\left(0,1-\frac{\operatorname{dist}(\hat{b}_i^t,g_j^t)}{\delta_{assoc}}\right).
$$

未匹配且包含高置信来源的 group 生成 `CANDIDATE`。未匹配且只包含低置信来源的 group 只有在达到低置信生成阈值时才生成 `LOW_CANDIDATE`，并且不会立即作为公开轨迹输出。

### 3.3 Evidence-Driven State Transition

**设计动机。** 在检测不稳定场景中，单帧置信度不足以决定轨迹是否可靠。一个真实小目标可能连续数帧只有低置信检测，也可能在短期内完全漏检；一个背景弱响应也可能偶然出现高分。因此，ELTTrack 使用跨帧证据累积和漏检衰减驱动状态转换，使候选确认、活跃至丢失转换、丢失保留和移除决策依赖历史观测。

**输入输出。** 输入为上一帧轨迹集合 \(\{\mathcal{T}_i^{t-1}\}\)、当前帧 observation groups \(G_t\) 以及匹配结果。输出为更新后的轨迹集合及生命周期事件，包括候选确认、活跃至丢失转换、丢失轨迹恢复、候选剪枝和移除。

**机制描述。** 对于被匹配的轨迹，ELTTrack 根据 observation group 中不同来源的置信度计算正证据增量。检测筛选阶段的 group 只包含 \(Q_{det}\) 来源；若 group 在重关联阶段恢复 `LOST` 轨迹，则其来源集合可扩展为 \(Q(g_j^t)\subseteq\mathcal{Q}\)：

$$
\Delta(g_j^t)=\sum_{q\in Q(g_j^t)}w_q s_q(g_j^t),
$$

其中 \(w_q\) 为来源权重，\(s_q(g_j^t)\) 为该来源对应的检测分数。证据得分采用衰减后的历史证据加当前正证据；未匹配轨迹则增加漏检次数，并施加负证据项：

$$
e_i^t=
\begin{cases}
\alpha e_i^{t-1}+\Delta(g_j^t), & \text{if track } i \text{ is matched},\\
\max(0,\alpha e_i^{t-1}-w_{neg}), & \text{otherwise}.
\end{cases}
$$

匹配到实际检测观测时，轨迹命中次数 \(h_i\) 增加，漏检次数 \(m_i\) 归零，真实检测命中次数 \(r_i\) 增加，并将 observation source 写入 \(\mathcal{S}_i\)。这里的实际检测观测指来源属于 \(\{\text{high\_det},\text{low\_det},\text{reacquire}\}\) 的 observation group，用于区别于未匹配帧中的预测保留。未匹配时，\(m_i\) 增加，证据随 \(\alpha\) 和 \(w_{neg}\) 衰减。观测来源历史用于区分候选是否曾获得高置信检测支持，低置信检测历史 \(\mathcal{L}_i\) 则记录低置信框的分数、面积和中心位置，用于低置信候选确认。

普通候选确认要求证据、总命中、真实检测命中和高置信来源同时满足条件，并且候选在当前帧被匹配：

$$
e_i^t\geq \theta_{conf},\quad
h_i^t\geq H_{conf},\quad
r_i^t\geq R_{conf},\quad
\text{high\_det}\in \mathcal{S}_i^t,\quad
m_i^t=0.
$$

`LOW_CANDIDATE` 使用更严格的确认条件。设 \(\mathcal{L}_i^t\) 为最近 \(W_{low}\) 帧内的低置信检测历史，则低置信候选需要满足低置信命中数量、平均分数、漏检次数、面积稳定性和中心步长稳定性：

$$
|\mathcal{L}_i^t|\geq H_{low},\quad
\frac{1}{|\mathcal{L}_i^t|}
\sum_{o\in \mathcal{L}_i^t}s(o)\geq \theta_{low},\quad
m_i^t\leq M_{low},
$$

$$
\frac{\max_{o\in\mathcal{L}_i^t}\operatorname{area}(o)}
{\min_{o\in\mathcal{L}_i^t}\operatorname{area}(o)}
\leq \rho_{area},\qquad
\max_k \|c(o_k)-c(o_{k-1})\|_2
\leq
\max(\delta_{assoc},\rho_{step}\cdot \operatorname{median}_k \|c(o_k)-c(o_{k-1})\|_2).
$$

其中 \(c(o)\) 为观测中心。该确认路径防止孤立低分框直接生成公开 track ID。

上述确认规则对应如下候选到活跃轨迹的状态转换：

$$
z_i^t=
\begin{cases}
\text{ACTIVE}, &
z_i^{t-1}=\text{CANDIDATE}
\land e_i^t\geq \theta_{conf}
\land h_i^t\geq H_{conf}
\land r_i^t\geq R_{conf}
\land \text{high\_det}\in \mathcal{S}_i^t
\land m_i^t=0,\\
\text{ACTIVE}, &
z_i^{t-1}=\text{LOW\_CANDIDATE}
\land C_{low}(i,t)
\land \neg A_{active}(i,t)
\land \neg I_{lost}(i,t).
\end{cases}
$$

其中 \(C_{low}(i,t)\) 表示上述低置信历史数量、平均分数、漏检、面积和中心步长条件同时满足；\(A_{active}(i,t)\) 表示该低置信候选与已有 `ACTIVE` 轨迹发生几何冲突；\(I_{lost}(i,t)\) 表示该低置信候选已在 3.4 节的继承机制中并入某条 `LOST` 轨迹。ELTTrack 的执行顺序是：先处理普通匹配和证据更新，再对满足 \(C_{low}\) 的 `LOW_CANDIDATE` 尝试 `LOST` 继承；继承成功时低置信候选被合并并转为 `REMOVED`，对应的 `LOST` 轨迹恢复为 `ACTIVE` 且沿用原 public ID；只有未继承、未与 `ACTIVE` 冲突且仍满足确认条件的 `LOW_CANDIDATE`，才独立晋升为新的 `ACTIVE` 并获得新的 public ID。因此，低置信候选的“继承旧 ID”和“独立生成新 ID”是按顺序判定的互斥路径。

已确认轨迹在连续漏检后进入 `LOST`。当 `ACTIVE` 轨迹的漏检次数超过容忍窗口 \(M_{active}\) 时，ELTTrack 将其转为 `LOST` 并保留有限帧数：

$$
z_i^t=
\begin{cases}
\text{LOST}, & z_i^{t-1}=\text{ACTIVE}\ \land\ m_i^t>M_{active},\\
\text{REMOVED}, & z_i^{t-1}=\text{LOST}\ \land\ t-t_i^{last}>T_{lost}.
\end{cases}
$$

`CANDIDATE` 和 `LOW_CANDIDATE` 若年龄超过候选窗口或证据低于剪枝阈值，则转为 `REMOVED`。`LOST` 若超过最大保留时间仍未恢复，也转为 `REMOVED`。进入 `REMOVED` 时，ELTTrack 保存短期 removed signature；后续新候选若与该签名满足冲突门控，则不会复用被移除轨迹的 public ID。

### 3.4 Geometry-Gated Track Reassociation

**设计动机。** 小目标短期漏检后可能重新出现，但在弱观测条件下直接重关联容易造成错误 ID 延续。ELTTrack 因此只允许满足几何和时间约束的 observation group 或已准备确认的低置信候选恢复 `LOST` 轨迹，并将恢复过程限制在在线保留窗口内。

**输入输出。** 输入为当前帧未匹配 observation groups、`LOST` 轨迹集合和已达到低置信确认条件的 `LOW_CANDIDATE` 集合。输出为恢复为 `ACTIVE` 的轨迹、被合并或移除的低置信候选，以及相应生命周期事件。

**机制描述。** 对于直接重关联，ELTTrack 按固定帧间隔检查 `LOST` 轨迹与未匹配 observation group。设 \(\hat{b}_i^t\) 为 `LOST` 轨迹预测框，\(g_j^t\) 为当前 observation group。候选配对必须满足 IoU 或中心距离门控：

$$
\operatorname{IoU}(\hat{b}_i^t,g_j^t)\geq \eta_{re}
\quad \text{or} \quad
\operatorname{dist}(\hat{b}_i^t,g_j^t)\leq \delta_{re}(i).
$$

中心距离阈值随轨迹框尺度受限放宽：

$$
\delta_{re}(i)=
\min\left(
\delta_{re}^{max},
\max(\delta_{re}^{base},\gamma_{scale}\max(w_i,h_i))
\right).
$$

候选重关联得分由 observation group 的证据增量给出：

$$
R(i,j)=\Delta(g_j^t).
$$

当 \(R(i,j)\geq \theta_{re}\) 时，`LOST` 轨迹可恢复为 `ACTIVE`，并保留原 public ID。若多个候选同时满足条件，按重关联得分从高到低贪心选择，保证一个 observation group 不同时恢复多条轨迹。

直接重关联采用“几何硬门控、证据排序”的规则。其原因是 \(g_j^t\) 是当前帧的实际 observation group，IoU 和中心距离已经用于排除几何不可信的配对；通过门控后，排序只使用 \(\Delta(g_j^t)\)，避免在单帧检测上重复放大几何项。相比之下，下面的低置信候选继承面对的是一个已累积多帧弱观测的候选轨迹，候选本身具有独立历史和运动趋势，因此需要在继承得分中同时考虑几何位置、运动一致性、低置信历史分数和 lost age。

对于低置信候选继承，ELTTrack 只考虑已经满足低置信确认条件的 `LOW_CANDIDATE`。给定 `LOST` 轨迹 \(i\) 和低置信候选 \(k\)，首先检查丢失时间：

$$
0\leq t-t_i^{last}\leq T_{inherit}.
$$

随后检查预测框与低置信候选框之间的 IoU 或中心距离门控。类别一致性和尺度一致性作为硬门控使用：当配置要求类别匹配时，类别不一致的候选被拒绝；尺度一致性由 IoU 门控和尺度相关的中心距离阈值共同约束。通过这些门控后，继承得分定义为：

$$
I(i,k)=
\beta_{iou}\operatorname{IoU}(\hat{b}_i^t,b_k^t)
+\beta_c\left(1-\frac{\operatorname{dist}(\hat{b}_i^t,b_k^t)}{\delta_{inherit}}\right)_+
+\beta_v V(i,k)
+\beta_s \bar{s}_k
+\beta_r\left(1-\frac{t-t_i^{last}}{T_{inherit}}\right)_+,
$$

其中 \(V(i,k)\) 为中心位移方向一致性，\(\bar{s}_k\) 为低置信候选近期平均检测分数，\((x)_+=\max(0,x)\)。若 \(I(i,k)\geq \theta_{inherit}\)，该低置信候选合并到 `LOST` 轨迹，后者恢复为 `ACTIVE` 并沿用原 public ID；被合并的低置信候选转为 `REMOVED`。

Removed-track guard 用于处理已移除轨迹附近的新候选。设 \(\mathcal{R}_t\) 为最近 \(T_{guard}\) 帧内的 removed signatures。若新候选 \(u\) 与某个 removed signature \(r\) 满足：

$$
\operatorname{IoU}(b(u),b(r))\geq \eta_{guard}
\quad \text{or} \quad
\operatorname{dist}(b(u),b(r))\leq \delta_{guard},
$$

则 ELTTrack 记录冲突并分配新的内部候选身份，不让该候选直接复用被移除轨迹的 public ID。该守护机制把重关联限制在 `LOST` 保留窗口内，避免已经退出生命周期的轨迹在背景弱响应附近被反复恢复。
