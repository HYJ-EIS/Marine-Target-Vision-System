# 方法 V2：MS-DC-ELT v3

## 3. 方法

MS-DC-ELT v3 是一个用于 tracking-by-detection 的在线轨迹生命周期层，面向检测置信度不稳定的场景。它接收检测器输出的目标框，维护未确认轨迹和已确认轨迹，并且默认只输出已确认的 active tracks。该方法的输入和状态更新仅依赖边界框几何关系、检测置信度和生命周期状态；不使用外观 ReID、全局运动补偿、ROI 重检测、模板匹配、RGB/IR 融合或离线 tracklet linking。

该设计采用偏召回的权衡。在 UAV/USV 空海小目标视频中，同一目标可能在高置信检测、低置信检测和短时漏检之间切换。因此，MS-DC-ELT 保留额外的低置信观测，使其能够进入生命周期层；同时通过证据累积、确认门控和删除规则延迟其对 public track ID 的影响。该设计目标是在弱检测被直接丢弃时减少 false negatives，但当低置信观测噪声较多时，也可能增加 false positives 或削弱身份相关指标。

### 3.1 双阈值观测预算

MS-DC-ELT 使用一次低阈值检测，并从同一检测输出中派生高置信检测子集。对于第 \(t\) 帧，令 \(D_t^{L}\) 表示置信度高于低阈值 \(\tau_L\) 的检测集合，令 \(D_t^{H}\) 表示置信度高于常规阈值 \(\tau_H\) 的检测集合：

$$
D_t^{H}=\{d\in D_t^{L}\mid s(d)\geq \tau_H\}.
$$

低置信独有检测集合会排除与高置信检测重叠的低阈值框：

$$
D_t^{lo}=
\left\{
d\in D_t^{L}
\mid
\max_{d'\in D_t^{H}}\operatorname{IoU}(d,d') < \eta_{HL}
\right\}.
$$

低置信独有检测比高置信检测包含更多噪声。为避免生命周期状态机被密集背景框主导，v3 根据置信度和与已有轨迹的邻近关系设置观测预算。一个低置信独有检测只有在分数超过最低低观测阈值，并且在存在 active 或 lost tracks 时靠近至少一条 active 或 lost track，才会被保留：

$$
\max_i\operatorname{IoU}(d,\hat{b}_i)\geq \eta_{prox}
\quad \text{or} \quad
\min_i\operatorname{dist}(d,\hat{b}_i)\leq \delta_{prox}.
$$

其中 \(\hat{b}_i\) 是轨迹 \(i\) 的一步预测框。通过上述筛选后，剩余低置信独有框按置信度排序。跟踪器保留固定数量的全局 Top-\(K\) 低置信框，并为每条 active 或 lost track 额外保留最近的低置信框。该规则为低置信检测提供受控入口，避免每个弱检测都直接生成或更新轨迹。

每个保留检测会被转换为带来源标签的 observation group，其中 \(q\in\{\text{high\_det},\text{low\_det}\}\)。Observation group 与已有 `LOW_CANDIDATE`、`CANDIDATE` 和 `ACTIVE` 轨迹通过几何关系关联：

$$
\operatorname{IoU}(\hat{b}_i,g_j)\geq \eta_{assoc}
\quad \text{or} \quad
\operatorname{dist}(\hat{b}_i,g_j)\leq \delta_{assoc}.
$$

满足门控的候选配对按以下分数排序：

$$
A(i,j)=\operatorname{IoU}(\hat{b}_i,g_j)
+\lambda_c\max\left(0,1-\frac{\operatorname{dist}(\hat{b}_i,g_j)}{\delta_{assoc}}\right),
$$

并按分数从高到低贪心匹配。未匹配且包含高置信检测的 observation group 会生成普通 `CANDIDATE` 轨迹。未匹配的低置信独有 observation group 只有在通过低分数和邻近关系筛选后，才允许生成 `LOW_CANDIDATE` 轨迹。这些低置信候选是未确认轨迹，不会作为 MOT 结果输出。

### 3.2 基于证据的状态转移

每条轨迹维护一个运动状态和一个生命周期状态描述：

$$
\mathcal{P}_i=\{z_i,e_i,h_i,m_i,r_i,\mathcal{S}_i,\mathcal{H}_i^{low}\},
$$

其中 \(z_i\) 是生命周期状态，\(e_i\) 是证据分数，\(h_i\) 是命中次数，\(m_i\) 是 miss 次数，\(r_i\) 是真实检测命中次数，\(\mathcal{S}_i\) 是观测来源历史，\(\mathcal{H}_i^{low}\) 是近期低置信检测历史。生命周期状态定义为：

$$
z_i\in
\{\text{LOW\_CANDIDATE},\text{CANDIDATE},\text{ACTIVE},\text{LOST},\text{REMOVED}\}.
$$

只有 `ACTIVE` 轨迹是已确认输出轨迹。`LOW_CANDIDATE` 和 `CANDIDATE` 是未确认轨迹。`LOST` 是短时缺测后仍被保留的已确认轨迹。`REMOVED` 是已退役轨迹，其近期签名仍可用于避免 ID 立即复用。

对于 observation group \(g_t\)，正证据增量由不同来源的加权置信度构成：

$$
\Delta(g_t)=\sum_{q\in Q(g_t)}w_qs_q.
$$

证据更新同时包含正证据累积和 missed-frame 衰减：

$$
e_t=
\max\left(
0,
\alpha e_{t-1}
+\mathbb{I}_{match}\Delta(g_t)
-(1-\mathbb{I}_{match})w_{neg}
\right).
$$

其中，当轨迹在当前帧被匹配时，\(\mathbb{I}_{match}=1\)，否则为 0。被匹配的轨迹会增加 hit count、重置 miss count，并将观测来源加入 source history。未匹配轨迹会增加 miss count，并通过负证据项衰减证据分数。

普通 `CANDIDATE` 只有在证据充分、总命中数足够、真实检测命中数足够，并且来源历史中至少包含一次高置信检测时，才允许转为 `ACTIVE`：

$$
e_t\geq \theta_{conf},\quad
h_t\geq H_{conf},\quad
r_t\geq R_{conf},\quad
\text{high\_det}\in\mathcal{S}_i.
$$

该 candidate 还必须在当前帧被匹配。这个确认条件防止轨迹仅凭低置信观测累积就成为公开输出轨迹。

`LOW_CANDIDATE` 为重复出现但置信度较低的检测提供更严格的确认路径。低置信候选只有在近期低置信检测历史中包含足够观测、平均置信度足够、miss 次数较少，并且面积与中心位置变化保持稳定时，才允许被确认。令 \(\mathcal{H}_i^{low}(t)\) 表示近期低置信检测窗口，则其基础条件为：

$$
|\mathcal{H}_i^{low}(t)|\geq H_{low},\quad
\frac{1}{|\mathcal{H}_i^{low}(t)|}\sum_{o\in\mathcal{H}_i^{low}(t)}s(o)
\geq \theta_{low},\quad
m_t\leq M_{low}.
$$

该窗口内的面积比例和最大中心步长还必须低于配置上限。该低置信路径用于表示连续弱检测，但当前消融和诊断结果不支持将其写成主要性能收益的独立来源。

如果 `CANDIDATE` 或 `LOW_CANDIDATE` 存活时间过长，或证据分数低于剪枝阈值，则转为 `REMOVED`。`ACTIVE` 轨迹在 miss 次数超过短时容忍窗口后转为 `LOST`。`LOST` 轨迹只在有限帧数内保留；若未被恢复，则转为 `REMOVED`。这一状态转移过程使 MS-DC-ELT 成为轨迹生命周期管理方法，而不是单纯的逐帧检测关联方法。

### 3.3 轨迹恢复机制的边界

MS-DC-ELT v3 还包含仅基于几何关系的 lost track 恢复机制。直接重获会以固定帧间隔尝试将未匹配 observation group 重新关联到 `LOST` 轨迹，使用 IoU、中心距离和证据分数作为门控。低置信候选继承则尝试将一个已满足条件的 `LOW_CANDIDATE` 合并到附近的 `LOST` 轨迹，从而保留 lost track 的 public ID。实现中还会删除与 active track 冲突的低置信候选，并保留短期 removed-track 签名以避免 ID 立即复用。

这些机制在本文中不作为主要贡献。在当前 full v3 诊断中，直接重获没有成功尝试或机会，低置信候选继承也没有机会。低置信候选路径会创建并确认少量轨迹，但确认 precision 较低，并且 `no_low_candidate` 消融没有损害已评估指标。这些结果说明，在缺少外观特征或全局运动补偿的情况下，仅依赖几何关系在轨迹已进入 lost 状态后进行恢复并不稳定。当前证据支持的 v3 机制是短时缺测前后连续的生命周期更新，而不是对较长 fragmentation 后可靠 ID 恢复的强声明。

### 3.4 实现细节

实验中的正式变体为 `v3_candidate_topk_no_roi_no_motion`，报告名称为 MS-DC-ELT v3。默认配置如下。

| 组件 | 默认值 |
| --- | --- |
| 可见光低/高阈值 | \(\tau_L=0.18,\ \tau_H=0.50\) |
| 红外低/高阈值 | \(\tau_L=0.22,\ \tau_H=0.65\) |
| 高低框去重 IoU | \(\eta_{HL}=0.5\) |
| 低观测最低分数 | 0.25 |
| 低观测邻近门控 | IoU \(0.01\) 或中心距离 \(240\) |
| 低观测预算 | 全局 Top 32，每条轨迹最近 1 个，每帧最多 64 个 |
| 关联门控 | IoU \(0.2\) 或中心距离 \(80\) |
| 关联中心距离权重 | \(\lambda_c=0.5\) |
| 证据衰减系数 | \(\alpha=0.85\) |
| 来源权重 | high 1.3，low 1.0，reacquire 1.0 |
| 负证据权重 | \(w_{neg}=0.5\) |
| 普通候选确认 | score 2.5，hits 4，真实检测 hits 4，要求高置信检测历史 |
| 低置信候选确认 | 8 帧内 hits 5，平均分 0.22，最大 miss 1 |
| 低置信候选稳定性 | 面积比例 1.8，中心步长因子 3.0 |
| Active missing patience | 2 帧 |
| Lost max age | 40 帧 |
| 直接重获 | 间隔 5，score 1.5，IoU 0.05，中心距离基准 160，尺度因子 4，最大中心距离 240 |
| 低置信继承 | score 0.40，IoU 0.02，中心距离 220，最大 lost age 120 |
| Removed signature guard | 80 帧 |
| 输出过滤 | 仅输出 `ACTIVE`，最大真实检测 age 3，最小输出框尺寸 12 |

跟踪器默认只输出 active tracks。正式配置启用 output NMS 和 real-detection-age filtering，但当前消融显示它们属于辅助实现选择，而不是主要性能来源。
