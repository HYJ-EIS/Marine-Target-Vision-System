## 1. 核心状态定义 (States)

跟踪器包含以下 5 种核心状态及其流转含义：

- **`LOW_CANDIDATE`**：由 `low_det`（低置信度检测）生成、未确认、等待低阈值时序确认，或继承 `LOST` 状态的 `public_id`。
- **`CANDIDATE`**：由 `high_det`（高置信度检测）或非 low-only 观测生成、未确认、等待证据确认。
- **`ACTIVE`**：已确认的轨迹；分配了 `public_id`；默认情况下**唯一可输出**的状态。
- **`LOST`**：`ACTIVE` 状态超过丢失容忍度（missing patience）后进入；保留该状态用于重新捕获（reacquire）或供 `LOW_CANDIDATE` 继承。
- **`REMOVED`**：剪枝、超时、合并或达到容量上限被淘汰后的退休状态；不再参与普通的关联和输出。

## 2. 状态分配逻辑 (Assign State)

这是跟踪器最核心的逻辑层。其本质是基于 4 个核心输入参数，结合阈值进行状态判定。

### 2.1 核心输入参数 (Inputs)

- **$p_t$** (`evidence_score`)：历史上看，像不像真目标。
- **$v_t$** (`observed_score`)：当前这一帧出现了吗。
- **$m_t$** (`missing_score`)：若当前没有出现，失联程度是否可接受。
- **$dt$**：`frame_idx - last_seen`（当前帧与上次可见帧的时间差）。

### 2.2 底层分数计算逻辑解析

#### A. 证据分数 $p_t$ 的推导与计算

$p_t$ 的计算依赖于当前帧的**正证据增量** ($s_t$) 和**历史证据** ($p_{t-1}$)。

**步骤 1：计算正证据增量 $s_t$ (`positive_score`)**
当前 track 匹配到了一个检测组 `groups[group_idx]`，根据检测来源和置信度，计算本帧给该轨迹增加的加权证据增量：
$$s_t = \operatorname{round}\left( \sum_{k \in \text{sources}} w_k \cdot c_k \right)$$
其中：
- $s_t$：当前帧的 `positive_score`。
- $k$：检测来源（如 `high_det`, `low_det`, `reacquire`）。
- $c_k$：该来源的有效分数（`obs.score * obs.reliability`，同来源取最大值）。
- $w_k$：来源对应的证据权重（默认 `high=1.3`, `low=1.0`, `reacquire=1.0`）。

> **举例说明：**
> 假设当前 group 里包含：`high_det` (0.80) 和 `low_det` (0.35)。
> 则 $s_t = \operatorname{round}(1.3 \cdot 0.80 + 1.0 \cdot 0.35) = \operatorname{round}(1.39)$。

**步骤 2：更新历史证据分数 $p_t$**

连续检测到目标时，证据累积；长时间没检测到时，乘以系数 $\alpha$ 衰减。
$$p_t = \operatorname{round}\left(\alpha \cdot p_{t-1} + s_t\right)$$
其中：
- $p_t$：当前帧更新后的 `track.evidence_score`。
- $p_{t-1}$：更新前的历史 `track.evidence_score`。
- $\alpha$：历史证据保留系数（`MSDC_EVIDENCE_ALPHA`，默认 `0.85`）。

**步骤 3：$p_t$ 在不同状态下的特殊处理规则**
- **常规更新**：`p_t = float(track.evidence_score)` (适用于 `CANDIDATE -> ACTIVE` 或 `ACTIVE -> LOST`)。
- **LOST 被重新捕获**：`p_t = max(track.evidence_score, MSDC_CONFIRM_SCORE + 1e-6)`。强制让 $p_t$ 略高于确认阈值，防止目标找回但历史分数太低导致无法回到 `ACTIVE`。
- **LOST 未匹配且超时**：`p_t = 0.0`。人为归零使其满足进入 `REMOVED` 的条件。未超时则保留真实分数。

#### B. 观测分数 $v_t$ 与 丢失分数 $m_t$ 的计算
- **$v_t$ 计算**：如果 `confirm_ready` 且 `matched`，则 $v_t = 1.0$，否则为 $0.0$。
- **$m_t$ 计算**：如果 `confirm_ready` 为 `True`，则 $m_t = 0.0$（允许升至 ACTIVE）；若为 `False`，则 $m_t = \text{theta\_m}$（保持原状态）。

### 2.3 判定阈值 (Thresholds)
- **`theta_p`** (`MSDC_CONFIRM_SCORE`)：确认阈值。
- **`theta_p_low`** (`MSDC_PRUNE_SCORE`)：剪枝阈值 / 最低证据分阈值。
- **`theta_v`** (`MSDC_STATE_THETA_V`)：检测置信度阈值，默认 `0.5`。
- **`theta_m`** (`MSDC_STATE_THETA_M`)：丢失状态阈值，默认 `1.0`。
- **`tau_max`** (`MSDC_LOST_MAX_AGE`)：最大丢失容忍帧数。

### 2.4 最终分配规则 (Rules)

程序按以下顺序进行逻辑短路判断：
```
# 规则 1：彻底丢失且分数极低，清理。
if not observed and p_t < theta_p_low and dt > tau_max:
return REMOVED

# 规则 2：未观测到，且失联程度超标，降级为 LOST。
elif not observed and m_t < theta_m:
return LOST

# 规则 3：观测有效，证据充足，晋升或保持 ACTIVE。
elif observed and p_t > theta_p and m_t < theta_m:
return ACTIVE

# 规则 4：证据不足，继续考察。
else:
return CANDIDATE
```

## 3. 状态转移矩阵 (Transitions)

| **起始状态**        | **目标状态**        | **触发事件 (Event)**       | **转移条件 (Condition)**                                         |
| --------------- | --------------- | ---------------------- | ------------------------------------------------------------ |
| `null`          | `LOW_CANDIDATE` | `NEW_LOW_CANDIDATE`    | 未匹配组可生成轨迹；`ENABLE=true`；无 `high_det`；`low_det` 分数 $\ge$ 生成阈值 |
| `null`          | `CANDIDATE`     | `NEW_CANDIDATE`        | 未匹配组可生成轨迹；含 `high_det` 或 `low_det` 分数 $\ge$ 生成阈值；非低置信候选      |
| `LOW_CANDIDATE` | `ACTIVE`        | `CONFIRM_LOW_ACTIVE`   | 满足低置信度候选确认条件（详见后文）                                           |
| `LOW_CANDIDATE` | `REMOVED`       | `PRUNED_LOW_CANDIDATE` | 存活 $>1$ 帧 且 (连续丢失超标 或 存活期超标 或 证据分 $<$ 剪枝阈值)                  |
| `LOW_CANDIDATE` | `REMOVED`       | `ACTIVE_CONFLICT`      | 候选轨迹与已有的 `ACTIVE` 轨迹在空间上发生冲突/重叠                              |
| `LOW_CANDIDATE` | `REMOVED`       | `MERGED`               | 合并到 `LOST` 轨迹中                                               |
| `LOST`          | `ACTIVE`        | `INHERITED_LOST`       | `LOW_CANDIDATE` 匹配到 `LOST`；`inherit_score` $\ge$ 继承阈值        |
| `CANDIDATE`     | `ACTIVE`        | `CONFIRM_ACTIVE`       | 确认就绪；匹配成功；状态分配返回 `ACTIVE`                                    |
| `CANDIDATE`     | `REMOVED`       | `PRUNED_CANDIDATE`     | 存活 $>1$ 帧 且 (存活期超限 或 证据分 $<$ 剪枝阈值)                           |
| `ACTIVE`        | `LOST`          | `ACTIVE_TO_LOST`       | 丢失次数 $>$ 容忍度 且 状态分配返回 `LOST`                                 |
| `LOST`          | `ACTIVE`        | `LOST_REACQUIRED`      | 匹配成功；`reacquire_score` $\ge$ 找回阈值；状态分配返回 `ACTIVE`            |
| `LOST`          | `REMOVED`       | `LOST_TO_REMOVED`      | 丢失时间 $dt >$ 最大丢失容忍帧数                                         |
| `任意存活状态`        | `REMOVED`       | `null`                 | 对应状态的轨迹池或总容量超过配置上限，溢出淘汰                                      |

## 4. 确认条件细节 (Confirm Conditions)

### 4.1 低置信度候选确认条件 (`LOW_CANDIDATE` -> `ACTIVE`)

必须**同时满足**以下 6 项条件：

1. `MSDC_LOW_CANDIDATE_ENABLE = true`
2. `misses <= MSDC_LOW_CONFIRM_MAX_MISSES`
3. 观测窗口内的命中次数：`count(history) >= MIN_HITS`
4. 历史平均分数：`avg(score) >= MIN_AVG_SCORE`
5. 面积突变限制：`max(area) / min(area) <= MAX_AREA_CHANGE`
6. 若历史长度 $\ge 4$：最大中心步长受限（防止漂移过大）。

### 4.2 候选就绪确认条件 (`CANDIDATE` -> 准备 `ACTIVE`)

必须**同时满足**以下条件：
1. 证据分：`evidence_score >= MSDC_CONFIRM_SCORE`
2. 命中数：`hits >= MSDC_CONFIRM_MIN_HITS`
3. 若要求真实检测：`real_det_hits >= MIN_REAL_DET_HITS`
4. 若要求高置信度检测：`source_history` 必须包含 `high_det`

## 5. 接口与核心数据结构

### 5.1 输入 API

- **方法:** `MSDCLifecycleTracker.update`
- **签名:** 接收当前帧图像 (`frame`)、帧索引 (`frame_idx`)、高/低置信度检测框列表 (`high_boxes`, `low_boxes`) 等。
- **返回:** `list[dict]` 格式的跟踪结果。

### 5.2 核心数据对象定义

- **`Observation` (单次观测)**
- 包含：`box` (xyxy), `source` (high/low/reacquire), `score`, `reliability`, `class_id` 等原始检测信息。
- **`EvidenceTrack` (证据轨迹实体)**
- **身份与状态：** `gid`, `public_id`, `state`
- **运动与几何：** `box`, `velocity`
- **生命周期统计：** `evidence_score`, `hits`, `misses`, `age`, `last_seen`
- **历史追溯：** `low_det_history`, `source_history`, `last_real_det_box`
- **`LifecycleEvent` (生命周期流转记录)**
- 记录轨迹从 `from_state` 变为 `to_state` 的时间点、触发原因及上下文环境。
- **`output_box` (输出格式)**
- 标准的跟踪结果输出格式，包含坐标、置信度、类别，以及 `track_id` (等同于 `public_id`) 和当前的 `lifecycle_state`。