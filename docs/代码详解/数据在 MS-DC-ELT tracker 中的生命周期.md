下面是针对 `msdc_elt` 追踪系统生命周期和架构的优化解析。解释性内容已全部替换为中文，同时保留了代码变量名以确保技术准确性。内容已按模块重新组织，将数据结构与生命周期行为分离，以提升逻辑连贯性和清晰度。

## 1. 轨迹 ID 管理 (Track ID Management)

### 数据结构

|**ID 类型**|**变量名**|**描述**|
|---|---|---|
|**内部 ID**|`gid`|在轨迹创建时分配的全局唯一内部 ID。|
|**外部 ID**|`public_id`|仅在轨迹达到稳定状态（如 ACTIVE）时才会正式分配。|
|**输出 ID**|`track_id`|如果存在 `public_id` 则优先使用；否则回退使用 `gid` 作为输出。|

### 生命周期阶段

- **创建 (Creation)：** 当一个新的 `EvidenceTrack` 生成时，系统只为其分配一个 `gid`。此时它通常还没有 `public_id`。
    - _参考: `evidence_state.py` (Line 357)_
        
- **更新 (Update)：** 在每一帧更新之前，系统会调用 `_ensure_next_gid` 和 `_ensure_next_public_id` 同步 ID 计数器，从而避免与已有轨迹发生 ID 冲突。
    - _参考: `evidence_state.py` (Line 1694, 1700)_
        
- **状态转移 (Transition)：** `public_id` 会在以下两种特定场景下被分配：
    
    1. 当轨迹晋升为 `ACTIVE` 状态时。
    2. 当一个低分候选轨迹 (`LOW_CANDIDATE`) 继承一个丢失轨迹 (`LOST`) 时，系统会确保被继承的旧轨迹拥有有效的外部 ID。
    - _参考: `evidence_state.py` (Line 695, 1533)_
        
- **删除 (Deletion)：** ID 和轨迹不会被立即物理删除。轨迹的状态会首先变更为 `REMOVED`（逻辑删除）。随后，系统通过 `_prune_stale_removed_tracks` 机制将这些过期的废弃轨迹从活跃列表中彻底清理。
    - _参考: `evidence_state.py` (Line 1224, 1536)_
        

## 2. 检测与观测值 (Detection & Observations)

### 数据结构

- **Raw Project Box (原始检测框)：** 包含基本的几何信息（`x/y/w/h` 或 `x1/y1/x2/y2`）、置信度 (confidence/score) 以及类别 (class)。
- **Observation Object (观测值对象)：** 解析后的内部表示，包含 `box`、`source`、`score`、`reliability`、`modality`、`frame_idx`、`class_id`、`class_name` 以及原始数据 `raw`。

### 生命周期阶段

- **创建 (Creation)：** 系统根据预设的阈值将原始检测框拆分为高分框 (`high_boxes`) 和低分框 (`low_boxes`)，然后将它们转换为 `Observation` 对象。
    - _参考: `msdc_detection.py` (Line 27, 36), `msdc_types.py` (Line 110)_
        
- **更新 (Update)：** 当前帧的观测值会被聚合并打包成一个 `_ObservationGroup`。如果该 Group 与已有轨迹匹配成功，它会更新轨迹的核心变量：包含边界框 (`box`)、速度 (`velocity`)、证据得分 (`evidence_score`)、命中/未命中次数 (`hits/misses`)、最后可见时间 (`last_seen`) 以及其他历史统计数据。
    - _参考: `evidence_state.py` (Line 187, 1047, 1368)_
        
- **状态转移 (Transition)：** 对于未能与现有轨迹匹配上的检测框 Group，系统会评估其是否满足创建新轨迹的条件 (`spawn_candidates`)。
    - _参考: `evidence_state.py` (Line 258)_
        
- **删除 (Garbage Collection)：** 观测值和 `_ObservationGroup` 对象是帧级别的临时数据，在 `update_tracks()` 函数返回后就会被自然丢弃（垃圾回收）。只有特定的核心属性（例如 `last_real_det_box`, `source_history`）会作为状态被持久化保留在轨迹实例内部。
    

## 3. 匹配与证据得分 (Matching & Evidence Scores)

### 数据结构

|**得分类型**|**构成 / 公式**|
|---|---|
|**Association Score (关联得分)**|`IoU + 0.5 * center_bonus`|
|**Evidence Increment (证据增量)**|`sum(source_weight * source_score)`|
|**Reacquire Score (重捕获得分)**|四舍五入取整后的证据增量|
|**Inherit Score (继承得分)**|IoU、中心点距离、速度一致性、低分惩罚和时间衰减等因素得分的加权总和。|

### 生命周期阶段

- **创建 (Creation)：** 基础的关联得分是通过批量的 IoU 计算和中心点距离计算函数得出的。
    - _参考: `evidence_state.py` (Line 304-313)_
        
- **更新 (Update)：** 轨迹的 `evidence_score` 是动态变化的。匹配成功时，通过 `_evidence_increment` 累加正向得分；未命中时，则通过 `_apply_negative_evidence` 扣减得分。
    - _参考: `evidence_state.py` (Line 226, 1384, 1405)_
        
- **状态转移 (Transition)：** 这些得分直接作为阈值条件驱动轨迹状态机的跃迁：
    - 当 `evidence_score >= MSDC_CONFIRM_SCORE` 时，允许从候选状态晋升为 `ACTIVE`。
    - 当 `reacquire_score >= MSDC_REACQUIRE_SCORE` 时，允许 `LOST` 轨迹重新激活为 `ACTIVE`。
    - 当 `inherit_score >= MSDC_LOW_INHERIT_SCORE` 时，允许 `LOW_CANDIDATE` 继承 `LOST` 轨迹。
    - _参考: `evidence_state.py` (Line 500, 1443, 1481)_
        
- **删除与保留 (Persistence)：** 普通的关联得分 (matching score) 是局部变量，计算完毕后即被丢弃。但是，诸如 `inherit_score` 和 `reacquire_score` 这样的关键事件得分，会作为调试数据 (extra) 被记录进事件日志中保存。

## 4. 轨迹状态机 (Track State Machine)

### 数据结构

系统维护了一个严格的 `TrackState` 枚举类：

- `LOW_CANDIDATE` (低分候选)
- `CANDIDATE` (候选)
- `ACTIVE` (活跃)
- `LOST` (丢失)
- `REMOVED` (移除)

### 生命周期阶段

- **创建 (Creation)：** 所有的全新轨迹只能从 `LOW_CANDIDATE` 或 `CANDIDATE` 这两个状态初始化。
    
    - _参考: `evidence_state.py` (Line 360-361)_
        
- **更新 (Update)：** 每次更新时，轨迹的存活帧数 (`age`) 递增，并应用当前帧的证据得分。随后系统会基于这些更新后的变量重新评估轨迹状态。
    
    - _参考: `evidence_state.py` (Line 217-244)_
        
- **状态转移 (Transition)：** 所有状态跃迁由 `_transition_track` 统一调度。
    
    - **常规路径:** `CANDIDATE` -> `ACTIVE` -> `LOST` -> `REMOVED`。
        
    - **继承路径:** 当低分候选匹配上丢失轨迹时，被继承的 `LOST` 轨迹变回 `ACTIVE` 状态（接管现有检测框），而用来匹配的 `LOW_CANDIDATE` 则被废弃转为 `REMOVED`。
        
    - **容量裁剪:** 如果当前维护的轨迹总数超过上限，多余的轨迹会被强制设置为溢出状态 (`REMOVED`) 以释放资源。
        
    - _参考: `evidence_state.py` (Line 715, 1282, 1415, 1523)_
        
- **删除 (Deletion)：** `REMOVED` 充当的是逻辑删除标志。一旦轨迹状态变为 `REMOVED`，系统会生成一个 `retired_signature`（退役签名）。它们只有在满足 `_prune_stale_removed_tracks` 函数中的清理条件后，才会被真正从内存列表中剔除。
    
    - _参考: `evidence_state.py` (Line 1224, 1536)_
        

> **架构结论：**
> 
> 当前 `msdc_elt` 的实现完全是一个 **“几何匹配 + 证据得分 + 生命周期状态机”** 的追踪器。它纯粹依赖于边界框 (Bounding Box) 的空间几何关系、检测框置信度以及时间序列上的状态转换逻辑。它**没有维护，也没有使用** 任何基于 ReID (重识别)、外观特征 (Appearance) 或视觉 Embedding 的关联机制。