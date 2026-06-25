**源码位置**

- State 定义：[msdc_types.py (line 17)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/msdc_types.py:17)
- State transition 主逻辑：[evidence_state.py (line 1415)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1415)
- Track 创建：[evidence_state.py (line 340)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:340)
- Track update cycle：[evidence_state.py (line 164)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:164)

## 1. 所有 State

```
LOW_CANDIDATE = "low_candidate"
CANDIDATE = "candidate"
ACTIVE = "active"
LOST = "lost"
REMOVED = "removed"
```

---

## 2. LOW_CANDIDATE

**进入条件**

`None → LOW_CANDIDATE`

位置：[evidence_state.py (line 360)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:360)

条件：

```
_group_should_spawn_low_candidate(group) == True
```

展开：

```
MSDC_LOW_CANDIDATE_ENABLE == True
group 不包含 high_det
group.source_scores["low_det"] >= MSDC_LOW_SPAWN_MIN_CONF
unmatched observation group 通过 spawn filter
candidate 数量未超过 MSDC_MAX_CANDIDATES
```

**退出条件**

`LOW_CANDIDATE → ACTIVE`

位置：[evidence_state.py (line 1428)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1428)

```
_low_candidate_can_confirm(track, frame_idx) == True
```

展开：

```
MSDC_LOW_CANDIDATE_ENABLE == True
misses <= MSDC_LOW_CONFIRM_MAX_MISSES
low_det_history window 内数量 >= MSDC_LOW_CONFIRM_MIN_HITS
avg(low_det_history.score) >= MSDC_LOW_CONFIRM_MIN_AVG_SCORE
area stable
center step stable
```

`LOW_CANDIDATE → REMOVED`

位置：[evidence_state.py (line 1433)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1433)

```
age > 1
AND (
  misses > MSDC_LOW_CONFIRM_MAX_MISSES
  OR age > MSDC_LOW_CONFIRM_WINDOW
  OR evidence_score < MSDC_PRUNE_SCORE
)
```

`LOW_CANDIDATE → REMOVED`

位置：[evidence_state.py (line 547)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:547)

```
ready LOW_CANDIDATE overlaps ACTIVE track
event_type = PRUNED_LOW_CANDIDATE_ACTIVE_CONFLICT
```

`LOW_CANDIDATE → REMOVED`

位置：[evidence_state.py (line 731)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:731)

```
LOW_CANDIDATE merged into LOST track
event_type = LOW_CANDIDATE_MERGED
```

**修改的变量**

创建时修改：

```
gid
state
box
velocity
evidence_score
hits
misses
age
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
next_gid
```

匹配时修改：

```
box
velocity
evidence_score
hits
misses
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
```

未匹配时修改：

```
misses
evidence_score
```

退出到 ACTIVE 时修改：

```
state
public_id
next_public_id
```

退出到 REMOVED 时修改：

```
state
retired_signature
```

---

## 3. CANDIDATE

**进入条件**

`None → CANDIDATE`

位置：[evidence_state.py (line 360)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:360)

条件：

```
_group_can_spawn_candidate(group) == True
_group_should_spawn_low_candidate(group) == False
```

展开：

```
group contains high_det
OR low_det score >= MSDC_LOW_SPAWN_MIN_CONF
```

并且：

```
unmatched observation group 通过 spawn filter
candidate 数量未超过 MSDC_MAX_CANDIDATES
```

**退出条件**

`CANDIDATE → ACTIVE`

位置：[evidence_state.py (line 1441)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1441)

```
confirm_ready == True
matched == True
assign_state(...) == ACTIVE
```

`confirm_ready`：

```
evidence_score >= MSDC_CONFIRM_SCORE
hits >= MSDC_CONFIRM_MIN_HITS
_candidate_has_required_detection(track) == True
```

`_candidate_has_required_detection`：

```
if MSDC_CONFIRM_REQUIRE_DET:
  real_det_hits >= MSDC_CONFIRM_MIN_REAL_DET_HITS

if MSDC_CONFIRM_REQUIRE_HIGH_DET:
  source_history contains "high_det"
```

`CANDIDATE → REMOVED`

位置：[evidence_state.py (line 1457)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1457)

```
age > 1
AND (
  age > MSDC_CANDIDATE_MAX_AGE
  OR evidence_score < MSDC_PRUNE_SCORE
)
```

`CANDIDATE → REMOVED`

位置：[evidence_state.py (line 1255)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1255)

```
CANDIDATE pool exceeds MSDC_MAX_CANDIDATES
```

`CANDIDATE → REMOVED`

位置：[evidence_state.py (line 1298)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1298)

```
live track count exceeds MSDC_MAX_TOTAL_TRACKS
```

**修改的变量**

创建时修改：

```
gid
state
box
velocity
evidence_score
hits
misses
age
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
next_gid
```

匹配时修改：

```
box
velocity
evidence_score
hits
misses
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
```

未匹配时修改：

```
misses
evidence_score
```

退出到 ACTIVE 时修改：

```
state
public_id
next_public_id
```

退出到 REMOVED 时修改：

```
state
retired_signature
```

---

## 4. ACTIVE

**进入条件**

`LOW_CANDIDATE → ACTIVE`

位置：[evidence_state.py (line 1429)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1429)

```
_low_candidate_can_confirm(track, frame_idx) == True
```

`CANDIDATE → ACTIVE`

位置：[evidence_state.py (line 1454)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1454)

```
confirm_ready == True
matched == True
assign_state(...) == ACTIVE
```

`LOST → ACTIVE`

位置：[evidence_state.py (line 1480)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1480)

```
matched == True
reacquire_score >= MSDC_REACQUIRE_SCORE
assign_state(...) == ACTIVE
```

`LOST → ACTIVE`

位置：[evidence_state.py (line 715)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:715)

```
LOW_CANDIDATE inherited into LOST
inherit_score >= MSDC_LOW_INHERIT_SCORE
```

**退出条件**

`ACTIVE → LOST`

位置：[evidence_state.py (line 1466)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1466)

```
misses > MSDC_ACTIVE_MISSING_PATIENCE
assign_state(...) == LOST
```

`ACTIVE → LOST`

位置：[evidence_state.py (line 1251)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1251)

```
ACTIVE pool exceeds MSDC_MAX_ACTIVE_TRACKS
```

`ACTIVE → REMOVED`

位置：[evidence_state.py (line 1298)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1298)

```
live track count exceeds MSDC_MAX_TOTAL_TRACKS
```

**修改的变量**

进入 ACTIVE 时修改：

```
state
public_id
next_public_id
```

LOST 继承进入 ACTIVE 时修改：

```
public_id
next_public_id
box
velocity
evidence_score
hits
misses
age
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
state
```

ACTIVE 匹配 high_det / reacquire 时修改：

```
box
velocity
evidence_score
hits
misses
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
```

ACTIVE 匹配 low_det-only 时修改：

```
evidence_score
hits
misses
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
```

ACTIVE 未匹配时修改：

```
misses
evidence_score
```

退出到 LOST 时修改：

```
state
misses
```

退出到 REMOVED 时修改：

```
state
retired_signature
```

---

## 5. LOST

**进入条件**

`ACTIVE → LOST`

位置：[evidence_state.py (line 1475)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1475)

```
misses > MSDC_ACTIVE_MISSING_PATIENCE
assign_state(
  p_t=evidence_score,
  v_t=0.0,
  m_t=0.0,
  dt=current_dt
) == LOST
```

`ACTIVE → LOST`

位置：[evidence_state.py (line 1252)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1252)

```
ACTIVE pool exceeds MSDC_MAX_ACTIVE_TRACKS
overflow_state = LOST
```

**退出条件**

`LOST → ACTIVE`

位置：[evidence_state.py (line 1481)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1481)

```
matched == True
reacquire_score >= MSDC_REACQUIRE_SCORE
assign_state(...) == ACTIVE
```

`LOST → ACTIVE`

位置：[evidence_state.py (line 686)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:686)

```
ready LOW_CANDIDATE matched to LOST
inherit_score >= MSDC_LOW_INHERIT_SCORE
```

`LOST → REMOVED`

位置：[evidence_state.py (line 1497)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1497)

```
current_dt > MSDC_LOST_MAX_AGE
assign_state(...) == REMOVED
```

`LOST → REMOVED`

位置：[evidence_state.py (line 1253)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1253)

```
LOST pool exceeds MSDC_MAX_LOST_TRACKS
```

`LOST → REMOVED`

位置：[evidence_state.py (line 1298)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1298)

```
live track count exceeds MSDC_MAX_TOTAL_TRACKS
```

**修改的变量**

进入 LOST 时修改：

```
state
misses
```

LOST reacquire 匹配时修改：

```
box
velocity
evidence_score
hits
misses
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
```

LOST 未 reacquire 时修改：

```
misses
evidence_score
```

LOW_CANDIDATE 继承 LOST 时修改：

```
public_id
next_public_id
box
velocity
evidence_score
hits
misses
age
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
state
```

退出到 ACTIVE 时修改：

```
state
public_id
next_public_id
misses
```

退出到 REMOVED 时修改：

```
state
retired_signature
```

---

## 6. REMOVED

**进入条件**

`LOW_CANDIDATE → REMOVED`

```
low candidate prune
low candidate active conflict
low candidate merged into LOST
LOW_CANDIDATE pool exceeds MSDC_MAX_LOW_CANDIDATES
live track count exceeds MSDC_MAX_TOTAL_TRACKS
```

`CANDIDATE → REMOVED`

```
candidate prune
CANDIDATE pool exceeds MSDC_MAX_CANDIDATES
live track count exceeds MSDC_MAX_TOTAL_TRACKS
```

`LOST → REMOVED`

```
lost timeout
LOST pool exceeds MSDC_MAX_LOST_TRACKS
live track count exceeds MSDC_MAX_TOTAL_TRACKS
```

`ACTIVE → REMOVED`

```
live track count exceeds MSDC_MAX_TOTAL_TRACKS
```

**退出条件**

无 state transition 退出。

列表清理条件：

位置：[evidence_state.py (line 1224)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1224)

```
guard_frames <= 0:
  remove all REMOVED tracks from list

or

removed_signature.skip_removed_guard == True:
  remove from list

or

frame_idx - removed_frame_idx > MSDC_REMOVED_GUARD_FRAMES:
  remove from list
```

**修改的变量**

进入 REMOVED 时修改：

```
state
retired_signature
```

作为 removed guard signature 使用时读取：

```
retired_signature.last_box
retired_signature.last_seen
retired_signature.removed_frame_idx
retired_signature.last_velocity
retired_signature.class_id
retired_signature.class_name
```

列表清理时修改：

```
tracks list
```

---

## 全局每帧变量修改

位置：[evidence_state.py (line 217)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:217)

所有非 `REMOVED` track：

```
age += 1
```

位置：[evidence_state.py (line 1368)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1368)

匹配成功 track：

```
velocity
box
evidence_score
hits
misses
last_seen
last_real_det_frame
real_det_hits
last_real_det_box
low_det_history
source_history
class_id
class_name
```

位置：[evidence_state.py (line 1405)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1405)

负证据 track：

```
misses
evidence_score
```

位置：[evidence_state.py (line 1523)](/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/target_module/image_detect_module/utils/evidence_state.py:1523)

发生 state transition：

```
state
public_id
next_public_id
retired_signature
```