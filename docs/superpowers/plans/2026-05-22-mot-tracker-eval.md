# MOT Tracker Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict, reproducible, fair MOT comparison protocol for UAV/USV small-target tracking using `E:\dataset\MOT`.

**Architecture:** The experiment is split into immutable data splits, reusable detection files, tracker parameter tuning, locked tracker-only comparison, detection threshold sweep, and one-shot final end-to-end testing. All results are recorded by configuration hash, split, sequence, tracker, detector threshold, and evaluator version.

**Tech Stack:** MOTChallenge-format data, detector-generated `det.txt`, ByteTrack, custom BoT-SORT-Lite, custom OC-SORT, official BoT-SORT, official OC-SORT, HOTA/TrackEval-style metrics, CSV/YAML experiment records.

---

## 1. 实验总目标

目标是在同一 UAV/USV 小目标数据集 `E:\dataset\MOT` 上，严格区分检测能力与跟踪能力，比较以下 tracker 在相同检测输入下的真实 Tracking-by-Detection 表现：

- ByteTrack
- 自定义 BoT-SORT-Lite
- 自定义 OC-SORT
- 官方 BoT-SORT
- 官方 OC-SORT

核心问题不是泛化地比较 MOTA，而是回答：

- 哪个 tracker 在小目标、弱纹理、检测断续条件下更能保持 ID？
- 哪个 tracker 的 IDSW 和 Frag 更稳定？
- 检测置信度阈值 `det_conf` 应设为多少，才能兼顾 FP/FN 与 ID 连续性？
- 最终完整系统在后 40% 测试集上的真实表现是多少？

所有结论必须来自比较集或最终测试集，不能根据最终测试集结果回改系统配置。

## 2. 数据划分方案

数据必须按 sequence 顺序和时间顺序划分，禁止随机抽帧。若数据集由多个 sequence 组成，先按自然顺序或采集时间排序 sequence；如果单个 sequence 很长，再按帧号连续切分。优先避免同一 sequence 同时落入多个集合；只有当 sequence 数量太少时，才允许按连续帧段切分，并在 `README.md` 中记录原因。

划分比例：

| 阶段 | 数据范围 | 用途 | 是否允许调参 |
|---|---:|---|---|
| Stage 0 | 全数据 | 只生成 `split_tune.txt`、`split_compare.txt`、`split_test.txt` | 否 |
| Stage 1 | 前 20% tune set | 每个 tracker 独立搜索内部最优参数 | 是 |
| Stage 2 | 中间 40% compare set | Tracker-only 主比较、方案选择、Threshold sweep 结论 | 否，不能继续细调单 tracker |
| Stage 3 | 后 40% test set | 最终报告、End-to-end 一次性确认 | 否，只允许运行一次 |

推荐 split 文件格式，每行一个 sequence 或连续帧段：

```text
# split_tune.txt
seq_0001
seq_0002

# 如果必须切帧段，使用闭区间并记录原因
seq_0003:000001-001200
```

Stage 0 产物：

- `experiments/mot_tracker_eval/dataset/split_tune.txt`
- `experiments/mot_tracker_eval/dataset/split_compare.txt`
- `experiments/mot_tracker_eval/dataset/split_test.txt`
- `experiments/mot_tracker_eval/dataset/split_manifest.csv`

`split_manifest.csv` 字段：

```csv
split,sequence,start_frame,end_frame,num_frames,source_path,split_reason
tune,seq_0001,1,900,900,E:/dataset/MOT/seq_0001,sequence_order
```

## 3. 三类实验的整体关系

三组实验的依赖关系：

```text
Stage 0: time-ordered split
        |
        v
固定检测模型 / 固定 NMS / 多个 det_conf 生成 detections
        |
        +--> Tracker-only 主实验
        |       tune set 搜索每个 tracker 的 best_config
        |       compare set 做 tracker 能力比较
        |       test set 只报告最终 tracker-only 结果
        |
        +--> Threshold sweep
        |       固定一个已选 tracker 和 best_config
        |       只改变 detector det_conf
        |       回答 det_conf 设多少更合适
        |
        +--> End-to-end
                使用最终 det_conf + 最终 tracker + 锁定参数
                只在 test set 运行一次
```

实验组定义：

| 实验组 | 改变什么 | 固定什么 | 目的 |
|---|---|---|---|
| Tracker-only | tracker 类型及其 tune set 锁定后的内部参数 | 检测模型、`det.txt`、数据、GT、评估工具、IoU 阈值 | 比较 tracker 本身 |
| Threshold sweep | 检测置信度阈值 `det_conf` | tracker 类型、tracker best_config、检测模型、NMS、GT、评估工具 | 找合理检测阈值 |
| End-to-end | 完整系统配置 | test set、GT、评估工具 | 评估系统实际效果 |

## 4. Tracker-only 主实验详细流程

主实验必须使用同一检测输入来控制检测变量。

流程：

```text
固定检测模型
固定检测阈值 det_conf，例如 0.3 或 0.4
固定 NMS
固定输入帧
生成同一份 det.txt
        |
        v
tune set:
  ByteTrack 搜索参数
  自定义 BoT-SORT-Lite 搜索参数
  自定义 OC-SORT 搜索参数
  官方 BoT-SORT 搜索参数
  官方 OC-SORT 搜索参数
        |
        v
写入:
  configs/best_bytetrack.yaml
  configs/best_botsort.yaml
  configs/best_ocsort.yaml
  configs/best_official_botsort.yaml
  configs/best_official_ocsort.yaml
        |
        v
锁定参数
        |
        v
compare set:
  使用锁定参数做 Tracker-only 对比
        |
        v
test set:
  只运行一次并报告最终结果
```

为什么每个 tracker 用自己的 tune set 最优参数比全部使用默认参数更公平：

- 默认参数通常来自不同数据集、不同检测器、不同帧率和不同目标尺度，不代表同等适配程度。
- 小目标 UAV/USV 场景下，lost buffer、匹配阈值、轨迹确认帧数会强烈影响 IDSW 和 Frag。
- 公平控制变量应固定检测输入、GT、评估工具和搜索预算，而不是强迫不同机制的 tracker 使用不适配默认值。
- 允许各 tracker 在同一 tune set、同一搜索预算内达到合理工作状态，比较的是“同等调参预算后的真实跟踪能力”。

Tracker-only 中的阈值边界：

| 类型 | 含义 | 是否 tracker 内部参数 | 在哪个实验中改变 |
|---|---|---:|---|
| `det_conf` | 检测器输出 `det.txt` 前过滤检测框的置信度 | 否 | Threshold sweep |
| NMS 阈值 | 检测器后处理阈值 | 否 | 通常固定 |
| `track_high_thresh` | tracker 内部高分检测匹配阈值 | 是 | Tracker 调参 |
| `track_low_thresh` | tracker 内部低分检测利用阈值 | 是 | Tracker 调参 |
| `new_track_thresh` | tracker 内部新建轨迹阈值 | 是 | Tracker 调参 |

这不违反控制变量原则：Tracker-only 固定的是 detector 输出给 tracker 的 `det.txt`；tracker 内部阈值属于关联机制本身，允许在 tune set 内优化。

## 5. 各 tracker 参数搜索空间

总体预算：

- 每个 tracker 最多 40 组候选配置。
- 每组配置必须跑完整 tune set。
- 每个 tracker 使用相同评估工具、相同 GT、相同 det 文件、相同 IoU 匹配阈值。
- 粗搜索最多 24 组，细化最多 16 组。
- 若某 tracker 参数无效或不支持，必须在 trials CSV 中记录 `status=invalid` 和 `error_message`。

### 5.1 ByteTrack

注意：ByteTrack 的 `minimum_matching_threshold` 通常不是简单“越大越严格”的通用 IoU 阈值语义，不应直接与 OC-SORT 的 `iou_threshold` 横向解释，只能作为 ByteTrack 内部候选参数比较。

粗搜索建议 24 组，使用固定组合表：

| 参数 | 候选值 |
|---|---|
| `track_activation_threshold` | 0.25, 0.35, 0.45 |
| `lost_track_buffer` | 30, 60 |
| `minimum_matching_threshold` | 0.70, 0.80 |
| `minimum_consecutive_frames` | 1, 3 |
| `frame_rate` | 使用数据真实 FPS；若未知固定 30，不参与搜索 |

细化策略：

- 取 tune set 综合排名前 3。
- 围绕 `track_activation_threshold` 做 `best ± 0.05`。
- 围绕 `lost_track_buffer` 做 `best ± 15`，限制在 `[15, 90]`。
- 围绕 `minimum_matching_threshold` 做 `best ± 0.05`，限制在 `[0.65, 0.90]`。
- 生成去重后的最多 16 组细化配置。

### 5.2 自定义 OC-SORT

粗搜索建议 24 组：

| 参数 | 候选值 |
|---|---|
| `max_age` | 15, 30, 60 |
| `min_hits` | 1, 3 |
| `iou_threshold` | 0.20, 0.30 |
| `dist_threshold` | 0.50, 0.70 |
| `use_oru` | true |
| `use_ocm` | true, false |

细化策略：

- 取前 3 配置。
- `max_age` 尝试 `best - 10`, `best`, `best + 10`，限制 `[10, 80]`。
- `iou_threshold` 尝试 `best ± 0.05`，限制 `[0.15, 0.45]`。
- `dist_threshold` 尝试 `best ± 0.10`，限制 `[0.40, 0.90]`。
- `min_hits` 只在 1 和 3 间切换。
- 最多 16 组。

### 5.3 自定义 BoT-SORT-Lite

可理解为自定义 OC-SORT + GMC。粗搜索建议先固定 OC-SORT 核心范围，再搜索 GMC：

| 参数 | 候选值 |
|---|---|
| `max_age` | 30, 60 |
| `min_hits` | 1, 3 |
| `iou_threshold` | 0.20, 0.30 |
| `dist_threshold` | 0.50, 0.70 |
| `use_oru` | true |
| `use_ocm` | true |
| `GMC_METHOD` | none, sparse_flow, orb |
| `GMC_DOWNSCALE` | 1, 2, 4 |

为控制预算，不做全笛卡尔积。固定 24 组采样：

- 8 组 OC-SORT 核心参数组合。
- 每组配 3 个 GMC 设置：`none/1`、`sparse_flow/2`、`orb/2`。
- 若小目标运动相机抖动明显，再在细化阶段加入 `sparse_flow/1` 和 `orb/4`。

细化策略：

- 取前 3 配置。
- 若 `GMC_METHOD=none` 进入前 3，保留一个无 GMC 分支作为强基线。
- 对进入前 3 的 GMC 方法搜索 `GMC_DOWNSCALE ∈ {1,2,4}`。
- 对 `max_age`、`iou_threshold`、`dist_threshold` 按自定义 OC-SORT 的细化规则微调。
- 最多 16 组。

### 5.4 官方 OC-SORT

粗搜索建议 24 组：

| 参数 | 候选值 |
|---|---|
| `det_thresh` | 0.20, 0.30, 0.40 |
| `max_age` | 30, 60 |
| `min_hits` | 1, 3 |
| `iou_threshold` | 0.20, 0.30 |
| `delta_t` | 3 |
| `asso_func` | iou |
| `inertia` | 0.20, 0.40 |
| `use_byte` | true, false |

预算控制方式：

- 优先遍历 `det_thresh × max_age × min_hits × iou_threshold` 的 24 组。
- 将 `inertia=0.20` 作为粗搜索默认值。
- 将 `use_byte=true` 作为主分支；额外替换 4 组代表配置为 `use_byte=false`，总数仍不超过 24。

细化策略：

- 取前 3 配置。
- `det_thresh` 尝试 `best ± 0.05`，限制 `[0.15, 0.50]`。
- `iou_threshold` 尝试 `best ± 0.05`，限制 `[0.15, 0.45]`。
- `inertia` 尝试 0.10, 0.20, 0.40, 0.60 中靠近 best 的值。
- `delta_t` 可在细化阶段尝试 2, 3, 5。
- 最多 16 组。

### 5.5 官方 BoT-SORT

小目标 ReID 特征可能不稳定，主实验默认 `with_reid=false`；但必须设置补充 ReID 对照。

主实验粗搜索建议 24 组：

| 参数 | 候选值 |
|---|---|
| `track_high_thresh` | 0.40, 0.50, 0.60 |
| `track_low_thresh` | 0.10, 0.20 |
| `new_track_thresh` | 0.50, 0.60 |
| `track_buffer` | 30, 60 |
| `match_thresh` | 0.70, 0.80 |
| `proximity_thresh` | 0.50 |
| `appearance_thresh` | 0.25 |
| `with_reid` | false |
| `cmc_method` | none, sparseOptFlow |

预算控制方式：

- 先固定 `cmc_method=sparseOptFlow` 搜索 18 组核心阈值组合。
- 再用排名前 6 的核心组合替换为 `cmc_method=none` 做对照。
- 总数不超过 24。

细化策略：

- 取前 3 配置。
- `track_high_thresh`、`new_track_thresh` 尝试 `best ± 0.05`。
- `track_low_thresh` 尝试 `best ± 0.05`，限制 `[0.05, track_high_thresh - 0.10]`。
- `track_buffer` 尝试 `best ± 15`，限制 `[15, 90]`。
- `match_thresh` 尝试 `best ± 0.05`，限制 `[0.65, 0.90]`。
- 最多 16 组。

ReID 补充对照：

- 只在 compare set 上对官方 BoT-SORT 做补充分析，不参与主实验调参回改。
- 使用主实验 best config，复制两份：
  - `with_reid=false`
  - `with_reid=true`
- 若 `with_reid=true` 需要外部 ReID 权重，必须记录权重路径、版本、输入尺寸和是否适配红外/可见光。
- 结论只回答“小目标场景是否值得开启 ReID”，不能回改 test set 前已锁定的主实验参数，除非在正式 End-to-end 锁定前已经通过 compare set 明确选择并记录。

## 6. best_config 选择规则

调参目标优先级：

| 优先级 | 指标 | 方向 | 说明 |
|---:|---|---|---|
| 1 | HOTA | 越高越好 | 综合检测、关联、定位质量 |
| 2 | IDF1 | 越高越好 | 身份保持能力 |
| 3 | IDSW | 越低越好 | ID 切换次数 |
| 4 | Frag | 越低越好 | 轨迹碎片数 |
| 5 | FPS | 越高越好 | 工程可用性 |

选择规则：

1. 优先最大化 HOTA。
2. 若 HOTA 差距 `< 0.5`，选择 IDF1 更高者。
3. 若 IDF1 差距 `< 0.5`，选择 IDSW 更低者。
4. 若 IDSW 接近，选择 Frag 更低者。
5. 若以上都接近，选择 FPS 更高、参数更简单者。

稳健性附加规则：

- 如果某配置 HOTA 只高于另一配置 `< 0.5`，但 IDSW 高出 `> 10%` 或 Frag 高出 `> 10%`，不选择该极端配置。
- 如果某配置全局指标较好，但最差 sequence 的 HOTA 低于候选中位数 `> 3.0`，且该 sequence IDSW 爆炸，不选择该配置。
- 如果某配置依赖非常长的 lost buffer 导致假轨迹明显增多，且 IDF1 没有提升 `>= 0.5`，优先选择更简单配置。

明确伪代码：

```python
def is_close(a, b, eps):
    return abs(a - b) < eps

def robust_penalty(candidate):
    penalty = 0
    if candidate.idsw > candidate.peer_min_idsw * 1.10:
        penalty += 1
    if candidate.frag > candidate.peer_min_frag * 1.10:
        penalty += 1
    if candidate.worst_sequence_hota < candidate.peer_median_worst_hota - 3.0:
        penalty += 1
    if candidate.sequence_idsw_max > candidate.sequence_idsw_median * 3:
        penalty += 1
    return penalty

def better(a, b):
    if robust_penalty(a) >= 2 and a.hota - b.hota < 0.5:
        return False
    if robust_penalty(b) >= 2 and b.hota - a.hota < 0.5:
        return True

    if not is_close(a.hota, b.hota, 0.5):
        return a.hota > b.hota
    if not is_close(a.idf1, b.idf1, 0.5):
        return a.idf1 > b.idf1
    if a.idsw != b.idsw:
        return a.idsw < b.idsw
    if a.frag != b.frag:
        return a.frag < b.frag
    if not is_close(a.fps, b.fps, 1.0):
        return a.fps > b.fps
    return a.num_tuned_params < b.num_tuned_params
```

每个 best config 必须写入：

```yaml
tracker_name: bytetrack
selected_from_split: tune
selection_date: "2026-05-22"
detection_conf_for_tuning: 0.3
nms_threshold: 0.5
metrics_iou_threshold: 0.5
trial_id: bytetrack_fine_007
config_hash: sha256:...
params:
  track_activation_threshold: 0.35
  lost_track_buffer: 60
  minimum_matching_threshold: 0.80
  minimum_consecutive_frames: 1
locked: true
do_not_modify_after_compare: true
```

## 7. Threshold sweep 实验设计

目的：回答当前检测器输出给 tracker 时，检测置信度阈值 `det_conf` 设多少更合适。

流程：

```text
从 Tracker-only compare set 选择表现最好的 tracker
读取该 tracker 的 best_config
固定 tracker 内部参数
固定检测模型和 NMS
只改变 det_conf: 0.2 / 0.3 / 0.4 / 0.5 / 0.6
分别生成 det.txt
在 compare set 上评估
记录 HOTA、IDF1、MOTA、FP、FN、IDSW、Frag、FPS
选择最终 det_conf
```

结论判据：

- 优先选择 compare set 上 HOTA 高、IDF1 高、IDSW/Frag 低的 `det_conf`。
- 如果低阈值显著降低 FN 并提升 IDF1，同时 FP 没有明显失控，可选低阈值。
- 如果低阈值造成 FP 大幅上升并导致 IDSW/Frag 上升，则选中等阈值。
- 若 HOTA 差距 `< 0.5` 且 IDF1 差距 `< 0.5`，选择 IDSW 和 Frag 更低的阈值。
- 最终阈值只由 compare set 决定，test set 不参与选择。

候选：

```text
det_conf = 0.2, 0.3, 0.4, 0.5, 0.6
```

补充低阈值 det 策略：

- 主实验策略：`det_conf=0.3` 或 `0.4` 生成统一 `det.txt`。目的是真正控制检测输入，直接比较 tracker 关联能力。适用于主论文/主报告。
- 补充实验策略：`det_conf=0.1` 或 `0.2` 生成统一低阈值 `det.txt`，所有 tracker 仍使用同一份输入，让 ByteTrack/BoT-SORT 类机制通过内部 high/low/new_track 阈值自行筛选。目的：观察低置信检测是否能提升 IDF1、降低 IDSW 和 Frag。适用于检测断续严重、小目标漏检明显的场景分析。

## 8. End-to-end 实验设计

End-to-end 只用于最终系统确认。

流程：

```text
读取 compare set 选择出的最终 tracker
读取该 tracker 的 locked best_config
读取 Threshold sweep 选择出的最终 det_conf
固定检测模型、NMS、输入尺寸、类别映射、评估工具版本
只在 test set 运行一次
输出最终系统指标和 per-sequence 指标
禁止根据 test set 结果修改配置
```

报告方式：

- `metrics/end_to_end_summary.csv` 记录全局指标。
- `metrics/per_sequence_metrics.csv` 记录每个 sequence 指标。
- `tracker_results/final_end_to_end/<sequence>/tracks.txt` 保存跟踪结果。
- `configs/final_system.yaml` 保存最终检测与跟踪配置。
- `logs/runtime/end_to_end_test.log` 保存运行命令、开始结束时间、硬件信息。

最终报告只能说：

- “在锁定配置下，最终测试集指标为……”
- “该结果验证/不验证 compare set 上的选择趋势……”

不能说：

- “根据 test set 结果又选择了另一个 tracker”
- “发现 test set 上阈值不好，所以改 det_conf 后重新报告”

## 9. 防止过拟合与公平性约束

必须按 sequence 统计：

- 每个 sequence 的 HOTA、IDF1、MOTA、FP、FN、IDSW、Frag、FPS。
- sequence 均值。
- sequence 方差。
- 最差 sequence 表现。
- `sequence_idsw_max / sequence_idsw_median`，用于发现 IDSW 爆炸。

公平性约束：

- 全部 tracker 使用相同 tune/compare/test 划分。
- Tracker-only 使用同一份 `det.txt`。
- 每个 tracker 最多 40 组候选参数。
- 每组配置必须在完整 tune set 上评估。
- 不得根据 compare set 或 test set 回改 tracker 内部参数。
- Threshold sweep 只能改变 `det_conf`，不能同时改 tracker 内部阈值。
- test set 只允许运行一次，用于最终确认。
- 所有实验记录 GPU、CPU、内存、运行时、代码 commit、配置 hash。
- 所有评估使用相同 GT 格式、相同 IoU 匹配阈值、相同 evaluator 版本。

## 10. 目录结构与文件命名规范

完整结构：

```text
experiments/mot_tracker_eval/
  dataset/
    split_tune.txt
    split_compare.txt
    split_test.txt
    split_manifest.csv
    dataset_audit.csv

  detections/
    conf_0.1/
      seq_0001/det.txt
      seq_0001/det_meta.yaml
    conf_0.2/
    conf_0.3/
    conf_0.4/
    conf_0.5/
    conf_0.6/

  tuning/
    bytetrack_trials.csv
    botsort_trials.csv
    ocsort_trials.csv
    official_botsort_trials.csv
    official_ocsort_trials.csv
    trial_configs/
      bytetrack_trial_001.yaml
      botsort_trial_001.yaml

  configs/
    best_bytetrack.yaml
    best_botsort.yaml
    best_ocsort.yaml
    best_official_botsort.yaml
    best_official_ocsort.yaml
    final_system.yaml

  tracker_results/
    bytetrack/
      tune/trial_001/seq_0001/tracks.txt
      compare/best/seq_0001/tracks.txt
      test/best/seq_0001/tracks.txt
    botsort/
    ocsort/
    official_botsort/
    official_ocsort/
    final_end_to_end/

  metrics/
    tuning_summary.csv
    tracker_only_summary.csv
    threshold_sweep_summary.csv
    end_to_end_summary.csv
    per_sequence_metrics.csv
    reid_ablation_summary.csv

  logs/
    runtime/
      bytetrack_tune_trial_001.log
      end_to_end_test.log
    gpu_memory/
      bytetrack_tune_trial_001.csv
    cpu_memory/
      bytetrack_tune_trial_001.csv

  README.md
```

文件命名规则：

- tracker 名统一小写：`bytetrack`、`botsort`、`ocsort`、`official_botsort`、`official_ocsort`。
- trial ID 格式：`<tracker>_<coarse|fine>_<three_digit_id>`，例如 `bytetrack_fine_007`。
- detection 目录格式：`conf_<one_decimal>`，例如 `conf_0.3`。
- 每个 sequence 的 detection 文件使用 MOTChallenge 格式：`detections/conf_0.3/<sequence>/det.txt`。
- 每个 sequence 的 tracking 文件：`tracker_results/<tracker>/<split>/<trial_or_best>/<sequence>/tracks.txt`。
- YAML 配置必须包含 `config_hash` 和 `locked` 字段。

`det_meta.yaml` 应记录：

```yaml
det_conf: 0.3
nms_threshold: 0.5
detector_name: fixed_detector_name
detector_weights: path_or_hash
input_size: [1280, 720]
class_map:
  0: target
generated_at: "2026-05-22T00:00:00+08:00"
code_commit: unknown
```

实验 README 必须记录：

- 数据集路径和划分规则。
- 是否按 sequence 划分；若按帧段切分，说明原因。
- 检测模型、权重、输入尺寸、类别映射。
- NMS 阈值、所有 `det_conf` 候选。
- evaluator 名称、版本、IoU 阈值。
- 每个 tracker 的代码来源、commit、配置文件。
- 搜索预算和是否有 invalid trial。
- 最终锁定时间点。
- test set 运行时间点和确认“只运行一次”。

## 11. 结果表格模板

### 11.1 tuning trials

```csv
trial_id,tracker,stage,split,det_conf,nms,config_hash,status,HOTA,IDF1,MOTA,DetA,AssA,FP,FN,IDSW,Frag,FPS,seq_mean_HOTA,seq_var_HOTA,worst_seq,worst_seq_HOTA,max_seq_IDSW,median_seq_IDSW,robust_penalty,error_message
bytetrack_coarse_001,bytetrack,coarse,tune,0.3,0.5,sha256:...,ok,0,0,0,0,0,0,0,0,0,0,0,0,seq_0001,0,0,0,0,
```

### 11.2 tracker-only summary

```csv
rank,tracker,split,det_conf,best_config,HOTA,IDF1,MOTA,DetA,AssA,FP,FN,IDSW,Frag,FPS,seq_mean_HOTA,seq_var_HOTA,worst_seq,worst_seq_HOTA,max_seq_IDSW,notes
1,bytetrack,compare,0.3,configs/best_bytetrack.yaml,0,0,0,0,0,0,0,0,0,0,0,0,seq_0001,0,0,
```

### 11.3 threshold sweep summary

```csv
det_conf,tracker,best_config,split,HOTA,IDF1,MOTA,FP,FN,IDSW,Frag,FPS,seq_mean_HOTA,seq_var_HOTA,worst_seq,conclusion_tag
0.2,selected_tracker,configs/best_xxx.yaml,compare,0,0,0,0,0,0,0,0,0,0,seq_0001,candidate
```

### 11.4 end-to-end summary

```csv
run_id,split,det_conf,tracker,config_path,detector_hash,tracker_config_hash,HOTA,IDF1,MOTA,DetA,AssA,FP,FN,IDSW,Frag,FPS,run_started_at,run_finished_at,code_commit,locked_before_test
e2e_test_001,test,0.3,selected_tracker,configs/final_system.yaml,sha256:...,sha256:...,0,0,0,0,0,0,0,0,0,0,2026-05-22T00:00:00+08:00,2026-05-22T00:00:00+08:00,unknown,true
```

### 11.5 per-sequence metrics

```csv
experiment_group,tracker,split,trial_id,sequence,det_conf,HOTA,IDF1,MOTA,DetA,AssA,FP,FN,IDSW,Frag,FPS,num_frames,notes
tracker_only,bytetrack,compare,best,seq_0001,0.3,0,0,0,0,0,0,0,0,0,0,900,
```

## 12. 最终报告模板

```markdown
# UAV/USV Small-Target MOT Tracker Evaluation Report

## 1. Experiment Objective

本实验比较 ByteTrack、自定义 BoT-SORT-Lite、自定义 OC-SORT、官方 BoT-SORT、官方 OC-SORT 在 UAV/USV 小目标持续跟踪任务中的 ID 保持能力和轨迹连续性。

## 2. Dataset Split

- Dataset: `E:\dataset\MOT`
- Split method: 按 sequence / 时间顺序连续划分
- Tune: 前 20%
- Compare: 中间 40%
- Test: 后 40%
- Random frame split: 未使用

## 3. Fixed Conditions

- Detector:
- Detector weights/hash:
- Detection input size:
- NMS threshold:
- Evaluator/version:
- Metrics IoU threshold:
- Hardware:
- Code commit:

## 4. Tracker Tuning Protocol

每个 tracker 在 tune set 上最多评估 40 组候选参数。选择规则按 HOTA、IDF1、IDSW、Frag、FPS 优先级执行，并加入 per-sequence 稳健性约束。compare set 和 test set 不用于回改 tracker 参数。

## 5. Best Configs From Tune Set

| Tracker | Best config | HOTA | IDF1 | IDSW | Frag | FPS | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| ByteTrack | `configs/best_bytetrack.yaml` |  |  |  |  |  |  |
| BoT-SORT-Lite | `configs/best_botsort.yaml` |  |  |  |  |  |  |
| OC-SORT | `configs/best_ocsort.yaml` |  |  |  |  |  |  |
| Official BoT-SORT | `configs/best_official_botsort.yaml` |  |  |  |  |  |  |
| Official OC-SORT | `configs/best_official_ocsort.yaml` |  |  |  |  |  |  |

## 6. Tracker-only Compare Set Results

| Rank | Tracker | HOTA | IDF1 | MOTA | FP | FN | IDSW | Frag | FPS | Worst sequence | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 |  |  |  |  |  |  |  |  |  |  |  |

Conclusion from compare set:

- Selected tracker:
- Reason:
- ID stability observation:
- Failure cases:

## 7. Detection Threshold Sweep

| det_conf | HOTA | IDF1 | MOTA | FP | FN | IDSW | Frag | FPS | Conclusion |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.2 |  |  |  |  |  |  |  |  |  |
| 0.3 |  |  |  |  |  |  |  |  |  |
| 0.4 |  |  |  |  |  |  |  |  |  |
| 0.5 |  |  |  |  |  |  |  |  |  |
| 0.6 |  |  |  |  |  |  |  |  |  |

Selected `det_conf`:

Reason:

## 8. ReID Ablation For Official BoT-SORT

| with_reid | HOTA | IDF1 | IDSW | Frag | FPS | Notes |
|---:|---:|---:|---:|---:|---:|---|
| false |  |  |  |  |  |  |
| true |  |  |  |  |  |  |

Conclusion:

## 9. Final End-to-end Test Set Result

This section reports the locked configuration on the final test set. The test set was not used for parameter selection.

| Tracker | det_conf | HOTA | IDF1 | MOTA | FP | FN | IDSW | Frag | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |  |

## 10. Per-sequence Robustness

| Sequence | HOTA | IDF1 | MOTA | IDSW | Frag | Failure mode |
|---|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |

## 11. Final Conclusion

- Best tracker under controlled detection:
- Recommended detection threshold:
- Main ID stability finding:
- Main failure cases:
- Engineering recommendation:

## 12. Reproducibility Checklist

- [ ] Split files saved
- [ ] Detection files saved
- [ ] All best configs locked
- [ ] All trial CSVs saved
- [ ] Per-sequence metrics saved
- [ ] Logs saved
- [ ] Final test run executed once
```

## 13. 可执行实验 checklist

- [ ] 创建 `experiments/mot_tracker_eval/` 目录结构。
- [ ] 扫描 `E:\dataset\MOT`，生成 `dataset_audit.csv`。
- [ ] 按 sequence / 时间顺序生成 20% / 40% / 40% split 文件。
- [ ] 固定检测模型、NMS、输入尺寸、类别映射。
- [ ] 生成主实验 `det_conf=0.3` 或 `0.4` 的统一 `det.txt`。
- [ ] 生成 threshold sweep 所需 `det_conf=0.2/0.3/0.4/0.5/0.6` 的 `det.txt`。
- [ ] 可选生成低阈值补充实验 `det_conf=0.1` 或 `0.2` 的 `det.txt`。
- [ ] 在 tune set 上运行 ByteTrack 粗搜索。
- [ ] 在 tune set 上运行 ByteTrack 细化搜索。
- [ ] 写入 `configs/best_bytetrack.yaml`。
- [ ] 在 tune set 上运行自定义 BoT-SORT-Lite 粗搜索。
- [ ] 在 tune set 上运行自定义 BoT-SORT-Lite 细化搜索。
- [ ] 写入 `configs/best_botsort.yaml`。
- [ ] 在 tune set 上运行自定义 OC-SORT 粗搜索。
- [ ] 在 tune set 上运行自定义 OC-SORT 细化搜索。
- [ ] 写入 `configs/best_ocsort.yaml`。
- [ ] 在 tune set 上运行官方 BoT-SORT 粗搜索。
- [ ] 在 tune set 上运行官方 BoT-SORT 细化搜索。
- [ ] 写入 `configs/best_official_botsort.yaml`。
- [ ] 在 tune set 上运行官方 OC-SORT 粗搜索。
- [ ] 在 tune set 上运行官方 OC-SORT 细化搜索。
- [ ] 写入 `configs/best_official_ocsort.yaml`。
- [ ] 汇总 `metrics/tuning_summary.csv`。
- [ ] 使用所有 best config 在 compare set 上运行 Tracker-only。
- [ ] 汇总 `metrics/tracker_only_summary.csv` 和 `metrics/per_sequence_metrics.csv`。
- [ ] 根据 compare set 选择 tracker，不回改内部参数。
- [ ] 固定该 tracker 的 best_config，运行 Threshold sweep。
- [ ] 汇总 `metrics/threshold_sweep_summary.csv`。
- [ ] 根据 compare set 选择最终 `det_conf`。
- [ ] 对官方 BoT-SORT 做 `with_reid=false/true` 补充对照，并记录是否用于最终系统选择。
- [ ] 写入 `configs/final_system.yaml`。
- [ ] 在 test set 上只运行一次 End-to-end。
- [ ] 汇总 `metrics/end_to_end_summary.csv`。
- [ ] 写最终报告。

## 14. Implementation Task List

### Task 1: Create Experiment Skeleton

**Files:**
- Create: `experiments/mot_tracker_eval/README.md`
- Create: `experiments/mot_tracker_eval/dataset/.gitkeep`
- Create: `experiments/mot_tracker_eval/detections/.gitkeep`
- Create: `experiments/mot_tracker_eval/tuning/.gitkeep`
- Create: `experiments/mot_tracker_eval/configs/.gitkeep`
- Create: `experiments/mot_tracker_eval/tracker_results/.gitkeep`
- Create: `experiments/mot_tracker_eval/metrics/.gitkeep`
- Create: `experiments/mot_tracker_eval/logs/runtime/.gitkeep`
- Create: `experiments/mot_tracker_eval/logs/gpu_memory/.gitkeep`
- Create: `experiments/mot_tracker_eval/logs/cpu_memory/.gitkeep`

- [ ] **Step 1: Create directories**

Run:

```bash
mkdir -p experiments/mot_tracker_eval/{dataset,detections,tuning/configs,configs,tracker_results,metrics,logs/runtime,logs/gpu_memory,logs/cpu_memory}
```

Expected: directories exist.

- [ ] **Step 2: Add README with fixed protocol**

Create `experiments/mot_tracker_eval/README.md` with:

```markdown
# MOT Tracker Evaluation

Dataset: `E:\dataset\MOT`

This experiment uses ordered 20% / 40% / 40% splits:

- Tune set: parameter search only
- Compare set: tracker selection and threshold sweep only
- Test set: final one-shot reporting only

Rules:

- No random frame split.
- Tracker-only uses the same `det.txt` for all trackers.
- Each tracker receives at most 40 tuning trials.
- Compare/test results must not be used to modify tracker internal parameters.
- Test set must be run once after `configs/final_system.yaml` is locked.
```

- [ ] **Step 3: Commit**

Run:

```bash
git add experiments/mot_tracker_eval
git commit -m "docs: add MOT tracker evaluation skeleton"
```

Expected: commit created.

### Task 2: Generate Ordered Dataset Splits

**Files:**
- Create: `experiments/mot_tracker_eval/dataset/split_tune.txt`
- Create: `experiments/mot_tracker_eval/dataset/split_compare.txt`
- Create: `experiments/mot_tracker_eval/dataset/split_test.txt`
- Create: `experiments/mot_tracker_eval/dataset/split_manifest.csv`

- [ ] **Step 1: List sequences in order**

Run:

```bash
conda run -n ship_detect python -c "from pathlib import Path; root=Path(r'E:\dataset\MOT'); seqs=sorted([p.name for p in root.iterdir() if p.is_dir()]); print('\n'.join(seqs))"
```

Expected: ordered sequence list printed.

- [ ] **Step 2: Write split files**

Use the ordered sequence list:

```python
from pathlib import Path

root = Path(r"E:\dataset\MOT")
out = Path("experiments/mot_tracker_eval/dataset")
seqs = sorted([p.name for p in root.iterdir() if p.is_dir()])

n = len(seqs)
tune_end = max(1, round(n * 0.20))
compare_end = tune_end + max(1, round(n * 0.40))

splits = {
    "tune": seqs[:tune_end],
    "compare": seqs[tune_end:compare_end],
    "test": seqs[compare_end:],
}

for split, names in splits.items():
    (out / f"split_{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

with (out / "split_manifest.csv").open("w", encoding="utf-8") as f:
    f.write("split,sequence,start_frame,end_frame,num_frames,source_path,split_reason\n")
    for split, names in splits.items():
        for name in names:
            seq_dir = root / name
            img_dir = seq_dir / "img1"
            frames = sorted(img_dir.glob("*")) if img_dir.exists() else []
            start = frames[0].stem if frames else ""
            end = frames[-1].stem if frames else ""
            f.write(f"{split},{name},{start},{end},{len(frames)},{seq_dir},sequence_order\n")
```

- [ ] **Step 3: Verify split counts**

Run:

```bash
wc -l experiments/mot_tracker_eval/dataset/split_tune.txt experiments/mot_tracker_eval/dataset/split_compare.txt experiments/mot_tracker_eval/dataset/split_test.txt
```

Expected: tune is about 20%, compare about 40%, test about 40%.

- [ ] **Step 4: Commit**

Run:

```bash
git add experiments/mot_tracker_eval/dataset
git commit -m "data: add ordered MOT evaluation splits"
```

Expected: commit created.

### Task 3: Run Tuning And Lock Configs

**Files:**
- Create: `experiments/mot_tracker_eval/tuning/*_trials.csv`
- Create: `experiments/mot_tracker_eval/configs/best_*.yaml`
- Create: `experiments/mot_tracker_eval/metrics/tuning_summary.csv`

- [ ] **Step 1: Generate trial configs**

For each tracker, create coarse trial YAML files from the parameter spaces in section 5. Do not exceed 24 coarse trials.

- [ ] **Step 2: Evaluate every coarse trial on full tune set**

Run each tracker through the same tune split and same detection directory.

Expected: every trial appends one row to the matching `*_trials.csv`.

- [ ] **Step 3: Select top 3 by section 6 rules**

Expected: top 3 trial IDs are identified per tracker.

- [ ] **Step 4: Generate fine trials**

Generate at most 16 fine trials around the top 3.

- [ ] **Step 5: Evaluate every fine trial on full tune set**

Expected: every fine trial appends one row to the matching `*_trials.csv`.

- [ ] **Step 6: Write best config**

Write the selected config to the tracker-specific `configs/best_*.yaml` with `locked: true`.

- [ ] **Step 7: Commit**

Run:

```bash
git add experiments/mot_tracker_eval/tuning experiments/mot_tracker_eval/configs experiments/mot_tracker_eval/metrics/tuning_summary.csv
git commit -m "exp: lock tuned MOT tracker configs"
```

Expected: commit created.

### Task 4: Run Tracker-only Compare And Test

**Files:**
- Create: `experiments/mot_tracker_eval/metrics/tracker_only_summary.csv`
- Update: `experiments/mot_tracker_eval/metrics/per_sequence_metrics.csv`
- Create: `experiments/mot_tracker_eval/tracker_results/<tracker>/compare/best/<sequence>/tracks.txt`
- Create: `experiments/mot_tracker_eval/tracker_results/<tracker>/test/best/<sequence>/tracks.txt`

- [ ] **Step 1: Run compare set with locked configs**

Expected: all five trackers have compare set results.

- [ ] **Step 2: Write tracker-only summary**

Use the CSV schema in section 11.2.

- [ ] **Step 3: Select tracker from compare set**

Record the selected tracker and rationale in `experiments/mot_tracker_eval/README.md`.

- [ ] **Step 4: Run tracker-only test once**

Expected: test set tracker-only result is recorded and no parameters are changed afterward.

- [ ] **Step 5: Commit**

Run:

```bash
git add experiments/mot_tracker_eval/tracker_results experiments/mot_tracker_eval/metrics experiments/mot_tracker_eval/README.md
git commit -m "exp: add tracker-only MOT comparison results"
```

Expected: commit created.

### Task 5: Run Threshold Sweep

**Files:**
- Create: `experiments/mot_tracker_eval/metrics/threshold_sweep_summary.csv`
- Update: `experiments/mot_tracker_eval/configs/final_system.yaml`

- [ ] **Step 1: Fix selected tracker and config**

Expected: selected tracker comes from compare set, config comes from tune set best config.

- [ ] **Step 2: Generate detection files**

Generate `det.txt` for `det_conf=0.2/0.3/0.4/0.5/0.6` with fixed detector and NMS.

- [ ] **Step 3: Evaluate compare set for each det_conf**

Expected: one summary row per threshold.

- [ ] **Step 4: Select final det_conf by section 7 rules**

Expected: selected threshold is recorded before final test.

- [ ] **Step 5: Write final system config**

Create `experiments/mot_tracker_eval/configs/final_system.yaml`:

```yaml
selected_from_split: compare
test_set_used_for_selection: false
det_conf: 0.3
nms_threshold: 0.5
tracker: selected_tracker_name
tracker_config: experiments/mot_tracker_eval/configs/best_selected_tracker.yaml
locked_before_test: true
```

- [ ] **Step 6: Commit**

Run:

```bash
git add experiments/mot_tracker_eval/detections experiments/mot_tracker_eval/metrics/threshold_sweep_summary.csv experiments/mot_tracker_eval/configs/final_system.yaml
git commit -m "exp: select MOT detection threshold on compare set"
```

Expected: commit created.

### Task 6: Run Final End-to-end Test Once

**Files:**
- Create: `experiments/mot_tracker_eval/metrics/end_to_end_summary.csv`
- Create: `experiments/mot_tracker_eval/tracker_results/final_end_to_end/<sequence>/tracks.txt`
- Create: `experiments/mot_tracker_eval/logs/runtime/end_to_end_test.log`

- [ ] **Step 1: Confirm final config is locked**

Run:

```bash
grep -n "locked_before_test: true" experiments/mot_tracker_eval/configs/final_system.yaml
```

Expected: the line is present.

- [ ] **Step 2: Run final test**

Run the complete detection + tracker system on `split_test.txt` once.

Expected: final tracking outputs and logs are created.

- [ ] **Step 3: Evaluate final outputs**

Expected: `end_to_end_summary.csv` and final per-sequence rows are written.

- [ ] **Step 4: Commit**

Run:

```bash
git add experiments/mot_tracker_eval/metrics/end_to_end_summary.csv experiments/mot_tracker_eval/tracker_results/final_end_to_end experiments/mot_tracker_eval/logs/runtime/end_to_end_test.log
git commit -m "exp: add final one-shot MOT end-to-end result"
```

Expected: commit created.

