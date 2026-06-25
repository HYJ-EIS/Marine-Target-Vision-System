# Method Review Round 1

Review time: 2026-06-25 16:41 CST

Input draft: `paper-prepare/Method_V1.md`

Reviewer: Gemini via `gemini-review`

Thread ID: `6fe35c402aaa46f8911a66342b5a380b`

Job ID: `cd59ce1532384b39b65af8c734d5cf83`

## Reviewer Verdict

Major Revision.

## Main Criticisms

1. `Method_V1.md` is faithful to the v3 implementation, but reads too much like an engineering report.
2. Low-candidate confirmation, direct reacquire, and low-candidate inheritance occupy too much space despite weak or absent experimental support.
3. Hard-coded thresholds are scattered throughout the main method text, making the method look over-tuned.
4. The core method should foreground the evidence-driven lifecycle transition and the recall/identity trade-off.
5. Track recovery mechanisms should be moved into a short discussion/boundary section rather than presented as main contributions.

## Actions For V2

1. Reframe the method as a lightweight state-machine lifecycle layer that favors recall under unstable detections while accepting FP/identity trade-offs.
2. Merge the setup and lifecycle update into a tighter evidence-driven state-transition section.
3. Keep dual-threshold low-confidence admission as a core module, but describe Top-K/proximity filtering symbolically in the main body.
4. Move concrete thresholds into an implementation details section.
5. Reduce direct reacquire and low-inheritance formulas to a short boundary discussion.
6. Remove self-answering module-summary paragraphs.

## Reviewer Raw Response

<details>
<summary>Full Gemini response</summary>

这是一份基于严格的机器视觉/MOT（多目标跟踪）领域标准的审稿意见。

### 总体 Verdict: Major Revision (大修)

**评价总结**：
当前 Method 初稿（V1）非常诚实且极其忠实于代码实现（MS-DC-ELT v3_candidate_topk_no_roi_no_motion），在避免“过度声称（Overclaiming）”方面做出了值得肯定的尝试（例如在 3.4 节主动承认某些模块没有收益）。
**但是，从学术论文的写作标准来看，该草稿存在严重的“结构性失衡”和“工程报告化”问题。** 作者将大量篇幅、复杂的公式（如 $I(i,j)$ 的 5 项加权）和详细的规则分配给了在实验中**触发次数为 0 或极低（如 low_candidate confirm, direct reacquire, low inheritance）的无效模块**。这不仅会严重分散审稿人对核心贡献（Evidence-based Lifecycle）的注意力，还会引发致命的质疑：“为什么要用一整节介绍一个不起作用的复杂策略？” 此外，大量硬编码的超参数（如 0.18, 240, 80）直接写在正文中，降低了方法的泛化感。

---

### 🔴 CRITICAL (致命问题：直接影响录用)

**1. 严重倒置的篇幅分配：无效模块占据了核心版面**
*   **问题**：3.4 节（Constrained Lost-Track Recovery）和 3.3 节中的 `LOW_CANDIDATE` 确认逻辑占据了大量篇幅。但根据实验事实，`low inheritance` 和 `direct reacquire` 成功次数为 0，`low_candidate` 仅确认了 1-2 次且 Precision 极低。在顶会论文中，详细列出无效模块的复杂公式（如 $I(i,j)$）是学术自杀，审稿人会认为作者缺乏对算法核心机制的提炼能力。
*   **修改建议**：
    *   **彻底删减 3.4 节的公式和繁琐细节**。不需要写出 $I(i,j)$ 和 $\rho_i$ 的具体计算。
    *   将这些机制降级为 **“探讨与负面结果 (Discussion / Implementation Details)”** 中的一小段。
    *   **话术转换**：将其包装为一个有价值的科学发现。例如：“*为了探究在极度不稳定检测下是否需要复杂的重识别或轨迹继承机制，我们设计了直接重捕获和低分继承策略。然而，广泛的实验表明，在缺乏外观特征的情况下，仅靠时空几何进行重捕获的成功率极低且容易引入 FP。因此，本方法的最终有效性完全来源于 3.3 节的连续证据累积，而非断裂后的重连。*” 这样既解释了代码里有这些逻辑，又拔高了立意。

**2. 核心定位的偏离：未充分呼应实验结论**
*   **问题**：实验表明该方法比 ByteTrack 强，FN 比 OC/BoT-SORT 低，但 ID 相关的指标（HOTA/IDF1/IDSW）和 FP 较差。Method 必须为这个结果“打好预防针”，即明确该方法的设计哲学是**“在极端不稳定条件下，宁可牺牲部分 ID 连续性（不使用 ReID/GMC），也要通过严格的生命周期管理最大化召回率（降低 FN）并抑制瞬态噪声”**。
*   **修改建议**：在 3.1 节开头明确声明这一 Trade-off。强调 MS-DC-ELT v3 是一个轻量级的、纯基于状态机的防御性策略，专门针对小目标/弱检测场景。

---

### 🟠 MAJOR (重要问题：影响逻辑与可读性)

**1. 符号与状态定义过于冗长（工程味太重）**
*   **问题**：3.1 节中轨迹状态元组 `(b_i, v_i, z_i, e_i, h_i, m_i, a_i, r_i, H_i^low, S_i)` 包含 10 个变量，过于繁琐，且后续并未全部在数学推导中严格使用。
*   **修改建议**：简化状态表示。只需提及核心状态：运动状态（Box, Velocity）、生命周期属性（State $z_i$, Evidence $e_i$）和历史统计（Hits, Misses）。可以写成：“Each track maintains its kinematic state and a lifecycle profile $\mathcal{P}_i = \{z_i, e_i, h_i, m_i, S_i\}$...”，将不重要的变量放入附录或代码开源说明中。

**2. 超参数硬编码破坏了方法的泛化性**
*   **问题**：正文中充斥着 `0.18`, `0.22`, `240`, `80`, `32`, `64` 等具体数字。这让 Method 看起来像是一份调参记录（Technical Report），而不是一个通用的学术方法。
*   **修改建议**：
    *   在 3.1 - 3.3 节的正文中，全部使用符号代替，如 $\tau_{low}, \tau_{high}, \theta_{dist}, N_{budget}$。
    *   新增一个 **"3.4 Implementation Details"** 小节，集中列出这些具体的数值，并说明可见光和红外模态的阈值差异。

**3. Low-only Budget 的描述缺乏理论支撑**
*   **问题**：3.2 节中关于 low-only 观测的预算控制（top 32, nearest 1, max 64）写得像拍脑门决定的规则。
*   **修改建议**：增加一句话解释其动机（Motivation）。例如：“*To prevent the state machine from being overwhelmed by dense background noise typical in low-confidence regimes, we introduce a spatial-aware observation budget...*” 强调这是为了在计算效率和抗噪性之间取得平衡。

---

### 🟡 MINOR (次要问题：学术写作规范)

**1. 术语规范性**
*   **问题**：“public output tracks” 和 “hidden candidates” 略显口语化。
*   **修改建议**：使用 MOT 领域的标准术语。将 “hidden candidates” 称为 “Tentative tracks” 或 “Unconfirmed tracks”；将 “public output tracks” 称为 “Confirmed active tracks”。

**2. 证据更新公式的表达**
*   **问题**：3.3 节的 $e_t = \alpha e_{t-1} + \Delta(g_t)$ 表达清晰，但负更新 $e_t = \max(0, \alpha e_{t-1} - w_{neg})$ 中的 $w_{neg}$ 突然出现。
*   **修改建议**：将正负更新统一为一个状态转移方程：
    $e_t = \max(0, \alpha e_{t-1} + \mathbb{I}_{match} \cdot \Delta(g_t) - (1 - \mathbb{I}_{match}) \cdot w_{neg})$
    这样在数学上更加紧凑和优雅。

**3. 冗余表达**
*   **问题**：3.2 节末尾“This module answers how low-confidence detections enter...” 和 3.3 节末尾“This module is the main lifecycle mechanism...” 这种自问自答式的总结过于啰嗦。
*   **修改建议**：直接删除这些段落总结，学术论文应保持客观陈述，让读者自己体会模块的作用。

---

### 📝 建议的 Revised Structure (重构大纲)

为了让文章更符合顶会标准，建议按照以下结构重写 Method：

*   **3. Method** (简短的 Overview，强调无 ReID/GMC，纯靠证据累积应对高 FN 场景)
*   **3.1 Dual-Threshold Observation Budgeting** (原 3.2 节)
    *   定义 $D^H$ 和 $D^{lo}$。
    *   重点讲**空间感知预算 (Spatial-aware budget)** 的动机和符号化逻辑（过滤掉远离现有轨迹的低分框，限制最大数量），不写死具体数字。
*   **3.2 Evidence-Driven State Transition** (原 3.1 和 3.3 节的融合，**核心贡献**)
    *   定义状态空间 $\mathcal{Z} = \{\text{Candidate, Active, Lost, Removed}\}$。
    *   给出统一的 Evidence 更新公式（正向累积与反向衰减）。
    *   详细说明 Candidate -> Active 的严格确认条件（强调需要 high_det 历史，解释这正是控制 FP 的关键）。
    *   *一笔带过* `LOW_CANDIDATE`：提及为了极端情况设计了纯低分确认路径，但条件极其苛刻（8帧等）。
*   **3.3 Discussion on Track Recovery Mechanisms** (原 3.4 节的**大幅缩减版**)
    *   简述系统包含了基于纯运动学的重捕获（Reacquire）和低分继承（Inheritance）机制。
    *   **关键反转**：明确指出在当前的不稳定检测设定下，这些机制几乎不触发。借此得出结论：在缺乏外观特征时，**维持连续的证据链（3.2节）比断裂后的盲目重连更重要**，这也解释了为什么本方法在 IDSW 上不如引入了复杂重识别的 SORT 变体，但在 FN 上表现更好。
*   **3.4 Implementation Details**
    *   集中列出所有硬编码参数：$\tau_{low}=0.18/0.22$, $\tau_{high}=0.50/0.65$, $\alpha=0.85$, 预算上限 64 等。
    *   说明实验中使用的具体配置（v3_candidate_topk_no_roi_no_motion）。

</details>
