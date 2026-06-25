# Method V1: MS-DC-ELT v3

## 3. Method

MS-DC-ELT v3 is an online lifecycle layer placed after the detector and before MOT result export. Given the same detector output, it decides how high-confidence detections, low-confidence detections, and missed frames update track states. The method does not introduce appearance ReID, global motion compensation, ROI redetection, template matching, or offline tracklet linking. Its scope is the state transition process among candidate, active, lost, and removed tracks under unstable detector confidence.

### 3.1 Problem Setup and Lifecycle States

For frame \(t\), the detector produces bounding boxes with confidence scores and class labels. MS-DC-ELT v3 first obtains a low-threshold detection set and derives the high-threshold subset from the same detector output. Let \(D_t^{L}\) denote detections above the low threshold and \(D_t^{H}\subseteq D_t^{L}\) denote detections above the normal detector threshold. For visible videos, the low threshold is 0.18 and the normal threshold is 0.50. For infrared videos, the low threshold is 0.22 and the normal threshold is 0.65.

Each detection is converted into an observation

$$
o_t = (b_t, s_t, q_t, c_t),
$$

where \(b_t\) is the box, \(s_t\) is the detector confidence, \(q_t\in\{\text{high\_det},\text{low\_det},\text{reacquire}\}\) is the observation source, and \(c_t\) is the class label. Observations that overlap strongly are merged into one observation group. In the implementation, observation groups are merged when their IoU is at least 0.5.

Each track \(\tau_i\) stores

$$
\tau_i=(b_i, v_i, z_i, e_i, h_i, m_i, a_i, r_i, \mathcal{H}^{low}_i, \mathcal{S}_i),
$$

where \(b_i\) is the current box, \(v_i\) is the one-step box displacement used by geometric gates, \(z_i\) is the lifecycle state, \(e_i\) is the evidence score, \(h_i\) is the hit count, \(m_i\) is the miss count, \(a_i\) is the age, \(r_i\) is the number of real detection hits, \(\mathcal{H}^{low}_i\) records recent low-confidence detections, and \(\mathcal{S}_i\) records recent observation sources. The lifecycle state is

$$
z_i\in\{\text{LOW\_CANDIDATE},\text{CANDIDATE},\text{ACTIVE},\text{LOST},\text{REMOVED}\}.
$$

`LOW_CANDIDATE` stores tracks initialized only from low-confidence observations. `CANDIDATE` stores ordinary candidate tracks that can be confirmed by sufficient evidence and high-confidence detection history. `ACTIVE` tracks are public output tracks. `LOST` tracks are confirmed tracks kept for a limited number of missed frames. `REMOVED` tracks are logically retired tracks whose recent signatures can still be used to avoid immediate ID reuse.

Only `ACTIVE` tracks are returned as MOT outputs by default. A public track ID is assigned when a track first becomes `ACTIVE`; hidden candidates keep only internal IDs. This separation prevents low-confidence observations from immediately appearing as output tracks.

### 3.2 Dual-Threshold Observation Admission

The first problem is how low-confidence detections enter the tracker without flooding the state machine with background boxes. MS-DC-ELT v3 uses a dual-threshold input path. A single low-threshold detector pass produces \(D_t^{L}\), and the high-confidence set is derived as

$$
D_t^{H}=\{d\in D_t^{L}\mid s(d)\geq \theta_H\}.
$$

Low-only detections are detections in \(D_t^L\) that do not overlap a high-confidence detection above the high-low IoU threshold 0.5:

$$
D_t^{lo}=\{d\in D_t^{L}\mid \max_{d'\in D_t^{H}}\operatorname{IoU}(d,d') < 0.5\}.
$$

Before low-only detections are converted into observations, v3 applies a budget. A low-only detection must have confidence at least 0.25. When active or lost tracks exist, it must also be near at least one such track according to a geometric gate:

$$
\max_i \operatorname{IoU}(d,\hat{b}_i)\geq 0.01
\quad \text{or} \quad
\min_i \operatorname{dist}(d,\hat{b}_i)\leq 240,
$$

where \(\hat{b}_i\) is the one-step predicted box of an active or lost track. Among the remaining low-only detections, v3 keeps the global top 32 by confidence, additionally keeps the nearest low-only detection for each active or lost track, and caps the final low-only observations at 64 per frame. These thresholds control the number of weak observations before they enter lifecycle association.

High detections and retained low-only detections are converted into observation groups. Existing `LOW_CANDIDATE`, `CANDIDATE`, and `ACTIVE` tracks are associated to observation groups using IoU and center distance. A pair is eligible if

$$
\operatorname{IoU}(\hat{b}_i,g_j)\geq 0.2
\quad \text{or} \quad
\operatorname{dist}(\hat{b}_i,g_j)\leq 80.
$$

Eligible pairs are ranked by

$$
A(i,j)=\operatorname{IoU}(\hat{b}_i,g_j)
 + 0.5\max\left(0,1-\frac{\operatorname{dist}(\hat{b}_i,g_j)}{80}\right),
$$

and greedily matched from high to low score. Unmatched observation groups may spawn new tracks. A group containing a high-confidence detection spawns an ordinary `CANDIDATE`. A low-only group spawns a `LOW_CANDIDATE` only if low-candidate mode is enabled and its low-confidence score is at least 0.30. New low-only candidates near existing active tracks are suppressed with a wider center-distance gate, because such detections often correspond to duplicate boxes around an already tracked target.

This module answers how low-confidence detections enter the tracker: they are not directly output, and they are not treated as ordinary high-confidence detections. They enter as budgeted observations and may create hidden low-confidence candidates only after geometric suppression.

### 3.3 Evidence-Based Lifecycle Update

The second problem is how tracks are confirmed, retained, and deleted when detections fluctuate. MS-DC-ELT v3 uses an evidence score updated from matched observations and missed frames. For an observation group \(g_t\), the positive evidence increment is

$$
\Delta(g_t)=\sum_{q\in Q(g_t)} w_q s_q,
$$

where \(Q(g_t)\) is the set of sources in the group. The default source weights are

$$
w_{\text{high\_det}}=1.3,\quad
w_{\text{low\_det}}=1.0,\quad
w_{\text{reacquire}}=1.0.
$$

When a track is matched by an observation group, its evidence score is updated as

$$
e_t=\alpha e_{t-1}+\Delta(g_t),
$$

with \(\alpha=0.85\). Its hit count increases, its miss count is reset, and the source history is updated. A matched group also updates the track box and velocity when it contains a high-confidence or reacquisition observation. For an active track matched only by a low-confidence observation, the implementation records the low-confidence history but does not refresh the primary output box. This prevents weak detections from immediately moving an active track.

When a track is unmatched, v3 applies negative evidence:

$$
e_t=\max(0,\alpha e_{t-1}-w_{neg}),
$$

where \(w_{neg}=0.5\). The miss count increases by one. This rule makes the state machine sensitive to both accumulated support and recent missed frames.

An ordinary candidate is confirmed as `ACTIVE` only when all of the following conditions hold:

$$
e_t\geq 2.5,\quad
h_t\geq 4,\quad
r_t\geq 4,\quad
\text{high\_det}\in \mathcal{S}_i,
$$

and the candidate is matched in the current frame. Thus, a candidate cannot become an output track from accumulated low-confidence observations alone under the default v3 configuration.

A low-confidence candidate uses a separate temporal gate. Let \(\mathcal{H}^{low}_i(t)\) be its low-detection history within the last 8 frames. It can be confirmed only if

$$
|\mathcal{H}^{low}_i(t)|\geq 5,\quad
\frac{1}{|\mathcal{H}^{low}_i(t)|}\sum_{o\in\mathcal{H}^{low}_i(t)}s(o)\geq 0.22,\quad
m_t\leq 1.
$$

It must also satisfy two geometric stability checks. First, its area ratio within the low-detection history must be at most 1.8:

$$
\frac{\max_{o\in\mathcal{H}^{low}_i(t)}\operatorname{area}(o)}
{\min_{o\in\mathcal{H}^{low}_i(t)}\operatorname{area}(o)}
\leq 1.8.
$$

Second, its maximum center displacement between consecutive low detections must not exceed the larger of the base association distance 80 and three times the median center step:

$$
\max_k \|c_k-c_{k-1}\|_2
\leq
\max\left(80,3\cdot \operatorname{median}_k\|c_k-c_{k-1}\|_2\right).
$$

If a `LOW_CANDIDATE` exceeds the allowed miss count, exceeds the 8-frame confirmation window, or its evidence score falls below 0.1 after age 1, it is removed. If an ordinary `CANDIDATE` is older than 5 frames or its evidence falls below 0.1 after age 1, it is also removed.

For confirmed tracks, an `ACTIVE` track becomes `LOST` when its miss count exceeds 2. A `LOST` track is kept for at most 40 frames. If it is not recovered within this window, it becomes `REMOVED`. When a track becomes `REMOVED`, v3 stores a recent retired signature. This signature is kept for 80 frames and is used to prevent immediate reuse of the same ID semantics by a new nearby candidate.

This module is the main lifecycle mechanism of MS-DC-ELT v3. It converts frame-level detection confidence into multi-frame state decisions: ordinary candidates require evidence and real high-confidence detection history, low-confidence candidates require repeated weak observations and geometric stability, active tracks tolerate short gaps, and stale tracks are eventually removed.

### 3.4 Constrained Lost-Track Recovery and ID Handling

The third problem is how a track can recover its ID after a short missed period. MS-DC-ELT v3 contains two constrained recovery paths: direct lost-track reacquisition and low-candidate inheritance. These paths are part of the implemented state machine, but the current experiments do not show stable independent gains from them. They should therefore be interpreted as auxiliary lifecycle constraints rather than the main source of the reported performance.

Direct reacquisition is attempted only at a fixed interval of 5 frames. For a lost track \(\tau_i\) and an unmatched observation group \(g_j\), v3 first computes a geometric gate using the predicted lost-track box. The center-distance threshold is scale-aware:

$$
\rho_i=\min\left(240,\max(160,4\max(w_i,h_i))\right),
$$

where \(w_i\) and \(h_i\) are the width and height of the lost track box. The observation group is eligible if

$$
\operatorname{IoU}(\hat{b}_i,g_j)\geq 0.05
\quad \text{or} \quad
\operatorname{dist}(\hat{b}_i,g_j)\leq \rho_i.
$$

Its reacquisition score is the same weighted evidence increment used in the lifecycle update:

$$
R(i,j)=\Delta(g_j).
$$

A `LOST` track returns to `ACTIVE` only if it is matched by such a group and \(R(i,j)\geq 1.5\). This preserves the old public ID rather than allocating a new one.

Low-candidate inheritance handles a different case: a stable low-confidence candidate may be spatially consistent with a recently lost track. For each ready low-confidence candidate and each lost track, v3 checks lost age, class compatibility, IoU or center-distance eligibility, and velocity consistency. The inheritance score is

$$
I(i,j)=
0.35\,\operatorname{IoU}(\hat{b}_i,b_j)
+0.25\,C(i,j)
+0.20\,V(i,j)
+0.15\,\bar{s}^{low}_j
+0.05\,T(i),
$$

where \(C(i,j)=\max(0,1-\operatorname{dist}(\hat{b}_i,b_j)/220)\), \(V(i,j)\) is the velocity-consistency score, \(\bar{s}^{low}_j\) is the average score of the low-candidate history, and \(T(i)\) is the recency score of the lost track. The inheritance path is allowed only when \(I(i,j)\geq 0.40\). If accepted, the lost track becomes `ACTIVE` and keeps its public ID, while the low-confidence candidate is merged and removed.

The implementation also contains two guards around ID handling. First, if a ready low-confidence candidate overlaps an active track, the low-confidence candidate is removed instead of becoming another active ID. Second, if a new candidate conflicts with a recent removed-track signature, v3 records that the old ID is blocked and creates a new internal ID. These guards address duplicate activation and immediate ID reuse, but the current ablation results indicate that they should be treated as supporting constraints rather than primary contributions.

Overall, MS-DC-ELT v3 is best understood as a lifecycle tracker for detection-unstable small targets. Its main mechanism is evidence-based state management under dual-threshold observations. The recovery and inheritance paths are implemented to constrain ID handling after missed detections, but the current formal evidence supports only a narrower claim: v3 improves substantially over ByteTrack in the same replay setting and reduces false negatives relative to OC-SORT and BoT-SORT, while increasing false positives and remaining weaker on several identity metrics.
