# MS-DC-ELT Gap Analysis

## 1. What existing methods already solve well

Online tracking-by-detection methods solve the base association problem well when detections are sufficiently reliable. SORT gives a fast and clear baseline for motion and IoU association. DeepSORT, BoT-SORT, GHOST and Deep OC-SORT show that appearance features can improve ID consistency. OC-SORT and Hybrid-SORT improve motion and association robustness using observations, velocity direction, confidence or other weak cues.

Low-confidence detections are no longer ignored by all modern trackers. ByteTrack is the strongest evidence: it shows that associating low-score detections to unmatched tracklets can recover true objects and reduce fragmentation. ConfTrack, BoostTrack and BoostTrack++ further show that confidence can be used as a matching cue, penalty or score-boosting term.

Track lifecycle is also a known MOT component. MDP Tracking models target lifetime explicitly. SORT-family trackers use tentative, confirmed, lost and removed style states. StrongSORT++ targets missing association and missing detection through AFLink and GSI. These works provide a valid basis for discussing track confirmation, lost buffers, reactivation and termination.

## 2. What remains insufficient for the current MS-DC-ELT problem

The most relevant gap is not that previous work ignores association or lost tracks. The gap is narrower:

1. Many methods use low-confidence detections as an association input but do not separately model low-confidence observations as delayed candidates requiring multi-frame confirmation.
2. Low-confidence detection usage is often tied to already established tracks. It is less often used to create a low-confidence candidate that can later become active only after satisfying hit count, average score, geometry stability and miss constraints.
3. Lost track recovery is usually handled by matching a later detection to a lost track, by motion correction, or by offline linking/interpolation. Fewer methods explicitly combine lost track retention with a low-confidence candidate inheritance path.
4. Detection confidence is frequently treated as a score threshold, a matching weight, or an uncertainty input. Fewer methods use a score-decay lifecycle variable to decide confirmation, lost retention and removal under intermittent detection.
5. In the specific UAV/USV sea-air small-target setting, the search did not find enough strong literature that directly studies the same combination of low-confidence detections, short missed detections, delayed candidate confirmation and ID continuation.

## 3. What MS-DC-ELT should not claim

Current MS-DC-ELT should not be described as solving all MOT association problems. Based on the project implementation and V6 evidence boundary, it does not implement:

- ReID or learned appearance embedding.
- Global camera-motion compensation.
- ROI redetection.
- Template locking.
- RGB/IR fusion.
- Offline global tracklet linking.
- Learned end-to-end temporal association.

It should not claim general superiority over OC-SORT or BoT-SORT. The current V6 evidence says MS-DC-ELT v3 has a recall-side advantage over OC-SORT/BoT-SORT in FN but weaker HOTA, IDF1, AssA, IDSW and FP. The related-work argument should therefore focus on the mechanism and problem setting, not on broad performance dominance.

## 4. How to position the research gap

The literature supports the following cautious gap statement:

Existing online MOT methods have well-developed detection-to-track association mechanisms, including motion, appearance, geometry, confidence-aware matching and low-score detection association. However, in detection-unstable small-target video, the tracker must also decide whether weak observations should create a candidate, whether a temporarily missing confirmed track should remain recoverable, and whether a later weak candidate should inherit a previous track ID. Existing methods address parts of this problem, but the combination of dual-confidence detection input, delayed low-confidence candidate confirmation, evidence decay under misses, and lost-track inheritance remains less directly studied, especially for UAV/USV sea-air small targets.

This gap is empirical and should be tested. It should not be written as proof that MS-DC-ELT is necessarily better than recent trackers. The paper should frame MS-DC-ELT as a method that explores this lifecycle-management design space and report both gains and costs.

## 5. Is MS-DC-ELT suitable as a "track lifecycle management method for detection instability"?

Yes, this is the safest and most accurate positioning, with a small wording adjustment:

**MS-DC-ELT is suitable to position as an online track lifecycle management method for detection-unstable small-target MOT.**

Reasons:

1. Its implemented state set is lifecycle-oriented: `LOW_CANDIDATE`, `CANDIDATE`, `ACTIVE`, `LOST`, and `REMOVED`.
2. Its core variables are lifecycle evidence and history: evidence score, hits, misses, real detection hits, source history and low-detection history.
3. Its low-confidence path is not just second-stage matching; it includes delayed low-candidate confirmation and pruning.
4. Its lost-track path includes retained lost states, reacquire scoring and low-candidate inheritance hooks.
5. This positioning avoids claiming unimplemented capabilities such as ReID, GMC or detector-level redetection.

The phrase "detection instability stream" is understandable, but "detection-unstable small-target MOT" or "under detection instability" is clearer and closer to common MOT wording.

