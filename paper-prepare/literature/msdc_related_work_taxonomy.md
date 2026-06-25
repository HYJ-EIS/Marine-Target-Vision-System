# MS-DC-ELT Related Work Taxonomy

## 1. Online tracking-by-detection association

### Research line

Tracking-by-detection first detects objects in each frame and then associates detections to existing tracks. SORT established the lightweight form of this pipeline: Kalman prediction, IoU cost and Hungarian assignment. DeepSORT added appearance embeddings to handle longer occlusions and reduce ID switches. Later SORT-family methods improved the association cue set: OC-SORT focuses on observation-centric motion recovery, BoT-SORT combines motion, ReID and camera-motion compensation, Deep OC-SORT adaptively introduces appearance, and Hybrid-SORT adds weak cues such as confidence and height state.

### Representative methods

- SORT: motion prediction plus IoU/Hungarian association.
- DeepSORT: motion gating plus ReID feature association.
- OC-SORT: observation-centric re-update and momentum for missing measurements and nonlinear motion.
- BoT-SORT: motion, appearance and camera-motion compensation.
- Deep OC-SORT and Hybrid-SORT: recent extensions using adaptive appearance or weak cues.

### What this line solves well

- Efficient online association under relatively reliable detection.
- Short-term motion extrapolation and frame-to-frame matching.
- Identity preservation when appearance embeddings or camera-motion compensation are available.
- Robustness to occlusion and nonlinear motion when observations are still recoverable.

### Remaining limitation for MS-DC-ELT problem

This line mainly improves association costs and motion/appearance cues. It does not always treat low-confidence detections as candidates requiring multi-frame confirmation, and its track lifecycle is often controlled by fixed buffers or heuristic state rules. For sea-air small targets, where the detector output itself is unstable, association cues alone may not decide when low-confidence observations should confirm a new track, extend an existing one, or be suppressed as noise.

## 2. Low-confidence detection and confidence-aware tracking

### Research line

ByteTrack changed the usual high-threshold filtering strategy by associating low-score detections after high-score detections. ConfTrack, BoostTrack, BoostTrack++ and Hybrid-SORT further show that detection confidence can affect matching order, similarity scaling, penalization and weak-cue association. UncertaintyTrack and UTrack go beyond scalar confidence by using localization uncertainty or predictive distributions.

### Representative methods

- ByteTrack: two-stage high/low score association.
- ConfTrack: low-confidence penalization and cascade matching for noisy detections.
- BoostTrack / BoostTrack++: detection-tracklet confidence and soft confidence boosting.
- Hybrid-SORT: confidence as a weak cue among velocity direction and height state.
- UTrack / UncertaintyTrack: uncertainty-aware association when detectors provide uncertainty.

### What this line solves well

- It recognizes that low-confidence detections can include true targets.
- It reduces FN and fragmentation when low-score boxes can be linked to existing tracks.
- It avoids treating all low-score detections equally by using similarity, confidence or uncertainty.

### Remaining limitation for MS-DC-ELT problem

Most methods use low-confidence detections inside association rather than as a delayed lifecycle state. ByteTrack, for example, uses low-score detections mainly to recover unmatched established tracklets, not to create a separate low-confidence candidate that must satisfy multi-frame evidence conditions before becoming output. BoostTrack-style methods adjust confidence or similarity but do not directly answer how long a low-confidence candidate should be retained, when it should inherit a lost ID, or when it should be removed.

## 3. Track lifecycle, lost recovery and fragmentation reduction

### Research line

Track management appears in most online trackers, but it is often simple: tentative tracks become confirmed after several hits, and unmatched tracks are deleted after an age threshold. MDP Tracking explicitly formulates target lifetime as states such as active, tracked, lost and inactive. StrongSORT++ addresses missing association and missing detection by adding AFLink and GSI. OC-SORT also contains recovery logic for tracks after missing measurements.

### Representative methods

- MDP Tracking: target lifetime as decision states.
- SORT/DeepSORT family: tentative/confirmed/lost or active/lost/removed style rules.
- OC-SORT: observation-centric re-update when a track is recovered.
- StrongSORT++: AFLink for missing association and GSI for missing detection.

### What this line solves well

- It provides standard mechanisms for track confirmation, lost buffer, reactivation and termination.
- It reduces trajectory fragmentation by linking tracklets or interpolating missing detections.
- It clarifies the trade-off between retaining a lost track and preventing false association.

### Remaining limitation for MS-DC-ELT problem

The literature contains lifecycle mechanisms, but fewer works focus specifically on low-confidence observations as first-class lifecycle evidence. Many methods either confirm new tracks after a fixed number of hits, recover lost tracks through association, or repair trajectories offline. The combination of high/low confidence detection streams, delayed low-confidence candidate confirmation, evidence decay under misses, and low-candidate inheritance of lost IDs is less directly covered.

## 4. End-to-end and query-based tracking

### Research line

Transformer trackers such as TransTrack, TrackFormer and MOTR reduce or remove explicit handcrafted association by using object queries, track queries or attention. They are important for a full MOT survey because they represent the alternative to explicit tracking-by-detection association.

### Representative methods

- TransTrack: query-key association and learned object queries.
- TrackFormer: track queries in a tracking-by-attention framework.
- MOTR: transferred track queries and temporal modeling.

### Relevance and limitation

These methods address identity persistence with learned temporal modeling. They are not close implementation priors for current MS-DC-ELT, which is an online, detector-output-level lifecycle layer. They should be mentioned briefly as a different MOT family, not as the main comparison for low-confidence track lifecycle management.

## 5. Search gap for UAV/USV sea-air small targets

The search found many generic pedestrian MOT, DanceTrack, MOT17/MOT20 and autonomous-driving MOT papers. It did not find enough strong, directly relevant papers that simultaneously satisfy all of the following:

- UAV/USV sea-air small target setting.
- Online 2D MOT.
- Explicit use of low-confidence detections.
- Delayed low-confidence candidate confirmation.
- Lost-track reactivation or ID inheritance based on multi-frame evidence.

Therefore, the related-work draft should not claim that this exact combination has been comprehensively studied. It should say that existing work has separately studied online association, low-score detection usage, confidence-aware matching, and track lifecycle management.

