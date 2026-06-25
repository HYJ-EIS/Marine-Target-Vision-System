# MS-DC-ELT Literature Search Protocol

## 1. 研究问题

本文调研围绕以下问题展开：在 UAV/USV 空海小目标视频中，检测置信度波动、短时漏检和低置信目标同时存在时，现有在线多目标跟踪方法如何完成 detection-to-track association、轨迹确认、lost track 保留、track reactivation 和 track termination，以及这些方法对 MS-DC-ELT 的问题定位有什么启发。

## 2. 检索关键词

第一类：基于检测关联范式的在线多目标跟踪方法

- `tracking-by-detection multi-object tracking`
- `online multi-object tracking`
- `detection-to-track association`
- `data association in MOT`
- `motion based association MOT`
- `appearance based association MOT`
- `geometry based association MOT`
- `high confidence low confidence detection association`
- `ByteTrack low score detections`
- `OC-SORT observation-centric recovery`
- `BoT-SORT robust association`
- `Deep OC-SORT adaptive re-identification`
- `Hybrid-SORT weak cues online multi-object tracking`

第二类：面向检测不稳定的轨迹延续与生命周期管理方法

- `low-confidence detections in MOT`
- `uncertain detections multi-object tracking`
- `track lifecycle management multi-object tracking`
- `track confirmation MOT`
- `lost track recovery`
- `track reactivation`
- `post-linking multi-object tracking`
- `missing detection missing association MOT`
- `trajectory fragmentation reduction`
- `ConfTrack confidence score detection box`
- `BoostTrack detection confidence`
- `StrongSORT AFLink GSI missing association missing detection`
- `MDP online multi-object tracking lost inactive`

补充边界检索

- `UAV USV small target multi-object tracking low confidence detection`
- `marine target multi-object tracking detection confidence`
- `small object MOT low confidence detections`

## 3. 检索数据库与来源

- Web search over arXiv, CVF Open Access, ACM Digital Library, Springer, Semantic Scholar pages, official project pages and paper PDFs.
- Local repository scan: `papers/` and `literature/` were checked; no local PDF paper library was found in this workspace.
- ARIS `arxiv_fetch.py` and `semantic_scholar_fetch.py` helpers were attempted but were not available in this repository or the installed skill paths. The search therefore used web results and primary paper pages where available.
- For method details, primary sources were preferred: arXiv/CVF/ACM/Springer paper pages and official repositories. Blogs were used only as secondary implementation context and not as the sole basis for conclusions.

## 4. 纳入标准

Papers were included when they satisfy at least one of the following:

1. They are widely used or foundational online tracking-by-detection MOT methods.
2. They directly describe detection-to-track association with motion, geometry, appearance, confidence score, or uncertainty.
3. They explicitly use low-confidence detections, detection confidence, or detection uncertainty in MOT.
4. They explicitly define track states, confirmation, lost track handling, reactivation, termination, post-linking, or missing-detection interpolation.
5. They are recent papers from 2022 to 2026 that update the above directions.
6. They help establish the boundary between MS-DC-ELT and methods based on ReID, GMC, end-to-end tracking, ROI redetection, or offline post-processing.

## 5. 排除标准

Papers were excluded from representative status when:

1. They are single-object tracking or multimodal single-object tracking papers.
2. They are 3D-only or multi-camera-only methods whose core setting does not transfer cleanly to 2D online image-plane MOT.
3. They require modalities or components not present in current MS-DC-ELT, such as RGB/IR fusion, LiDAR, radar, learned ReID embeddings, global camera-motion compensation, ROI redetection, or template locking.
4. They mention confidence in passing but do not use it for association, lifecycle, or track management.
5. They are application papers that simply plug in ByteTrack/DeepSORT without a clear contribution to detection instability, lifecycle management, or association.

## 6. 分类标准

Final papers were classified by the role played in the tracking pipeline:

1. **Online tracking-by-detection association**: methods whose main contribution is frame-by-frame matching between detections and existing tracks. Representative cues include IoU, Kalman prediction, Mahalanobis distance, velocity direction, appearance embedding, camera-motion compensation, and Hungarian matching.
2. **Confidence-aware and low-confidence detection usage**: methods that explicitly keep, penalize, boost, or re-rank detections below the usual high-confidence threshold.
3. **Track lifecycle, lost recovery, and fragmentation reduction**: methods that define track states, confirmation policies, lost buffers, reactivation, termination, tracklet linking, or interpolation for missing detections.
4. **Boundary methods**: end-to-end query or transformer trackers and strong ReID/GMC trackers that are relevant to MOT but should not be described as solving the same engineering problem as current MS-DC-ELT.

## 7. Representative-paper selection rule

A paper is marked representative only if it satisfies both:

1. It is methodologically central to one of the above categories.
2. Its relation to MS-DC-ELT can be stated without adding components not implemented in the current project.

For example, ByteTrack is representative for low-score detection association. BoT-SORT is representative for motion, appearance and camera-motion compensation association, but it should not be used as evidence that MS-DC-ELT has ReID or GMC. StrongSORT++ is representative for post-linking and interpolation, but its AFLink/GSI components are not the current online MS-DC-ELT mechanism.

