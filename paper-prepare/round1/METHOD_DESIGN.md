# METHOD_DESIGN.md

## Paper title

Evidence-Guided Track Lifecycle Management for Detection-Unstable Small-Target MOT

## Paper method name

ELTTrack: Evidence-guided Lifecycle Tracking

## Relation to implementation name

The current implementation may still use the name MS-DC-ELT in code, scripts, and experiment records. In the paper, the method should be referred to as ELTTrack.

## Method positioning

ELTTrack is an online track lifecycle management method for detection-unstable small-target MOT. It is built under the tracking-by-detection paradigm and focuses on how high-confidence detections, low-confidence detections, and missed frames jointly affect track initialization, confirmation, retention, reassociation, and removal.

The core problem addressed by ELTTrack is lifecycle-level decision making under unstable detections. In small-target UAV/USV videos, the same object may alternate between high-confidence detection, low-confidence detection, and short-term missed detection. ELTTrack therefore treats weak observations and missed frames as evidence that should be accumulated, decayed, and constrained before affecting public track IDs.

## Method modules

### 1. Dual-threshold Observation Filtering

This module converts detector outputs into controlled observations. A low detection threshold is used to preserve weak target evidence, while a high detection threshold defines reliable detections. Low-confidence detections are deduplicated against high-confidence boxes and further filtered by confidence, proximity to existing tracks, and Top-K budget constraints.

The purpose of this module is not to directly output low-confidence detections as tracks. Instead, it provides a controlled observation entrance for weak detections, allowing them to participate in the lifecycle layer without letting background clutter directly create public tracks.

### 2. Evidence-Accumulated State Update

This module maintains each track using an evidence score, hit count, miss count, real-detection hit count, and observation source history. Matched observations increase track evidence according to their source confidence, while unmatched frames decrease evidence through missed-frame decay.

Candidate tracks are confirmed as active tracks only after satisfying evidence, hit, real-detection, and source-history constraints. Active tracks are moved to lost when missing exceeds a short tolerance window, while low-quality candidates or long-missing tracks are removed.

The key idea is that track states should not be determined only by a single-frame detection score or a fixed age threshold. Instead, the state transition should depend on accumulated observation evidence and missing-frame penalties.

### 3. Geometry-Gated Track Reassociation

This module handles short-term target disappearance and reappearance. Lost tracks are allowed to reconnect with new observations or qualified low-confidence candidates only when geometric constraints are satisfied, including IoU, center distance, scale consistency, and temporal age constraints.

This reassociation is constrained to reduce ID reuse and erroneous reconnection under weak detections. It should be described as geometry-gated reassociation, not as appearance-based re-identification or global tracklet linking.

## Method structure in the paper

# 3. Method

## 3.1 Dual-threshold Observation Filtering

## 3.2 Evidence-Accumulated State Update

## 3.3 Geometry-Gated Track Reassociation

## Writing constraints

- Write Introduction, Related Work, and Method only.
- Do not write Experiment.
- Do not use current weak experiment results to weaken the method design.
- Do not claim overall superiority over OC-SORT or BoT-SORT.
- Do not invent modules not implemented in the current code.
- Do not write ReID, GMC, RGB/IR fusion, ROI redetection, or template matching as core modules.
- Use formal MOT terminology.
- Avoid over-designed or artificial terms.
- All formulas must use $$ $$.