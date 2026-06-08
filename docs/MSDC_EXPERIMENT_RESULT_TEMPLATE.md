# MS-DC-ELT Experiment Result Template

This template is intentionally blank. Do not fill values until the corresponding experiment has actually been run.

## 1. Main Results

| Dataset / Sequence | Modality | Method | Detector | Tracker | MOTA | IDF1 | HOTA | DetA | AssA | FP | FN | IDSW | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|  |  | FFCA-YOLO + OC-SORT |  | ocsort |  |  |  |  |  |  |  |  |  |
|  |  | FFCA-YOLO + BoT-SORT |  | botsort |  |  |  |  |  |  |  |  |  |
|  |  | MS-DC-ELT |  | msdc_elt |  |  |  |  |  |  |  |  |  |

## 2. Ablation Results

| Dataset / Sequence | Variant | MSDC_USE_LOW_DET | MSDC_USE_MOTION | MSDC_USE_TEMPLATE | MSDC_USE_REACQUIRE | MSDC_REUSE_GUARD_ENABLE | MOTA | IDF1 | HOTA | IDSW | Notes |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
|  | Ours-lite-no-motion | True | False | True | True | True |  |  |  |  |  |
|  | Ours-lite-no-low-det | False | True | True | True | True |  |  |  |  |  |
|  | Ours-no-template | True | True | False | True | True |  |  |  |  |  |
|  | Ours-no-reacquire | True | True | True | False | True |  |  |  |  |  |
|  | Ours-no-removed-guard | True | True | True | True | False |  |  |  |  |  |
|  | Ours-full | True | True | True | True | True |  |  |  |  |  |

## 3. Stress Tests

| Dataset / Sequence | Stress Slice | Method | Frames | Target Count | Occlusion Level | Camera Motion | MOTA | IDF1 | IDSW | Failure Notes |
|---|---|---|---:|---:|---|---|---:|---:|---:|---|
|  | Low contrast |  |  |  |  |  |  |  |  |  |
|  | Small target |  |  |  |  |  |  |  |  |  |
|  | Heavy motion |  |  |  |  |  |  |  |  |  |
|  | Long lost gap |  |  |  |  |  |  |  |  |  |

## 4. Speed Tests

| Hardware | Dataset / Sequence | Method | Resolution | Frames | Mean FPS | P50 Latency ms | P95 Latency ms | Peak Memory MB | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|
|  |  | FFCA-YOLO + OC-SORT |  |  |  |  |  |  |  |
|  |  | FFCA-YOLO + BoT-SORT |  |  |  |  |  |  |  |
|  |  | MS-DC-ELT |  |  |  |  |  |  |  |

## 5. Failure Cases

| Dataset / Sequence | Frame Range | Method | Failure Type | Description | Suspected Cause | Debug File | Follow-up |
|---|---|---|---|---|---|---|---|
|  |  |  | False confirmation |  |  |  |  |
|  |  |  | Template drift |  |  |  |  |
|  |  |  | False reacquire |  |  |  |  |
|  |  |  | ID reuse guard miss |  |  |  |  |

## 6. Lifecycle Diagnostics

| Dataset / Sequence | Method | Candidate Count Mean | Active Count Mean | Lost Count Mean | Removed Count | LOST_REACQUIRED | PREVENT_removed_ID_REUSE | Avg Reacquire Score | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|  | MS-DC-ELT |  |  |  |  |  |  |  |  |

## 7. Artifact Paths

| Run Name | MOT Result Path | Lifecycle Events JSONL | Tracks JSONL | Candidate Pool JSONL | Command |
|---|---|---|---|---|---|
|  |  |  |  |  |  |
