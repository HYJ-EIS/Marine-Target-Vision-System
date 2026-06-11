# MS-DC-ELT Experiment Report

## Metadata and Paths

- Main root: `/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full`
- Ablation root: `/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full`
- Summary output root: `/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/summary`

## Supplemental Run Provenance

| item | value |
| --- | --- |
| Git branch at run | MSDC |
| Formal experiment commit hash | 157ed52aa3d4d83df2a9c0c67a8d99b1cc8e51d1 |
| Conda environment | ship_detect |
| Python version | Python 3.11.4 |
| OS/kernel | Linux LAPTOP-48MSG30L 6.6.114.1-microsoft-standard-WSL2 x86_64 |
| Dataset roots | `/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集`; `/home/hyj/Anti_Drone_Project/USV_MOT标注数据集` |
| Actual source videos resolved from dataset metadata | `/mnt/d/Desktop/UAV_USV标注数据集/multi_target_source_videos/USV/RGB/DJI_20250916100639_0001_V.MP4`; `/mnt/d/Desktop/UAV_USV标注数据集/multi_target_source_videos/USV/RGB/DJI_20250711140128_0002_V.MP4` |
| Formal sequences | `DJI_20250916100639_0001_V`: 26757 frames, 3840x2160, 29.9566 FPS; `DJI_20250711140128_0002_V`: 4772 frames, 3840x2160, 29.97 FPS |
| Main experiment wall time | 2026-06-09T01:58:44+08:00 to 2026-06-09T12:37:06+08:00, 38302 s |
| Ablation experiment wall time | 2026-06-09T12:37:06+08:00 to 2026-06-10T15:32:27+08:00, 96921 s |
| Speed experiment scope | `DJI_20250916100639_0001_V`, first 1000/26757 frames, 3840x2160 |
| Speed measured processing time | OC-SORT 86.107156 s; BoT-SORT 113.874484 s; MS-DC-ELT 392.956127 s |
| Formal result status | Main, ablation, and speed commands completed with status `ok`; smoke runs were not used for the tables below |

## Run Metadata

| path | metadata |
| --- | --- |
| /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/metadata/run_metadata.json | {"commit_hash": "157ed52aa3d4d83df2a9c0c67a8d99b1cc8e51d1", "dataset_roots": ["/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集", "/home/hyj/Anti_Drone_Project/USV_MOT标注数据集"], "duration_seconds": 0.0, "formal": true, "max_frames": 0, "mode": "main", "output_root": "/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full", "render": true, "run_id": "main_full", "started_at": "2026-06-09T01:58:44+08:00", "trackers": ["ocsort", "botsort", "msdc_elt"], "variants": ["Ours-full"]} |
| /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/metadata/run_metadata.json | {"commit_hash": "157ed52aa3d4d83df2a9c0c67a8d99b1cc8e51d1", "dataset_roots": ["/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集", "/home/hyj/Anti_Drone_Project/USV_MOT标注数据集"], "duration_seconds": 0.0, "formal": true, "max_frames": 0, "mode": "ablation", "output_root": "/home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full", "render": true, "run_id": "ablation_full", "started_at": "2026-06-09T12:37:06+08:00", "trackers": ["msdc_elt"], "variants": ["Ours-full", "Ours-lite-no-motion", "Ours-lite-no-low-det", "Ours-no-template", "Ours-no-reacquire", "Ours-no-removed-guard"]} |

## Main Results

| run_name | method | tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | IDTP | IDFP | IDFN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_full | FFCA-YOLO + MS-DC-ELT | Ours-full | 55.78748 | 63.57144 | 49.75196 | 72.19103 | 52.83628 | 154 | 4172 | 26832 | 53213 | 36170 | 58830 |
| main_full | FFCA-YOLO + BoT-SORT | botsort | 75.37659 | 69.1015 | 82.32799 | 75.64506 | 82.25078 | 33 | 3327 | 23928 | 83684 | 7758 | 28359 |
| main_full | FFCA-YOLO + OC-SORT | ocsort | 69.10581 | 69.08914 | 69.45903 | 75.63168 | 69.58317 | 39 | 3332 | 23932 | 70796 | 20647 | 41247 |

## Ablation Results

| run_name | variant | MSDC_USE_LOW_DET | MSDC_USE_MOTION | MSDC_USE_TEMPLATE | MSDC_USE_REACQUIRE | MSDC_USE_ROI_REDETECT | MSDC_REUSE_GUARD_ENABLE | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | IDTP | IDFP | IDFN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ablation_full | Ours-full | True | True | True | True | True | True | 55.78748 | 63.57144 | 49.75196 | 72.19103 | 52.83628 | 154 | 4172 | 26832 | 53213 | 36170 | 58830 |
| ablation_full | Ours-lite-no-low-det | False | True | True | True | True | True | 59.54011 | 62.26641 | 57.64661 | 69.78214 | 55.29901 | 150 | 3060 | 30647 | 54331 | 30125 | 57712 |
| ablation_full | Ours-lite-no-motion | True | False | True | True | True | True | 55.46328 | 64.26132 | 48.50895 | 72.565 | 55.03717 | 147 | 4143 | 26449 | 55527 | 34210 | 56516 |
| ablation_full | Ours-no-reacquire | True | True | True | False | True | True | 57.56636 | 66.94641 | 50.79018 | 76.72322 | 49.22904 | 208 | 4919 | 20953 | 51211 | 44798 | 60832 |
| ablation_full | Ours-no-removed-guard | True | True | True | True | True | False | 55.78748 | 63.57144 | 49.75196 | 72.19103 | 52.83628 | 154 | 4172 | 26832 | 53213 | 36170 | 58830 |
| ablation_full | Ours-no-template | True | True | False | True | True | True | 67.31212 | 70.07451 | 64.93125 | 76.98294 | 66.1078 | 64 | 4184 | 21541 | 68332 | 26354 | 43711 |

## Speed Results

| run_id | commit_hash | video_path | seq_name | tracker | method | resolution | requested_frames | processed_frames | total_time_s | mean_fps | mean_latency_ms | p50_latency_ms | p95_latency_ms | peak_memory_mb | detector_calls_total | detector_calls_high_det | detector_calls_low_det | detector_calls_tracker_update | mean_read_ms | mean_high_det_ms | mean_low_det_ms | mean_tracker_ms | status | failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| speed_1000 | 157ed52aa3d4d83df2a9c0c67a8d99b1cc8e51d1 | /mnt/d/Desktop/UAV_USV标注数据集/multi_target_source_videos/USV/RGB/DJI_20250916100639_0001_V.MP4 | DJI_20250916100639_0001_V | ocsort | FFCA-YOLO + OC-SORT | 3840x2160 | 1000 | 1000 | 86.107156 | 11.613437 | 86.107156 | 81.536032 | 92.844712 | 1864.586000 | 1000 | 1000 | 0 | 0 | 10.332352 | 75.420219 | 0.000000 | 0.346783 | ok | N/A |
| speed_1000 | 157ed52aa3d4d83df2a9c0c67a8d99b1cc8e51d1 | /mnt/d/Desktop/UAV_USV标注数据集/multi_target_source_videos/USV/RGB/DJI_20250916100639_0001_V.MP4 | DJI_20250916100639_0001_V | botsort | FFCA-YOLO + BoT-SORT | 3840x2160 | 1000 | 1000 | 113.874484 | 8.781598 | 113.874484 | 111.117192 | 126.796210 | 1892.461000 | 1000 | 1000 | 0 | 0 | 8.030046 | 79.282737 | 0.000000 | 26.552481 | ok | N/A |
| speed_1000 | 157ed52aa3d4d83df2a9c0c67a8d99b1cc8e51d1 | /mnt/d/Desktop/UAV_USV标注数据集/multi_target_source_videos/USV/RGB/DJI_20250916100639_0001_V.MP4 | DJI_20250916100639_0001_V | msdc_elt | FFCA-YOLO + MS-DC-ELT | 3840x2160 | 1000 | 1000 | 392.956127 | 2.544813 | 392.956127 | 324.689275 | 613.038122 | 2113.047000 | 2952 | 1000 | 1000 | 952 | 9.950573 | 77.153770 | 79.892962 | 225.948916 | ok | N/A |

## Output Paths

| kind | path |
| --- | --- |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/data/DJI_20250711140128_0002_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/data/DJI_20250916100639_0001_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/botsort/data/DJI_20250711140128_0002_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/botsort/data/DJI_20250916100639_0001_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/ocsort/data/DJI_20250711140128_0002_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/ocsort/data/DJI_20250916100639_0001_V.txt |
| trackeval_summary | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/eval/motchallenge_summary.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250711140128_0002_V/Ours-full_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250711140128_0002_V/Ours-full_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250711140128_0002_V/Ours-full_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250711140128_0002_V/botsort_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250711140128_0002_V/botsort_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250711140128_0002_V/ocsort_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250711140128_0002_V/ocsort_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250916100639_0001_V/Ours-full_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250916100639_0001_V/Ours-full_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250916100639_0001_V/Ours-full_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250916100639_0001_V/botsort_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250916100639_0001_V/botsort_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250916100639_0001_V/ocsort_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/diagnostics/DJI_20250916100639_0001_V/ocsort_per_gt_diagnostics.csv |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/diagnostics/DJI_20250711140128_0002_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/diagnostics/DJI_20250711140128_0002_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/diagnostics/DJI_20250711140128_0002_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/diagnostics/DJI_20250711140128_0002_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/diagnostics/DJI_20250916100639_0001_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/diagnostics/DJI_20250916100639_0001_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/diagnostics/DJI_20250916100639_0001_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/trackers/Ours-full/diagnostics/DJI_20250916100639_0001_V/stage_observations.jsonl |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/visualizations/DJI_20250711140128_0002_V/Ours-full.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/visualizations/DJI_20250711140128_0002_V/botsort.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/visualizations/DJI_20250711140128_0002_V/ocsort.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/visualizations/DJI_20250916100639_0001_V/Ours-full.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/visualizations/DJI_20250916100639_0001_V/botsort.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/visualizations/DJI_20250916100639_0001_V/ocsort.mp4 |
| metadata_json | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/metadata/run_metadata.json |
| commands_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/metadata/commands.jsonl |
| failure_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/metadata/failures.csv |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/data/DJI_20250711140128_0002_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/data/DJI_20250916100639_0001_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/data/DJI_20250711140128_0002_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/data/DJI_20250916100639_0001_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/data/DJI_20250711140128_0002_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/data/DJI_20250916100639_0001_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/data/DJI_20250711140128_0002_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/data/DJI_20250916100639_0001_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/data/DJI_20250711140128_0002_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/data/DJI_20250916100639_0001_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/data/DJI_20250711140128_0002_V.txt |
| mot_result | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/data/DJI_20250916100639_0001_V.txt |
| trackeval_summary | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/eval/motchallenge_summary.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-full_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-full_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-full_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-lite-no-low-det_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-lite-no-low-det_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-lite-no-low-det_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-lite-no-motion_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-lite-no-motion_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-lite-no-motion_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-no-reacquire_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-no-reacquire_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-no-reacquire_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-no-removed-guard_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-no-removed-guard_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-no-removed-guard_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-no-template_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-no-template_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250711140128_0002_V/Ours-no-template_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-full_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-full_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-full_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-lite-no-low-det_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-lite-no-low-det_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-lite-no-low-det_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-lite-no-motion_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-lite-no-motion_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-lite-no-motion_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-no-reacquire_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-no-reacquire_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-no-reacquire_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-no-removed-guard_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-no-removed-guard_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-no-removed-guard_per_gt_stage_coverage.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-no-template_box_stats.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-no-template_per_gt_diagnostics.csv |
| diagnostic_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/diagnostics/DJI_20250916100639_0001_V/Ours-no-template_per_gt_stage_coverage.csv |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/diagnostics/DJI_20250711140128_0002_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/diagnostics/DJI_20250711140128_0002_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/diagnostics/DJI_20250711140128_0002_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/diagnostics/DJI_20250711140128_0002_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/diagnostics/DJI_20250916100639_0001_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/diagnostics/DJI_20250916100639_0001_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/diagnostics/DJI_20250916100639_0001_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-full/diagnostics/DJI_20250916100639_0001_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/diagnostics/DJI_20250711140128_0002_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/diagnostics/DJI_20250711140128_0002_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/diagnostics/DJI_20250711140128_0002_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/diagnostics/DJI_20250711140128_0002_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/diagnostics/DJI_20250916100639_0001_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/diagnostics/DJI_20250916100639_0001_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/diagnostics/DJI_20250916100639_0001_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-low-det/diagnostics/DJI_20250916100639_0001_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/diagnostics/DJI_20250711140128_0002_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/diagnostics/DJI_20250711140128_0002_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/diagnostics/DJI_20250711140128_0002_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/diagnostics/DJI_20250711140128_0002_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/diagnostics/DJI_20250916100639_0001_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/diagnostics/DJI_20250916100639_0001_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/diagnostics/DJI_20250916100639_0001_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-lite-no-motion/diagnostics/DJI_20250916100639_0001_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/diagnostics/DJI_20250711140128_0002_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/diagnostics/DJI_20250711140128_0002_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/diagnostics/DJI_20250711140128_0002_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/diagnostics/DJI_20250711140128_0002_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/diagnostics/DJI_20250916100639_0001_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/diagnostics/DJI_20250916100639_0001_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/diagnostics/DJI_20250916100639_0001_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-reacquire/diagnostics/DJI_20250916100639_0001_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/diagnostics/DJI_20250711140128_0002_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/diagnostics/DJI_20250711140128_0002_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/diagnostics/DJI_20250711140128_0002_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/diagnostics/DJI_20250711140128_0002_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/diagnostics/DJI_20250916100639_0001_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/diagnostics/DJI_20250916100639_0001_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/diagnostics/DJI_20250916100639_0001_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-removed-guard/diagnostics/DJI_20250916100639_0001_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/diagnostics/DJI_20250711140128_0002_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/diagnostics/DJI_20250711140128_0002_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/diagnostics/DJI_20250711140128_0002_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/diagnostics/DJI_20250711140128_0002_V/stage_observations.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/diagnostics/DJI_20250916100639_0001_V/candidate_pool_stats.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/diagnostics/DJI_20250916100639_0001_V/lifecycle_events.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/diagnostics/DJI_20250916100639_0001_V/msdc_tracks.jsonl |
| diagnostic_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/trackers/Ours-no-template/diagnostics/DJI_20250916100639_0001_V/stage_observations.jsonl |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250711140128_0002_V/Ours-full.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250711140128_0002_V/Ours-lite-no-low-det.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250711140128_0002_V/Ours-lite-no-motion.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250711140128_0002_V/Ours-no-reacquire.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250711140128_0002_V/Ours-no-removed-guard.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250711140128_0002_V/Ours-no-template.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250916100639_0001_V/Ours-full.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250916100639_0001_V/Ours-lite-no-low-det.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250916100639_0001_V/Ours-lite-no-motion.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250916100639_0001_V/Ours-no-reacquire.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250916100639_0001_V/Ours-no-removed-guard.mp4 |
| visualization_mp4 | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/visualizations/DJI_20250916100639_0001_V/Ours-no-template.mp4 |
| metadata_json | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/metadata/run_metadata.json |
| commands_jsonl | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/metadata/commands.jsonl |
| failure_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/metadata/failures.csv |
| summary_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/summary/main_results.csv |
| summary_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/summary/ablation_results.csv |
| speed_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/summary/speed_results.csv |
| analysis_markdown | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/summary/analysis_ablation.md |
| analysis_markdown | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/summary/analysis_main.md |
| analysis_markdown | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/summary/analysis_speed.md |
| speed_csv | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/speed/speed_1000/speed_results.csv |
| docs_markdown | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/docs/MSDC_EXPERIMENT_RESULT.md |
| report_markdown | /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/MSDC_EXPERIMENT_REPORT.md |

## Missing Metrics and Failures

- /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/main/main_full/metadata/failures.csv: present but empty
- /home/hyj/Anti_Drone_Project/Marine-Target-Vision-System/results/msdc_paper_phase1/formal_20260609_015836/ablation/ablation_full/metadata/failures.csv: present but empty

## Cautious Effectiveness Conclusion

Comparative evidence from available rows: IDF1 vs FFCA-YOLO + BoT-SORT: 52.83628 vs 82.25078; worse; IDF1 vs FFCA-YOLO + OC-SORT: 52.83628 vs 69.58317; worse; HOTA vs FFCA-YOLO + BoT-SORT: 55.78748 vs 75.37659; worse; HOTA vs FFCA-YOLO + OC-SORT: 55.78748 vs 69.10581; worse; AssA vs FFCA-YOLO + BoT-SORT: 49.75196 vs 82.32799; worse; AssA vs FFCA-YOLO + OC-SORT: 49.75196 vs 69.45903; worse; IDSW vs FFCA-YOLO + BoT-SORT: 154 vs 33; worse; IDSW vs FFCA-YOLO + OC-SORT: 154 vs 39; worse; FP vs FFCA-YOLO + BoT-SORT: 4172 vs 3327; worse; FP vs FFCA-YOLO + OC-SORT: 4172 vs 3332; worse; FN vs FFCA-YOLO + BoT-SORT: 26832 vs 23928; worse; FN vs FFCA-YOLO + OC-SORT: 26832 vs 23932; worse.
