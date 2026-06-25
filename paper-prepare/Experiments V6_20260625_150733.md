# Experiments

This section evaluates MS-DC-ELT v3 under the formal replay setting. The main run is `results/msdc_paper_phase1/20260624_190710`; the supplementary run is `results/msdc_paper_phase1/20260624_190710_supplement_cache_only`. The supplementary experiments reuse the detection cache from the main run to execute the trackers and compute TrackEval metrics. Therefore, results from ablation, diagnostics, slice analysis, and sensitivity analysis should be interpreted as cache-replay evidence rather than as a fresh detector run.

All methods use the same FFCA-YOLO detector outputs. Baseline trackers read the same high-confidence detection stream. MS-DC-ELT v3 additionally reads the low-confidence stream for low-confidence observations, evidence update, and lost-track handling. We report HOTA, MOTA, IDF1, AssA, IDSW, FP, and FN. The main comparison is not framed as a claim of overall superiority; the goal is to identify where evidence-based lifecycle management improves recall and where it incurs FP or identity-preservation costs.

## Main Comparison

Table 1 reports the formal main results. MS-DC-ELT v3 obtains higher main tracking and identity metrics than ByteTrack in this replay setting. HOTA increases from 10.728 to 72.512, IDF1 from 6.192 to 76.727, AssA from 2.503 to 76.198, IDSW decreases from 6384 to 16, and FN decreases from 13515 to 4710. This improvement is not free: FP increases from 698 to 1809.

Compared with OC-SORT and BoT-SORT, MS-DC-ELT v3 prioritizes recall. Its FN is lower than OC-SORT by 329 frames and lower than BoT-SORT by 323 frames. This recall-side gain is accompanied by higher FP and weaker identity metrics: FP increases from 1415 and 1412 to 1809, IDF1 decreases from 86.323 and 86.523 to 76.727, AssA decreases from 88.462 and 88.900 to 76.198, and IDSW increases from 13 and 10 to 16. Thus, the main result supports a local recall benefit, not overall superiority over OC-SORT or BoT-SORT.

| Method | HOTA | MOTA | IDF1 | AssA | IDSW | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FFCA-YOLO + ByteTrack | 10.728 | 29.643 | 6.192 | 2.503 | 6384 | 698 | 13515 |
| FFCA-YOLO + OC-SORT | 77.935 | 77.909 | 86.323 | 88.462 | 13 | 1415 | 5039 |
| FFCA-YOLO + BoT-SORT | 78.138 | 77.950 | 86.523 | 88.900 | 10 | 1412 | 5033 |
| FFCA-YOLO + MS-DC-ELT v3 | 72.512 | 77.677 | 76.727 | 76.198 | 16 | 1809 | 4710 |

## Ablation Study

Table 2 summarizes the cache-replay ablations. Disabling low-confidence candidates does not hurt the evaluated metrics. The `no_low_candidate` variant slightly improves HOTA, MOTA, and IDF1, reduces FP from 1809 to 1715, and keeps FN unchanged at 4710, with the same AssA and IDSW as full v3. This result does not support low-confidence candidate confirmation as a source of overall metric gain in the current configuration.

Disabling direct reacquisition produces mixed behavior. The `no_direct_reacquire` variant reduces FN from 4710 to 4429, but this FN reduction is coupled with higher FP, lower AssA, and worse IDSW: FP increases from 1809 to 1888, AssA decreases from 76.198 to 75.519, and IDSW increases from 16 to 17. IDF1 changes only slightly, from 76.727 to 76.791. This result does not justify a claim that direct reacquisition consistently improves tracking quality.

Disabling low-confidence inheritance has no measurable effect in this run: `no_low_inheritance` is identical to full v3 across HOTA, MOTA, IDF1, AssA, IDSW, FP, and FN. Replacing score-based evidence with `hits_only` shows a narrow identity-side trade-off. Full v3 improves AssA from 75.919 to 76.198 and reduces IDSW from 18 to 16, but has lower HOTA and MOTA, higher FN, and higher FP than `hits_only`: FN increases from 4613 to 4710, FP increases from 1795 to 1809, and HOTA decreases from 72.540 to 72.512. Output NMS and real-detection age gating are not major factors in this run: disabling output NMS changes the metrics only marginally, and disabling the real-detection age gate gives identical results. The low-observation budget variants are also identical to full v3 in this replay.

| Variant | HOTA | MOTA | IDF1 | AssA | IDSW | FP | FN | Main interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full v3 | 72.512 | 77.677 | 76.727 | 76.198 | 16 | 1809 | 4710 | Default |
| no low candidate | 72.618 | 77.998 | 76.857 | 76.198 | 16 | 1715 | 4710 | Low-candidate path not beneficial here |
| no direct reacquire | 72.499 | 78.364 | 76.791 | 75.519 | 17 | 1888 | 4429 | Lower FN with higher FP and weaker identity association |
| no low inheritance | 72.512 | 77.677 | 76.727 | 76.198 | 16 | 1809 | 4710 | No observed effect |
| hits-only evidence | 72.540 | 78.050 | 76.620 | 75.919 | 18 | 1795 | 4613 | Full v3 only improves AssA and IDSW |
| no output NMS | 72.532 | 77.694 | 76.736 | 76.221 | 16 | 1810 | 4704 | Marginal change |
| no real-det age gate | 72.512 | 77.677 | 76.727 | 76.198 | 16 | 1809 | 4710 | No observed effect |

## Event-Level Diagnostics

The event-level diagnostics explain why the low-confidence and lost-track mechanisms should be treated cautiously. For full v3, low-confidence candidates are created but rarely confirmed: 8 created and 2 confirmed in `DJI_20250916100639_0001_V`, and 5 created and 1 confirmed in `DJI_20250711140128_0002_V`. The confirmed precision is 0.25 and 0.20, respectively. Recall cannot be computed because the current diagnostic logging lacks the required stage-to-ground-truth overlap denominator.

The diagnostics do not show successful direct reacquisition or inheritance in full v3. Reacquire attempts, opportunities, and successes are all zero, and inherit opportunities are also zero. This supports the interpretation from the ablation study: direct reacquisition and low-confidence inheritance exist in the implementation, but the present run does not demonstrate successful recovery events for these mechanisms.

One diagnostic signal is still informative. In the first sequence, disabling direct reacquisition increases track breaks from 17 to 36. However, the aggregate ablation simultaneously shows lower FN with higher FP, lower AssA, and higher IDSW. This should be reported as a diagnostic trade-off rather than as evidence that direct reacquisition improves overall identity preservation.

| Sequence | Variant | Low candidates created / confirmed | Precision | Reacquire success / opportunities | Inherit opportunities | Fragmentation | Track breaks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DJI_20250916100639_0001_V | full v3 | 8 / 2 | 0.25 | 0 / 0 | 0 | 20 | 17 |
| DJI_20250711140128_0002_V | full v3 | 5 / 1 | 0.20 | 0 / 0 | 0 | 1 | 0 |
| DJI_20250916100639_0001_V | no direct reacquire | 63 / 20 | 0.318 | 0 / 0 | 0 | 22 | 36 |
| DJI_20250711140128_0002_V | no direct reacquire | 15 / 4 | 0.267 | 0 / 0 | 0 | 1 | 0 |

## Short-Miss Slice Analysis

The short-miss slice analysis does not support a positive claim for full v3. In `DJI_20250711140128_0002_V`, all variants obtain zero HOTA, IDF1, AssA, and MOTA over the selected two slices, with 12 FN and no FP. This indicates that the current tracker variants fail to recover these short-miss slices.

In `DJI_20250916100639_0001_V`, full v3 obtains HOTA 64.470, IDF1 66.790, AssA 67.356, MOTA 60.423, IDSW 4, FP 2, and FN 125. The `hits_only` variant has higher HOTA, IDF1, and AssA, but its FN is worse: HOTA increases to 69.695, IDF1 to 72.207, AssA to 74.504, while FN increases from 125 to 152; FP remains 2 and IDSW remains 4. Thus, full v3 reduces FN on this slice relative to `hits_only`, but the reduction is paired with weaker HOTA, IDF1, and AssA. The slice results are sequence-dependent and should be used as an error-analysis result rather than as evidence of robust short-miss recovery.

| Sequence | Variant | Slices / frames | HOTA | MOTA | IDF1 | AssA | IDSW | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DJI_20250711140128_0002_V | full v3 | 2 / 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 12 |
| DJI_20250916100639_0001_V | full v3 | 10 / 81 | 64.470 | 60.423 | 66.790 | 67.356 | 4 | 2 | 125 |
| DJI_20250916100639_0001_V | hits-only evidence | 12 / 110 | 69.695 | 64.253 | 72.207 | 74.504 | 4 | 2 | 152 |
| DJI_20250916100639_0001_V | no direct reacquire | 31 / 293 | 67.508 | 65.789 | 62.786 | 68.781 | 4 | 15 | 371 |

## Sensitivity Analysis

The sensitivity analysis shows that several parameters have limited effect, while the direct-reacquisition score threshold is important. Changing the low-confirmation minimum hits from 4 to 6 produces nearly identical metrics. Changing the low-inheritance score from 0.32 to 0.48 has no effect, consistent with the zero inherit opportunities in the diagnostics.

The evidence decay factor controls the trade-off between identity metrics and missed detections. Reducing the evidence alpha to 0.68 improves AssA to 78.315, IDF1 to 77.526, and IDSW to 1, but FN increases sharply from 4710 to 5781, while FP decreases to 1476. Increasing alpha to 1.02 slightly reduces FN from 4710 to 4631, but FP increases from 1809 to 1823, IDF1 decreases from 76.727 to 76.610, AssA decreases from 76.198 to 75.968, and IDSW increases from 16 to 18.

The reacquisition score threshold is the most consequential parameter in the current sweep. Setting `MSDC_REACQUIRE_SCORE=1.2` increases HOTA to 77.970, IDF1 to 85.662, and AssA to 88.308, while reducing FN from 4710 to 4682. Unlike other variants, this FN reduction does not incur an FP penalty: FP remains 1809, and IDSW improves from 16 to 15. However, this is a sensitivity point, not the default v3 configuration. It should be presented as evidence that the default threshold may be conservative and that a tuned variant requires a separate formal run.

| Variant | HOTA | MOTA | IDF1 | AssA | IDSW | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default v3 | 72.512 | 77.677 | 76.727 | 76.198 | 16 | 1809 | 4710 |
| alpha = 0.68 | 72.126 | 75.208 | 77.526 | 78.315 | 1 | 1476 | 5781 |
| alpha = 1.02 | 72.507 | 77.892 | 76.610 | 75.968 | 18 | 1823 | 4631 |
| confirm score = 3.0 | 72.519 | 77.448 | 76.854 | 76.484 | 13 | 1759 | 4830 |
| reacquire score = 1.2 | 77.970 | 77.776 | 85.662 | 88.308 | 15 | 1809 | 4682 |
| reacquire score = 1.8 | 72.512 | 77.677 | 76.727 | 76.198 | 16 | 1809 | 4710 |

## Runtime and Stage Timing

Runtime is measured on one 5400-frame video at 3840 x 2160 resolution. MS-DC-ELT v3 processes 5400 frames in 668.339 seconds, corresponding to 8.080 FPS and 123.766 ms mean latency. This is slower than ByteTrack at 8.805 FPS, faster than OC-SORT at 7.689 FPS, and faster than BoT-SORT at 5.503 FPS in this run.

The lifecycle update itself is lightweight relative to detection. MS-DC-ELT v3 spends 1.391 ms per frame in the MS-DC update, including 0.896 ms for evidence update, 0.320 ms for low-observation filtering and budget control, 0.042 ms for observation construction, and 0.112 ms for output/NMS. The dominant cost is low-threshold detection at 94.786 ms per frame, followed by video read/decode at 27.482 ms. These results support an efficiency claim for the lifecycle module, but not a broad cross-dataset real-time claim.

| Method | Frames | Total time (s) | FPS | Mean latency (ms) | Tracker/update time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FFCA-YOLO + ByteTrack | 5400 | 613.302 | 8.805 | 113.574 | 1.769 |
| FFCA-YOLO + OC-SORT | 5400 | 702.305 | 7.689 | 130.056 | 0.749 |
| FFCA-YOLO + BoT-SORT | 5400 | 981.310 | 5.503 | 181.724 | 44.817 |
| FFCA-YOLO + MS-DC-ELT v3 | 5400 | 668.339 | 8.080 | 123.766 | 1.391 |

| MS-DC-ELT v3 stage | Mean time (ms/frame) |
| --- | ---: |
| Read/decode | 27.482 |
| Low-threshold detection | 94.786 |
| High-confidence split | 0.024 |
| Low-observation filter/budget | 0.320 |
| Observation construction | 0.042 |
| Evidence update | 0.896 |
| Output/NMS | 0.112 |
| Render/write | 0.000 |
| Total MS-DC update | 1.391 |

## Summary of Supported Claims

The experiments support three claims. First, MS-DC-ELT v3 obtains higher main MOT and identity metrics than ByteTrack in this replay setting, while producing more FP. Second, relative to OC-SORT and BoT-SORT, it reduces FN but has lower HOTA, IDF1, and AssA, higher IDSW, and higher FP. Third, the lifecycle update overhead is small compared with detector runtime on the measured video. The ablation, diagnostic, and slice analyses do not support strong claims for low-confidence candidate confirmation, default direct reacquisition, or low-confidence inheritance as independent performance drivers in the current v3 configuration.
