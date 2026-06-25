# V6 Remaining Risks

生成日期：2026-06-25  
对应草稿：`paper-prepare/9 个科研问题 V6.md`

## 总体风险

V6 已将叙事从“大而全空海 MOT”收缩为“检测不稳定流的 evidence lifecycle tracker”，并删除了平台运动补偿、ReID、GMC、ROI redetect、template lock 等未实现贡献。当前最大剩余风险不是叙事边界，而是实验支撑仍偏局部：默认 MS-DC-ELT v3 只支持相对 ByteTrack 的主要指标优势和相对 OC-SORT/BoT-SORT 的 FN 降低，但两类结论都伴随 FP 代价；它不支持整体性能优于 OC-SORT/BoT-SORT。

## 仍需谨慎的 Claim

| 风险 | 当前证据 | 建议处理 |
| --- | --- | --- |
| 默认 v3 整体弱于 OC-SORT/BoT-SORT | HOTA、IDF1、AssA 低于 OC-SORT/BoT-SORT，FP/IDSW 更高 | 摘要和贡献中只写 ByteTrack 主要指标优势、FN 局部收益与代价 |
| ByteTrack 对比存在 FP 代价 | v3 的 HOTA/IDF1/IDSW/FN 显著更好，但 FP 1809 高于 ByteTrack 698 | 写“主要指标优于 ByteTrack”，不要写所有指标均优 |
| low candidate 未证明正向收益 | `no_low_candidate_replay` 略优于 full，low_candidate precision 0.20-0.25，recall N/A | 仅写为可控入口和诊断机制 |
| direct reacquire 证据混合 | full track break 局部更好，但 success/opportunity 为 0，关闭后 FN/MOTA/IDF1 更好；`MSDC_REACQUIRE_SCORE=1.2` 显著优于默认 1.5 | 写为默认阈值可能过保守的待调参恢复路径，不写稳定提升 |
| low inheritance 无有效证据 | inherit opportunities 为 0，关闭后与 full 相同 | 不进入主贡献，只保留方法边界或 future work |
| short-miss slice 不支持正向 claim | 一段序列指标为 0，另一段 hits-only 的 HOTA/IDF1/AssA 高于 full | 写 limitation 和诊断发现 |
| 默认参数非最优 | `MSDC_REACQUIRE_SCORE=1.2` 显著优于默认 1.5 | 若要作为结果，需要冻结 tuned variant 并重跑主实验 |
| 速度结论覆盖范围窄 | speed 只覆盖一个 5400 帧视频 | 写“开销可接受”，不写跨数据集实时性结论 |

## 投稿前建议补强

1. 冻结 `MSDC_REACQUIRE_SCORE=1.2` 或其他 tuned variant，重新运行主实验、消融、slice、diagnostic 和 speed，确认是否能稳定接近 OC-SORT/BoT-SORT 身份指标。
2. 修复 low_candidate recall 的 denominator 连接，使 `missing_stage_gt_overlap` 不再阻断 recall 计算。
3. 构造或标注能触发 reacquire/inherit opportunity 的短时漏检片段，报告 success/correct/wrong/ambiguous，而不是只报告 0 opportunity。
4. 增加错误案例图或诊断视频，展示 v3 何时降低 FN、何时因低置信入口增加 FP 或身份错误；任何 FN 降低结论都必须在同句或下一句绑定 FP/IDF1/AssA/IDSW 代价。
5. 若继续面向高水平会议，需要补充更强 baseline 对比解释：默认 v3 不是 OC-SORT/BoT-SORT 的替代，而是检测不稳定场景下的可解释生命周期层。

## 禁止回退的表述

后续修改中不要恢复以下说法：

* “全面优于 OC-SORT/BoT-SORT”
* “解决平台运动补偿误差”
* “解决目标外观相似性”
* “使用 ReID / GMC / ROI redetect / template lock”
* “low_candidate、reacquire、inherit 已被消融证明是主要增益来源”
* “默认 v3 参数稳定且最优”
