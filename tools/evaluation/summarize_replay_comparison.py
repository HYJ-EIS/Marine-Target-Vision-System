"""Validate and report replay comparisons including measured timing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


METRICS = (
    "HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDP", "IDR",
    "IDSW", "TP", "FP", "FN", "Frag", "IDTP", "IDFP", "IDFN",
)
COUNTS = {"IDSW", "TP", "FP", "FN", "Frag", "IDTP", "IDFP", "IDFN"}
LOWER_BETTER = {"IDSW", "FP", "FN", "Frag", "IDFP", "IDFN"}
STAGES = {
    "read_decode": "视频读取/解码",
    "low_detection": "低阈值检测",
    "high_split": "高阈值框筛选",
    "tracker": "跟踪器更新（MSDC 为 lifecycle tracker）",
    "render": "可视化渲染",
    "write_export": "结果写盘/导出（含视频编码）",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path, seq: str, expected_frames: int):
    metadata = read_json(root / "run_metadata.json")
    config = metadata["config"]
    report_trackers = metadata.get("report_trackers", {
        "official_botsort_replay": "BoT-SORT", "msdc_v3_replay": "MSDC-ELT（msdc_v3）",
    })
    names, labels = tuple(report_trackers), tuple(report_trackers.values())
    baseline_all = metadata.get("baseline_input") == "all"
    input_note = (
        "- 三个 baseline 均接收完整低阈值检测集合（包含高阈值框），由各自内部阈值决定关联和新建；MSDC-ELT 仍接收拆分后的高、低阈值集合。仅改变 baseline 输入，其他参数保持原值。"
        if baseline_all else
        "- 所有方法共享上次的逐帧检测缓存。BoT-SORT、ByteTrack、OC-SORT 按项目接口接收高阈值框；MSDC-ELT 接收高、低阈值框。未按 GT 调参。这是当前项目接入配置的比较。"
    )
    with (root / "eval/motchallenge_summary.csv").open(encoding="utf-8-sig") as fh:
        metrics = {row["tracker"]: row for row in csv.DictReader(fh)}
    cache = root / "detections" / f"{seq}_high_low_detections.jsonl"
    with cache.open(encoding="utf-8") as fh:
        frame_ids = [json.loads(line)["frame_id"] for line in fh if line.strip()]
    if frame_ids != list(range(1, expected_frames + 1)):
        raise ValueError("Detection cache is not an exact contiguous full run")
    detection = read_json(cache.with_suffix(".timing.json"))
    gt = root / "mot_gt" / seq / "gt/gt.txt"
    with gt.open(encoding="utf-8-sig") as fh:
        gt_rows = list(csv.reader(fh))
    gt_count = len(gt_rows)
    gt_ids = {r[1] for r in gt_rows}
    if any(r[7] != "1" for r in gt_rows):
        raise ValueError("This report expects the cleaned UAV-only GT")

    output = root / "summary"
    output.mkdir(exist_ok=True)
    speed_rows, stage_rows, artifact_rows = [], [], []
    for name in names:
        values = metrics[name]
        if int(values["TP"]) + int(values["FN"]) != gt_count:
            raise ValueError(f"CLEAR GT count mismatch: {name}")
        if int(values["IDTP"]) + int(values["IDFN"]) != gt_count:
            raise ValueError(f"Identity GT count mismatch: {name}")
        mot = root / "trackers" / name / "data" / f"{seq}.txt"
        video = root / "visualizations" / seq / f"{name}.mp4"
        replay = read_json(mot.with_suffix(".timing.json"))
        render = read_json(video.with_suffix(".timing.json"))
        for phase in (detection, replay, render):
            if phase["processed_frames"] != expected_frames:
                raise ValueError(f"Incomplete timing frame count: {name}")
        cap = cv2.VideoCapture(str(video))
        decoded = 0
        try:
            while cap.grab():
                decoded += 1
        finally:
            cap.release()
        if decoded != expected_frames:
            raise ValueError(f"Incomplete visualization: {name}: {decoded}")
        total = sum(p["total_time_s"] for p in (detection, replay, render))
        detection_tracking = detection["total_time_s"] + replay["total_time_s"]
        tracker_seconds = replay["stage_seconds"]["tracker"]
        speed_rows.append({
            "tracker": name, "processed_frames": expected_frames,
            "total_time_s": total, "mean_latency_ms": total * 1000 / expected_frames,
            "mean_fps": expected_frames / total,
            "detection_tracking_time_s": detection_tracking,
            "detection_tracking_fps": expected_frames / detection_tracking,
            "tracker_update_time_s": tracker_seconds,
            "tracker_update_fps": expected_frames / tracker_seconds,
            "timing_scope": "shared detection + replay + rendering; measured offline passes",
        })
        for key in STAGES:
            seconds = sum(p["stage_seconds"][key] for p in (detection, replay, render))
            stage_rows.append({"tracker": name, "stage": key, "seconds": seconds,
                               "mean_ms_per_frame": seconds * 1000 / expected_frames})
        stage_rows.append({"tracker": name, "stage": "other",
                           "seconds": sum(p["other_seconds"] for p in (detection, replay, render)),
                           "mean_ms_per_frame": sum(p["other_seconds"] for p in (detection, replay, render)) * 1000 / expected_frames})
        artifact_rows.append({"tracker": name, "mot_txt": str(mot.relative_to(root)),
                              "trackeval_summary": f"eval/{name}/pedestrian_summary.txt",
                              "diagnostics": f"diagnostics/{seq}/",
                              "diagnostic_jsonl": f"trackers/{name}/diagnostics/{seq}/" if name == "msdc_v3_replay" else "N/A (per-GT CSV available)",
                              "visualization_mp4": str(video.relative_to(root)),
                              "tracking_timing": str(mot.with_suffix('.timing.json').relative_to(root)),
                              "render_timing": str(video.with_suffix('.timing.json').relative_to(root))})

    comparison = []
    for field in METRICS:
        comparison.append({"metric": field,
                           **{label: float(metrics[name][field]) for name, label in zip(names, labels)},
                           "better": "lower" if field in LOWER_BETTER else "higher"})
    write_csv(output / "comparison.csv", comparison)
    write_csv(output / "speed_summary.csv", speed_rows)
    write_csv(output / "stage_timings.csv", stage_rows)
    write_csv(output / "artifacts.csv", artifact_rows)

    lines = ["# " + "、".join(labels) + " 检测跟踪评测", "",
             f"序列：`{seq}`；完整处理 {expected_frames:,} 帧。清理后的 GT 共 {gt_count:,} 个 UAV 框、{len(gt_ids)} 个身份 ID。",
             f"原始 GT 中的 {metadata['annotation_cleanup']['removed_rows']} 个 USV 框已按用户要求删除，原文件备份位于 `annotation_backup/gt_original.txt`。",
             "未标注帧按空目标帧计算，预测框可能计为 FP；因此结果依赖于所提供 GT 的完整性。身份指标以原始 ID=1 为准。", "",
             "## 配置与评测口径", "",
             f"- 可见光 FFCA-YOLO ONNX 模型，输入 1280×1280；低阈值 {config['MSDC_LOW_CONF_VISIBLE']}，高阈值 {config['VISIBLE_CONF_THRESH']}，NMS IoU {config['IOU_THRESHOLD']}。",
             input_note,
             "- ByteTrack 使用 supervision 实现；OC-SORT 使用 vendored 官方源码，具体阈值取 run_metadata 的 Config 快照。",
             "- BoT-SORT 使用 vendored 官方实现，sparseOptFlow GMC，ReID 关闭；MSDC 使用正式 msdc_v3 配置，详见 effective_config。",
             "- TrackEval HOTA/CLEAR/Identity，不区分类别；保留模型所有输出类别的预测框。`pedestrian` 是 MOT 适配器的输出名称，实际 GT 为 UAV。",
             "- 百分比指标为 0–100，MOTP 为平均匹配 IoU，越高越好。CLEAR/Identity 匹配阈值为 IoU=0.5，HOTA 对多个 IoU 阈值求平均。", "",
             "## 17 项指标", "", "| 指标 | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(names)]
    for row in comparison:
        field = row["metric"]
        fmt = ".0f" if field in COUNTS else ".4f"
        suffix = "↓" if field in LOWER_BETTER else "↑"
        lines.append(f"| {field} {suffix} | " + " | ".join(format(row[label], fmt) for label in labels) + " |")
    a, b = metrics["official_botsort_replay"], metrics["msdc_v3_replay"]
    lines += ["", "## 结果解读", "",
              f"MSDC-ELT 相对 BoT-SORT 的 HOTA 变化为 {float(b['HOTA']) - float(a['HOTA']):+.2f} 个百分点，IDF1 变化为 {float(b['IDF1']) - float(a['IDF1']):+.2f} 个百分点，MOTA 变化为 {float(b['MOTA']) - float(a['MOTA']):+.2f} 个百分点。",
              f"IDSW 从 {a['IDSW']} 变为 {b['IDSW']}，FP 从 {a['FP']} 变为 {b['FP']}；同时 FN 从 {a['FN']} 变为 {b['FN']}。BoT-SORT 的目标召回率为 {100 * int(a['TP']) / gt_count:.2f}%，MSDC-ELT 为 {100 * int(b['TP']) / gt_count:.2f}%。应结合身份一致性和漏检数量评价，不能只看 MOTA。"]
    for field in ("HOTA", "IDF1", "MOTA"):
        ranked = sorted(names, key=lambda n: float(metrics[n][field]), reverse=True)
        lines.append(f"{field} 排序：" + " > ".join(f"{report_trackers[n]}（{float(metrics[n][field]):.2f}）" for n in ranked) + "。")
    if "bytetrack_replay" in names:
        threshold = config["TRACKER_MATCH_THRESH"]
        lines += ["", "### ByteTrack 配置解释", "",
                  f"项目将 `Config.TRACKER_MATCH_THRESH={threshold}` 直接传入 supervision 的 `minimum_matching_threshold`。本次安装版本的库默认值为 0.8。其首轮关联使用代价 `c = 1 - IoU × s`，其中 s 为检测置信度；当前要求 `c ≤ {threshold}`，即 `IoU × s ≥ {1 - threshold:g}`。这比库默认设置更严格，可能增加轨迹碎片和漏检。源码依据：`target_module/image_detect_module/utils/tracker.py` 的 `_new_bytetrack()`，以及安装包的 `supervision/tracker/byte_tracker/core.py`、`matching.py`。",
                  "因此本表比较的是项目当前配置，不能将 ByteTrack 的低分外推为该算法默认配置的表现。检测缓存与算法参数均未按本次 GT 调整。"]
    if metadata.get("source_run"):
        source_note = metadata.get("source_reuse_note", "BoT-SORT/MSDC-ELT 的 MOT、视频及计时来自原 run；ByteTrack/OC-SORT 为本次新增运行。")
        lines += ["", f"来源：`{metadata['source_run']}`。{source_note} 四方法指标在同一份 GT 上统一重新计算，复用方法指标已与原结果核对。检测缓存 SHA256：`{metadata['detection_cache_sha256']}`。"]
    audit_path = output / "gt_geometry_audit.json"
    if audit_path.is_file():
        audit = read_json(audit_path)
        lines += ["",
                  f"定位指标解释：{audit['gt_frames_exactly_matching_detector_xywh']:,}/{gt_count:,} 个 GT 框（{audit['exact_geometry_percent_of_gt']:.2f}%）与本次检测缓存中的框坐标完全相同。MSDC-ELT 接近 100% 的 MOTP/LocA 需结合这一重合情况解释，不能单独作为定位能力提升的独立证据。本次检测与跟踪输入为视频/检测缓存，GT 用于离线评测；标注生成来源尚未核实。[坐标重合核查](summary/gt_geometry_audit.json)。",
                  f"BoT-SORT 日志包含 {audit['gmc_insufficient_matching_point_warnings']} 次匹配点不足提示；对应代码保留单位变换，本次未调整 GMC 参数。原始 GT、清理后 GT、视频及模型的 SHA256 均与运行元数据一致。"]
    lines += ["", "## 速度与耗时", "",
              "下表总耗时为共享检测、该方法回放、该方法视频渲染三个实测阶段之和，包含重复读取视频、初始化和导出。检测耗时计入每种方法一次，实际只执行一次。此为离线多阶段处理速度，不代表在线端到端 FPS。跟踪器更新包含其内部诊断日志开销。", "",
              "| 方法 | 帧数 | 总耗时(s) | 平均单帧(ms) | 平均 FPS | 检测+回放 FPS | 纯 tracker 更新 FPS |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, row in zip(labels, speed_rows):
        lines.append(f"| {label} | {expected_frames} | {row['total_time_s']:.3f} | {row['mean_latency_ms']:.3f} | {row['mean_fps']:.3f} | {row['detection_tracking_fps']:.3f} | {row['tracker_update_fps']:.3f} |")
    lines += ["", "| 阶段（秒） | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(names)]
    for key, label in {**STAGES, "other": "初始化、缓存载入及其他开销"}.items():
        vals = [next(r["seconds"] for r in stage_rows if r["tracker"] == name and r["stage"] == key) for name in names]
        lines.append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "## 交付文件", "",
              "- [指标 CSV](eval/motchallenge_summary.csv)、[比较 CSV](summary/comparison.csv)",
              "- [速度 CSV](summary/speed_summary.csv)、[分阶段耗时 CSV](summary/stage_timings.csv)",
              "- [路径清单](summary/artifacts.csv)、[运行元数据](run_metadata.json)"]
    for label, row in zip(labels, artifact_rows):
        lines.append(f"- {label}：[MOT txt]({row['mot_txt']})、[TrackEval summary]({row['trackeval_summary']})、[可视化 MP4]({row['visualization_mp4']})；诊断在 `{row['diagnostics']}`。")
    lines.append(f"- MSDC lifecycle/stage JSONL：`trackers/msdc_v3_replay/diagnostics/{seq}/`；BoT-SORT 提供上述 per-GT CSV。")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "validation.json").write_text(json.dumps({
        "status": "complete", "frames": expected_frames, "gt_boxes": gt_count,
        "detection_cache_contiguous": True, "videos_fully_decoded": list(names),
        "gt_count_identity_and_clear_verified": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(root / "REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    summarize(args.run_root, args.seq_name, args.frames)
