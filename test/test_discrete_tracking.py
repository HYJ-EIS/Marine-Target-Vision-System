"""
离散图片输入追踪测试

从视频中按 1s / 2s / 3s 间隔抽帧，模拟离散图片输入场景，
评估追踪器 ID 一致性。

指标：
    - ID 切换次数（同一目标被分配不同 ID 的次数）
    - ID 一致性率（最长连续 ID / 总帧数）
    - 平均轨迹长度
"""

import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.detectors import OnnxDetector
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from target_module.image_detect_module.utils.kalman_bbox import KalmanBoxTracker


def extract_frames(video_path: str, interval_sec: float, max_frames: int = 99999) -> list[tuple[int, np.ndarray]]:
    """从视频按指定间隔抽帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval_sec)

    print(f"  视频: {fps:.1f} FPS, 共 {total_frames} 帧, 抽帧间隔: {frame_interval} 帧 ({interval_sec}s)")

    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((frame_idx, frame))
        frame_idx += frame_interval

    cap.release()
    print(f"  抽取 {len(frames)} 帧")
    return frames


def run_detection(detector: OnnxDetector, frame: np.ndarray) -> list[dict]:
    """对单帧执行检测，返回 tracker 需要的 xywh 格式"""
    results = detector.detect(frame)
    detections = []
    for d in results:
        x1, y1 = d["x1"], d["y1"]
        x2, y2 = d["x2"], d["y2"]
        detections.append({
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "confidence": float(d["confidence"]),
            "class": d.get("class", Config.CLASSES[0]),
        })
    return detections


def compute_spatial_overlap(box_a: dict, box_b: dict) -> float:
    """计算两个检测框的 IoU"""
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["w"], ay1 + box_a["h"]
    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["w"], by1 + box_b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = box_a["w"] * box_a["h"]
    area_b = box_b["w"] * box_b["h"]
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def analyze_tracking(all_tracked: list[list[dict]], interval_sec: float) -> dict:
    """
    分析追踪结果的 ID 一致性。

    通过空间位置匹配追踪同一个物理目标在不同帧的 ID 变化。
    """
    if not all_tracked or all(len(t) == 0 for t in all_tracked):
        return {"error": "无追踪结果"}

    # 用空间位置建立物理目标的跨帧关联
    # physical_targets[target_idx] = [(frame_idx, track_id, box), ...]
    physical_targets: list[list[tuple[int, int, dict]]] = []

    for frame_i, tracked in enumerate(all_tracked):
        for t in tracked:
            tid = t["track_id"]
            # 尝试匹配到已有的物理目标
            best_match = -1
            best_iou = 0.3  # 最低匹配阈值

            for pt_idx, pt_history in enumerate(physical_targets):
                # 只看最近几帧
                for fi, _, prev_box in reversed(pt_history[-5:]):
                    if fi == frame_i:
                        continue
                    iou = compute_spatial_overlap(t, prev_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = pt_idx

            if best_match >= 0:
                physical_targets[best_match].append((frame_i, tid, t))
            else:
                physical_targets.append([(frame_i, tid, t)])

    # 统计每个物理目标的 ID 切换
    total_switches = 0
    total_frames_tracked = 0
    target_stats = []

    for pt_idx, history in enumerate(physical_targets):
        if len(history) < 2:
            continue

        ids_over_time = [tid for _, tid, _ in history]
        n_frames = len(ids_over_time)

        # 计算 ID 切换次数
        switches = sum(1 for i in range(1, n_frames) if ids_over_time[i] != ids_over_time[i - 1])

        # 最长连续相同 ID
        max_streak = 1
        current_streak = 1
        for i in range(1, n_frames):
            if ids_over_time[i] == ids_over_time[i - 1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1

        # 最常见 ID
        from collections import Counter
        id_counts = Counter(ids_over_time)
        dominant_id, dominant_count = id_counts.most_common(1)[0]

        consistency = dominant_count / n_frames

        target_stats.append({
            "target_idx": pt_idx,
            "n_frames": n_frames,
            "unique_ids": len(set(ids_over_time)),
            "id_switches": switches,
            "dominant_id": dominant_id,
            "consistency": consistency,
            "max_streak": max_streak,
            "id_sequence": ids_over_time,
        })

        total_switches += switches
        total_frames_tracked += n_frames

    # 汇总
    n_targets = len(target_stats)
    avg_consistency = np.mean([s["consistency"] for s in target_stats]) if target_stats else 0
    avg_switches = np.mean([s["id_switches"] for s in target_stats]) if target_stats else 0

    return {
        "interval_sec": interval_sec,
        "total_frames": len(all_tracked),
        "n_physical_targets": n_targets,
        "total_id_switches": total_switches,
        "avg_switches_per_target": round(avg_switches, 2),
        "avg_id_consistency": round(avg_consistency * 100, 1),
        "target_details": target_stats,
    }


def run_test(video_path: str, interval_sec: float, tracker_type: str = "botsort",
             max_frames: int = 60) -> dict:
    """对指定间隔运行完整的检测 + 追踪测试"""
    print(f"\n{'='*70}")
    print(f"测试: 间隔 {interval_sec}s, 追踪器 {tracker_type}")
    print(f"{'='*70}")

    # 1. 抽帧
    frames = extract_frames(video_path, interval_sec, max_frames)
    if len(frames) < 5:
        return {"error": f"抽取帧数不足: {len(frames)}"}

    # 2. 初始化检测器
    detector = OnnxDetector(
        onnx_path=Config.ONNX_VISIBLE_MODEL_PATH,
        conf_thres=Config.VISIBLE_CONF_THRESH,
        iou_thres=Config.IOU_THRESHOLD,
        classes=Config.CLASSES,
    )
    print(f"  检测器已加载: {Config.ONNX_VISIBLE_MODEL_PATH}")

    # 3. 初始化追踪器 — 按离散帧率
    effective_fps = 1.0 / interval_sec
    KalmanBoxTracker.reset_count()
    tracker = MultiObjectTracker(frame_rate=effective_fps, tracker_type=tracker_type)

    # 4. 逐帧检测 + 追踪
    all_tracked = []
    all_detections = []

    for i, (frame_idx, frame) in enumerate(frames):
        detections = run_detection(detector, frame)
        tracked = tracker.update(detections, frame.shape, frame)

        all_detections.append(detections)
        all_tracked.append(tracked)

        det_str = f"{len(detections)} det"
        trk_str = ", ".join(f"ID{t['track_id']}" for t in tracked) if tracked else "无"
        if i < 10 or i % 10 == 0:
            print(f"  帧 {i:3d} (vid #{frame_idx:5d}): {det_str} -> [{trk_str}]")

    # 5. 分析
    result = analyze_tracking(all_tracked, interval_sec)
    result["tracker_type"] = tracker_type

    return result


def print_report(results: list[dict]):
    """打印汇总报告"""
    print(f"\n{'='*70}")
    print("汇总报告: 离散图片输入追踪 ID 一致性测试")
    print(f"{'='*70}")

    print(f"\n{'间隔':>6s} | {'追踪器':>8s} | {'物理目标':>6s} | {'ID切换':>6s} | "
          f"{'平均切换/目标':>10s} | {'ID一致性':>8s}")
    print("-" * 70)

    for r in results:
        if "error" in r:
            print(f"  {r.get('interval_sec', '?')}s: {r['error']}")
            continue
        print(f"  {r['interval_sec']:4.0f}s | {r['tracker_type']:>8s} | "
              f"{r['n_physical_targets']:>6d} | {r['total_id_switches']:>6d} | "
              f"{r['avg_switches_per_target']:>10.2f} | "
              f"{r['avg_id_consistency']:>7.1f}%")

    # 打印每个目标的详细 ID 序列
    print(f"\n--- 各目标 ID 序列详情 ---")
    for r in results:
        if "error" in r or not r.get("target_details"):
            continue
        print(f"\n间隔 {r['interval_sec']}s ({r['tracker_type']}):")
        for t in r["target_details"]:
            seq = t["id_sequence"]
            # 显示前30个
            seq_str = " ".join(str(s) for s in seq[:30])
            if len(seq) > 30:
                seq_str += " ..."
            switches_str = f"切换{t['id_switches']}次" if t['id_switches'] > 0 else "稳定"
            print(f"  目标{t['target_idx']}: [{switches_str}] "
                  f"一致性{t['consistency']*100:.0f}% | {seq_str}")


def visualize_tracking(video_path: str, interval_sec: float, tracker_type: str = "botsort",
                       max_frames: int = 30, output_dir: str = "results/discrete_test"):
    """生成可视化图片，每帧标注 track_id"""
    os.makedirs(output_dir, exist_ok=True)

    frames = extract_frames(video_path, interval_sec, max_frames)
    detector = OnnxDetector(
        onnx_path=Config.ONNX_VISIBLE_MODEL_PATH,
        conf_thres=Config.VISIBLE_CONF_THRESH,
        iou_thres=Config.IOU_THRESHOLD,
        classes=Config.CLASSES,
    )
    effective_fps = 1.0 / interval_sec
    KalmanBoxTracker.reset_count()
    tracker = MultiObjectTracker(frame_rate=effective_fps, tracker_type=tracker_type)

    colors = {}
    def get_color(tid: int) -> tuple:
        if tid not in colors:
            rng = np.random.RandomState(tid * 37)
            colors[tid] = tuple(int(c) for c in rng.randint(50, 255, 3))
        return colors[tid]

    sub_dir = os.path.join(output_dir, f"{tracker_type}_{interval_sec}s")
    os.makedirs(sub_dir, exist_ok=True)

    for i, (frame_idx, frame) in enumerate(frames):
        detections = run_detection(detector, frame)
        tracked = tracker.update(detections, frame.shape, frame)

        vis = frame.copy()
        for t in tracked:
            x1, y1 = t["x"], t["y"]
            x2, y2 = x1 + t["w"], y1 + t["h"]
            tid = t["track_id"]
            color = get_color(tid)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID{tid} {t['class']} {t['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 帧信息
        info = f"Frame {i} (vid #{frame_idx}) | {interval_sec}s interval | {tracker_type}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out_path = os.path.join(sub_dir, f"frame_{i:04d}.jpg")
        cv2.imencode('.jpg', vis)[1].tofile(out_path)

    print(f"  可视化已保存到: {sub_dir}")


if __name__ == "__main__":
    VIDEO_PATH = r"D:\Desktop\烟台项目数据\原始数据集\视频\0916\DJI_202509161103_004\DJI_20250916110623_0001_S.MP4"

    if not os.path.isfile(VIDEO_PATH):
        print(f"错误: 视频文件不存在 {VIDEO_PATH}")
        sys.exit(1)

    intervals = [1.0, 2.0, 3.0]
    tracker_types = ["botsort", "ocsort", "bytetrack"]

    # max_frames=0 表示不限制，测试整个视频
    MAX_FRAMES = 99999

    results = []

    for tracker_type in tracker_types:
        for interval in intervals:
            try:
                r = run_test(VIDEO_PATH, interval, tracker_type=tracker_type, max_frames=MAX_FRAMES)
                results.append(r)
            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()
                results.append({"error": str(e), "interval_sec": interval, "tracker_type": tracker_type})

    print_report(results)

    # 生成可视化（botsort + ocsort）
    print("\n--- 生成可视化 ---")
    for tracker_type in ["botsort", "ocsort"]:
        for interval in intervals:
            try:
                visualize_tracking(VIDEO_PATH, interval, tracker_type, max_frames=MAX_FRAMES)
            except Exception as e:
                print(f"  可视化错误 ({tracker_type} {interval}s): {e}")
