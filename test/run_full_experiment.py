"""
完整实验：三算法 × 四视频 × 七帧率梯度 对比

优化策略：每个视频只做一次 ONNX 检测（耗时最长），
缓存检测结果后，对 3 种追踪器 × 7 个 FPS 级别复用缓存跑追踪。

输出:
    results/exp/                          实验结果目录
    results/exp/<tag>_detections.pkl      检测缓存
    results/exp/<tag>_<tracker>.csv       各算法 FPS 梯度 CSV
    results/exp/comparison_<tag>.png      三算法对比折线图
    results/exp/summary.csv              总汇总表

用法:
    python run_full_experiment.py [--max-frames 500]
"""

import argparse
import csv
import os
import pickle
import sys
import tempfile
import time

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import target_module.image_detect_module.target_detection as td
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from target_module.image_detect_module.utils.kalman_bbox import KalmanBoxTracker

# ── 配置 ──────────────────────────────────────────────────────────────────
VIDEOS = {
    "0812_S": r"D:\Desktop\烟台项目数据\原始数据集\视频\0812\DJI_202508121028_009\DJI_20250812103257_0003_S.MP4",
    "0812_T": r"D:\Desktop\烟台项目数据\原始数据集\视频\0812\DJI_202508121028_009\DJI_20250812103257_0003_T.MP4",
    "0915_S": r"D:\Desktop\烟台项目数据\原始数据集\视频\0915\DJI_202509151638_002\DJI_20250915164329_0001_S.MP4",
    "0915_T": r"D:\Desktop\烟台项目数据\原始数据集\视频\0915\DJI_202509151638_002\DJI_20250915164329_0001_T.MP4",
}
TRACKERS = ["bytetrack", "ocsort", "botsort"]
FPS_LEVELS = [25.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0]
OUT_DIR = os.path.join(_ROOT, "results", "exp")


# ── Phase 1: 检测缓存 ────────────────────────────────────────────────────

def detect_and_cache(tag: str, video_path: str, max_frames: int) -> dict:
    """对视频做逐帧检测，缓存到 pkl 文件。返回缓存 dict。"""
    cache_path = os.path.join(OUT_DIR, f"{tag}_detections.pkl")
    if os.path.exists(cache_path):
        print(f"[{tag}] 加载已有检测缓存: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"[{tag}] 开始检测: {w}x{h} @ {orig_fps:.1f}fps, "
          f"共 {total} 帧, 处理前 {min(max_frames, total)} 帧")

    detector = td.get_detector()
    tmp_dir = tempfile.mkdtemp(prefix="exp_")
    tmp_path = os.path.join(tmp_dir, "frame.jpg")

    frames_data = []  # list of (shape_tuple, boxes_list, frame_bgr_or_None)
    t0 = time.time()
    idx = 0

    while idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(tmp_path, frame)
        result = detector.detect_from_image_file(tmp_path)
        boxes = []
        if result and result.get("success"):
            boxes = result.get("data", {}).get("boxes", [])
        # 保存帧数据用于 GMC（下采样到 1/4 减小 pkl 体积）
        small = cv2.resize(frame, (w // 4, h // 4))
        frames_data.append({
            "shape": frame.shape,
            "boxes": boxes,
            "frame_small": small,
        })
        idx += 1
        if idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{tag}] {idx}/{min(max_frames, total)} 帧 "
                  f"({elapsed:.0f}s, {idx/elapsed:.1f} fps)")

    cap.release()
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    cache = {
        "tag": tag,
        "orig_fps": orig_fps,
        "frame_count": len(frames_data),
        "resolution": (w, h),
        "frames": frames_data,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[{tag}] 检测完成: {len(frames_data)} 帧, "
          f"耗时 {time.time()-t0:.0f}s, 缓存: {cache_path}")
    return cache


# ── Phase 2: 追踪统计 ────────────────────────────────────────────────────

def run_tracking(cache: dict, tracker_type: str, target_fps: float) -> dict:
    """用缓存的检测结果跑追踪，返回统计 dict。"""
    orig_fps = cache["orig_fps"]
    frames = cache["frames"]
    step = max(1, round(orig_fps / target_fps))
    w, h = cache["resolution"]

    # 重置 Kalman ID 计数器
    KalmanBoxTracker.reset_count()
    tracker = MultiObjectTracker(frame_rate=target_fps, tracker_type=tracker_type)

    id_switch_count = 0
    displacements = []
    all_track_ids = set()
    prev_ids = set()
    prev_centers = {}
    processed = 0

    for i, fd in enumerate(frames):
        if i % step != 0:
            continue

        # 为 GMC 恢复帧（从下采样恢复到原尺寸）
        frame_for_gmc = cv2.resize(fd["frame_small"], (w, h)) if tracker_type == "botsort" else None

        tracked = tracker.update(fd["boxes"], fd["shape"], frame=frame_for_gmc)
        processed += 1

        cur_ids = {b["track_id"] for b in tracked}
        all_track_ids.update(cur_ids)

        disappeared = prev_ids - cur_ids
        appeared = cur_ids - prev_ids
        id_switch_count += min(len(disappeared), len(appeared))

        cur_centers = {}
        for b in tracked:
            tid = b["track_id"]
            cx, cy = b["x"] + b["w"] // 2, b["y"] + b["h"] // 2
            cur_centers[tid] = (cx, cy)
            if tid in prev_centers:
                dx = cx - prev_centers[tid][0]
                dy = cy - prev_centers[tid][1]
                displacements.append(float(np.hypot(dx, dy)))

        prev_ids = cur_ids
        prev_centers = cur_centers

    duration = processed / target_fps if target_fps > 0 else 0
    rate = (id_switch_count / duration * 60) if duration > 0 else 0

    return {
        "fps": target_fps,
        "tracker": tracker_type,
        "processed_frames": processed,
        "duration_sec": round(duration, 1),
        "id_switch_count": id_switch_count,
        "id_switch_rate_per_min": round(rate, 2),
        "avg_displacement_px": round(float(np.mean(displacements)), 2) if displacements else 0.0,
        "max_displacement_px": round(float(np.max(displacements)), 2) if displacements else 0.0,
        "p90_displacement_px": round(float(np.percentile(displacements, 90)), 2) if displacements else 0.0,
        "track_count": len(all_track_ids),
    }


# ── Phase 3: 图表生成 ────────────────────────────────────────────────────

def generate_comparison_chart(tag: str, all_results: dict):
    """生成三算法对比折线图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import fontManager, FontProperties
        import glob

        # 中文字体
        for pattern in ["C:/Windows/Fonts/msyh*.ttc", "C:/Windows/Fonts/simhei.ttf"]:
            hits = glob.glob(pattern)
            if hits:
                fontManager.addfont(hits[0])
                fp = FontProperties(fname=hits[0])
                plt.rcParams["font.sans-serif"] = [fp.get_name(), "DejaVu Sans"]
                break
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"三算法追踪对比 — {tag}", fontsize=14, fontweight="bold")

        colors = {"bytetrack": "#2196F3", "ocsort": "#FF9800", "botsort": "#4CAF50"}
        markers = {"bytetrack": "s", "ocsort": "^", "botsort": "o"}

        # 图1: ID 切换率
        ax = axes[0]
        for tracker_type in TRACKERS:
            rows = all_results[tracker_type]
            fps_vals = [r["fps"] for r in rows]
            switch_rates = [r["id_switch_rate_per_min"] for r in rows]
            ax.plot(fps_vals, switch_rates,
                    f"{markers[tracker_type]}-",
                    color=colors[tracker_type],
                    linewidth=2, markersize=7,
                    label=tracker_type)
        ax.set_xlabel("处理帧率 (fps)")
        ax.set_ylabel("ID 切换次数/分钟")
        ax.set_title("ID 切换率 vs 帧率")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # 图2: 平均位移
        ax = axes[1]
        for tracker_type in TRACKERS:
            rows = all_results[tracker_type]
            fps_vals = [r["fps"] for r in rows]
            avg_disps = [r["avg_displacement_px"] for r in rows]
            ax.plot(fps_vals, avg_disps,
                    f"{markers[tracker_type]}-",
                    color=colors[tracker_type],
                    linewidth=2, markersize=7,
                    label=tracker_type)
        ax.set_xlabel("处理帧率 (fps)")
        ax.set_ylabel("平均帧间位移 (像素)")
        ax.set_title("平均帧间位移 vs 帧率")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # 图3: 轨迹总数 (越接近真实目标数越好)
        ax = axes[2]
        for tracker_type in TRACKERS:
            rows = all_results[tracker_type]
            fps_vals = [r["fps"] for r in rows]
            track_counts = [r["track_count"] for r in rows]
            ax.plot(fps_vals, track_counts,
                    f"{markers[tracker_type]}-",
                    color=colors[tracker_type],
                    linewidth=2, markersize=7,
                    label=tracker_type)
        ax.set_xlabel("处理帧率 (fps)")
        ax.set_ylabel("唯一轨迹总数")
        ax.set_title("轨迹数 vs 帧率 (越少越好)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        plt.tight_layout()
        png_path = os.path.join(OUT_DIR, f"comparison_{tag}.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  图表: {png_path}")
    except ImportError:
        print("  [WARN] matplotlib 未安装，跳过图表")


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="三算法完整对比实验")
    parser.add_argument("--max-frames", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    summary_rows = []

    for tag, vpath in VIDEOS.items():
        if not os.path.isfile(vpath):
            print(f"[SKIP] {tag}: 文件不存在")
            continue

        print(f"\n{'#'*60}")
        print(f"# 视频: {tag}")
        print(f"{'#'*60}")

        # Phase 1: 检测（或加载缓存）
        cache = detect_and_cache(tag, vpath, args.max_frames)

        # Phase 2: 三算法 × 七帧率
        all_results = {}
        for tracker_type in TRACKERS:
            print(f"\n  [{tag}] 追踪器: {tracker_type}")
            rows = []
            for fps in FPS_LEVELS:
                if fps > cache["orig_fps"] * 1.1:
                    continue
                stats = run_tracking(cache, tracker_type, fps)
                rows.append(stats)
                summary_rows.append({"tag": tag, **stats})
                print(f"    FPS={fps:5.1f}  IDSW={stats['id_switch_count']:3d}  "
                      f"rate={stats['id_switch_rate_per_min']:6.1f}/min  "
                      f"disp={stats['avg_displacement_px']:6.1f}px  "
                      f"tracks={stats['track_count']}")
            all_results[tracker_type] = rows

            # 保存单算法 CSV
            csv_path = os.path.join(OUT_DIR, f"{tag}_{tracker_type}.csv")
            fields = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

        # Phase 3: 对比图表
        generate_comparison_chart(tag, all_results)

    # 总汇总 CSV
    if summary_rows:
        summary_path = os.path.join(OUT_DIR, "summary.csv")
        fields = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n总汇总: {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  全部实验完成！总耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  结果目录: {OUT_DIR}")
    print(f"{'='*60}")

    # 打印精简对比表
    print("\n=== 三算法 ID 切换率对比 (次/分钟) ===\n")
    print(f"{'视频':>8s} {'FPS':>5s} {'ByteTrack':>10s} {'OC-SORT':>10s} {'BoT-SORT':>10s}")
    print("-" * 50)
    for tag in VIDEOS:
        for fps in [25.0, 5.0, 1.0]:
            bt = [r for r in summary_rows if r["tag"]==tag and r["tracker"]=="bytetrack" and r["fps"]==fps]
            oc = [r for r in summary_rows if r["tag"]==tag and r["tracker"]=="ocsort" and r["fps"]==fps]
            bo = [r for r in summary_rows if r["tag"]==tag and r["tracker"]=="botsort" and r["fps"]==fps]
            bt_v = f"{bt[0]['id_switch_rate_per_min']:.1f}" if bt else "-"
            oc_v = f"{oc[0]['id_switch_rate_per_min']:.1f}" if oc else "-"
            bo_v = f"{bo[0]['id_switch_rate_per_min']:.1f}" if bo else "-"
            print(f"{tag:>8s} {fps:5.0f} {bt_v:>10s} {oc_v:>10s} {bo_v:>10s}")
        print()


if __name__ == "__main__":
    main()
