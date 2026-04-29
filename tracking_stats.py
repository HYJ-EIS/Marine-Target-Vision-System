"""
帧率梯度 ID 稳定性统计工具

对同一视频以不同 FPS 运行追踪，统计每种 FPS 下的 ID 稳定性指标，
输出 CSV 报告和折线图，帮助确定「保证 ID 稳定的最小帧率」。

用法:
    python tracking_stats.py ^
        --input <视频路径> ^
        [--fps-levels 25 12 8 5 3 2 1] ^
        [--output results/stats.csv]

输出:
    - results/stats_<stem>.csv   各 FPS 级别统计数据
    - results/stats_<stem>.png   可视化折线图
"""

import argparse
import csv
import os
import sys
import tempfile
import time

# Windows 控制台默认 GBK 编码，无法输出 UTF-8 中文，强制切换为 UTF-8
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import target_module.image_detect_module.target_detection as td
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from target_module.image_detect_module.utils.file_utils import get_file_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="追踪 ID 稳定性统计（帧率梯度分析）")
    parser.add_argument("--input", required=True, help="输入视频文件路径")
    parser.add_argument(
        "--fps-levels",
        type=float, nargs="+",
        default=[25.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0],
        help="要测试的帧率列表（默认: 25 12 8 5 3 2 1）",
    )
    parser.add_argument("--output", default="", help="输出 CSV 路径（留空自动生成）")
    parser.add_argument(
        "--tracker", type=str, default="",
        choices=["", "bytetrack", "ocsort", "botsort"],
        help="追踪算法（留空使用 Config 默认值）",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="最大处理帧数（0 = 不限制）",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 核心：在给定抽样步长下跑一遍追踪，返回统计指标
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking_at_fps(
    frames_cache: list,
    frame_detections: list[list[dict]],
    orig_fps: float,
    target_fps: float,
    tracker_type: str = "",
) -> dict:
    """
    用已缓存的帧和检测结果，以 target_fps 抽样重跑追踪。

    Parameters
    ----------
    frames_cache    : 所有帧（np.ndarray 列表或 shape tuple 列表）
    frame_detections: 每帧的原始检测 boxes（无追踪 ID）
    orig_fps        : 视频原始帧率
    target_fps      : 目标帧率
    tracker_type    : 追踪器类型（留空使用 Config 默认值）

    Returns
    -------
    dict with keys:
        fps, id_switch_count, id_switch_rate, avg_displacement, max_displacement,
        track_count, processed_frames, duration_sec
    """
    step = max(1, round(orig_fps / target_fps))
    tracker = MultiObjectTracker(
        frame_rate=target_fps,
        tracker_type=tracker_type if tracker_type else None,
    )

    id_switch_count = 0
    displacements: list[float] = []
    all_track_ids: set[int] = set()

    prev_track_ids: set[int] = set()
    prev_centers: dict[int, tuple[int, int]] = {}

    processed = 0
    for i, boxes in enumerate(frame_detections):
        if i % step != 0:
            continue

        frame_item = frames_cache[i]
        if isinstance(frame_item, tuple):
            frame_shape = frame_item
            frame = None
        else:
            frame_shape = frame_item.shape
            frame = frame_item
        tracked = tracker.update(boxes, frame_shape, frame=frame)
        processed += 1

        current_ids = {b["track_id"] for b in tracked}
        all_track_ids.update(current_ids)

        # ID 切换：消失 ID 与同时出现新 ID 的最小值
        disappeared = prev_track_ids - current_ids
        appeared = current_ids - prev_track_ids
        id_switch_count += min(len(disappeared), len(appeared))

        # 帧间位移（相邻抽样帧，同 track_id）
        current_centers: dict[int, tuple[int, int]] = {}
        for b in tracked:
            tid = b["track_id"]
            cx = b["x"] + b["w"] // 2
            cy = b["y"] + b["h"] // 2
            current_centers[tid] = (cx, cy)
            if tid in prev_centers:
                dx = cx - prev_centers[tid][0]
                dy = cy - prev_centers[tid][1]
                displacements.append(float(np.hypot(dx, dy)))

        prev_track_ids = current_ids
        prev_centers = current_centers

    duration_sec = processed / target_fps if target_fps > 0 else 0
    id_switch_rate = (id_switch_count / duration_sec * 60) if duration_sec > 0 else 0

    return {
        "fps": target_fps,
        "step": step,
        "processed_frames": processed,
        "id_switch_count": id_switch_count,
        "id_switch_rate_per_min": round(id_switch_rate, 2),
        "avg_displacement_px": round(float(np.mean(displacements)), 2) if displacements else 0.0,
        "max_displacement_px": round(float(np.max(displacements)), 2) if displacements else 0.0,
        "p90_displacement_px": round(float(np.percentile(displacements, 90)), 2) if displacements else 0.0,
        "track_count": len(all_track_ids),
        "duration_sec": round(duration_sec, 1),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input

    if not os.path.isfile(input_path):
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    basename = os.path.basename(input_path)
    stem = os.path.splitext(basename)[0]

    # ── 输出路径 ───────────────────────────────────────────────────────────
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = args.output if args.output else os.path.join(out_dir, f"stats_{stem}.csv")
    png_path = csv_path.replace(".csv", ".png")

    # ── 打开视频，读取帧 & 检测结果 ────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {input_path}")
        sys.exit(1)

    orig_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"[INFO] 视频: {basename}, {frame_w}x{frame_h} @ {orig_fps:.1f} fps, {total_frames} 帧")

    # 初始化检测器
    print("[INFO] 初始化检测器（首次加载 ONNX 模型）...")
    detector = td.get_detector()

    # 临时文件目录
    tmp_dir = tempfile.mkdtemp(prefix="yolo_stats_")
    tmp_frame_path = os.path.join(tmp_dir, "frame.jpg")

    max_frames = args.max_frames if args.max_frames > 0 else float("inf")
    print(f"[INFO] 读取全帧 + 执行检测（全帧率，最多 {int(max_frames) if max_frames < float('inf') else '全部'} 帧）...")
    frames_shape: list[tuple] = []      # 只存 shape 节省内存
    frame_detections: list[list[dict]] = []

    t0 = time.time()
    frame_idx = 0
    while True:
        if frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames_shape.append(frame.shape)

        # 检测
        cv2.imwrite(tmp_frame_path, frame)
        result = detector.detect_from_image_file(tmp_frame_path)
        boxes = []
        if result and result.get("success"):
            boxes = result.get("data", {}).get("boxes", [])
        frame_detections.append(boxes)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  已预处理 {frame_idx}/{total_frames} 帧 ({time.time()-t0:.1f}s)")

    cap.release()
    print(f"[INFO] 全帧预处理完成，共 {frame_idx} 帧，耗时 {time.time()-t0:.1f}s")

    # ── 对各 FPS 级别运行追踪 ─────────────────────────────────────────────
    fps_levels = sorted(set(args.fps_levels), reverse=True)
    # 过滤掉超过原始帧率的值
    fps_levels = [f for f in fps_levels if f <= orig_fps * 1.1]
    if not fps_levels:
        fps_levels = [orig_fps]

    results = []
    for target_fps in fps_levels:
        print(f"\n[STATS] 测试 FPS = {target_fps:.1f} ...")
        stats = run_tracking_at_fps(
            frames_shape, frame_detections, orig_fps, target_fps,
            tracker_type=args.tracker,
        )
        results.append(stats)
        print(
            f"  处理帧数: {stats['processed_frames']}  "
            f"ID切换: {stats['id_switch_count']} ({stats['id_switch_rate_per_min']:.1f}/min)  "
            f"平均位移: {stats['avg_displacement_px']:.1f}px  "
            f"最大位移: {stats['max_displacement_px']:.1f}px  "
            f"P90位移: {stats['p90_displacement_px']:.1f}px  "
            f"轨迹数: {stats['track_count']}"
        )

    # ── 输出 CSV ───────────────────────────────────────────────────────────
    fieldnames = [
        "fps", "step", "processed_frames", "duration_sec",
        "id_switch_count", "id_switch_rate_per_min",
        "avg_displacement_px", "max_displacement_px", "p90_displacement_px",
        "track_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[输出] CSV: {os.path.abspath(csv_path)}")

    # ── 输出折线图（需要 matplotlib）────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties

        # Windows 中文字体：优先 Microsoft YaHei，回退 SimHei
        _zh_font = None
        for fname in ["Microsoft YaHei", "SimHei", "SimSun", "DengXian"]:
            try:
                fp = FontProperties(family=fname)
                if fp.get_name() != fname:
                    continue
                _zh_font = fname
                break
            except Exception:
                continue
        if _zh_font:
            plt.rcParams["font.sans-serif"] = [_zh_font, "DejaVu Sans"]
        else:
            # 直接搜索系统字体文件
            import glob as _glob
            for pattern in [
                "C:/Windows/Fonts/msyh*.ttc",
                "C:/Windows/Fonts/simhei.ttf",
            ]:
                hits = _glob.glob(pattern)
                if hits:
                    _zh_font = hits[0]
                    break
            if _zh_font:
                matplotlib.font_manager.fontManager.addfont(_zh_font)
                fp = FontProperties(fname=_zh_font)
                plt.rcParams["font.sans-serif"] = [fp.get_name(), "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

        fps_vals = [r["fps"] for r in results]
        switch_rates = [r["id_switch_rate_per_min"] for r in results]
        avg_disps = [r["avg_displacement_px"] for r in results]
        max_disps = [r["max_displacement_px"] for r in results]
        p90_disps = [r["p90_displacement_px"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"追踪稳定性分析 — {basename}", fontsize=13)

        # 图1：FPS vs ID switch rate
        ax1.plot(fps_vals, switch_rates, "o-r", linewidth=2, markersize=6, label="ID切换/分钟")
        ax1.axhline(0.1, color="gray", linestyle="--", alpha=0.6, label="稳定阈值 0.1/min")
        ax1.set_xlabel("处理帧率 (fps)")
        ax1.set_ylabel("ID切换次数/分钟")
        ax1.set_title("ID 切换率 vs 帧率")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()

        # 图2：FPS vs 位移
        ax2.plot(fps_vals, avg_disps, "o-b", linewidth=2, markersize=6, label="平均帧间位移")
        ax2.plot(fps_vals, p90_disps, "s--g", linewidth=1.5, markersize=5, label="P90 位移")
        ax2.plot(fps_vals, max_disps, "^:m", linewidth=1, markersize=5, label="最大位移")
        ax2.set_xlabel("处理帧率 (fps)")
        ax2.set_ylabel("帧间位移 (像素)")
        ax2.set_title("帧间位移分布 vs 帧率")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()

        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[输出] 图表: {os.path.abspath(png_path)}")

    except ImportError:
        print("[WARN] matplotlib 未安装，跳过图表生成。安装: pip install matplotlib")

    # ── 结论摘要 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[ 分析结论 ]")
    stable_fps = None
    for r in sorted(results, key=lambda x: x["fps"]):
        if r["id_switch_rate_per_min"] <= 0.1:
            stable_fps = r["fps"]
            stable_disp = r["avg_displacement_px"]
            stable_max_disp = r["max_displacement_px"]

    if stable_fps is not None:
        print(f"  保证 ID 稳定 (切换率 ≤ 0.1/min) 的最低帧率: {stable_fps:.1f} fps")
        print(f"  该帧率下的平均帧间位移: {stable_disp:.1f} px")
        print(f"  该帧率下的最大帧间位移: {stable_max_disp:.1f} px")
    else:
        min_switch = min(results, key=lambda x: x["id_switch_rate_per_min"])
        print(f"  所有测试 FPS 均不稳定，最稳定的是 {min_switch['fps']:.1f} fps")
        print(f"  建议提高帧率或调整追踪参数（增大 TRACKER_LOST_BUFFER）")
    print("=" * 60)

    # 清理临时文件
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
