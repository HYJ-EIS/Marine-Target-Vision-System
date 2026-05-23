"""
本地 MP4 视频追踪测试脚本

用法:
    python video_test_tracking.py --input <视频路径> [--output <输出路径>] [--fps-override N]

参数:
    --input       输入视频文件路径（.mp4 / .MP4）
    --output      输出视频路径（默认: results/tracking_<原始文件名>.mp4）
    --fps-override N  模拟低帧率：每隔 round(orig_fps/N) 帧取一帧（不影响输出视频的播放速度）

示例:
    python video_test_tracking.py ^
        --input "D:\\Desktop\\烟台项目数据\\原始数据集\\视频\\0915\\DJI_202509151638_002\\DJI_20250915164329_0001_S.MP4" ^
        --output results/test_S_full.mp4

    # 模拟 5fps
    python video_test_tracking.py ^
        --input "...DJI_20250915164329_0001_S.MP4" ^
        --output results/test_S_5fps.mp4 ^
        --fps-override 5
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Windows 控制台默认 GBK 编码，无法输出 UTF-8 中文，强制切换为 UTF-8
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

import cv2

# ── 路径修复：确保项目根目录在 sys.path 中 ──────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import target_module.image_detect_module.target_detection as td
import visualization as vis
from cv_utils import imwrite_unicode
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from target_module.image_detect_module.utils.file_utils import get_file_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 MP4 视频追踪测试")
    parser.add_argument("--input", required=True, help="输入视频文件路径")
    parser.add_argument("--output", default="", help="输出视频路径（留空自动生成）")
    parser.add_argument(
        "--fps-override",
        type=float,
        default=0.0,
        metavar="N",
        help="模拟低帧率：按此 FPS 抽帧处理（0 = 使用视频原始帧率）",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="",
        choices=["", "bytetrack", "ocsort", "botsort", "official_ocsort", "official_botsort"],
        help="追踪算法（留空使用 Config 默认值）",
    )
    parser.add_argument("--show", action="store_true", help="实时窗口预览（需要 GUI）")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最大处理帧数（0 = 不限制，处理完整视频）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    # ── 打开视频 ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {input_path}")
        sys.exit(1)

    orig_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_fps = float(args.fps_override) if args.fps_override > 0 else orig_fps
    # 每隔 step 帧取一帧（模拟低帧率）
    step = max(1, round(orig_fps / target_fps))

    print(f"[INFO] 视频信息: {frame_w}x{frame_h} @ {orig_fps:.1f} fps, 共 {total_frames} 帧")
    print(f"[INFO] 处理帧率: {target_fps:.1f} fps (每 {step} 帧取一帧)")

    # ── 从文件名判断图像类型（_T → infrared，否则 visible）────────────────
    basename = os.path.basename(input_path)
    file_type = get_file_type(os.path.splitext(basename)[0])
    if file_type == "unknown":
        file_type = "visible"
    print(f"[INFO] 检测类型: {file_type}")

    # ── 输出视频 ───────────────────────────────────────────────────────────
    if not args.output:
        os.makedirs("results", exist_ok=True)
        stem = os.path.splitext(basename)[0]
        fps_tag = f"_{int(target_fps)}fps" if args.fps_override > 0 else ""
        args.output = os.path.join("results", f"tracking_{stem}{fps_tag}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, target_fps, (frame_w, frame_h))

    # ── 初始化检测器 & 追踪器 ─────────────────────────────────────────────
    print("[INFO] 初始化检测器...")
    detector = td.get_detector()
    tracker_type = args.tracker if args.tracker else None
    tracker = MultiObjectTracker(frame_rate=target_fps, tracker_type=tracker_type)

    # 轨迹历史：{track_id: [(cx, cy), ...]}（保留最近 30 个点）
    track_history: dict[int, list[tuple[int, int]]] = {}
    TRAIL_LEN = 30

    # ── 临时文件（避免中文路径问题）─────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="yolo_track_")
    tmp_frame_path = os.path.join(tmp_dir, "frame.jpg")

    # ── 统计变量 ──────────────────────────────────────────────────────────
    processed = 0
    total_detections = 0
    start_time = time.time()

    print("[INFO] 开始处理...")
    frame_idx = 0

    max_frames = args.max_frames if args.max_frames > 0 else float("inf")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按步长跳帧
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        processed += 1
        frame_idx += 1

        if processed > max_frames:
            break

        # 将当前帧写入临时文件供检测器读取
        if not imwrite_unicode(tmp_frame_path, frame):
            print(f"[WARN] 无法写入临时帧: {tmp_frame_path}")
            writer.write(frame)
            continue

        # 检测（传入 file_type 避免临时文件名无法判断类型）
        result = detector.detect_from_image_file(tmp_frame_path, file_type=file_type)
        if result is None or not result.get("success"):
            writer.write(frame)
            continue

        boxes = result.get("data", {}).get("boxes", [])
        total_detections += len(boxes)

        # 追踪（传入 frame 供 GMC 使用）
        tracked_boxes = tracker.update(boxes, frame.shape, frame=frame)

        # 更新轨迹历史
        for b in tracked_boxes:
            tid = b["track_id"]
            cx = b["x"] + b["w"] // 2
            cy = b["y"] + b["h"] // 2
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append((cx, cy))
            if len(track_history[tid]) > TRAIL_LEN:
                track_history[tid].pop(0)

        # 可视化
        result["data"]["boxes"] = tracked_boxes
        vis_frame = vis.visualize_detections(
            frame, "video", result, frame_num=processed,
            track_history=track_history
        )

        writer.write(vis_frame)

        if args.show:
            cv2.imshow("Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 每 50 帧打印一次进度
        if processed % 50 == 0:
            elapsed = time.time() - start_time
            fps_real = processed / elapsed if elapsed > 0 else 0
            active_track_count = len({b["track_id"] for b in tracked_boxes})
            print(
                f"  帧 {processed}/{total_frames // step} | "
                f"检测: {len(boxes)} | 活跃轨迹: {active_track_count} | "
                f"实际处理fps: {fps_real:.1f}"
            )

    # ── 清理 & 统计 ────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"[完成] 处理帧数: {processed}")
    print(f"[完成] 总耗时:   {elapsed:.1f}s  ({processed / elapsed:.1f} fps)")
    print(f"[完成] 总检测数: {total_detections}")
    print(f"[完成] 输出文件: {os.path.abspath(args.output)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
