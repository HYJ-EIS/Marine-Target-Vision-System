"""
视频目标检测 + 多目标追踪系统

支持两种输入模式：
  1. RTSP 流（默认）
  2. 本地 MP4 文件（通过 VIDEO_FILE 配置或 --input 参数）

支持多种追踪算法（通过 Config.TRACKER_TYPE 或 --tracker 参数）：
  - bytetrack: 两阶段 IoU 匹配
  - ocsort:    OC-SORT 虚拟轨迹 + 方向一致性 + 距离回捞
  - botsort:   OC-SORT + 全局相机运动补偿 GMC（推荐）
  - dist_tracker: Dist-Tracker FLIT L2-IoU 融合匹配
  - official_ocsort / official_botsort: vendored 官方源码适配层

用法：
  # RTSP 模式（默认）
  python video_main.py

  # 本地文件模式
  python video_main.py --input "D:/path/to/video.mp4"

  # 指定追踪算法
  python video_main.py --input "video.mp4" --tracker ocsort
"""

import sys
import argparse

# Windows 控制台 UTF-8 编码
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

import cv2
import os
import time
import tempfile
import threading
import queue

import target_module.image_detect_module.target_detection as td
import messaging.mq_publisher as mq
import visualization as vis
from output_rtsp_video.video_output_manager import create_video_output_manager
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.tracker import MultiObjectTracker
from target_module.image_detect_module.utils.file_utils import get_file_type


# 帧获取方式：仅在 RTSP 模式下有效
USEOPENCV = False
FFMPEG = False


def parse_args():
    parser = argparse.ArgumentParser(description="视频目标检测 + 多目标追踪")
    parser.add_argument("--input", default="", help="本地视频文件路径（留空使用 RTSP 流）")
    parser.add_argument("--tracker", default="", choices=["", "bytetrack", "ocsort", "botsort", "dist_tracker", "official_ocsort", "official_botsort"],
                        help="追踪算法（留空使用 Config 默认值）")
    parser.add_argument("--output", default="", help="输出视频路径（留空使用默认值）")
    parser.add_argument("--no-display", action="store_true", help="不显示窗口（无 GUI 环境）")
    return parser.parse_args()


# ── RTSP 辅助函数 ─────────────────────────────────────────────────────────

def save_latest_frame_with_ffmpeg(rtsp_url, output_path):
    """使用 FFmpeg 从 RTSP 流抓取最新帧"""
    import subprocess
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_path = os.path.join(current_dir, 'output_rtsp_video', 'ffmpeg', 'bin', 'ffmpeg.exe')
    cmd = [ffmpeg_path, '-y', '-rtsp_transport', 'tcp', '-i', rtsp_url,
           '-frames:v', '1', '-q:v', '15', '-update', '1', output_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg 异常: {e}")
        return False


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    is_local_file = bool(args.input)

    # ── 初始化检测器 ──────────────────────────────────────────────────
    print("[INFO] 初始化检测器...")
    detector = td.get_detector()

    # ── 打开视频源 ────────────────────────────────────────────────────
    if is_local_file:
        if not os.path.isfile(args.input):
            print(f"[ERROR] 文件不存在: {args.input}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {args.input}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        basename = os.path.basename(args.input)
        file_type = get_file_type(os.path.splitext(basename)[0])
        if file_type == "unknown":
            file_type = "visible"
        print(f"[INFO] 本地文件: {basename}")
        print(f"[INFO] {frame_w}x{frame_h} @ {fps:.1f} fps, 共 {total_frames} 帧, 类型: {file_type}")
    else:
        cap = cv2.VideoCapture(Config.VIDEO_RTSP_INPUT)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开 RTSP 流: {Config.VIDEO_RTSP_INPUT}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = 0  # RTSP 流无总帧数
        print(f"[INFO] RTSP 流: {Config.VIDEO_RTSP_INPUT}")
        print(f"[INFO] {frame_w}x{frame_h} @ {fps:.1f} fps")

    # ── 初始化追踪器 ──────────────────────────────────────────────────
    tracker_type = args.tracker if args.tracker else None
    tracker = MultiObjectTracker(frame_rate=fps, tracker_type=tracker_type)

    # 轨迹历史（用于可视化轨迹线）
    track_history: dict[int, list[tuple[int, int]]] = {}
    TRAIL_LEN = 30

    # ── 初始化输出 ────────────────────────────────────────────────────
    output_path = args.output or Config.VIDEO_OUTPUT_PATH
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if is_local_file:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        output_manager = None
    else:
        writer = None
        # 读取首帧初始化 output_manager
        ret_init, frame_init = cap.read()
        if not ret_init:
            print("[ERROR] 无法读取首帧")
            sys.exit(1)
        output_manager = create_video_output_manager(
            rtsp_url=Config.VIDEO_RTSP_OUTPUT,
            video_save_path=output_path,
            fps=int(fps),
            enable_rtsp=True,
            enable_save=True,
        )
        output_manager.initialize(frame_init)
        # 把首帧放回处理（通过 seek 回到开头）
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── MQ 消息队列 ───────────────────────────────────────────────────
    message_queue = queue.Queue(maxsize=100)
    mq_thread = threading.Thread(target=_mq_sender_worker, args=(message_queue,), daemon=True)
    mq_thread.start()

    # ── 临时文件（检测器需要文件路径）─────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="video_main_")
    tmp_frame_path = os.path.join(tmp_dir, "frame.jpg")

    # ── 主循环 ────────────────────────────────────────────────────────
    frame_count = 0
    start_time = time.time()
    mq_timer = time.time()

    print("[INFO] 开始处理...")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_local_file:
                break  # 本地文件读完
            else:
                print("[WARN] RTSP 帧读取失败，重试...")
                time.sleep(0.1)
                continue

        frame_count += 1

        # 保存帧到临时文件供检测器使用
        cv2.imwrite(tmp_frame_path, frame)

        # 检测
        result = detector.detect_from_image_file(tmp_frame_path, file_type=file_type)
        if result is None or not result.get("success"):
            if writer:
                writer.write(frame)
            elif output_manager:
                output_manager.write_frame(frame)
            continue

        boxes = result.get("data", {}).get("boxes", [])

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

        # 将追踪结果写回 result
        result["data"]["boxes"] = tracked_boxes

        # 可视化
        vis_frame = vis.visualize_detections(
            frame, "video", result, frame_num=frame_count,
            track_history=track_history,
        )

        # 输出
        if writer:
            writer.write(vis_frame)
        if output_manager:
            output_manager.write_frame(vis_frame)

        # 显示窗口
        if not args.no_display:
            cv2.imshow("Video Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # MQ 上报（每 2 秒一次）
        if time.time() - mq_timer > 2:
            try:
                message_queue.put_nowait(result.copy())
            except queue.Full:
                pass
            mq_timer = time.time()

        # 进度打印
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            real_fps = frame_count / elapsed if elapsed > 0 else 0
            active_tracks = len({b["track_id"] for b in tracked_boxes})
            progress = f"{frame_count}/{total_frames}" if total_frames > 0 else str(frame_count)
            print(f"  帧 {progress} | 检测: {len(boxes)} | "
                  f"追踪: {active_tracks} | {real_fps:.1f} fps")

    # ── 清理 ──────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    message_queue.put(None)  # 通知 MQ 线程退出

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    print(f"\n[完成] 处理帧数: {frame_count}")
    print(f"[完成] 总耗时: {elapsed:.1f}s ({frame_count / elapsed:.1f} fps)")
    print(f"[完成] 输出: {os.path.abspath(output_path)}")


def _mq_sender_worker(mq_queue: queue.Queue):
    """MQ 消息发送线程"""
    while True:
        try:
            result = mq_queue.get(timeout=1)
            if result is None:
                break
            mq.send_video_result_to_mq(result)
            mq_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[MQ] 发送错误: {e}")


if __name__ == "__main__":
    main()
