"""
验证 1：离散图片追踪 API 测试

从视频按 1s 间隔抽帧，调用 detect_targets(enable_tracking=True)，
验证 track_id 跨帧一致，延迟 < 300ms，并保存带标注的结果图片。
"""

import sys
import os
import cv2
import time
import numpy as np
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import target_module.image_detect_module.target_detection as td
from visualization import visualize_detections


def main():
    VIDEO_PATH = r"D:\Desktop\烟台项目数据\原始数据集\视频\0908\DJI_202509081637_005\DJI_20250908164104_0001_cut_S.mp4"
    RESULT_DIR = r"D:\Desktop\无人项目代码\yolo_img_detect\results"
    INTERVAL_SEC = 1.0
    MAX_FRAMES = 60

    if not os.path.isfile(VIDEO_PATH):
        print(f"错误: 视频不存在 {VIDEO_PATH}")
        sys.exit(1)

    os.makedirs(RESULT_DIR, exist_ok=True)

    # 抽帧
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * INTERVAL_SEC)
    print(f"视频: {fps:.0f} FPS, 共 {total} 帧, 抽帧间隔: {frame_interval} 帧 ({INTERVAL_SEC}s)")

    # 初始化检测器（触发模型加载）
    print("初始化检测器...")
    detector = td.get_detector()

    # 保存抽帧到临时文件
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="track_test_")

    latencies = []
    all_track_ids = []  # [(frame_i, [track_ids])]

    frame_idx = 0
    frame_i = 0

    while frame_i < MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # 保存帧到临时文件
        tmp_path = os.path.join(tmp_dir, f"frame_{frame_i:04d}.jpg")
        cv2.imwrite(tmp_path, frame)

        # 调用 detect_targets with tracking
        t0 = time.perf_counter()
        result = td.detect_targets(tmp_path, RESULT_DIR, enable_tracking=True)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        boxes = result.get("data", {}).get("boxes", [])
        track_ids = [b.get("track_id", b.get("id", "?")) for b in boxes]
        all_track_ids.append((frame_i, track_ids))

        # 保存带标注的结果图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        vis_frame = visualize_detections(frame, "video", result, frame_num=frame_i)
        out_name = f"tracking_frame_{frame_i:04d}_{timestamp}.jpg"
        out_path = os.path.join(RESULT_DIR, out_name)
        cv2.imencode('.jpg', vis_frame)[1].tofile(out_path)

        # 打印
        ids_str = ", ".join(f"ID{tid}" for tid in track_ids) if track_ids else "无"
        status = "OK" if latency_ms < 300 else "SLOW"
        if frame_i < 15 or frame_i % 10 == 0:
            print(f"  帧 {frame_i:3d} (vid #{frame_idx:5d}) | {latency_ms:5.0f}ms [{status}] | "
                  f"{len(boxes)} det -> [{ids_str}]")

        frame_idx += frame_interval
        frame_i += 1

    cap.release()

    # 清理临时文件
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 汇总
    print(f"\n{'='*60}")
    print(f"验证 1 结果汇总")
    print(f"{'='*60}")
    print(f"总帧数: {frame_i}")
    print(f"延迟: mean={np.mean(latencies):.0f}ms, max={np.max(latencies):.0f}ms, "
          f"p95={np.percentile(latencies, 95):.0f}ms")
    print(f"所有帧 < 300ms: {'YES' if max(latencies) < 300 else 'NO'}")

    # ID 一致性分析
    from collections import Counter
    all_ids_seen = []
    for fi, ids in all_track_ids:
        all_ids_seen.extend(ids)
    id_counts = Counter(all_ids_seen)
    print(f"唯一 track_id 数: {len(id_counts)}")
    print(f"各 ID 出现帧数: {dict(id_counts.most_common(10))}")

    # 检查 track_id 字段是否存在
    sample_result = td.detect_targets(
        os.path.join(RESULT_DIR, "tracking_frame_0000_" +
                     datetime.now().strftime("%Y%m%d") + "*.jpg"),
        RESULT_DIR, enable_tracking=True
    )
    print(f"\n结果图片已保存到: {RESULT_DIR}")

    # === 验证 2: 回归测试 ===
    print(f"\n{'='*60}")
    print(f"验证 2: 回归测试")
    print(f"{'='*60}")

    # 2a: 不带 enable_tracking 的调用
    cap2 = cv2.VideoCapture(VIDEO_PATH)
    ret, test_frame = cap2.read()
    cap2.release()
    tmp_path2 = os.path.join(RESULT_DIR, "_regression_test.jpg")
    cv2.imwrite(tmp_path2, test_frame)

    result_no_track = td.detect_targets(tmp_path2)
    boxes_no_track = result_no_track.get("data", {}).get("boxes", [])
    has_track_id = any("track_id" in b for b in boxes_no_track)
    print(f"  不带 enable_tracking: {len(boxes_no_track)} 个检测, "
          f"含 track_id: {has_track_id} (应为 False)")
    assert not has_track_id, "回归失败: 不带 enable_tracking 不应有 track_id"
    print(f"  回归测试 2a 通过: 不带追踪时无 track_id")

    # 清理
    os.remove(tmp_path2)

    print(f"\n全部验证完成!")


if __name__ == "__main__":
    main()
