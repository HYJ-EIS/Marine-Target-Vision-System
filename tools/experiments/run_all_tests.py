"""
批量追踪验证脚本

对 4 个测试视频（0812/0915 可见光+红外）运行:
1. 全帧率 BoT-SORT 追踪（前 500 帧）
2. 5fps 低帧率 BoT-SORT 追踪（前 500 帧）

用法:
    python run_all_tests.py [--max-frames 500]
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

PYTHON = sys.executable
ROOT = str(Path(__file__).resolve().parents[2])
VIDEO_TEST_TRACKING = os.path.join(ROOT, "tools", "validation", "video_test_tracking.py")

VIDEOS = [
    r"D:\Desktop\烟台项目数据\原始数据集\视频\0812\DJI_202508121028_009\DJI_20250812103257_0003_S.MP4",
    r"D:\Desktop\烟台项目数据\原始数据集\视频\0812\DJI_202508121028_009\DJI_20250812103257_0003_T.MP4",
    r"D:\Desktop\烟台项目数据\原始数据集\视频\0915\DJI_202509151638_002\DJI_20250915164329_0001_S.MP4",
    r"D:\Desktop\烟台项目数据\原始数据集\视频\0915\DJI_202509151638_002\DJI_20250915164329_0001_T.MP4",
]


def run_cmd(desc: str, args: list[str]) -> int:
    """运行子进程并打印输出"""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(args, cwd=ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"[{status}] {desc} ({elapsed:.0f}s)")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=500)
    args = parser.parse_args()
    mf = str(args.max_frames)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    # 检查视频是否存在
    for v in VIDEOS:
        if not os.path.isfile(v):
            print(f"[WARN] 视频不存在，跳过: {v}")

    total_start = time.time()
    results_summary = []

    for video_path in VIDEOS:
        if not os.path.isfile(video_path):
            continue

        basename = os.path.basename(video_path)
        stem = os.path.splitext(basename)[0]
        # 提取关键信息: 0812/0915, _S/_T
        suffix = "_S" if "_S" in basename else "_T"
        date = "0812" if "0812" in video_path else "0915"
        tag = f"{date}{suffix}"

        print(f"\n\n{'#'*60}")
        print(f"# 视频: {tag} ({basename})")
        print(f"{'#'*60}")

        # ── 1. 全帧率 BoT-SORT 追踪 ──────────────────────────────────
        out_full = os.path.join(ROOT, "results", f"tracking_{tag}_full.mp4")
        run_cmd(
            f"[{tag}] 全帧率 BoT-SORT 追踪",
            [PYTHON, VIDEO_TEST_TRACKING,
             "--input", video_path,
             "--output", out_full,
             "--tracker", "botsort",
             "--max-frames", mf],
        )

        # ── 2. 5fps 低帧率追踪 ──────────────────────────────────────
        out_5fps = os.path.join(ROOT, "results", f"tracking_{tag}_5fps.mp4")
        run_cmd(
            f"[{tag}] 5fps BoT-SORT 追踪",
            [PYTHON, VIDEO_TEST_TRACKING,
             "--input", video_path,
             "--output", out_5fps,
             "--tracker", "botsort",
             "--fps-override", "5",
             "--max-frames", mf],
        )

        results_summary.append(tag)

    total_elapsed = time.time() - total_start

    print(f"\n\n{'='*60}")
    print(f"  全部测试完成！总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  已处理视频: {', '.join(results_summary)}")
    print(f"  输出目录: {os.path.join(ROOT, 'results')}")
    print(f"{'='*60}")
    print("\n结果文件:")
    for tag in results_summary:
        print(f"  - results/tracking_{tag}_full.mp4   (全帧率追踪视频)")
        print(f"  - results/tracking_{tag}_5fps.mp4   (5fps 追踪视频)")


if __name__ == "__main__":
    main()
