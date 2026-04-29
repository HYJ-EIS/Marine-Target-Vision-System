"""
OC-SORT vs BoT-SORT 对比测试

对 4 个测试视频分别用 ocsort 和 botsort 跑追踪，输出对比表格。
"""
import os, sys, subprocess, time, csv

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

PYTHON = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_FRAMES = "500"

VIDEOS = {
    "0812_S": r"D:\Desktop\烟台项目数据\原始数据集\视频\0812\DJI_202508121028_009\DJI_20250812103257_0003_S.MP4",
    "0812_T": r"D:\Desktop\烟台项目数据\原始数据集\视频\0812\DJI_202508121028_009\DJI_20250812103257_0003_T.MP4",
    "0915_S": r"D:\Desktop\烟台项目数据\原始数据集\视频\0915\DJI_202509151638_002\DJI_20250915164329_0001_S.MP4",
    "0915_T": r"D:\Desktop\烟台项目数据\原始数据集\视频\0915\DJI_202509151638_002\DJI_20250915164329_0001_T.MP4",
}

TRACKERS = ["ocsort", "botsort"]


def run(desc, args):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    r = subprocess.run(args, cwd=ROOT)
    print(f"  -> {'OK' if r.returncode == 0 else 'FAIL'} ({time.time()-t0:.0f}s)")
    return r.returncode


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    t0 = time.time()

    # 1. 对每个视频 x 每个追踪器，跑全帧率追踪 + 5fps追踪
    for tag, vpath in VIDEOS.items():
        if not os.path.isfile(vpath):
            print(f"[SKIP] {tag}: file not found")
            continue
        for tracker in TRACKERS:
            # 全帧率
            run(
                f"[{tag}] {tracker} 全帧率",
                [PYTHON, "video_test_tracking.py",
                 "--input", vpath,
                 "--output", f"results/cmp_{tag}_{tracker}_full.mp4",
                 "--tracker", tracker,
                 "--max-frames", MAX_FRAMES],
            )
            # 5fps
            run(
                f"[{tag}] {tracker} 5fps",
                [PYTHON, "video_test_tracking.py",
                 "--input", vpath,
                 "--output", f"results/cmp_{tag}_{tracker}_5fps.mp4",
                 "--tracker", tracker,
                 "--fps-override", "5",
                 "--max-frames", MAX_FRAMES],
            )

    # 2. 汇总对比表
    print(f"\n\n{'='*80}")
    print(f"  OC-SORT vs BoT-SORT 对比总结 (总耗时 {time.time()-t0:.0f}s)")
    print(f"{'='*80}")
    print()

    # 从输出文件提取结果不方便，直接让用户看终端输出中的 [完成] 行
    print("请查看上方各视频的 [完成] 统计行进行对比。")
    print("关键指标: ID切换估计 越低越好")
    print()
    print("输出视频文件:")
    for tag in VIDEOS:
        for tracker in TRACKERS:
            print(f"  results/cmp_{tag}_{tracker}_full.mp4")
            print(f"  results/cmp_{tag}_{tracker}_5fps.mp4")


if __name__ == "__main__":
    main()
