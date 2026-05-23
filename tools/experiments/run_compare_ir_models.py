"""
红外模型对比测试: A_S_F_ir_FFCA.onnx vs A_S_F_ir_FFCA_v2.onnx

对所有红外视频分别用两个模型跑追踪，解析处理速度与检测数量并输出对比表格。

用法:
    python run_compare_ir_models.py
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

PYTHON = sys.executable
ROOT = str(Path(__file__).resolve().parents[2])
VIDEO_TEST_TRACKING = os.path.join(ROOT, "tools", "validation", "video_test_tracking.py")

# ── 模型配置 ──────────────────────────────────────────────────────────────────
MODELS = {
    "ir_v1": r"target_module\models\A_S_F_ir_FFCA.onnx",
    "ir_v2": r"target_module\models\A_S_F_ir_FFCA_v2.onnx",
}

# ── 红外测试视频 ──────────────────────────────────────────────────────────────
VIDEOS = {
    "0812_T": r"D:\Desktop\烟台项目数据\原始数据集\视频\0812\DJI_202508121028_009\DJI_20250812103257_0003_T.MP4",
    "0915_T": r"D:\Desktop\烟台项目数据\原始数据集\视频\0915\DJI_202509151638_002\DJI_20250915164329_0001_T.MP4",
}

# ── 解析 video_test_tracking.py 的 [完成] 统计行 ─────────────────────────────
_PATTERNS = {
    "frames":     re.compile(r"\[完成\]\s*处理帧数:\s*(\d+)"),
    "time":       re.compile(r"\[完成\]\s*总耗时:\s*([\d.]+)s"),
    "fps":        re.compile(r"\[完成\]\s*总耗时:.*\(([\d.]+)\s*fps\)"),
    "detections": re.compile(r"\[完成\]\s*总检测数:\s*(\d+)"),
}


def parse_stats(output: str) -> dict:
    stats = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(output)
        if m:
            stats[key] = float(m.group(1))
    # 计算每帧平均检测数
    if "detections" in stats and "frames" in stats and stats["frames"] > 0:
        stats["det_per_frame"] = stats["detections"] / stats["frames"]
    return stats


def run_one(model_name: str, model_path: str, video_tag: str, video_path: str) -> dict:
    output_file = os.path.join("results", f"ir_cmp_{video_tag}_{model_name}.mp4")
    desc = f"[{video_tag}] {model_name}"
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")

    env = os.environ.copy()
    env["IR_MODEL_OVERRIDE"] = model_path

    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, VIDEO_TEST_TRACKING,
         "--input", video_path,
         "--output", output_file],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = time.time() - t0

    # 打印子进程输出（方便查看进度）
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        # 过滤掉 ONNX Runtime 的 CUDA 警告噪声
        for line in proc.stderr.splitlines():
            if "onnxruntime" not in line.lower():
                print(line, file=sys.stderr)

    ok = proc.returncode == 0
    print(f"  -> {'OK' if ok else 'FAIL'} ({elapsed:.0f}s)")

    if ok:
        return parse_stats(proc.stdout)
    return {}


def main() -> None:
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    t_start = time.time()

    # 收集结果: results[(video_tag, model_name)] = stats
    results: dict[tuple[str, str], dict] = {}

    for video_tag, video_path in VIDEOS.items():
        if not os.path.isfile(video_path):
            print(f"[SKIP] {video_tag}: 文件不存在")
            continue
        for model_name, model_path in MODELS.items():
            stats = run_one(model_name, model_path, video_tag, video_path)
            results[(video_tag, model_name)] = stats

    # ── 汇总对比表 ────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print(f"  红外模型对比: ir_v1 vs ir_v2  (总耗时 {time.time() - t_start:.0f}s)")
    print(f"{'=' * 80}\n")

    header = f"{'视频':<10} {'模型':<8} {'处理帧数':>8} {'总检测数':>8} {'检测/帧':>8} {'速度(fps)':>10}"
    print(header)
    print("-" * len(header))

    for video_tag in VIDEOS:
        for model_name in MODELS:
            s = results.get((video_tag, model_name), {})
            if not s:
                print(f"{video_tag:<10} {model_name:<8} {'(无数据)':>8}")
                continue
            print(
                f"{video_tag:<10} {model_name:<8} "
                f"{int(s.get('frames', 0)):>8} "
                f"{int(s.get('detections', 0)):>8} "
                f"{s.get('det_per_frame', 0):>8.2f} "
                f"{s.get('fps', 0):>10.1f}"
            )
        print()  # 视频间空行

    print("\n输出视频文件（可目视对比）:")
    for video_tag in VIDEOS:
        for model_name in MODELS:
            print(f"  results/ir_cmp_{video_tag}_{model_name}.mp4")


if __name__ == "__main__":
    main()
