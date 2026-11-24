"""Render selected intervals from complete final MOT runs, without retracking."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation.render_mot_video import render_mot_video


def export_cases(selection_file, output_dir):
    selection_file, output_dir = Path(selection_file), Path(output_dir)
    with selection_file.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output_dir.mkdir(parents=True, exist_ok=True)
    exported, seen = [], {}
    for row in rows:
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if end - start + 1 != 40:
            raise ValueError("A paper case must cover exactly 40 original frames")
        for side in ("a", "b"):
            method = row[f"method_{side}"]
            key = (row["sequence"], start, end, method)
            if key in seen:
                continue
            output = output_dir / f'{row["sequence"]}_{start:06d}_{end:06d}_{method}.mp4'
            render_mot_video(row["video_path"], row[f"mot_{side}"], output,
                             file_type="visible", class_source="mot", progress_interval=0,
                             start_frame=start, end_frame=end)
            timing = json.loads(output.with_suffix(".timing.json").read_text(encoding="utf-8"))
            if timing["processed_frames"] != 40:
                raise ValueError(f"Case video is incomplete: {output}")
            seen[key] = str(output)
            exported.append({"sequence": row["sequence"], "method": method,
                             "start_frame": start, "end_frame": end,
                             "source_mot": row[f"mot_{side}"], "video": str(output)})
    (output_dir / "manifest.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_cases(args.selection, args.output_dir)


if __name__ == "__main__":
    main()
