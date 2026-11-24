import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluation.render_mot_video import render_mot_video


def test_interval_keeps_original_mot_frame_and_id_without_detector(tmp_path, monkeypatch):
    import tools.evaluation.render_mot_video as renderer
    video, mot, output = tmp_path / "synthetic.avi", tmp_path / "mot.txt", tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full((64, 64, 3), 20 * i, dtype=np.uint8))
    writer.release()
    mot.write_text("".join(f"{frame},83,5,5,10,10,0.8,3,-1,-1\n" for frame in range(1, 7)),
                   encoding="utf-8")
    observed = []
    def visualize(frame, mode, result, frame_num, track_history):
        observed.append((frame_num, result["data"]["boxes"][0]["track_id"],
                         result["data"]["boxes"][0]["class"]))
        return frame
    monkeypatch.setattr(renderer.vis, "visualize_detections", visualize)
    monkeypatch.setattr(renderer.td, "get_detector", lambda: (_ for _ in ()).throw(AssertionError("detector")))
    render_mot_video(video, mot, output, class_source="mot", start_frame=3, end_frame=5,
                     progress_interval=0)
    assert observed == [(3, 83, "UAV"), (4, 83, "UAV"), (5, 83, "UAV")]
    timing = json.loads(output.with_suffix(".timing.json").read_text())
    assert timing["processed_frames"] == 3
    assert timing["source_start_frame"] == 3 and timing["source_last_frame"] == 5
