import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation import render_mot_video as renderer
from tools.evaluation.video_encoding import H264VideoWriter
from tools.experiments import paper_queue


@pytest.fixture
def ffmpeg():
    path = os.environ.get("PAPER_FFMPEG_EXECUTABLE")
    if not path or not Path(path).is_file():
        pytest.skip("Set PAPER_FFMPEG_EXECUTABLE to run the real H.264 encoder tests")
    return path


class Capture:
    def __init__(self, length=8, width=256, height=128):
        self.length, self.width, self.height = length, width, height
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 29.97, cv2.CAP_PROP_FRAME_WIDTH: self.width,
                cv2.CAP_PROP_FRAME_HEIGHT: self.height}.get(prop, 0)

    def set(self, *args):
        pytest.fail("Rendering must use sequential reads, never seek")

    def read(self):
        if self.read_count == self.length:
            return False, None
        self.read_count += 1
        return True, np.full((self.height, self.width, 3), self.read_count * 20, np.uint8)

    def release(self):
        self.released = True


def inputs(tmp_path):
    source, mot = tmp_path / "source.mp4", tmp_path / "tracks.txt"
    source.touch()
    mot.write_text("1,12,20.5,10,40,20,1,1,1\n2,12,22.5,12,40,20,1,2,1\n5,99,30,20,10,10,1,3,1\n")
    return source, mot


def decoded_count(path, factory=cv2.VideoCapture):
    capture = factory(str(path))
    assert capture.isOpened()
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def test_h264_all_frames_geometry_classes_and_empty_tail(tmp_path, monkeypatch, ffmpeg):
    source, mot = inputs(tmp_path)
    before = mot.read_bytes()
    capture, rendered = Capture(), []
    monkeypatch.setattr(renderer.cv2, "VideoCapture", lambda *args: capture)
    def annotate(frame, mode, payload, **kwargs):
        rendered.append((kwargs["frame_num"], payload["data"]["boxes"]))
        return frame
    monkeypatch.setattr(renderer.vis, "visualize_detections", annotate)
    output = tmp_path / "result.mp4"
    renderer.render_mot_video(source, mot, output, class_source="mot", video_codec="h264",
                              ffmpeg_executable=ffmpeg, max_output_width=128, progress_interval=0)
    assert [frame for frame, _ in rendered] == list(range(1, 9))
    assert rendered[0][1][0]["x"] == 10.25
    assert rendered[0][1][0]["w"] == 20
    assert rendered[0][1][0]["track_id"] == rendered[1][1][0]["track_id"] == 12
    assert [rendered[index][1][0]["mot_class_id"] for index in (0, 1, 4)] == [1, 2, 3]
    assert all(not boxes for frame, boxes in rendered if frame >= 6)
    assert decoded_count(output) == 8
    timing = json.loads(output.with_suffix(".timing.json").read_text())
    assert timing["processed_frames"] == timing["source_last_frame"] == 8
    assert timing["source_width"] == 256 and timing["output_width"] == 128
    assert timing["video_codec"] == "h264" and timing["encoder_crf"] == 20
    assert timing["stage_seconds"]["write_export"] > 0 and timing["stage_seconds"]["tracker"] == 0
    assert capture.released and mot.read_bytes() == before


def test_h264_complete_frame_after_short_pipe_writes(tmp_path, ffmpeg):
    writer = H264VideoWriter(tmp_path / "short-writes.mp4", 25, (16, 16), executable=ffmpeg)
    original = writer.process.stdin
    class ShortPipe:
        def write(self, buffer):
            return original.write(buffer[:17])
        def close(self):
            original.close()
    writer.process.stdin = ShortPipe()
    writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    assert writer.frames_written == decoded_count(writer.output) == 1


@pytest.mark.parametrize("consume", [False, True])
def test_encoder_failure_never_publishes_video_or_timing(tmp_path, monkeypatch, consume):
    source, mot = inputs(tmp_path)
    capture = Capture()
    monkeypatch.setattr(renderer.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(renderer.vis, "visualize_detections", lambda frame, *a, **k: frame)
    fake = tmp_path / "bad-ffmpeg"
    fake.write_text(f"#!{sys.executable}\nimport sys\n" +
                    ("sys.stdin.buffer.read()\n" if consume else "") +
                    "sys.stderr.write('synthetic encoder failure\\n')\nsys.exit(17)\n")
    fake.chmod(0o755)
    output = tmp_path / "failed.mp4"
    with pytest.raises(RuntimeError):
        renderer.render_mot_video(source, mot, output, class_source="mot", video_codec="h264",
                                  ffmpeg_executable=str(fake), progress_interval=0)
    assert not output.exists() and not output.with_suffix(".timing.json").exists()
    assert not list(tmp_path.glob(".*.encoding-*.mp4"))
    assert "synthetic encoder failure" in output.with_suffix(".ffmpeg.log").read_text()
    assert capture.released


def test_render_exception_aborts_encoder(tmp_path, monkeypatch, ffmpeg):
    source, mot = inputs(tmp_path)
    capture, writers = Capture(), []
    monkeypatch.setattr(renderer.cv2, "VideoCapture", lambda *args: capture)
    def construct(*args, **kwargs):
        writer = H264VideoWriter(*args, **kwargs)
        writers.append(writer)
        return writer
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic annotation failure")
    monkeypatch.setattr(renderer, "H264VideoWriter", construct)
    monkeypatch.setattr(renderer.vis, "visualize_detections", fail)
    output = tmp_path / "failed-annotation.mp4"
    with pytest.raises(RuntimeError, match="synthetic annotation failure"):
        renderer.render_mot_video(source, mot, output, class_source="mot", video_codec="h264",
                                  ffmpeg_executable=ffmpeg, progress_interval=0)
    assert writers[0].closed and writers[0].process.poll() is not None
    assert not output.exists() and not output.with_suffix(".timing.json").exists()
    assert not list(tmp_path.glob(".*.encoding-*.mp4"))


def test_zero_frame_export_is_rejected(tmp_path, ffmpeg):
    writer = H264VideoWriter(tmp_path / "zero.mp4", 25, (16, 16), executable=ffmpeg)
    with pytest.raises(RuntimeError):
        writer.release()
    assert not writer.output.exists() and not writer.temporary.exists()


def test_failed_replacement_preserves_existing_video(tmp_path, ffmpeg):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"previous successful artifact")
    writer = H264VideoWriter(output, 25, (16, 16), executable=ffmpeg)
    with pytest.raises(ValueError, match="dimensions"):
        writer.write(np.zeros((15, 16, 3), np.uint8))
    assert output.read_bytes() == b"previous successful artifact"
    assert not writer.temporary.exists()


def test_missing_frozen_encoder_preflight_records_failure_without_work(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    missing = tmp_path / "absent-ffmpeg"
    args = SimpleNamespace(run_root=tmp_path, spec=spec, stage="core", sequence="seq",
                           job_id="core_01", ffmpeg_executable=missing)
    with pytest.raises(FileNotFoundError, match="Frozen render encoder"):
        paper_queue.run_job(args, execute=lambda command: pytest.fail("Must not start tracking with an absent encoder"))
    failures = list((tmp_path / "queue/failures").glob("*.json"))
    assert len(failures) == 1
    assert str(missing) in json.loads(failures[0].read_text())["error"]
    config = json.loads((tmp_path / "queue/job_configs/core_01.json").read_text())
    assert config["render_ffmpeg_executable"] == str(missing)
    assert not (tmp_path / "queue/markers/core_01.completed.json").exists()
