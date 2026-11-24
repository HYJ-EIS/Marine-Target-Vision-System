"""Transactional H.264 export for already annotated BGR frames."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import numpy as np


class H264VideoWriter:
    """Publish the target MP4 only after FFmpeg has accepted every frame."""

    FINALIZE_TIMEOUT_SECONDS = 600
    FINALIZE_POLL_SECONDS = 5

    def __init__(self, output, fps, size, *, executable=None, crf=20, preset="veryfast"):
        self.output = Path(output)
        self.width, self.height = map(int, size)
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("H.264 export requires a positive finite frame rate")
        if self.width < 2 or self.height < 2 or self.width % 2 or self.height % 2:
            raise ValueError("H.264 yuv420p export requires positive even width and height")
        if not 0 <= int(crf) <= 20:
            raise ValueError("Paper H.264 export requires CRF in 0..20")
        selected = executable or os.environ.get("PAPER_FFMPEG_EXECUTABLE") or shutil.which("ffmpeg")
        if not selected or not Path(selected).is_file():
            raise FileNotFoundError("Set PAPER_FFMPEG_EXECUTABLE to the frozen libx264-capable FFmpeg binary")
        self.executable = str(Path(selected).resolve())
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.output.with_name(f".{self.output.stem}.encoding-{uuid.uuid4().hex}.mp4")
        self.stderr_path = self.output.with_suffix(".ffmpeg.log")
        self.frames_written = 0
        self.closed = False
        self.command = [self.executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
                        "-s:v", f"{self.width}x{self.height}", "-r", repr(self.fps), "-i", "pipe:0",
                        "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
                        "-pix_fmt", "yuv420p", "-threads", "2", "-vsync", "0",
                        "-movflags", "+faststart", str(self.temporary)]
        self.metadata = {"video_codec": "h264", "encoder": "libx264", "encoder_crf": int(crf),
                         "encoder_preset": preset, "encoder_pixel_format": "yuv420p",
                         "encoder_threads": 2, "encoder_executable": self.executable,
                         "encoder_sha256": hashlib.sha256(Path(self.executable).read_bytes()).hexdigest(),
                         "encoder_command": list(self.command), "encoder_stderr": str(self.stderr_path),
                         "encoder_frame_rate": self.fps}
        self.metadata['encoder_finalize_timeout_s'] = self.FINALIZE_TIMEOUT_SECONDS
        self._stderr = self.stderr_path.open("wb")
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self._stderr, bufsize=0)
        except BaseException:
            self._stderr.close()
            self.temporary.unlink(missing_ok=True)
            raise

    def isOpened(self):
        return not self.closed and self.process.poll() is None

    def _error(self, detail):
        text = self.stderr_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        return RuntimeError(f"H.264 encoder failed ({detail}); log={self.stderr_path}\n{text}")

    def write(self, frame):
        if self.closed:
            raise RuntimeError("H.264 writer is already closed")
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            self.abort()
            raise ValueError("H.264 frame dimensions or dtype differ from the declared BGR24 stream")
        payload = memoryview(np.ascontiguousarray(frame)).cast("B")
        position = 0
        try:
            while position < len(payload):
                count = self.process.stdin.write(payload[position:])
                if not count:
                    raise BrokenPipeError("FFmpeg stdin accepted no bytes")
                position += count
        except (BrokenPipeError, OSError) as error:
            self.abort()
            raise self._error(f"stdin at frame {self.frames_written + 1}: {error}") from error
        self.frames_written += 1

    def abort(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=10)
        finally:
            self._stderr.close()
            self.temporary.unlink(missing_ok=True)

    def _wait_for_finalization(self):
        started = time.monotonic()
        deadline = started + self.FINALIZE_TIMEOUT_SECONDS
        next_progress = started + 30
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._error(f'finalization exceeded {self.FINALIZE_TIMEOUT_SECONDS}s')
            try:
                code = self.process.wait(timeout=min(self.FINALIZE_POLL_SECONDS, remaining))
                self.metadata['encoder_finalize_wait_s'] = time.monotonic() - started
                return code
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if now >= next_progress:
                    size = self.temporary.stat().st_size if self.temporary.is_file() else 0
                    print(f'[PROGRESS] H.264 finalizing: frames={self.frames_written} '
                          f'elapsed={now-started:.0f}s temporary_bytes={size}', flush=True)
                    next_progress = now + 30

    def release(self):
        if self.closed:
            return
        try:
            self.process.stdin.close()
            code = self._wait_for_finalization()
            self._stderr.close()
            if code != 0:
                raise self._error(f"exit {code}")
            if self.frames_written == 0 or not self.temporary.is_file() or self.temporary.stat().st_size == 0:
                raise self._error("no completed video frames")
            os.replace(self.temporary, self.output)
            self.closed = True
        except BaseException:
            self.abort()
            raise
