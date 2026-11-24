"""Measured stage totals for detector replay and MOT video rendering."""

import json
import time
from pathlib import Path


class StageTiming:
    def __init__(self):
        self.started = time.perf_counter()
        self.frames = 0
        self.seconds = dict.fromkeys((
            "read_decode", "low_detection", "high_split", "tracker",
            "render", "write_export",
        ), 0.0)

    def add(self, stage, started):
        self.seconds[stage] += time.perf_counter() - started

    def write(self, path, **metadata):
        elapsed = time.perf_counter() - self.started
        payload = {
            **metadata,
            "processed_frames": self.frames,
            "total_time_s": elapsed,
            "mean_latency_ms": elapsed * 1000 / self.frames if self.frames else 0,
            "mean_fps": self.frames / elapsed if elapsed else 0,
            "stage_seconds": self.seconds,
            "other_seconds": elapsed - sum(self.seconds.values()),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
