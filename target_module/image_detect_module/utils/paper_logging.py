"""Complete per-frame research logs, separate from bounded debug summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PAPER_LOG_KINDS = ("observations", "associations", "states", "events", "outputs")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


class PaperFrameLogger:
    """Write all five streams with one envelope per processed video frame.

    A logger is scoped to one method and sequence. Opening it starts a fresh
    replay for that pair; no GT is accepted by this runtime logger.
    """

    def __init__(self, directory: str | Path, sequence: str, method: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sequence = str(sequence)
        self.method = str(method)
        self._last_frame = 0
        self._streams = {
            kind: (self.directory / f"{kind}.jsonl").open("w", encoding="utf-8")
            for kind in PAPER_LOG_KINDS
        }

    def write_frame(self, frame: int, records: dict[str, list]) -> None:
        frame = int(frame)
        if frame != self._last_frame + 1:
            raise ValueError(f"Paper frames must be contiguous from 1, got {frame} after {self._last_frame}")
        for kind in PAPER_LOG_KINDS:
            envelope = {
                "seq": self.sequence, "method": self.method,
                "frame": frame, "records": records.get(kind, []),
            }
            self._streams[kind].write(
                json.dumps(_json_value(envelope), ensure_ascii=False, allow_nan=False) + "\n"
            )
        self._last_frame = frame

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
