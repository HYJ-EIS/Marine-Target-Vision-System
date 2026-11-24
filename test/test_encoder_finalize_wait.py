"""Slow trailer completion must preserve frames and retain a finite failure gate."""
from pathlib import Path
import subprocess

import pytest

from tools.evaluation import video_encoding as encoding


class Clock:
    value = 0

    def monotonic(self):
        return self.value


class DelayedProcess:
    def __init__(self, clock, complete_at, exit_code=0):
        self.clock, self.complete_at, self.exit_code = clock, complete_at, exit_code
        self.returncode = None
        self.waits = []
        self.killed = False
        self.stdin = self
        self.input_closed = False

    def close(self):
        self.input_closed = True

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is not None:
            return self.returncode
        self.clock.value += timeout
        if self.clock.value < self.complete_at:
            raise subprocess.TimeoutExpired(['synthetic-encoder'], timeout)
        self.returncode = self.exit_code
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def prepared_writer(tmp_path, monkeypatch, complete_at, exit_code=0):
    clock = Clock()
    monkeypatch.setattr(encoding.time, 'monotonic', clock.monotonic)
    writer = object.__new__(encoding.H264VideoWriter)
    writer.output = tmp_path / 'result.mp4'
    writer.temporary = tmp_path / '.result.encoding-test.mp4'
    writer.temporary.write_bytes(b'complete-encoded-frames-and-trailer')
    writer.stderr_path = tmp_path / 'result.ffmpeg.log'
    writer._stderr = writer.stderr_path.open('wb')
    writer.process = DelayedProcess(clock, complete_at, exit_code)
    writer.frames_written = 26757
    writer.metadata = {}
    writer.closed = False
    return writer, clock


def test_success_after_old_sixty_second_limit(tmp_path, monkeypatch, capsys):
    writer, clock = prepared_writer(tmp_path, monkeypatch, complete_at=65)
    writer.release()
    assert writer.closed and writer.process.input_closed
    assert not writer.process.killed and writer._stderr.closed
    assert writer.output.read_bytes() == b'complete-encoded-frames-and-trailer'
    assert not writer.temporary.exists()
    assert clock.value == writer.metadata['encoder_finalize_wait_s'] == 65
    assert max(writer.process.waits) <= 5
    assert 'elapsed=60s' in capsys.readouterr().out


def test_stalled_finalization_is_bounded_and_preserves_previous_output(tmp_path, monkeypatch):
    writer, clock = prepared_writer(tmp_path, monkeypatch, complete_at=float('inf'))
    writer.output.write_bytes(b'previous-successful-video')
    with pytest.raises(RuntimeError, match='finalization exceeded 600s'):
        writer.release()
    assert clock.value == 600
    assert writer.closed and writer.process.killed and writer._stderr.closed
    assert not writer.temporary.exists()
    assert writer.output.read_bytes() == b'previous-successful-video'
    assert 'encoder_finalize_wait_s' not in writer.metadata


def test_late_encoder_failure_does_not_publish(tmp_path, monkeypatch):
    writer, clock = prepared_writer(tmp_path, monkeypatch, complete_at=65, exit_code=17)
    with pytest.raises(RuntimeError, match='exit 17'):
        writer.release()
    assert not writer.output.exists() and not writer.temporary.exists()
    assert writer.closed and writer._stderr.closed
    assert clock.value == 65


def test_successful_release_is_idempotent(tmp_path, monkeypatch):
    writer, clock = prepared_writer(tmp_path, monkeypatch, complete_at=5)
    writer.release()
    waits = list(writer.process.waits)
    writer.release()
    assert writer.process.waits == waits and clock.value == 5
