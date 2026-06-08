import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.msdc_speed_benchmark import (
    CountingProcessor,
    percentile,
    summarize_latencies,
)


class DummyProcessor:
    def process_frame(self, frame, file_type, conf_override=None):
        return {"boxes": [], "processing_time": 0.01}


def test_percentile_uses_sorted_nearest_rank():
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 95) == 38.5


def test_summarize_latencies_reports_mean_p50_p95():
    summary = summarize_latencies([0.01, 0.02, 0.03], processed_frames=3)
    assert summary["processed_frames"] == 3
    assert round(summary["mean_latency_ms"], 4) == 20.0
    assert round(summary["p50_latency_ms"], 4) == 20.0
    assert round(summary["p95_latency_ms"], 4) == 29.0


def test_counting_processor_records_calls_by_stage():
    processor = CountingProcessor(DummyProcessor())
    processor.stage = "high_det"
    processor.process_frame(frame=object(), file_type="visible", conf_override=None)
    processor.stage = "low_det"
    processor.process_frame(frame=object(), file_type="visible", conf_override=0.18)
    assert processor.call_counts == {"high_det": 1, "low_det": 1}
