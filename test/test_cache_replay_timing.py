"""Timing provenance checks using synthetic files; no video is decoded."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation.paper_statistics import STAGES
from tools.evaluation.paper_statistics_run import Method, timing_row


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture_run(tmp_path, batch, replay_metadata):
    sequence, length = 'synthetic_no_video', 4
    method = Method('Full' if batch == 'main' else '−A', batch, 'synthetic_replay', 'synthetic', True)
    shared = {'processed_frames': length, 'stage_seconds': dict.fromkeys(STAGES, .1), 'total_time_s': 1}
    replay = {'processed_frames': length, 'stage_seconds': dict.fromkeys(STAGES, .2), 'total_time_s': 2,
              **replay_metadata}
    render = {'processed_frames': length, 'stage_seconds': dict.fromkeys(STAGES, .3), 'total_time_s': 3}
    mot = tmp_path/batch/'trackers'/method.tracker/'data'/f'{sequence}.txt'
    write_json(mot.with_suffix('.timing.json'), replay)
    mot.write_text('', encoding='utf-8')
    mot.with_suffix('.frames.jsonl').write_text(''.join(json.dumps({'frame_id': frame})+'\n' for frame in range(1, length+1)), encoding='utf-8')
    video = tmp_path/batch/'visualizations'/sequence/(method.tracker+'.mp4')
    video.parent.mkdir(parents=True)
    video.write_bytes(b'synthetic placeholder; timing_row never decodes video')
    write_json(video.with_suffix('.timing.json'), render)
    return method, sequence, length, shared, replay, render


@pytest.mark.parametrize('batch,metadata,source,pixels,scope', [
    ('main', {}, 'decode', 4, 'main_comparison'),
    ('core', {'frame_source': 'cache_only', 'pixel_frames_decoded': 0}, 'cache_only', 0, 'ablation_execution'),
    ('sensitivity', {'frame_source': 'decode', 'pixel_frames_decoded': 4}, 'decode', 4, 'ablation_execution'),
])
def test_timing_provenance_does_not_change_existing_values(tmp_path, batch, metadata, source, pixels, scope):
    method, sequence, length, shared, replay, render = fixture_run(tmp_path, batch, metadata)
    if source == 'cache_only':
        replay['stage_seconds']['read_decode'] = 0
        write_json(tmp_path/batch/'trackers'/method.tracker/'data'/f'{sequence}.timing.json', replay)
    row = timing_row(tmp_path, method, sequence, length, shared)
    assert row['frame_source'] == source
    assert row['pixel_frames_decoded'] == pixels
    assert row['speed_comparison_scope'] == scope
    assert row['replay_read_decode_s'] == replay['stage_seconds']['read_decode']
    # All pre-existing numeric values retain the frozen aggregation formula.
    assert row['processed_frames'] == length
    assert row['total_time_s'] == 6
    assert row['mean_latency_ms'] == 1500
    assert row['mean_fps'] == length/6
    assert row['tracker_time_s'] == .2
    assert row['tracker_ms_per_frame'] == 50
    assert row['tracker_fps'] == 20
    assert row['shared_detection_time_s'] == 1
    expected_stages = {key: sum(data['stage_seconds'][key] for data in (shared, replay, render)) for key in STAGES}
    for key, expected in expected_stages.items():
        assert row[key+'_s'] == expected
    assert row['other_s'] == 6-sum(expected_stages.values())
    # Aggregated read/decode still includes shared detection and rendering;
    # cache-only provenance exposes the replay portion separately.
    assert row['read_decode_s'] == shared['stage_seconds']['read_decode']+replay['stage_seconds']['read_decode']+render['stage_seconds']['read_decode']
