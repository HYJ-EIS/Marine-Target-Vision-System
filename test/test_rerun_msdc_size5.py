import json

import pytest

from tools.evaluation.rerun_msdc_size5 import restore_cache


def test_restores_raw_input_not_filtered_or_output_boxes(tmp_path):
    box=dict(x=1,y=2,w=3,h=4,confidence=.3,**{'class':'UAV'})
    rows=[dict(frame_idx=0,file_type='visible',boxes=[dict(box,stage=s) for s in
          ('high_det','low_det','low_only','output')]),
          dict(frame_idx=1,file_type='visible',boxes=[])]
    source=tmp_path/'log.jsonl';dest=tmp_path/'cache.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    assert restore_cache(source,dest)==2
    restored=[json.loads(line) for line in dest.read_text().splitlines()]
    assert restored[0]['high_boxes']==restored[0]['low_boxes']==[box]
    assert restored[1]['low_boxes']==[]
    assert [r['frame_id'] for r in restored]==[1,2]


def test_incomplete_log_is_rejected(tmp_path):
    source=tmp_path/'log.jsonl'
    source.write_text(json.dumps(dict(frame_idx=1,file_type='visible',boxes=[]))+'\n')
    with pytest.raises(ValueError,match='Non-contiguous'):
        restore_cache(source,tmp_path/'cache.jsonl')
