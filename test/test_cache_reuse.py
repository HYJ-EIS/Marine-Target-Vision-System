import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.cache_reuse import detection_cache_name, link_detection_cache_for_sequence


def test_link_detection_cache_for_sequence_symlinks_readable_cache(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    seq_name = "DJI_20250711140128_0002_V"
    source_root.mkdir()
    source_cache = source_root / detection_cache_name(seq_name)
    source_cache.write_text('{"frame_id": 1, "high_boxes": [], "low_boxes": []}\n', encoding="utf-8")

    linked_cache = link_detection_cache_for_sequence(source_root, target_root, seq_name)

    assert linked_cache == target_root / source_cache.name
    assert linked_cache.is_symlink()
    assert linked_cache.resolve() == source_cache.resolve()
    assert linked_cache.read_text(encoding="utf-8") == source_cache.read_text(encoding="utf-8")


def test_link_detection_cache_for_sequence_replaces_stale_symlink(tmp_path):
    old_source_root = tmp_path / "old_source"
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    seq_name = "seq_a"
    old_source_root.mkdir()
    source_root.mkdir()
    old_cache = old_source_root / detection_cache_name(seq_name)
    source_cache = source_root / detection_cache_name(seq_name)
    old_cache.write_text('{"frame_id": 1}\n', encoding="utf-8")
    source_cache.write_text('{"frame_id": 1, "high_boxes": []}\n', encoding="utf-8")
    target_root.mkdir()
    target_cache = target_root / detection_cache_name(seq_name)
    target_cache.symlink_to(old_cache)

    linked_cache = link_detection_cache_for_sequence(source_root, target_root, seq_name)

    assert linked_cache == target_cache
    assert linked_cache.is_symlink()
    assert linked_cache.resolve() == source_cache.resolve()
    assert linked_cache.read_text(encoding="utf-8") == source_cache.read_text(encoding="utf-8")


def test_link_detection_cache_for_sequence_does_not_replace_directory(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    seq_name = "seq_a"
    source_root.mkdir()
    target_root.mkdir()
    source_cache = source_root / detection_cache_name(seq_name)
    source_cache.write_text('{"frame_id": 1}\n', encoding="utf-8")
    target_cache = target_root / detection_cache_name(seq_name)
    target_cache.mkdir()

    with pytest.raises(IsADirectoryError):
        link_detection_cache_for_sequence(source_root, target_root, seq_name)

    assert target_cache.is_dir()
