"""Helpers for reusing detection cache files across evaluation runs."""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path


def detection_cache_name(seq_name: str) -> str:
    return f"{seq_name}_high_low_detections.jsonl"


def link_detection_cache_for_sequence(
    source_cache_root,
    target_cache_root,
    seq_name,
    *,
    symlink=True,
) -> Path:
    source = Path(source_cache_root) / detection_cache_name(seq_name)
    if not source.is_file():
        raise FileNotFoundError(f"Source detection cache is missing: {source}")
    target_root = Path(target_cache_root)
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / source.name
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return target
        target.unlink()
    elif target.exists():
        if target.is_dir():
            raise IsADirectoryError(f"Detection cache target is a directory: {target}")
        try:
            if target.samefile(source) or filecmp.cmp(target, source, shallow=False):
                return target
        except OSError:
            pass
        target.unlink()
    if symlink:
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)
    return target
