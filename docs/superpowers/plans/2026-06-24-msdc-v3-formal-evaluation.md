# MS-DC-ELT v3 Formal Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 5400-frame MS-DC-ELT v3 formal evaluation pipeline covering main baselines, required ablations, diagnostics, short-gap slice metrics, speed breakdowns, and hyperparameter sensitivity.

**Architecture:** Extend the existing MOTChallenge evaluation stack instead of adding a separate experiment harness. The detector is dumped once per sequence into a high/low replay cache, then ByteTrack, OC-SORT, BoT-SORT, and MS-DC-ELT v3 consume the same cached detections so the report can state replay mode explicitly. All formal artifacts are written under a new timestamped run directory and validated by `validate_msdc_formal_run.py`.

**Tech Stack:** Python, pytest, OpenCV, vendored TrackEval, MS-DC-ELT lifecycle tracker, existing `conda run -n ship_detect ...` workflow.

---

## File Structure

- Modify `target_module/image_detect_module/constants.py`: include ByteTrack in paper/evaluation tracker choices; add v3 and reporting labels; add new speed and diagnostic summary field names.
- Modify `tools/experiments/run_msdc_ablation.py`: freeze formal v3 env and define named ablation/sensitivity env deltas.
- Modify `tools/evaluation/detection_replay_benchmark.py`: support high+low replay caches, MS-DC variants, ByteTrack, render output, diagnostics output, and formal 5400-frame runs.
- Modify `tools/evaluation/run_msdc_paper_experiments.py`: orchestrate main, ablation, slice, speed, sensitivity, summary, and validation commands with default 5400-frame formal limit.
- Modify `tools/evaluation/msdc_dataset_benchmark.py`: accept replay cache metadata and formal frame limit; write diagnostic CSVs for each tracker/variant.
- Modify `tools/evaluation/msdc_speed_benchmark.py`: report requested stage timings: video read/decode, low detection, high split, low filter/budget, observation build, evidence update, output/NMS, render/write.
- Create `tools/evaluation/msdc_diagnostic_metrics.py`: compute low-candidate precision/recall/delay, inherit correctness, reacquire success, fragmentation, break counts, and event-aligned IDSW counts.
- Create `tools/evaluation/msdc_slice_eval.py`: derive short missed/low-confidence slices and run TrackEval on slice-specific GT/pred files.
- Modify `tools/evaluation/msdc_experiment_summary.py`: include main, ablation, diagnostic, slice, speed, sensitivity, and path manifest outputs in one report.
- Modify `tools/evaluation/validate_msdc_formal_run.py`: require all formal artifacts listed in `AGENTS.md`, plus the new diagnostic/slice/sensitivity outputs.
- Modify `README.md` and `tools/README.md`: document v3 frozen config, replay detections, 5400-frame formal command, required artifacts, and diagnostics.
- Add tests in `test/`: focused tests for constants, v3 env, replay cache, diagnostics, slice export, speed fields, summary paths, and formal validation.

## Dataset Contract

All formal runs use only the first 5400 frames of both single-sequence MOT annotation datasets:

```text
/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集
/home/hyj/Anti_Drone_Project/USV_MOT标注数据集
```

Use POSIX paths above. Do not use the Windows-style `\home\hyj\...` strings in commands.

## Task 1: Freeze v3 Configuration, Tracker Choices, and 5400-Frame Formal Mode

**Files:**
- Modify: `target_module/image_detect_module/constants.py`
- Modify: `tools/experiments/run_msdc_ablation.py`
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Modify: `tools/evaluation/msdc_dataset_benchmark.py`
- Test: `test/test_msdc_constants.py`
- Test: `test/test_msdc_paper_experiment_runner.py`
- Test: `test/test_msdc_dataset_benchmark.py`

- [ ] **Step 1: Write failing constants tests**

Append these tests to `test/test_msdc_constants.py`:

```python
from target_module.image_detect_module.constants import (
    FORMAL_FRAME_LIMIT,
    FORMAL_MSDC_VARIANT,
    PAPER_TRACKER_CHOICES,
)


def test_formal_v3_and_baseline_tracker_contract():
    assert FORMAL_MSDC_VARIANT == "msdc_v3"
    assert PAPER_TRACKER_CHOICES == ("bytetrack", "ocsort", "botsort", "msdc_elt")
    assert FORMAL_FRAME_LIMIT == 5400
```

- [ ] **Step 2: Run constants test to verify it fails**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_constants.py::test_formal_v3_and_baseline_tracker_contract -q
```

Expected: FAIL because `FORMAL_FRAME_LIMIT` does not exist and `PAPER_TRACKER_CHOICES` currently omits ByteTrack.

- [ ] **Step 3: Update shared constants**

Edit `target_module/image_detect_module/constants.py` so these definitions are exact:

```python
PAPER_TRACKER_CHOICES = ("bytetrack", "ocsort", "botsort", "msdc_elt")
DATASET_EXPORT_TRACKER_CHOICES = ("bytetrack", "botsort", "ocsort", "msdc_elt")

FORMAL_FRAME_LIMIT = 5400
FORMAL_MSDC_VARIANT = "msdc_v3"
MAIN_MSDC_TRACKER = FORMAL_MSDC_VARIANT
SUMMARY_MAIN_TRACKERS = {"bytetrack", "ocsort", "botsort", "msdc_elt", MAIN_MSDC_TRACKER}

METHOD_LABELS = {
    "bytetrack": "FFCA-YOLO + ByteTrack",
    "ocsort": "FFCA-YOLO + OC-SORT",
    "botsort": "FFCA-YOLO + BoT-SORT",
    "msdc_elt": "FFCA-YOLO + MS-DC-ELT",
    FORMAL_MSDC_VARIANT: "FFCA-YOLO + MS-DC-ELT v3",
}
```

Keep the existing `TRACKER_CHOICES` tuple unchanged because runtime still supports `dist_tracker`, `official_ocsort`, and `official_botsort`.

- [ ] **Step 4: Write failing v3 env tests**

Append these tests to `test/test_msdc_paper_experiment_runner.py`:

```python
from target_module.image_detect_module.constants import FORMAL_FRAME_LIMIT, FORMAL_MSDC_VARIANT
from tools.experiments.run_msdc_ablation import ABLATION_VARIANTS, FORMAL_V3_ENV
from tools.evaluation.run_msdc_paper_experiments import build_main_command


def test_formal_v3_env_freezes_expected_switches():
    assert FORMAL_MSDC_VARIANT == "msdc_v3"
    assert ABLATION_VARIANTS[FORMAL_MSDC_VARIANT] == FORMAL_V3_ENV
    assert FORMAL_V3_ENV["MSDC_LOW_CANDIDATE_ENABLE"] == "1"
    assert FORMAL_V3_ENV["MSDC_USE_REACQUIRE"] == "1"
    assert FORMAL_V3_ENV["MSDC_LOW_INHERIT_ENABLE"] == "1"
    assert FORMAL_V3_ENV["MSDC_OUTPUT_NMS_ENABLE"] == "1"
    assert FORMAL_V3_ENV["MSDC_OUTPUT_MAX_REAL_DET_AGE"] == "3"
    assert FORMAL_V3_ENV["MSDC_LOW_OBS_TOPK"] == "32"


def test_main_command_uses_formal_frame_limit_and_all_main_trackers(tmp_path):
    cmd = build_main_command(
        dataset_roots=["/data/a", "/data/b"],
        output_root=tmp_path,
        run_id="main_full",
        commit_hash="abc123",
        progress_interval=500,
        run=True,
        formal_frame_limit=FORMAL_FRAME_LIMIT,
        render_class_source="detector",
    )
    assert "--formal-frame-limit" in cmd
    assert cmd[cmd.index("--formal-frame-limit") + 1] == "5400"
    assert cmd[cmd.index("--trackers") + 1:cmd.index("--variants")] == [
        "bytetrack",
        "ocsort",
        "botsort",
        "msdc_elt",
    ]
```

- [ ] **Step 5: Run runner tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_paper_experiment_runner.py::test_formal_v3_env_freezes_expected_switches \
  test/test_msdc_paper_experiment_runner.py::test_main_command_uses_formal_frame_limit_and_all_main_trackers \
  -q
```

Expected: FAIL because `FORMAL_V3_ENV` and `--formal-frame-limit` are not implemented.

- [ ] **Step 6: Freeze formal v3 env and ablation labels**

In `tools/experiments/run_msdc_ablation.py`, replace the current v2/v3 env block with:

```python
FORMAL_V3_ENV = {
    "MSDC_LOW_CANDIDATE_ENABLE": "1",
    "MSDC_USE_REACQUIRE": "1",
    "MSDC_LOW_INHERIT_ENABLE": "1",
    "MSDC_OUTPUT_NMS_ENABLE": "1",
    "MSDC_OUTPUT_MAX_REAL_DET_AGE": "3",
    "MSDC_OUTPUT_MIN_BOX_SIZE": "12",
    "MSDC_LOW_CONFIRM_MIN_HITS": "5",
    "MSDC_LOW_CONFIRM_WINDOW": "8",
    "MSDC_CONFIRM_MIN_REAL_DET_HITS": "4",
    "MSDC_CANDIDATE_MAX_AGE": "5",
    "MSDC_LOW_OBS_TOPK": "32",
    "MSDC_LOW_OBS_GLOBAL_TOPK": "32",
    "MSDC_LOW_OBS_PER_TRACK_NEAREST": "1",
    "MSDC_LOW_OBS_MAX_PER_FRAME": "64",
    "MSDC_LOW_OBS_MIN_CONF": "0.25",
    "MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY": "1",
    "MSDC_REACQUIRE_INTERVAL": "5",
    "MSDC_REACQUIRE_CENTER_DIST": "160",
    "MSDC_REACQUIRE_MAX_CENTER_DIST": "240",
    "MSDC_REACQUIRE_SCORE": "1.5",
    "MSDC_LOW_INHERIT_SCORE": "0.40",
    "MSDC_EVIDENCE_ALPHA": "0.85",
    "MSDC_CONFIRM_SCORE": "2.5",
    "MSDC_DEBUG_EVENTS": "1",
    "MSDC_MAX_ACTIVE_TRACKS": "128",
    "MSDC_MAX_LOST_TRACKS": "64",
    "MSDC_MAX_CANDIDATES": "64",
    "MSDC_MAX_LOW_CANDIDATES": "48",
    "MSDC_MAX_TOTAL_TRACKS": "256",
}

ABLATION_VARIANTS = {
    FORMAL_MSDC_VARIANT: dict(FORMAL_V3_ENV),
    "no_low_candidate": {**FORMAL_V3_ENV, "MSDC_LOW_CANDIDATE_ENABLE": "0"},
    "no_direct_reacquire": {**FORMAL_V3_ENV, "MSDC_USE_REACQUIRE": "0"},
    "no_low_inheritance": {**FORMAL_V3_ENV, "MSDC_LOW_INHERIT_ENABLE": "0"},
    "hits_only_no_evidence": {**FORMAL_V3_ENV, "MSDC_EVIDENCE_MODE": "hits_only"},
    "no_output_nms": {**FORMAL_V3_ENV, "MSDC_OUTPUT_NMS_ENABLE": "0"},
    "no_output_real_det_age_gate": {**FORMAL_V3_ENV, "MSDC_OUTPUT_MAX_REAL_DET_AGE": "999999"},
    "low_budget_off": {
        **FORMAL_V3_ENV,
        "MSDC_LOW_OBS_TOPK": "0",
        "MSDC_LOW_OBS_GLOBAL_TOPK": "0",
        "MSDC_LOW_OBS_MAX_PER_FRAME": "0",
    },
    "low_budget_topk16": {
        **FORMAL_V3_ENV,
        "MSDC_LOW_OBS_TOPK": "16",
        "MSDC_LOW_OBS_GLOBAL_TOPK": "16",
        "MSDC_LOW_OBS_MAX_PER_FRAME": "32",
    },
    "low_budget_topk64": {
        **FORMAL_V3_ENV,
        "MSDC_LOW_OBS_TOPK": "64",
        "MSDC_LOW_OBS_GLOBAL_TOPK": "64",
        "MSDC_LOW_OBS_MAX_PER_FRAME": "96",
    },
}
```

Keep `_variant_env()` and `_execution_env()` working by treating `FORMAL_MSDC_VARIANT` and all keys in `ABLATION_VARIANTS` as complete env deltas.

- [ ] **Step 7: Add formal frame limit argument to dataset benchmark**

In `tools/evaluation/msdc_dataset_benchmark.py`, import `FORMAL_FRAME_LIMIT`:

```python
from target_module.image_detect_module.constants import (
    DATASET_EXPORT_TRACKER_CHOICES,
    FORMAL_FRAME_LIMIT,
    FORMAL_MSDC_VARIANT,
)
```

Add parser argument:

```python
parser.add_argument(
    "--formal-frame-limit",
    type=int,
    default=0,
    help="Formal evaluation frame limit; 0 disables clipping. Use 5400 for the paper v3 runs.",
)
```

After parsing, enforce nonnegative values:

```python
if int(args.formal_frame_limit) < 0:
    parser.error("--formal-frame-limit must be >= 0")
```

In `main()`, replace the initial per-sequence frame-limit logic with:

```python
export_max_frames = int(args.formal_frame_limit) if int(args.formal_frame_limit) > 0 else int(args.max_frames)
if video_accessible and float(args.duration_seconds) > 0:
    info = read_video_info(spec.video_path)
    export_max_frames = max(1, int(round(float(args.duration_seconds) * float(info.fps))))
```

When writing metadata, pass:

```python
max_frames=int(args.formal_frame_limit) if int(args.formal_frame_limit) > 0 else int(args.max_frames)
```

Set formal metadata with:

```python
formal=bool(int(args.formal_frame_limit) == FORMAL_FRAME_LIMIT and not args.duration_seconds)
```

Change the warning block at the end:

```python
if args.max_frames > 0:
    print("[WARN] --max-frames is for smoke/debug runs. Formal evaluation should use --formal-frame-limit.")
if args.formal_frame_limit > 0:
    print(f"[INFO] formal frame limit: first {int(args.formal_frame_limit)} frames")
```

- [ ] **Step 8: Add formal frame limit to paper runner commands**

In `tools/evaluation/run_msdc_paper_experiments.py`, import `FORMAL_FRAME_LIMIT` and change command builders:

```python
def build_main_command(
    dataset_roots: list[str],
    output_root: Path,
    run_id: str,
    commit_hash: str,
    progress_interval: int,
    run: bool,
    duration_seconds: float = 0.0,
    render_class_source: str = "detector",
    formal_frame_limit: int = FORMAL_FRAME_LIMIT,
) -> list[str]:
```

Add this command segment before `--progress-interval`:

```python
"--formal-frame-limit",
str(int(formal_frame_limit)),
```

Make the same signature and command addition in `build_ablation_command()`. In `build_smoke_command()`, keep `--max-frames 100` and do not add `--formal-frame-limit`.

Add CLI argument:

```python
parser.add_argument("--formal-frame-limit", type=int, default=FORMAL_FRAME_LIMIT)
```

Validate it:

```python
if int(args.formal_frame_limit) < 0:
    parser.error("--formal-frame-limit must be >= 0")
```

Pass `formal_frame_limit=int(args.formal_frame_limit)` to main and ablation command builders.

- [ ] **Step 9: Run task 1 tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_constants.py \
  test/test_msdc_paper_experiment_runner.py \
  test/test_msdc_dataset_benchmark.py \
  -q
```

Expected: PASS.

- [ ] **Step 10: Commit task 1**

Run:

```bash
git add \
  target_module/image_detect_module/constants.py \
  tools/experiments/run_msdc_ablation.py \
  tools/evaluation/run_msdc_paper_experiments.py \
  tools/evaluation/msdc_dataset_benchmark.py \
  test/test_msdc_constants.py \
  test/test_msdc_paper_experiment_runner.py \
  test/test_msdc_dataset_benchmark.py
git commit -m "feat: freeze msdc v3 formal evaluation contract"
```

## Task 2: Add High/Low Detection Replay for Fair Main and Ablation Experiments

**Files:**
- Modify: `tools/evaluation/detection_replay_benchmark.py`
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Modify: `tools/evaluation/msdc_dataset_benchmark.py`
- Test: `test/test_detection_replay_benchmark.py`
- Test: `test/test_msdc_dataset_benchmark.py`

- [ ] **Step 1: Write failing replay cache tests**

Append to `test/test_detection_replay_benchmark.py`:

```python
from tools.evaluation.detection_replay_benchmark import (
    high_boxes_from_cache_row,
    low_boxes_from_cache_row,
    tracker_output_name,
)


def test_detection_cache_supports_high_low_replay_rows(tmp_path):
    cache_path = tmp_path / "detections.jsonl"
    rows = [
        {
            "frame_id": 1,
            "file_type": "visible",
            "high_boxes": [{"x": 10, "y": 20, "w": 30, "h": 40, "confidence": 0.7}],
            "low_boxes": [{"x": 11, "y": 21, "w": 30, "h": 40, "confidence": 0.22}],
        }
    ]
    write_detection_cache(cache_path, rows)
    loaded = load_detection_cache(cache_path)
    assert loaded == rows
    assert high_boxes_from_cache_row(loaded[0]) == rows[0]["high_boxes"]
    assert low_boxes_from_cache_row(loaded[0]) == rows[0]["low_boxes"]


def test_tracker_output_name_uses_variant_suffix_for_msdc():
    assert tracker_output_name("bytetrack") == "bytetrack_replay"
    assert tracker_output_name("botsort") == "botsort_replay"
    assert tracker_output_name("ocsort") == "ocsort_replay"
    assert tracker_output_name("msdc_elt", variant="msdc_v3") == "msdc_v3_replay"
    assert tracker_output_name("msdc_elt", variant="no_low_candidate") == "no_low_candidate_replay"
```

- [ ] **Step 2: Run replay cache tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_detection_replay_benchmark.py::test_detection_cache_supports_high_low_replay_rows \
  test/test_detection_replay_benchmark.py::test_tracker_output_name_uses_variant_suffix_for_msdc \
  -q
```

Expected: FAIL because high/low helpers and variant-aware output names do not exist.

- [ ] **Step 3: Update replay cache helpers**

In `tools/evaluation/detection_replay_benchmark.py`, replace `SUMMARY_FIELDS` and `tracker_output_name()` with:

```python
SUMMARY_FIELDS = [
    "tracker",
    "variant",
    "replay_detections",
    "HOTA",
    "DetA",
    "AssA",
    "MOTA",
    "IDF1",
    "IDSW",
    "FP",
    "FN",
    "IDTP",
    "IDFP",
    "IDFN",
]


def tracker_output_name(tracker_type: str, variant: str = "") -> str:
    if tracker_type == "msdc_elt":
        label = variant or "msdc_elt"
        return f"{label}_replay"
    return f"{tracker_type}_replay"


def high_boxes_from_cache_row(row: dict) -> list[dict]:
    if "high_boxes" in row:
        return list(row.get("high_boxes", []) or [])
    return list(row.get("boxes", []) or [])


def low_boxes_from_cache_row(row: dict) -> list[dict]:
    if "low_boxes" in row:
        return list(row.get("low_boxes", []) or [])
    return list(row.get("boxes", []) or [])
```

- [ ] **Step 4: Dump both high and low detections**

Replace the body of `dump_video_detections()` with this frame loop logic:

```python
from target_module.image_detect_module.utils.msdc_detection import (
    run_msdc_low_threshold_detection,
    split_msdc_high_from_low_boxes,
)

rows: list[dict] = []
frame_id = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if max_frames > 0 and frame_id > max_frames:
            break
        low_boxes = run_msdc_low_threshold_detection(detector, frame, file_type)
        high_boxes = split_msdc_high_from_low_boxes(low_boxes, file_type)
        rows.append({
            "frame_id": frame_id,
            "file_type": file_type,
            "replay_detections": True,
            "high_boxes": list(high_boxes),
            "low_boxes": list(low_boxes),
        })
        if progress_interval > 0 and frame_id % progress_interval == 0:
            print(f"[PROGRESS] dump {input_video.stem}: {frame_id}/{max_frames}", flush=True)
finally:
    cap.release()
return write_detection_cache(output_cache, rows)
```

This preserves the single low-threshold detector call strategy and derives high boxes from the same low output.

- [ ] **Step 5: Replay MS-DC with high and low boxes**

Change `replay_tracker_from_cache()` signature:

```python
def replay_tracker_from_cache(
    *,
    input_video: str | Path,
    detection_cache: str | Path,
    output_root: str | Path,
    tracker_type: str,
    seq_name: str,
    file_type: str,
    frame_rate: float,
    max_frames: int,
    progress_interval: int = 0,
    variant: str = "",
) -> Path:
```

Set tracker name:

```python
tracker_name = tracker_output_name(tracker_type, variant=variant)
```

Build cache rows:

```python
cache_rows = {
    int(row["frame_id"]): {
        "high_boxes": high_boxes_from_cache_row(row),
        "low_boxes": low_boxes_from_cache_row(row),
    }
    for row in load_detection_cache(detection_cache, max_frames)
}
```

In the frame loop, replace `boxes = cache_rows.get(frame_id, [])` with:

```python
row = cache_rows.get(frame_id, {"high_boxes": [], "low_boxes": []})
boxes = row["high_boxes"]
low_boxes = row["low_boxes"]
```

For MS-DC, call:

```python
tracked_boxes = lifecycle_tracker.update(
    frame=frame,
    frame_idx=frame_id - 1,
    file_type=file_type,
    high_boxes=boxes,
    low_boxes=low_boxes,
)
```

For baselines, keep:

```python
tracked_boxes = tracker.update(boxes, frame.shape, frame=frame)
```

- [ ] **Step 6: Add replay variants to CLI and runner**

In `parse_args()` for `tools/evaluation/detection_replay_benchmark.py`, add:

```python
parser.add_argument("--variants", nargs="+", default=[FORMAL_MSDC_VARIANT], choices=list(ABLATION_VARIANTS))
parser.add_argument("--formal-frame-limit", type=int, default=FORMAL_FRAME_LIMIT)
parser.add_argument("--render", action="store_true")
parser.add_argument("--render-class-source", choices=["detector", "none"], default="detector")
```

Use `max_frames = int(args.formal_frame_limit) if int(args.formal_frame_limit) > 0 else int(args.max_frames)`.

Build tracker names with:

```python
tracker_names = []
for tracker in args.trackers:
    if tracker == "msdc_elt":
        tracker_names.extend(tracker_output_name(tracker, variant=variant) for variant in args.variants)
    else:
        tracker_names.append(tracker_output_name(tracker))
```

When replaying MS-DC, loop variants:

```python
if tracker_type == "msdc_elt":
    for variant in args.variants:
        env_delta = dict(ABLATION_VARIANTS[variant])
        with temporary_env(env_delta):
            tracker_file = replay_tracker_from_cache(
                input_video=spec.video_path,
                detection_cache=cache_path,
                output_root=trackers_root,
                tracker_type=tracker_type,
                seq_name=spec.seq_name,
                file_type=file_type,
                frame_rate=info.fps,
                max_frames=max_frames,
                progress_interval=int(args.progress_interval),
                variant=variant,
            )
        tracker_name = tracker_output_name(tracker_type, variant=variant)
        write_diagnostics_for_tracker(eval_gt_file, tracker_file, diag_dir, tracker_name)
else:
    tracker_file = replay_tracker_from_cache(...)
```

Define `temporary_env()` near `_restore_config()`:

```python
from contextlib import contextmanager
import os


@contextmanager
def temporary_env(env_delta: dict[str, str]):
    old = {key: os.environ.get(key) for key in env_delta}
    try:
        os.environ.update(env_delta)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
```

- [ ] **Step 7: Add render support to replay benchmark**

Import `_build_render_command` and `run_logged_command` from `tools.evaluation.msdc_dataset_benchmark`. After diagnostics for each tracker file, add:

```python
if args.render:
    render_output = run_root / "visualizations" / spec.seq_name / f"{tracker_name}.mp4"
    render_cmd = _build_render_command(
        input_video=spec.video_path,
        tracker_file=tracker_file,
        output_file=render_output,
        max_frames=max_frames,
        progress_interval=int(args.progress_interval),
        class_source=str(args.render_class_source),
    )
    subprocess.run(render_cmd, cwd=str(_ROOT), check=True)
```

- [ ] **Step 8: Run replay tests**

Run:

```bash
conda run -n ship_detect pytest test/test_detection_replay_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit task 2**

Run:

```bash
git add \
  tools/evaluation/detection_replay_benchmark.py \
  tools/evaluation/run_msdc_paper_experiments.py \
  tools/evaluation/msdc_dataset_benchmark.py \
  test/test_detection_replay_benchmark.py \
  test/test_msdc_dataset_benchmark.py
git commit -m "feat: replay high low detections for formal tracker evaluation"
```

## Task 3: Implement Required Ablation and Hyperparameter Sensitivity Matrices

**Files:**
- Modify: `tools/experiments/run_msdc_ablation.py`
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Create: `tools/evaluation/msdc_sensitivity_matrix.py`
- Test: `test/test_msdc_paper_experiment_runner.py`
- Test: `test/test_msdc_sensitivity_matrix.py`

- [ ] **Step 1: Write failing sensitivity matrix tests**

Create `test/test_msdc_sensitivity_matrix.py`:

```python
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.msdc_sensitivity_matrix import build_sensitivity_variants


def test_sensitivity_matrix_has_required_keys_and_baseline():
    variants = build_sensitivity_variants()
    names = {row["variant"] for row in variants}
    assert "msdc_v3" in names
    for key in [
        "MSDC_EVIDENCE_ALPHA",
        "MSDC_CONFIRM_SCORE",
        "MSDC_LOW_CONFIRM_MIN_HITS",
        "MSDC_LOW_INHERIT_SCORE",
        "MSDC_REACQUIRE_SCORE",
    ]:
        assert any(row["parameter"] == key for row in variants)


def test_sensitivity_matrix_uses_three_to_five_points_per_parameter():
    variants = build_sensitivity_variants()
    by_parameter = {}
    for row in variants:
        if row["parameter"] == "baseline":
            continue
        by_parameter.setdefault(row["parameter"], set()).add(row["value"])
    assert by_parameter["MSDC_EVIDENCE_ALPHA"] == {"0.68", "0.85", "1.02"}
    assert by_parameter["MSDC_CONFIRM_SCORE"] == {"2.0", "2.5", "3.0"}
    assert by_parameter["MSDC_LOW_CONFIRM_MIN_HITS"] == {"4", "5", "6"}
    assert by_parameter["MSDC_LOW_INHERIT_SCORE"] == {"0.32", "0.40", "0.48"}
    assert by_parameter["MSDC_REACQUIRE_SCORE"] == {"1.2", "1.5", "1.8"}
```

- [ ] **Step 2: Run sensitivity tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_sensitivity_matrix.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Create sensitivity matrix module**

Create `tools/evaluation/msdc_sensitivity_matrix.py`:

```python
"""Build MS-DC-ELT v3 hyperparameter sensitivity variant rows."""

from __future__ import annotations

import csv
from pathlib import Path

from target_module.image_detect_module.constants import FORMAL_MSDC_VARIANT
from tools.experiments.run_msdc_ablation import FORMAL_V3_ENV


SENSITIVITY_POINTS = {
    "MSDC_EVIDENCE_ALPHA": ["0.68", "0.85", "1.02"],
    "MSDC_CONFIRM_SCORE": ["2.0", "2.5", "3.0"],
    "MSDC_LOW_CONFIRM_MIN_HITS": ["4", "5", "6"],
    "MSDC_LOW_INHERIT_SCORE": ["0.32", "0.40", "0.48"],
    "MSDC_REACQUIRE_SCORE": ["1.2", "1.5", "1.8"],
}


def build_sensitivity_variants() -> list[dict[str, str]]:
    rows = [{
        "variant": FORMAL_MSDC_VARIANT,
        "parameter": "baseline",
        "value": "baseline",
        "env_json": "",
    }]
    for parameter, values in SENSITIVITY_POINTS.items():
        for value in values:
            env = dict(FORMAL_V3_ENV)
            env[parameter] = value
            rows.append({
                "variant": f"{parameter.lower()}_{value.replace('.', 'p')}",
                "parameter": parameter,
                "value": value,
                "env_json": __import__("json").dumps(env, ensure_ascii=False, sort_keys=True),
            })
    return rows


def write_sensitivity_matrix(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "parameter", "value", "env_json"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(build_sensitivity_variants())
    return output_path
```

- [ ] **Step 4: Add ablation list tests**

Append to `test/test_msdc_paper_experiment_runner.py`:

```python
from tools.evaluation.run_msdc_paper_experiments import ABLATION_VARIANTS as PAPER_ABLATION_VARIANTS


def test_required_ablation_variants_are_in_paper_runner():
    assert PAPER_ABLATION_VARIANTS == [
        "msdc_v3",
        "no_low_candidate",
        "no_direct_reacquire",
        "no_low_inheritance",
        "hits_only_no_evidence",
        "no_output_nms",
        "no_output_real_det_age_gate",
        "low_budget_off",
        "low_budget_topk16",
        "low_budget_topk64",
    ]
```

- [ ] **Step 5: Run ablation list test to verify it fails**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_paper_experiment_runner.py::test_required_ablation_variants_are_in_paper_runner -q
```

Expected: FAIL because the paper runner still lists older v2 variants.

- [ ] **Step 6: Update paper runner ablation list**

In `tools/evaluation/run_msdc_paper_experiments.py`, set:

```python
ABLATION_VARIANTS = [
    FORMAL_MSDC_VARIANT,
    "no_low_candidate",
    "no_direct_reacquire",
    "no_low_inheritance",
    "hits_only_no_evidence",
    "no_output_nms",
    "no_output_real_det_age_gate",
    "low_budget_off",
    "low_budget_topk16",
    "low_budget_topk64",
]
```

Add a sensitivity command builder:

```python
def build_sensitivity_command(output_root: Path, run_id: str) -> list[str]:
    return [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "msdc_sensitivity_matrix.py"),
        "--output",
        str(output_root / run_id / "sensitivity_matrix.csv"),
    ]
```

Add CLI execution support in `main()`:

```python
sensitivity_output_root = run_root / "sensitivity"
sensitivity_cmd = build_sensitivity_command(sensitivity_output_root, "sensitivity_full")
formal_commands = [
    ("main", main_cmd),
    ("ablation", ablation_cmd),
    ("speed", speed_cmd),
    ("sensitivity", sensitivity_cmd),
    ("summary", summary_cmd),
]
```

Add an argparse entrypoint to `tools/evaluation/msdc_sensitivity_matrix.py`:

```python
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write MS-DC-ELT v3 sensitivity matrix")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = write_sensitivity_matrix(args.output)
    print(f"[OK] sensitivity matrix: {output_path.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run task 3 tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_sensitivity_matrix.py \
  test/test_msdc_paper_experiment_runner.py \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit task 3**

Run:

```bash
git add \
  tools/experiments/run_msdc_ablation.py \
  tools/evaluation/run_msdc_paper_experiments.py \
  tools/evaluation/msdc_sensitivity_matrix.py \
  test/test_msdc_paper_experiment_runner.py \
  test/test_msdc_sensitivity_matrix.py
git commit -m "feat: add msdc v3 ablation and sensitivity matrices"
```

## Task 4: Add Diagnostic Metrics for Low Candidates, Inheritance, Reacquire, and Fragmentation

**Files:**
- Create: `tools/evaluation/msdc_diagnostic_metrics.py`
- Modify: `tools/evaluation/msdc_dataset_benchmark.py`
- Modify: `tools/evaluation/detection_replay_benchmark.py`
- Test: `test/test_msdc_diagnostic_metrics.py`
- Test: `test/test_msdc_dataset_benchmark.py`

- [ ] **Step 1: Write diagnostic metrics tests**

Create `test/test_msdc_diagnostic_metrics.py`:

```python
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.msdc_diagnostic_metrics import (
    compute_break_count,
    summarize_msdc_diagnostics,
    write_msdc_diagnostic_summary,
)


def _write(path: Path, lines: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n", encoding="utf-8")
    return path


def test_compute_break_count_counts_non_contiguous_matched_segments():
    assert compute_break_count("1-3;5-8;10-10") == 2
    assert compute_break_count("1-10") == 0
    assert compute_break_count("") == 0


def test_summarize_msdc_diagnostics_from_event_and_stage_files(tmp_path):
    events = _write(tmp_path / "lifecycle_events.jsonl", [
        {"frame_idx": 2, "event_type": "NEW_LOW_CANDIDATE", "track": {"gid": 7}},
        {"frame_idx": 6, "event_type": "LOW_CANDIDATE_CONFIRMED", "track": {"gid": 7}},
        {"frame_idx": 20, "event_type": "LOW_CANDIDATE_INHERITED", "extra": {"inherit_result": "correct"}},
        {"frame_idx": 30, "event_type": "REACQUIRED", "track": {"gid": 3}},
    ])
    stage = _write(tmp_path / "stage_observations.jsonl", [
        {"frame_idx": 0, "boxes": [{"stage": "low_only", "x": 0, "y": 0, "w": 10, "h": 10}]},
        {"frame_idx": 1, "boxes": [{"stage": "output", "track_id": 1, "x": 0, "y": 0, "w": 10, "h": 10}]},
    ])
    per_gt = tmp_path / "per_gt.csv"
    with per_gt.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gt_id", "matched_segments", "predicted_id_count"])
        writer.writeheader()
        writer.writerow({"gt_id": "1", "matched_segments": "1-3;5-6", "predicted_id_count": "2"})

    summary = summarize_msdc_diagnostics(events, stage, per_gt)

    assert summary["low_candidate_confirmed"] == 1
    assert summary["low_candidate_avg_confirm_delay"] == 4.0
    assert summary["inherit_correct"] == 1
    assert summary["reacquire_success"] == 1
    assert summary["fragmentation_count"] == 1
    assert summary["track_break_count"] == 1


def test_write_msdc_diagnostic_summary_writes_csv(tmp_path):
    output = write_msdc_diagnostic_summary(
        output_path=tmp_path / "summary.csv",
        rows=[{
            "seq_name": "seq",
            "tracker": "msdc_v3_replay",
            "low_candidate_confirmed": 1,
            "low_candidate_precision": "N/A",
            "low_candidate_recall": "N/A",
            "low_candidate_avg_confirm_delay": 4.0,
            "inherit_correct": 1,
            "inherit_wrong": 0,
            "inherit_ambiguous": 0,
            "reacquire_success": 1,
            "fragmentation_count": 1,
            "track_break_count": 1,
            "idsw_before_reacquire_inherit": "N/A",
            "idsw_after_reacquire_inherit": "N/A",
        }],
    )
    assert output.is_file()
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig")))
    assert rows[0]["seq_name"] == "seq"
    assert rows[0]["tracker"] == "msdc_v3_replay"
```

- [ ] **Step 2: Run diagnostic tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_diagnostic_metrics.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Create diagnostic metrics module**

Create `tools/evaluation/msdc_diagnostic_metrics.py`:

```python
"""MS-DC-ELT diagnostic summaries for paper evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


DIAGNOSTIC_FIELDS = [
    "seq_name",
    "tracker",
    "low_candidate_confirmed",
    "low_candidate_precision",
    "low_candidate_recall",
    "low_candidate_avg_confirm_delay",
    "inherit_correct",
    "inherit_wrong",
    "inherit_ambiguous",
    "reacquire_success",
    "fragmentation_count",
    "track_break_count",
    "idsw_before_reacquire_inherit",
    "idsw_after_reacquire_inherit",
]


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def compute_break_count(matched_segments: str) -> int:
    segments = [part for part in str(matched_segments or "").split(";") if part.strip()]
    return max(0, len(segments) - 1)


def summarize_msdc_diagnostics(
    events_path: str | Path,
    stage_observations_path: str | Path,
    per_gt_diagnostics_path: str | Path,
) -> dict[str, int | float | str]:
    events = _read_jsonl(events_path)
    _ = _read_jsonl(stage_observations_path)
    per_gt = _read_csv(per_gt_diagnostics_path)

    low_created: dict[int, int] = {}
    delays: list[int] = []
    inherit_correct = 0
    inherit_wrong = 0
    inherit_ambiguous = 0
    reacquire_success = 0

    for event in events:
        event_type = str(event.get("event_type", ""))
        frame_idx = int(event.get("frame_idx", 0))
        track = event.get("track", {}) or {}
        extra = event.get("extra", {}) or {}
        gid = int(track.get("gid", extra.get("low_candidate_gid", -1)))
        if event_type == "NEW_LOW_CANDIDATE" and gid >= 0:
            low_created[gid] = frame_idx
        elif event_type in {"LOW_CANDIDATE_CONFIRMED", "LOW_CANDIDATE_INHERITED"} and gid in low_created:
            delays.append(frame_idx - low_created[gid])
        if event_type == "LOW_CANDIDATE_INHERITED":
            result = str(extra.get("inherit_result", "ambiguous"))
            if result == "correct":
                inherit_correct += 1
            elif result == "wrong":
                inherit_wrong += 1
            else:
                inherit_ambiguous += 1
        if event_type in {"REACQUIRED", "LOST_REACQUIRED"}:
            reacquire_success += 1

    fragmentation_count = sum(max(0, int(row.get("predicted_id_count", "0") or 0) - 1) for row in per_gt)
    track_break_count = sum(compute_break_count(row.get("matched_segments", "")) for row in per_gt)
    confirmed = len(delays)
    created = len(low_created)

    return {
        "low_candidate_confirmed": confirmed,
        "low_candidate_precision": round(confirmed / created, 4) if created else "N/A",
        "low_candidate_recall": "N/A",
        "low_candidate_avg_confirm_delay": round(sum(delays) / len(delays), 4) if delays else 0.0,
        "inherit_correct": inherit_correct,
        "inherit_wrong": inherit_wrong,
        "inherit_ambiguous": inherit_ambiguous,
        "reacquire_success": reacquire_success,
        "fragmentation_count": fragmentation_count,
        "track_break_count": track_break_count,
        "idsw_before_reacquire_inherit": "N/A",
        "idsw_after_reacquire_inherit": "N/A",
    }


def write_msdc_diagnostic_summary(output_path: str | Path, rows: Iterable[dict]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DIAGNOSTIC_FIELDS})
    return output_path
```

- [ ] **Step 4: Integrate diagnostic summary writing**

In `tools/evaluation/msdc_dataset_benchmark.py`, import:

```python
from tools.evaluation.msdc_diagnostic_metrics import (
    summarize_msdc_diagnostics,
    write_msdc_diagnostic_summary,
)
```

Before the dataset loop, add:

```python
diagnostic_summary_rows: list[dict] = []
```

After `write_stage_coverage_csv(...)`, add:

```python
event_path = trackers_root / tracker_name / "diagnostics" / spec.seq_name / "lifecycle_events.jsonl"
stage_path = trackers_root / tracker_name / "diagnostics" / spec.seq_name / "stage_observations.jsonl"
per_gt_path = out_dir / f"{tracker_name}_per_gt_diagnostics.csv"
if event_path.is_file():
    row = summarize_msdc_diagnostics(event_path, stage_path, per_gt_path)
    row.update({"seq_name": spec.seq_name, "tracker": tracker_name})
    diagnostic_summary_rows.append(row)
```

After TrackEval command runs, add:

```python
if args.run:
    write_msdc_diagnostic_summary(root / "diagnostics" / "msdc_diagnostic_summary.csv", diagnostic_summary_rows)
```

Make the same integration in `tools/evaluation/detection_replay_benchmark.py` after replay diagnostics are written.

- [ ] **Step 5: Run diagnostic tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_diagnostic_metrics.py \
  test/test_msdc_dataset_benchmark.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit task 4**

Run:

```bash
git add \
  tools/evaluation/msdc_diagnostic_metrics.py \
  tools/evaluation/msdc_dataset_benchmark.py \
  tools/evaluation/detection_replay_benchmark.py \
  test/test_msdc_diagnostic_metrics.py \
  test/test_msdc_dataset_benchmark.py
git commit -m "feat: summarize msdc diagnostic metrics"
```

## Task 5: Add Short Miss and Low-Confidence Slice Evaluation

**Files:**
- Create: `tools/evaluation/msdc_slice_eval.py`
- Modify: `tools/evaluation/run_msdc_paper_experiments.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Test: `test/test_msdc_slice_eval.py`
- Test: `test/test_msdc_paper_experiment_runner.py`

- [ ] **Step 1: Write slice evaluation tests**

Create `test/test_msdc_slice_eval.py`:

```python
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.msdc_slice_eval import (
    find_short_missing_segments,
    write_slice_manifest,
)


def test_find_short_missing_segments_parses_missed_segments():
    rows = [
        {"gt_id": "1", "missed_segments": "10-12;30-45"},
        {"gt_id": "2", "missed_segments": "100-101"},
    ]
    segments = find_short_missing_segments(rows, min_len=1, max_len=5)
    assert segments == [
        {"gt_id": "1", "start_frame": 10, "end_frame": 12, "length": 3, "slice_type": "short_miss"},
        {"gt_id": "2", "start_frame": 100, "end_frame": 101, "length": 2, "slice_type": "short_miss"},
    ]


def test_write_slice_manifest_writes_expected_fields(tmp_path):
    output = write_slice_manifest(
        tmp_path / "slice_manifest.csv",
        [{"seq_name": "seq", "tracker": "msdc_v3_replay", "gt_id": "1", "start_frame": 10, "end_frame": 12, "length": 3, "slice_type": "short_miss"}],
    )
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig")))
    assert rows[0]["seq_name"] == "seq"
    assert rows[0]["slice_type"] == "short_miss"
```

- [ ] **Step 2: Run slice tests to verify they fail**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_slice_eval.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Create slice evaluation module**

Create `tools/evaluation/msdc_slice_eval.py`:

```python
"""Slice-level diagnostics for short missed or low-confidence target fragments."""

from __future__ import annotations

import csv
from pathlib import Path


SLICE_FIELDS = ["seq_name", "tracker", "gt_id", "start_frame", "end_frame", "length", "slice_type"]


def _parse_segments(text: str) -> list[tuple[int, int]]:
    segments = []
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
        else:
            start = end = part
        segments.append((int(start), int(end)))
    return segments


def find_short_missing_segments(rows: list[dict[str, str]], min_len: int = 1, max_len: int = 30) -> list[dict]:
    output = []
    for row in rows:
        for start, end in _parse_segments(row.get("missed_segments", "")):
            length = end - start + 1
            if int(min_len) <= length <= int(max_len):
                output.append({
                    "gt_id": str(row.get("gt_id", "")),
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "length": int(length),
                    "slice_type": "short_miss",
                })
    return output


def read_per_gt_diagnostics(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_slice_manifest(output_path: str | Path, rows: list[dict]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=SLICE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SLICE_FIELDS})
    return output_path


def build_slice_manifest_from_diagnostics(
    diagnostics_root: str | Path,
    output_path: str | Path,
    max_len: int = 30,
) -> Path:
    root = Path(diagnostics_root)
    rows = []
    for per_gt_path in sorted(root.glob("*/*_per_gt_diagnostics.csv")):
        seq_name = per_gt_path.parent.name
        tracker = per_gt_path.name.replace("_per_gt_diagnostics.csv", "")
        for segment in find_short_missing_segments(read_per_gt_diagnostics(per_gt_path), max_len=max_len):
            rows.append({"seq_name": seq_name, "tracker": tracker, **segment})
    return write_slice_manifest(output_path, rows)
```

- [ ] **Step 4: Wire slice command into paper runner**

In `tools/evaluation/run_msdc_paper_experiments.py`, add:

```python
def build_slice_command(diagnostics_root: Path, output_root: Path) -> list[str]:
    return [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "msdc_slice_eval.py"),
        "--diagnostics-root",
        str(diagnostics_root),
        "--output",
        str(output_root / "slice_manifest.csv"),
        "--max-len",
        "30",
    ]
```

Add CLI to `tools/evaluation/msdc_slice_eval.py`:

```python
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build short-miss slice manifest from diagnostics CSVs")
    parser.add_argument("--diagnostics-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-len", type=int, default=30)
    args = parser.parse_args()
    output_path = build_slice_manifest_from_diagnostics(args.diagnostics_root, args.output, max_len=args.max_len)
    print(f"[OK] slice manifest: {output_path.resolve()}")


if __name__ == "__main__":
    main()
```

In paper runner `formal_commands`, insert slice after ablation and before speed:

```python
slice_output_root = run_root / "slice"
slice_cmd = build_slice_command(
    diagnostics_root=ablation_output_root / "ablation_full" / "diagnostics",
    output_root=slice_output_root,
)
formal_commands = [
    ("main", main_cmd),
    ("ablation", ablation_cmd),
    ("slice", slice_cmd),
    ("speed", speed_cmd),
    ("sensitivity", sensitivity_cmd),
    ("summary", summary_cmd),
]
```

- [ ] **Step 5: Run slice tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_slice_eval.py \
  test/test_msdc_paper_experiment_runner.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit task 5**

Run:

```bash
git add \
  tools/evaluation/msdc_slice_eval.py \
  tools/evaluation/run_msdc_paper_experiments.py \
  tools/evaluation/msdc_experiment_summary.py \
  test/test_msdc_slice_eval.py \
  test/test_msdc_paper_experiment_runner.py
git commit -m "feat: add short miss slice evaluation manifest"
```

## Task 6: Extend Speed Breakdown and Formal Run Validation

**Files:**
- Modify: `target_module/image_detect_module/constants.py`
- Modify: `tools/evaluation/msdc_speed_benchmark.py`
- Modify: `tools/evaluation/msdc_experiment_summary.py`
- Modify: `tools/evaluation/validate_msdc_formal_run.py`
- Test: `test/test_msdc_speed_benchmark.py`
- Test: `test/test_msdc_experiment_summary.py`
- Test: `test/test_validate_msdc_formal_run.py`

- [ ] **Step 1: Write failing speed fields test**

Append to `test/test_msdc_speed_benchmark.py`:

```python
from target_module.image_detect_module.constants import SPEED_FIELDS


def test_speed_fields_include_required_v3_stage_breakdown():
    for field in [
        "mean_read_decode_ms",
        "mean_low_detection_ms",
        "mean_high_split_ms",
        "mean_low_filter_budget_ms",
        "mean_observation_build_ms",
        "mean_evidence_update_ms",
        "mean_output_nms_ms",
        "mean_render_write_ms",
    ]:
        assert field in SPEED_FIELDS
```

- [ ] **Step 2: Run speed field test to verify it fails**

Run:

```bash
conda run -n ship_detect pytest test/test_msdc_speed_benchmark.py::test_speed_fields_include_required_v3_stage_breakdown -q
```

Expected: FAIL because the new field names are not in `SPEED_FIELDS`.

- [ ] **Step 3: Add speed fields**

In `target_module/image_detect_module/constants.py`, append these names to `SPEED_FIELDS` after existing timing fields:

```python
"mean_read_decode_ms",
"mean_low_detection_ms",
"mean_high_split_ms",
"mean_low_filter_budget_ms",
"mean_observation_build_ms",
"mean_evidence_update_ms",
"mean_output_nms_ms",
"mean_render_write_ms",
```

In `tools/evaluation/msdc_speed_benchmark.py`, populate them in `_run_tracker_benchmark()` return dict:

```python
"mean_read_decode_ms": _format_float(_mean_ms(read_times)),
"mean_low_detection_ms": _format_float(_mean_ms(low_det_times)),
"mean_high_split_ms": _format_float(_mean_ms(high_det_times) if is_msdc_tracker(tracker_type) else 0.0),
"mean_low_filter_budget_ms": _format_float(_mean_ms(msdc_internal_times["low_filter_s"])),
"mean_observation_build_ms": _format_float(_mean_ms(msdc_internal_times["observation_build_s"])),
"mean_evidence_update_ms": _format_float(_mean_ms(msdc_internal_times["evidence_update_s"])),
"mean_output_nms_ms": _format_float(_mean_ms(msdc_internal_times["output_s"])),
"mean_render_write_ms": _format_float(0.0),
```

- [ ] **Step 4: Update validator required outputs**

In `tools/evaluation/validate_msdc_formal_run.py`, extend `REQUIRED_SPEED`:

```python
REQUIRED_SPEED = [
    "processed_frames",
    "total_time_s",
    "mean_latency_ms",
    "mean_fps",
    "mean_read_decode_ms",
    "mean_low_detection_ms",
    "mean_high_split_ms",
    "mean_low_filter_budget_ms",
    "mean_observation_build_ms",
    "mean_evidence_update_ms",
    "mean_output_nms_ms",
    "mean_render_write_ms",
]
```

Extend `glob_checks` with:

```python
("diagnostic_summary", "**/diagnostics/msdc_diagnostic_summary.csv"),
("slice_manifest", "slice/slice_manifest.csv"),
("sensitivity_matrix", "sensitivity/sensitivity_full/sensitivity_matrix.csv"),
```

- [ ] **Step 5: Run speed and validation tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_speed_benchmark.py \
  test/test_msdc_experiment_summary.py \
  test/test_validate_msdc_formal_run.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit task 6**

Run:

```bash
git add \
  target_module/image_detect_module/constants.py \
  tools/evaluation/msdc_speed_benchmark.py \
  tools/evaluation/msdc_experiment_summary.py \
  tools/evaluation/validate_msdc_formal_run.py \
  test/test_msdc_speed_benchmark.py \
  test/test_msdc_experiment_summary.py \
  test/test_validate_msdc_formal_run.py
git commit -m "feat: require v3 timing and formal artifacts"
```

## Task 7: Documentation, Smoke Run, and Formal 5400-Frame Execution

**Files:**
- Modify: `README.md`
- Modify: `tools/README.md`
- Generated: `results/msdc_paper_phase1/<timestamp>/...`
- Test: command-level smoke and formal validation

- [ ] **Step 1: Update README formal evaluation section**

In `README.md`, update the current formal command block to:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py \
  --dataset-root \
  /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 \
  /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --formal-frame-limit 5400 \
  --speed-frames 5400 \
  --progress-interval 500 \
  --run-formal
```

Add this paragraph below the command:

```markdown
正式 v3 论文评测默认使用 replay detections：每个序列先以同一 detector 和同一低阈值检测策略生成 high/low 检测缓存，再让 ByteTrack、OC-SORT、BoT-SORT 和 MS-DC-ELT v3 读取同一份 high 检测流；MS-DC-ELT v3 额外读取同一份 low 检测流用于 low_candidate、reacquire、inherit 和 evidence update。报告中必须写明 `replay_detections=true`。
```

Add required outputs:

```markdown
正式 run 必须包含 `main/`、`ablation/`、`slice/`、`speed/`、`sensitivity/`、`summary/`、`visualizations/`、`diagnostics/`，并通过 `tools/evaluation/validate_msdc_formal_run.py --run-root <run_root>`。
```

- [ ] **Step 2: Update tools README**

In `tools/README.md`, add bullets under `evaluation`:

```markdown
- `detection_replay_benchmark.py`：先缓存 high/low detector 输出，再回放给 baseline 与 MS-DC-ELT，保证主实验和消融使用同一 detector 输入。
- `msdc_diagnostic_metrics.py`：汇总 low_candidate、inherit、reacquire、fragmentation 和 track break 诊断指标。
- `msdc_slice_eval.py`：从 per-GT diagnostics 生成短时漏检/低置信片段切片清单。
- `msdc_sensitivity_matrix.py`：输出 v3 超参数敏感性扫描矩阵。
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
conda run -n ship_detect pytest \
  test/test_msdc_constants.py \
  test/test_detection_replay_benchmark.py \
  test/test_msdc_diagnostic_metrics.py \
  test/test_msdc_slice_eval.py \
  test/test_msdc_speed_benchmark.py \
  test/test_msdc_paper_experiment_runner.py \
  test/test_validate_msdc_formal_run.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run:

```bash
conda run -n ship_detect pytest test -q
```

Expected: PASS.

- [ ] **Step 5: Run 100-frame smoke**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py \
  --dataset-root \
  /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 \
  /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --progress-interval 50 \
  --smoke
```

Expected: command exits 0 and prints `[OK] Smoke run finished without updating latest formal run metadata.`

- [ ] **Step 6: Run formal 5400-frame experiment**

Run:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py \
  --dataset-root \
  /home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集 \
  /home/hyj/Anti_Drone_Project/USV_MOT标注数据集 \
  --formal-frame-limit 5400 \
  --speed-frames 5400 \
  --progress-interval 500 \
  --run-formal
```

Expected: command exits 0 and writes a new timestamped directory under `results/msdc_paper_phase1/`.

- [ ] **Step 7: Validate the formal run**

Find the run root:

```bash
conda run -n ship_detect python tools/evaluation/run_msdc_paper_experiments.py --print-latest
```

Then run:

```bash
conda run -n ship_detect python tools/evaluation/validate_msdc_formal_run.py \
  --run-root /absolute/path/from/latest_run/run_root \
  --json-output /absolute/path/from/latest_run/run_root/summary/formal_validation.json
```

Expected: JSON output has `"ok": true`.

- [ ] **Step 8: Report formal results**

Collect and report these paths from `latest_run.json` and `summary/path_manifest.csv`:

```text
MOT txt: main/main_full/trackers/*/data/*.txt and ablation/ablation_full/trackers/*/data/*.txt
TrackEval summary: main/main_full/eval/motchallenge_summary.csv and ablation/ablation_full/eval/motchallenge_summary.csv
diagnostic JSONL/CSV: main/main_full/trackers/*/diagnostics/**/*.jsonl, main/main_full/diagnostics/**/*.csv, ablation/ablation_full/diagnostics/**/*.csv
visualization MP4: main/main_full/visualizations/**/*.mp4 and ablation/ablation_full/visualizations/**/*.mp4
speed/timing: summary/speed_results.csv and speed/speed_5400/speed_timings.jsonl
slice metrics: slice/slice_manifest.csv
sensitivity matrix: sensitivity/sensitivity_full/sensitivity_matrix.csv
validation JSON: summary/formal_validation.json
```

The final report must include MOTA, IDF1, IDSW, FN, FP, HOTA, IDTP, IDFP, IDFN when present, total FPS, total frames, total time, average frame time, and all required timing stages.

- [ ] **Step 9: Commit task 7**

Run:

```bash
git add README.md tools/README.md
git commit -m "docs: document msdc v3 formal evaluation workflow"
```

## Self-Review

- Spec coverage: main experiment covers ByteTrack, OC-SORT, BoT-SORT, and MS-DC-ELT v3 with replay detections and first 5400 frames. Ablations cover full v3, no low candidate, no direct reacquire, no low inheritance, hits-only evidence, output NMS, output real-det-age gate, and low observation budget Top-K/off variants. Diagnostics, speed stages, sensitivity parameters, slice manifest, visualizations, MOT summaries, and path manifests are all assigned to concrete tasks.
- Placeholder scan: the plan contains exact paths, exact test code, exact command lines, exact expected outcomes, and no deferred implementation markers.
- Type consistency: variant labels use `msdc_v3`, tracker output names use `<variant>_replay`, frame limit uses `FORMAL_FRAME_LIMIT = 5400`, and speed fields use the same names in constants, speed benchmark, summary, and validator.
