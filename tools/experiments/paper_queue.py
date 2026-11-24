"""Build and execute the fixed, serial, 21-job visible-nine paper queue."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
CONDA = "/home/hyj/anaconda3/bin/conda"
STAGES = ("read_decode", "low_detection", "high_split", "tracker", "render", "write_export")
LOGS = ("observations", "associations", "states", "events", "outputs")
BASELINES = ("official_botsort_replay", "bytetrack_replay", "official_ocsort_replay")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def marker_path(run_root, job_id):
    return Path(run_root) / "queue" / "markers" / f"{job_id}.completed.json"


def build_manifest(run_root, spec_path, repo=REPO):
    run_root, spec_path, repo = Path(run_root).resolve(), Path(spec_path).resolve(), Path(repo).resolve()
    spec = read_json(spec_path)
    assert len(spec["sequences"]) == 9 and len(spec["core_variants"]) == len(spec["sensitivity_variants"]) == 8
    phases = []
    for stage, dependency in (("main", []), ("core", ["main"]), ("core_stats", ["core"]),
                              ("sensitivity", ["core_stats"]), ("all_stats", ["sensitivity"])):
        jobs = []
        sequences = spec["sequences"] if stage in {"core", "sensitivity"} else [None]
        for index, row in enumerate(sequences, 1):
            job_id = stage if row is None else f"{stage}_{index:02d}"
            command = [CONDA, "run", "--no-capture-output", "-n", "ship_detect", "python", "-B",
                       str(repo / "tools/experiments/paper_queue.py"), "run",
                       "--run-root", str(run_root), "--spec", str(spec_path), "--stage", stage,
                       "--job-id", job_id]
            if row is not None:
                command += ["--sequence", row["sequence"]]
            jobs.append({"id": job_id, "cmd": shlex.join(command),
                         "expected_output": str(marker_path(run_root, job_id)),
                         "oom_requires_manual_resume": stage == "main"})
        phases.append({"name": stage, "depends_on": dependency, "jobs": jobs})
    return {"project": "paper_visible9_fixed180", "backend": "local_wsl",
            "cwd": str(repo), "conda": "ship_detect", "gpus": [0], "max_parallel": 1,
            "gpu_minimum_free_mib": 4096, "gpu_utilization_policy": "record_only",
            "oom_retry": {"delay": 120, "max_attempts": 3},
            "non_oom_manual_retry_limit": 1, "run_root": str(run_root),
            "spec": str(spec_path), "formal_unique_results": 180, "phases": phases}


def frame_lines(path, length, frame_key):
    count = 0
    with Path(path).open(encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            if int(row[frame_key]) != count:
                raise ValueError(f"Non-contiguous frame log: {path}: {count}")
            if frame_key == "frame" and not isinstance(row.get("records"), list):
                raise ValueError(f"Missing complete records array: {path}:{count}")
    if count != length:
        raise ValueError(f"Frame log length {count} != {length}: {path}")


def timing_check(path, length):
    timing = read_json(path)
    if int(timing["processed_frames"]) != length:
        raise ValueError(f"Timing length differs: {path}")
    if set(STAGES) - set(timing["stage_seconds"]):
        raise ValueError(f"Missing timing stages: {path}")
    return timing


def validate_protocol(directory):
    protocol = read_json(Path(directory) / "evaluation_protocol.json")
    if not protocol.get("class_consistency_required") or protocol.get("matching") != "same_class_and_iou":
        raise ValueError(f"Wrong class matching protocol: {directory}")


def validate_result(run_root, batch, tracker, sequence, length, paper_logs=True):
    root = Path(run_root) / batch
    mot = root / "trackers" / tracker / "data" / f"{sequence}.txt"
    if not mot.is_file():
        raise ValueError(f"Missing MOT: {mot}")
    with mot.open(encoding="utf-8-sig", newline="") as stream:
        seen = set()
        for row in csv.reader(stream):
            if not row:
                continue
            frame, identity = int(row[0]), int(row[1])
            cls = float(row[7])
            if not (1 <= frame <= length and cls.is_integer() and 1 <= cls <= 3
                    and float(row[4]) > 0 and float(row[5]) > 0):
                raise ValueError(f"Invalid frame/class in {mot}: {row[:8]}")
            if (frame, identity) in seen:
                raise ValueError(f"Duplicate public ID in frame: {mot}")
            seen.add((frame, identity))
    frame_lines(mot.with_suffix(".frames.jsonl"), length, "frame_id")
    timing_check(mot.with_suffix(".timing.json"), length)
    video = root / "visualizations" / sequence / f"{tracker}.mp4"
    timing_check(video.with_suffix(".timing.json"), length)
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != length:
            raise ValueError(f"Rendered video length differs: {video}")
    finally:
        capture.release()
    if paper_logs:
        log_root = root / "trackers" / tracker / "paper_logs" / sequence
        for name in LOGS:
            frame_lines(log_root / f"{name}.jsonl", length, "frame")
    return {"batch": batch, "tracker": tracker, "sequence": sequence, "frames": length,
            "mot": str(mot), "video": str(video)}


def archive_metadata(run_root, batch, sequence, trackers, replayed_variants):
    """Archive current-sequence evidence, distinguishing a partial replay call."""
    root = Path(run_root) / batch
    target = root / "metadata" / sequence
    target.mkdir(parents=True, exist_ok=True)
    # target/eval was just recomputed for this sequence and all eight trackers.
    # Global eval/summary may contain earlier sequences or only the retry subset.
    config_target = target / "effective_config"
    config_target.mkdir(exist_ok=True)
    for tracker in trackers:
        source = root / "effective_config" / f"{tracker}.json"
        if source.is_file():
            shutil.copyfile(source, config_target / source.name)
    for name in ("baseline_input_policy.json", "paper_environment.json"):
        if (root / name).is_file():
            shutil.copyfile(root / name, target / name)
    diagnostic = root / "diagnostics" / sequence
    if diagnostic.is_dir():
        shutil.copytree(diagnostic, target / "diagnostics", dirs_exist_ok=True)
    if replayed_variants:
        call = target / "replay_calls" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        call.mkdir(parents=True)
        if (root / "summary").is_dir():
            shutil.copytree(root / "summary", call / "summary", dirs_exist_ok=True)
        for name in ("run_metadata.json", "phase_results.json"):
            if (root / name).is_file():
                shutil.copyfile(root / name, call / name)
        source = root / "diagnostics/msdc_diagnostic_summary.csv"
        if source.is_file():
            with source.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = [row for row in reader if row.get("seq_name") == sequence
                        and row.get("tracker") in trackers]
            with (call / source.name).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields or ["seq_name", "tracker"])
                writer.writeheader()
                writer.writerows(rows)
        write_json(call / "scope.json", {"sequence": sequence, "replayed_variants": replayed_variants,
                   "scope": "last replay command subset, not the complete eight-tracker sequence evaluation"})
    write_json(target / "metadata_scope.json", {
        "sequence": sequence, "evaluated_trackers": trackers,
        "evaluation": "eval contains only this sequence and all specified trackers",
        "replayed_variants": replayed_variants,
        "reused_trackers": [tracker for tracker in trackers
                            if tracker.removesuffix("_replay") not in replayed_variants],
        "summary_scope": "replay_calls/* preserves each actual command subset; global summary is not a full-sequence table",
    })


def preserve_partial_result(run_root, batch, tracker, sequence, job_id):
    """Move only this failed result's generated files; retain valid peers/cache."""
    run_root = Path(run_root).resolve()
    root = run_root / batch
    destination = run_root / "queue/partial_results" / job_id / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = root / "trackers" / tracker
    mot = base / "data" / f"{sequence}.txt"
    video = root / "visualizations" / sequence / f"{tracker}.mp4"
    owned = [mot, mot.with_suffix(".frames.jsonl"), mot.with_suffix(".timing.json"),
             video, video.with_suffix(".timing.json"), base / "paper_logs" / sequence,
             base / "diagnostics" / sequence]
    for source in owned:
        if source.exists():
            assert source.resolve().is_relative_to(run_root), source
            target = destination / source.relative_to(run_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)


def csv_count(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def validate_statistics(run_root, phase):
    root = Path(run_root) / "paper_statistics"
    expected = {"main_by_sequence.csv": 36, "ablation_by_sequence.csv": 45,
                "recovery_s0_by_sequence.csv": 36, "timing_by_sequence.csv": 108 if phase == "core" else 180}
    if phase == "all":
        expected.update({"sensitivity_distance.csv": 45, "sensitivity_confirmation.csv": 45})
    for name, count in expected.items():
        if csv_count(root / name) != count:
            raise ValueError(f"Wrong row count: {name}; expected {count}")
    for name in ("interactions", "a_actions", "e_confirmation", "r_recovery", "confirmation_events",
                 "recovery_events", "identity_by_gt", "identity_by_sequence", "duplicate_outputs",
                 "duplicate_events", "output_matches"):
        if not (root / f"{name}.csv").is_file():
            raise ValueError(f"Missing statistics table: {name}")
    manifest = read_json(root / "statistics_manifest.json")
    if manifest["status"] != "complete" or manifest["unique_method_sequence_results"] != (108 if phase == "core" else 180):
        raise ValueError("Statistics manifest incomplete")
    return expected


def validate_cases(run_root):
    root = Path(run_root) / "paper_statistics"
    with (root / "case_selection.csv").open(encoding="utf-8-sig", newline="") as stream:
        selection = list(csv.DictReader(stream))
    expected = {}
    for row in selection:
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if end - start + 1 != 40:
            raise ValueError("Case selection must cover 40 original frames")
        for side in ("a", "b"):
            expected[(row["sequence"], row[f"method_{side}"], start, end)] = str(Path(row[f"mot_{side}"]).resolve())
    exported = read_json(root / "cases/manifest.json")
    found = {}
    import cv2
    for row in exported:
        key = (row["sequence"], row["method"], int(row["start_frame"]), int(row["end_frame"]))
        if key in found or str(Path(row["source_mot"]).resolve()) != expected.get(key):
            raise ValueError("Case manifest differs from the fixed selected intervals/MOT")
        video = Path(row["video"])
        timing_check(video.with_suffix(".timing.json"), 40)
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != 40:
                raise ValueError(f"Case video is not 40 frames: {video}")
        finally:
            cap.release()
        found[key] = str(video)
    if set(found) != set(expected):
        raise ValueError("Missing selected case videos")
    return {"case_videos": len(found)}


def run_job(args, execute=None):
    root, spec_path = Path(args.run_root).resolve(), Path(args.spec).resolve()
    spec = read_json(spec_path)
    env = dict(os.environ, PAPER_EXPERIMENT_SPEC=str(spec_path))
    if getattr(args, "ffmpeg_executable", None):
        encoder = Path(args.ffmpeg_executable).resolve()
        env["PAPER_FFMPEG_EXECUTABLE"] = str(encoder)
    commands = []
    artifacts = []
    write_json(root / "queue/job_configs" / f"{args.job_id}.json", {
        "job_id": args.job_id, "stage": args.stage, "sequence": args.sequence,
        "spec": str(spec_path), "frozen_parameters": spec.get("frozen_parameters"),
        "variants": spec.get("variants"), "counts": spec.get("counts"),
        "render_ffmpeg_executable": env.get("PAPER_FFMPEG_EXECUTABLE"),
    })
    def run(command):
        full = [CONDA, "run", "--no-capture-output", "-n", "ship_detect", "python", "-B", *map(str, command)]
        commands.append(full)
        if execute is None:
            subprocess.run(full, cwd=REPO, env=env, check=True)
        else:
            execute(full)
    marker = marker_path(root, args.job_id)
    if marker.exists():
        marker.unlink()
    try:
        if env.get("PAPER_FFMPEG_EXECUTABLE") and not Path(env["PAPER_FFMPEG_EXECUTABLE"]).is_file():
            raise FileNotFoundError(f"Frozen render encoder is absent: {env['PAPER_FFMPEG_EXECUTABLE']}")
        if args.stage == "main":
            if (root / "main").exists():
                raise RuntimeError("Preserved partial main exists; perform targeted resume/repair before requeue, do not restart main blindly")
            run(["tools/evaluation/run_unified_comparison.py", "--dataset-root", root / "inputs",
                 "--run-root", root / "main", "--msdc-variant", "msdc_v3"])
            inventory = read_json(root / "main/inventory.json")
            if {r["sequence"] for r in inventory} != {r["sequence"] for r in spec["sequences"]}:
                raise ValueError("Main inventory must contain exactly nine specified sequences")
            for row in inventory:
                seq, length = row["sequence"], int(row["frames"])
                cache = root / "main/detections" / f"{seq}_high_low_detections.jsonl"
                frame_lines(cache, length, "frame_id")
                timing_check(cache.with_suffix(".timing.json"), length)
                for tracker in ("msdc_v3_replay", *BASELINES):
                    artifacts.append(validate_result(root, "main", tracker, seq, length, tracker == "msdc_v3_replay"))
            validate_protocol(root / "main/eval")
            (root / "sequence_frames.tsv").write_text("".join(
                f'{r["sequence"]}\t{r["frames"]}\n' for r in inventory), encoding="utf-8")
        elif args.stage in {"core", "sensitivity"}:
            lengths = {r["sequence"]: int(r["frames"]) for r in read_json(root / "main/inventory.json")}
            if args.sequence not in lengths:
                raise ValueError("Sequence absent from main inventory")
            length, batch = lengths[args.sequence], args.stage
            variants = spec["core_variants" if batch == "core" else "sensitivity_variants"]
            needed = []
            for variant in variants:
                tracker = variant + "_replay"
                mot = root / batch / "trackers" / tracker / "data" / f"{args.sequence}.txt"
                try:
                    validate_result(root, batch, tracker, args.sequence, length)
                except (ValueError, OSError, KeyError, json.JSONDecodeError):
                    needed.append(variant)
                    if mot.exists():
                        preserve_partial_result(root, batch, tracker, args.sequence, args.job_id)
            if needed:
                run(["tools/evaluation/detection_replay_benchmark.py", "--dataset-root", root / "inputs" / args.sequence,
                     "--output-root", root, "--run-id", batch, "--trackers", "msdc_elt", "--variants", *needed,
                     "--formal-frame-limit", str(length), "--file-type", "visible", "--baseline-input", "all",
                     "--source-detection-cache-root", root / "main/detections", "--render",
                     "--frame-source", "cache_only",
                     "--render-class-source", "cache", "--progress-interval", "500"])
            for variant in variants:
                artifacts.append(validate_result(root, batch, variant + "_replay", args.sequence, length))
            trackers = [variant + "_replay" for variant in variants]
            sequence_eval = root / batch / "metadata" / args.sequence / "eval"
            run(["tools/evaluation/motchallenge_eval.py", "--gt-root", root / batch / "mot_gt",
                 "--trackers-root", root / batch / "trackers", "--output-root", sequence_eval,
                 "--trackers", *trackers, "--sequences", args.sequence, "--quiet"])
            validate_protocol(sequence_eval)
            archive_metadata(root, batch, args.sequence, trackers, needed)
        else:
            batch = "core" if args.stage == "core_stats" else "sensitivity"
            phase = "core" if batch == "core" else "all"
            run(["tools/evaluation/motchallenge_eval.py", "--gt-root", root / batch / "mot_gt",
                 "--trackers-root", root / batch / "trackers", "--output-root", root / batch / "eval_all", "--quiet"])
            validate_protocol(root / batch / "eval_all")
            run(["tools/evaluation/export_paper_statistics.py", "--run-root", root, "--spec", spec_path, "--phase", phase])
            artifacts.append(validate_statistics(root, phase))
            if phase == "core":
                run(["tools/evaluation/export_paper_cases.py", "--selection", root / "paper_statistics/case_selection.csv",
                     "--output-dir", root / "paper_statistics/cases"])
            artifacts.append(validate_cases(root))
        write_json(marker, {"status": "completed", "job_id": args.job_id, "stage": args.stage,
                            "sequence": args.sequence, "spec": str(spec_path), "commands": commands,
                            "artifacts": artifacts, "completed": datetime.utcnow().isoformat() + "Z"})
        return marker
    except Exception as error:
        write_json(root / "queue/failures" / f"{args.job_id}.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json",
                   {"job_id": args.job_id, "stage": args.stage, "sequence": args.sequence,
                    "spec": str(spec_path), "commands": commands, "error": repr(error),
                    "job_config": str(root / "queue/job_configs" / f"{args.job_id}.json"),
                    "traceback": traceback.format_exc()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("build", "run"):
        item = sub.add_parser(mode)
        item.add_argument("--run-root", type=Path, required=True)
        item.add_argument("--spec", type=Path, required=True)
        if mode == "build":
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--repo", type=Path, default=REPO)
        else:
            item.add_argument("--stage", choices=("main", "core", "core_stats", "sensitivity", "all_stats"), required=True)
            item.add_argument("--sequence", default="")
            item.add_argument("--job-id", required=True)
            item.add_argument("--ffmpeg-executable", type=Path, help="Frozen FFmpeg binary for paper H.264 export")
    args = parser.parse_args(argv)
    if args.mode == "build":
        write_json(args.output, build_manifest(args.run_root, args.spec, args.repo))
    else:
        print(run_job(args), flush=True)


if __name__ == "__main__":
    main()
