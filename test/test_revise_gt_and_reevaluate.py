import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
MODULE = REPOSITORY / "tools/experiments/revise_gt_and_reevaluate.py"
spec = importlib.util.spec_from_file_location("gt_revision", MODULE)
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)
NAMES = {"official_botsort": "official_botsort_replay", "msdc_elt": "msdc_v3_replay",
         "bytetrack": "bytetrack_replay", "official_ocsort": "official_ocsort_replay"}


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(tmp_path):
    root, original = tmp_path / "run", tmp_path / "original_dataset"
    records, inventory, input_inventory = [], [], []
    for index in range(9):
        seq = f"seq{index}"
        raw = original / seq / "gt.txt"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"1,1,1.25,2.50,10,11,1,3,1\r\n2,1,1.25,2.50,10,11,1,3,1\r\n3,1,1.25,2.50,10,11,1,3,1\r\n")
        for path in (root / "inputs" / seq / "gt/gt.txt", root / "main/mot_gt" / seq / "gt/gt.txt"):
            path.parent.mkdir(parents=True)
            shutil.copyfile(raw, path)
            path.with_name("labels.txt").write_text("USV\nfishship\nUAV\n")
        (root / "main/mot_gt" / seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=25\nseqLength=3\nimWidth=32\nimHeight=32\nimExt=.jpg\n")
        records.append({"sequence": seq, "gt": str(raw)})
        active = root / "inputs" / seq / "gt/gt.txt"
        stats = rev.gt_statistics(active)
        inventory.append({"sequence": seq, "frames": 3, "container_frames": 3,
                          "video": "/original/video.mp4", "gt": str(active), **stats})
        input_inventory.append({"sequence": seq, "gt": str(raw), "gt_rows": 3, "gt_classes": {"3": 3}})
        for tracker in NAMES.values():
            mot = root / "main/trackers" / tracker / "data" / f"{seq}.txt"
            mot.parent.mkdir(parents=True, exist_ok=True)
            mot.write_text("".join(f"{frame},20,1.25,2.50,10,11,.9,3,-1,-1\n" for frame in range(1,4)))
        cache = root / "main/detections" / f"{seq}_high_low_detections.jsonl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("cache is not rerun")
    put(root / "experiment_spec.json", {"sequences": records})
    put(root / "input_inventory.json", input_inventory)
    put(root / "main/inventory.json", inventory)
    put(root / "main/run_metadata.json", {"tracker_names": NAMES})
    put(root / "main/validation.json", {"passed": True, "gt_identical": True, "old_gt_count": 27})
    put(root / "main/status.json", {"status": "complete"})
    put(root / "main/provenance.json", {"original": True})
    put(root / "main/phase_results.json", {"original_timing": 123})
    put(root / "main/config.json", {"unchanged": True})
    (root / "main/summary").mkdir()
    (root / "main/summary/speed.csv").write_text("original unchanged speed")
    (root / "main/summary/stages.csv").write_text("original unchanged stages")
    (root / "main/summary/performance.csv").write_text("old_gt_count=27")
    (root / "main/REPORT.md").write_text("old_gt_count=27")
    (root / "main/eval").mkdir()
    (root / "main/eval/old.txt").write_text("old evaluation")
    marker, exit_file = root / "queue/markers/main.completed.json", root / "queue/logs/main.exit"
    put(marker, {"status": "completed", "job_id": "main", "old_gt_count": 27})
    exit_file.parent.mkdir(parents=True)
    exit_file.write_text("0")
    put(root / "queue/queue_state.json", {"jobs": [
        {"id": "main", "phase": "main", "status": "running", "attempts": 1, "pid": 23,
         "exit_file": str(exit_file), "expected_output": str(marker)},
        {"id": "core_01", "phase": "core", "status": "pending", "attempts": 0}]})
    put(root / "queue/scheduler_pid.json", {"pid": 2681})
    put(root / "queue/control_hold.json", {"status": "held", "scheduler_pid": 2681, "reason": "GT review"})
    return root, original


def csv_file(path, records):
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def edit(seq="seq0", frame="2", **extra):
    return {"sequence": seq, "frame": frame, "gt_id": "1",
            "old_x": "1.25", "old_y": "2.50", "old_w": "10", "old_h": "11",
            "action": "drop", "reason": "Confirmed in a reviewed frame", "evidence_path": "/review/frame.png", **extra}


def add(seq="seq0", frame="2", gt_id="2", **extra):
    return {"sequence": seq, "frame": frame, "gt_id": gt_id, "action": "add",
            "new_x": "20.125", "new_y": "20.5", "new_w": "5", "new_h": "6",
            "reason": "Visible target has no GT box in this reviewed frame",
            "evidence_path": "/review/missing_target.png", **extra}


def candidate(tmp_path, root, records=None):
    changes, target = tmp_path / "changes.csv", tmp_path / "candidate"
    csv_file(changes, records or [edit()])
    rev.prepare(root, changes, target)
    return target


class FakeEvaluation:
    def __init__(self, root, fail=False):
        self.root, self.fail, self.calls = root, fail, []

    def run(self, revision, archive):
        self.calls.append("evaluation_only")
        root = self.root
        if self.fail:
            (root / "main/eval").mkdir()
            (root / "main/eval/new_partial.txt").write_text("failed fresh evaluation")
            raise RuntimeError("Synthetic evaluator failure")
        counts = {f"seq{i}": rev.gt_statistics(root / f"main/mot_gt/seq{i}/gt/gt.txt")["valid_gt_rows"] for i in range(9)}
        total = sum(counts.values())
        for tracker in NAMES.values():
            output = root / "main/eval" / tracker / "pedestrian_detailed.csv"
            output.parent.mkdir(parents=True)
            csv_file(output, [{"seq": seq, "CLR_TP": count, "CLR_FN": 0, "IDTP": count, "IDFN": 0}
                              for seq, count in {**counts, "COMBINED": total}.items()])
        (root / "main/summary/performance.csv").write_text(f"new_gt_count={total}")
        (root / "main/REPORT.md").write_text(f"new_gt_count={total}")
        return {"metrics": {name: {"TP": total, "FN": 0, "IDTP": total, "IDFN": 0} for name in NAMES.values()},
                "count_checks": rev.count_consistency(root, list(NAMES.values()), counts),
                "artifacts": [{"tracker": name, "sequence": f"seq{i}"} for name in NAMES.values() for i in range(9)],
                "frames_per_method": 27}


def test_prepare_exact_drop_and_replace_preserves_other_fields_and_originals(tmp_path):
    root, original = make_run(tmp_path)
    before = (root / "inputs/seq0/gt/gt.txt").read_bytes()
    target = candidate(tmp_path, root, [edit(), edit(seq="seq1", frame="3", action="replace",
                       new_x="4.125", new_y="5.5", new_w="12", new_h="13")])
    assert (root / "inputs/seq0/gt/gt.txt").read_bytes() == before
    assert (original / "seq0/gt.txt").read_bytes() == before
    assert (target / "original/seq0/gt.txt").read_bytes() == before
    assert (target / "revised/seq0/gt.txt").read_bytes() == before.splitlines(keepends=True)[0] + before.splitlines(keepends=True)[2]
    rows = rev.gt_rows(target / "revised/seq1/gt.txt")
    assert rows[-1][2:6] == ["4.125", "5.5", "12", "13"]
    assert rows[-1][:2] == ["3", "1"] and rows[-1][6:] == ["1", "3", "1"]
    log = rev.read_json(target / "change_log.json")
    assert log[0]["reason"] and log[0]["evidence_path"] == "/review/frame.png"


def test_prepare_rejects_inexact_preimage_and_missing_reason(tmp_path):
    root, _ = make_run(tmp_path)
    changes = tmp_path / "bad.csv"
    csv_file(changes, [edit(old_x="1.25000000000000001")])
    with pytest.raises(ValueError, match="Original box differs"):
        rev.prepare(root, changes, tmp_path / "bad_candidate")
    assert not (tmp_path / "bad_candidate").exists()
    csv_file(changes, [edit(reason="")])
    with pytest.raises(ValueError, match="reason"):
        rev.prepare(root, changes, tmp_path / "bad_candidate")


def test_add_missing_row_inherits_class_and_inserts_sorted_without_rewriting_other_rows(tmp_path):
    root, original = make_run(tmp_path)
    source = root / "inputs/seq0/gt/gt.txt"
    raw = source.read_bytes().splitlines(keepends=True)
    source.write_bytes(raw[0] + raw[2])
    before = source.read_bytes()
    target = candidate(tmp_path, root, [add(gt_id="4", class_id="1"), add(gt_id="1"),
                                         add(gt_id="2", class_id="2")])
    revised = target / "revised/seq0/gt.txt"
    rows = rev.gt_rows(revised)
    assert [(int(row[0]), int(row[1])) for row in rows] == [(1, 1), (2, 1), (2, 2), (2, 4), (3, 1)]
    assert rev.gt_statistics(revised)["valid_gt_rows"] == 5
    assert [row[7] for row in rows[1:4]] == ["3", "2", "1"]
    assert all(len(row) == 9 and row[6] == row[8] == "1" for row in rows[1:4])
    lines = revised.read_bytes().splitlines(keepends=True)
    assert lines[0] == raw[0] and lines[-1] == raw[2]
    assert all(line.endswith(b"\r\n") for line in lines)
    assert source.read_bytes() == before
    assert len(rev.gt_rows(original / "seq0/gt.txt")) == 3
    audit = rev.read_json(target / "change_log.json")
    assert all(row["before"] is None and row["old_box"] is None and "source_line" not in row for row in audit)
    assert all(row["reason"] and row["evidence_path"] == "/review/missing_target.png" for row in audit)
    assert audit[1]["class_source"] == "existing_gt" and audit[1]["class_id"] == 3
    assert audit[0]["class_source"] == "explicit_csv"
    manifest = rev.read_json(target / "manifest.json")
    assert manifest["existing_row_ids_and_classes_unchanged"]
    assert "id_and_class_unchanged" not in manifest
    assert manifest["id_changes_by_sequence"]["seq0"] == {"added_gt_ids": [2, 4], "removed_gt_ids": []}
    assert all(value == {"added_gt_ids": [], "removed_gt_ids": []}
               for sequence, value in manifest["id_changes_by_sequence"].items() if sequence != "seq0")


@pytest.mark.parametrize("records,message", [
    ([add(gt_id="1", class_id="3")], "frame-ID already exists"),
    ([add()], "requires explicit class_id"),
    ([add(frame="4", gt_id="1", class_id="2")], "cannot change existing ID class"),
    ([add(gt_id="2", class_id="1"), add(frame="3", gt_id="2", class_id="2")], "inconsistent class_id"),
    ([add(class_id="1", old_x="20")], "original box empty"),
    ([add(class_id="1", evidence_path="")], "evidence_path"),
])
def test_add_rejects_occupied_frame_id_missing_class_and_class_changes(tmp_path, records, message):
    root, _ = make_run(tmp_path)
    before = (root / "inputs/seq0/gt/gt.txt").read_bytes()
    changes = tmp_path / "changes.csv"
    csv_file(changes, records)
    with pytest.raises(ValueError, match=message):
        rev.prepare(root, changes, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()
    assert (root / "inputs/seq0/gt/gt.txt").read_bytes() == before


def test_promote_rejects_retracted_candidate_without_changing_run(tmp_path):
    root, _ = make_run(tmp_path)
    target = candidate(tmp_path, root)
    manifest = rev.read_json(target / "manifest.json")
    manifest["status"] = "invalidated_identity_conflict"
    put(target / "manifest.json", manifest)
    before = (root / "inputs/seq0/gt/gt.txt").read_bytes()
    with pytest.raises(ValueError, match="not active"):
        rev.promote(root, target, backend=FakeEvaluation(root), proc_state=lambda _: "T")
    assert (root / "inputs/seq0/gt/gt.txt").read_bytes() == before
    assert not (root / "gt_revision_archives").exists()


def test_promote_updates_only_run_gt_copies_archives_old_counts_and_preserves_work(tmp_path):
    root, original = make_run(tmp_path)
    target = candidate(tmp_path, root)
    protected, original_bytes = rev.protected_snapshot(root), (original / "seq0/gt.txt").read_bytes()
    state_bytes = (root / "queue/queue_state.json").read_bytes()
    backend = FakeEvaluation(root)
    result = rev.promote(root, target, backend=backend, proc_state=lambda _: "T")
    assert result["status"] == "promoted" and backend.calls == ["evaluation_only"]
    assert len(rev.gt_rows(root / "inputs/seq0/gt/gt.txt")) == 2
    assert (root / "inputs/seq0/gt/gt.txt").read_bytes() == (root / "main/mot_gt/seq0/gt/gt.txt").read_bytes()
    assert (original / "seq0/gt.txt").read_bytes() == original_bytes
    assert (root / "queue/queue_state.json").read_bytes() == state_bytes
    rev.assert_preserved(protected)
    archive = Path(result["archive"])
    assert (archive / "main/REPORT.md").read_text() == "old_gt_count=27"
    assert rev.read_json(archive / "main/validation.json")["old_gt_count"] == 27
    assert (archive / "main/eval/old.txt").read_text() == "old evaluation"
    assert (root / "main/REPORT.md").read_text() == "new_gt_count=26"
    inventory = rev.read_json(root / "main/inventory.json")
    assert inventory[0]["gt_rows"] == 2 and inventory[0]["gt_sha256"] == rev.sha(root / "inputs/seq0/gt/gt.txt")
    validation = rev.read_json(root / "main/validation.json")
    assert validation["passed"] and len(validation["count_checks"]) == 40
    assert all(row["TP_plus_FN"] == row["valid_gt"] == row["IDTP_plus_IDFN"] for row in validation["count_checks"])
    marker = rev.read_json(root / "queue/markers/main.completed.json")
    assert marker["re_evaluation_only"] and len(marker["artifacts"]) == 36
    assert rev.read_json(root / "queue/control_hold.json")["status"] == "held"


@pytest.mark.parametrize("state,exit_code", [("S", "0"), ("T", "1")])
def test_promote_requires_actual_stopped_scheduler_and_successful_main(tmp_path, state, exit_code):
    root, _ = make_run(tmp_path)
    target = candidate(tmp_path, root)
    (root / "queue/logs/main.exit").write_text(exit_code)
    before = (root / "inputs/seq0/gt/gt.txt").read_bytes()
    with pytest.raises(ValueError):
        rev.promote(root, target, backend=FakeEvaluation(root), proc_state=lambda _: state)
    assert (root / "inputs/seq0/gt/gt.txt").read_bytes() == before
    assert not (root / "gt_revision_archives").exists()


def test_failed_fresh_eval_restores_old_gt_reports_and_marker_but_keeps_failed_attempt(tmp_path):
    root, _ = make_run(tmp_path)
    target = candidate(tmp_path, root)
    gt_before = (root / "inputs/seq0/gt/gt.txt").read_bytes()
    with pytest.raises(RuntimeError, match="Synthetic"):
        rev.promote(root, target, backend=FakeEvaluation(root, fail=True), proc_state=lambda _: "T")
    assert (root / "inputs/seq0/gt/gt.txt").read_bytes() == gt_before
    assert (root / "main/REPORT.md").read_text() == "old_gt_count=27"
    assert rev.read_json(root / "queue/markers/main.completed.json")["old_gt_count"] == 27
    archive = next((root / "gt_revision_archives").iterdir())
    assert (archive / "failed_attempt/main/eval/new_partial.txt").is_file()
    assert rev.read_json(archive / "promotion_receipt.json")["status"] == "rolled_back"
    assert rev.read_json(root / "queue/control_hold.json")["status"] == "held"


@pytest.mark.parametrize("records,first_count,total", [([edit()], 2, 26), ([add(class_id="1")], 4, 28)])
def test_actual_official_four_by_nine_counts_use_revised_gt_without_tracking(tmp_path, records, first_count, total):
    root, _ = make_run(tmp_path)
    target = candidate(tmp_path, root, records)
    shutil.copyfile(target / "revised/seq0/gt.txt", root / "main/mot_gt/seq0/gt/gt.txt")
    sys.path.insert(0, str(REPOSITORY))
    from tools.evaluation.motchallenge_eval import run_motchallenge_eval
    result = run_motchallenge_eval(root / "main/mot_gt", root / "main/trackers", root / "main/eval",
                                  trackers=list(NAMES.values()), sequences=[f"seq{i}" for i in range(9)], print_results=False)
    assert all(row["TP"] + row["FN"] == total and row["IDTP"] + row["IDFN"] == total for row in result.values())
    checks = rev.count_consistency(root, list(NAMES.values()), {f"seq{i}": first_count if i == 0 else 3 for i in range(9)})
    assert len(checks) == 40


def test_full_default_re_evaluation_refreshes_reports_without_detector_tracker_or_video_decode(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    root, original = make_run(tmp_path)
    repository = REPOSITORY
    (root / "source_code").symlink_to(repository, target_is_directory=True)
    sys.path.insert(0, str(repository))
    from target_module.image_detect_module import target_detection
    from target_module.image_detect_module.utils.tracker import MultiObjectTracker
    from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker
    from tools.evaluation import render_mot_video
    def forbidden(*args, **kwargs):
        pytest.fail("GT-only re-evaluation must not detect, track, render or decode video")
    monkeypatch.setattr(target_detection, "get_detector", forbidden)
    monkeypatch.setattr(MultiObjectTracker, "__init__", forbidden)
    monkeypatch.setattr(MSDCLifecycleTracker, "__init__", forbidden)
    monkeypatch.setattr(render_mot_video, "render_mot_video", forbidden)
    stages = dict.fromkeys(("read_decode", "low_detection", "high_split", "tracker", "render", "write_export"), .01)
    timing = {"processed_frames": 3, "total_time_s": 1, "stage_seconds": stages, "other_seconds": .94}
    phases = {name: [] for name in NAMES.values()}
    video_template = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video_template), cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 32))
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()
    box = {"x": 1.25, "y": 2.50, "w": 10, "h": 11, "confidence": .9, "class": "UAV"}
    for i in range(9):
        seq = f"seq{i}"
        cache = root / "main/detections" / f"{seq}_high_low_detections.jsonl"
        cache.write_text("".join(json.dumps({"frame_id": frame, "low_boxes": [box], "high_boxes": [box]})+"\n" for frame in range(1,4)))
        put(cache.with_suffix(".timing.json"), timing)
        for name in NAMES.values():
            mot = root / "main/trackers" / name / "data" / f"{seq}.txt"
            put(mot.with_suffix(".timing.json"), timing)
            mot.with_suffix(".frames.jsonl").write_text("".join(json.dumps({"frame_id": frame})+"\n" for frame in range(1,4)))
            video = root / "main/visualizations" / seq / f"{name}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video_template, video)
            put(video.with_suffix(".timing.json"), timing)
            phases[name].append([3, timing, timing, timing])
        for kind in ("observations", "associations", "states", "events", "outputs"):
            stream = root / "main/trackers/msdc_v3_replay/paper_logs" / seq / f"{kind}.jsonl"
            stream.parent.mkdir(parents=True, exist_ok=True)
            stream.write_text("".join(json.dumps({"seq": seq, "method": "msdc_v3_replay", "frame": frame, "records": []})+"\n" for frame in range(1,4)))
    put(root / "main/phase_results.json", phases)
    original_capture = cv2.VideoCapture
    class MetadataOnlyCapture:
        def __init__(self, *args, **kwargs):
            self.inner = original_capture(*args, **kwargs)
        def isOpened(self):
            return self.inner.isOpened()
        def get(self, *args):
            return self.inner.get(*args)
        def release(self):
            self.inner.release()
        def read(self):
            forbidden()
    monkeypatch.setattr(cv2, "VideoCapture", MetadataOnlyCapture)
    target = candidate(tmp_path, root)
    before = rev.protected_snapshot(root)
    outcome = rev.promote(root, target, proc_state=lambda _: "T")
    assert outcome["status"] == "promoted"
    rev.assert_preserved(before)
    with (root / "main/summary/per_dataset_metrics.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    assert all(int(row["TP"]) + int(row["FN"]) == (2 if row["Dataset"] == "seq0" else 3) for row in rows)
    assert rev.read_json(root / "main/gt_detection_overlap.json")[0]["gt_boxes"] == 2
    assert "old_gt_count=27" not in (root / "main/REPORT.md").read_text()
    assert len(rev.gt_rows(original / "seq0/gt.txt")) == 3
