"""Prepare exact GT-copy edits, then promote and re-evaluate a held completed main."""
from __future__ import annotations
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import traceback


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def integer(value):
    value = Decimal(str(value).strip())
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f"Expected an exact integer: {value}")
    return int(value)


def gt_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream) if row and any(v.strip() for v in row)]
    seen = set()
    for row in rows:
        if len(row) < 8:
            raise ValueError(f"Missing GT fields: {path}")
        frame, identity, cls = integer(row[0]), integer(row[1]), integer(row[7])
        if frame < 1 or not 1 <= cls <= 3 or Decimal(row[4]) <= 0 or Decimal(row[5]) <= 0:
            raise ValueError(f"Invalid GT row: {path}: {row}")
        if (frame, identity) in seen:
            raise ValueError(f"Duplicate frame-ID: {path}: {frame}/{identity}")
        seen.add((frame, identity))
    return rows


def gt_statistics(path):
    rows = gt_rows(path)
    valid = [r for r in rows if Decimal(r[6]) != 0]
    return {"gt_rows": len(rows), "valid_gt_rows": len(valid),
            "gt_ids": sorted({r[1] for r in rows}),
            "gt_classes": sorted({r[7] for r in rows}),
            "gt_class_counts": dict(Counter(r[7] for r in rows)), "gt_sha256": sha(path)}


def field(row, canonical, alias=None):
    value = row.get(canonical, row.get(alias, "") if alias else "")
    if not str(value).strip():
        raise ValueError(f"Missing correction field: {canonical}")
    return str(value).strip()


def prepare(run_root, changes_file, candidate_dir):
    root, candidate = Path(run_root).resolve(), Path(candidate_dir).resolve()
    if candidate == root or candidate.is_relative_to(root):
        raise ValueError("prepare writes only a candidate directory outside the formal RUN")
    if candidate.exists():
        raise ValueError("Use a new candidate directory")
    spec = read_json(root / "experiment_spec.json")
    sequences = [row["sequence"] for row in spec["sequences"]]
    corrections = defaultdict(list)
    with Path(changes_file).open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), 2):
            seq, action = field(row, "sequence"), field(row, "action")
            if seq not in sequences or action not in {"drop", "replace", "add"}:
                raise ValueError(f"Unknown sequence/action at correction CSV row {index}")
            if row.get("new_id", "").strip() or row.get("new_class", "").strip():
                raise ValueError("Existing ID/class cannot be changed; additions use gt_id and class_id")
            record = {"sequence": seq, "frame": integer(field(row, "frame")),
                      "gt_id": integer(field(row, "gt_id", "id")), "action": action,
                      "reason": field(row, "reason"), "evidence_path": row.get("evidence_path", "").strip(),
                      "csv_row": index}
            if action == "add":
                record["evidence_path"] = field(row, "evidence_path")
                if record["frame"] < 1:
                    raise ValueError(f"Invalid addition frame at correction row {index}")
                if any(str(row.get("old_" + key, row.get(key, ""))).strip() for key in ("x", "y", "w", "h")):
                    raise ValueError(f"Addition must leave original box empty at correction row {index}")
                record["old_box"] = None
                record["class_id"] = integer(row["class_id"]) if row.get("class_id", "").strip() else None
                if record["class_id"] is not None and record["class_id"] not in {1, 2, 3}:
                    raise ValueError(f"Invalid addition class_id at correction row {index}")
            else:
                record["old_box"] = [field(row, "old_" + key, key) for key in ("x", "y", "w", "h")]
            if action in {"replace", "add"}:
                record["new_box"] = [field(row, "new_" + key) for key in ("x", "y", "w", "h")]
                if any(not Decimal(v).is_finite() for v in record["new_box"]) or any(Decimal(v) <= 0 for v in record["new_box"][2:]):
                    raise ValueError(f"Invalid new box at correction row {index}")
            corrections[seq].append(record)
    # Validate every preimage before creating candidate output.
    data, audit = {}, []
    for seq in sequences:
        source = root / "inputs" / seq / "gt/gt.txt"
        content = source.read_bytes()
        text = content.decode("utf-8-sig")
        lines = text.splitlines(keepends=True)
        parsed, classes = {}, defaultdict(set)
        for line_number, line in enumerate(lines):
            row = next(csv.reader([line]))
            if row and any(value.strip() for value in row):
                key = (integer(row[0]), integer(row[1]))
                if key in parsed:
                    raise ValueError(f"Duplicate GT preimage: {seq}: {key}")
                parsed[key] = (line_number, row)
                classes[key[1]].add(integer(row[7]))
        touched, additions, added_classes = set(), [], {}
        for edit in corrections[seq]:
            key = (edit["frame"], edit["gt_id"])
            if key in touched:
                raise ValueError(f"Repeated or missing exact correction row: {seq}: {key}")
            touched.add(key)
            if edit["action"] == "add":
                if key in parsed:
                    raise ValueError(f"Addition frame-ID already exists: {seq}: {key}")
                inherited = classes.get(key[1], set())
                if inherited:
                    if len(inherited) != 1:
                        raise ValueError(f"Existing ID has no unique class to inherit: {seq}: {key[1]}")
                    cls = next(iter(inherited))
                    if cls not in {1, 2, 3} or edit["class_id"] not in {None, cls}:
                        raise ValueError(f"Addition cannot change existing ID class: {seq}: {key[1]}")
                    edit["class_source"] = "existing_gt"
                else:
                    cls = edit["class_id"]
                    if cls is None:
                        raise ValueError(f"New ID addition requires explicit class_id: {seq}: {key[1]}")
                    if key[1] in added_classes and added_classes[key[1]] != cls:
                        raise ValueError(f"New ID additions have inconsistent class_id: {seq}: {key[1]}")
                    added_classes[key[1]] = cls
                    edit["class_source"] = "explicit_csv"
                edit["class_id"] = cls
                after = [str(key[0]), str(key[1]), *edit["new_box"], "1", str(cls), "1"]
                additions.append((key, after))
                audit.append({**edit, "before": None, "after": after})
                continue
            if key not in parsed:
                raise ValueError(f"Repeated or missing exact correction row: {seq}: {key}")
            offset, row = parsed[key]
            if any(Decimal(a) != Decimal(b) for a, b in zip(row[2:6], edit["old_box"])):
                raise ValueError(f"Original box differs at {seq}: {key}")
            before, after = list(row), None
            if edit["action"] == "drop":
                lines[offset] = ""
            else:
                after = list(row)
                after[2:6] = edit["new_box"]
                ending = "\r\n" if lines[offset].endswith("\r\n") else "\n" if lines[offset].endswith("\n") else ""
                buffer = io.StringIO(newline="")
                csv.writer(buffer, lineterminator=ending).writerow(after)
                lines[offset] = buffer.getvalue()
            audit.append({**edit, "before": before, "after": after})
        if additions:
            keys = list(parsed)
            if keys != sorted(keys):
                raise ValueError(f"Stable additions require original frame-ID order: {seq}")
            ending = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
            insertions = defaultdict(list)
            for key, row in sorted(additions):
                index = bisect_left(keys, key)
                offset = parsed[keys[index]][0] if index < len(keys) else len(lines)
                buffer = io.StringIO(newline="")
                csv.writer(buffer, lineterminator=ending).writerow(row)
                insertions[offset].append(buffer.getvalue())
            output = []
            for offset in range(len(lines) + 1):
                if insertions[offset]:
                    last = next((line for line in reversed(output) if line), "")
                    if last and not last.endswith(("\n", "\r")):
                        output.append(ending)
                    output.extend(insertions[offset])
                if offset < len(lines):
                    output.append(lines[offset])
            lines = output
        revised = "".join(lines).encode("utf-8")
        if content.startswith(b"\xef\xbb\xbf"):
            revised = b"\xef\xbb\xbf" + revised
        data[seq] = (source, content, revised)
    candidate.mkdir(parents=True)
    records = []
    for seq, (source, original, revised) in data.items():
        first, second = candidate / "original" / seq / "gt.txt", candidate / "revised" / seq / "gt.txt"
        first.parent.mkdir(parents=True)
        second.parent.mkdir(parents=True)
        first.write_bytes(original)
        second.write_bytes(revised)
        records.append({"sequence": seq, "input_gt": str(source),
                        "original": str(first), "revised": str(second),
                        "original_statistics": gt_statistics(first), "revised_statistics": gt_statistics(second)})
    shutil.copyfile(changes_file, candidate / "changes.csv")
    write_json(candidate / "change_log.json", audit)
    id_changes = {}
    for record in records:
        original_ids = {integer(value) for value in record["original_statistics"]["gt_ids"]}
        revised_ids = {integer(value) for value in record["revised_statistics"]["gt_ids"]}
        id_changes[record["sequence"]] = {"added_gt_ids": sorted(revised_ids - original_ids),
                                         "removed_gt_ids": sorted(original_ids - revised_ids)}
    manifest = {"revision_id": candidate.name, "run_root": str(root), "prepared_utc": utc(),
                "changes": len(audit), "existing_row_ids_and_classes_unchanged": True,
                "id_changes_by_sequence": id_changes, "records": records,
                "changes_csv": str(candidate / "changes.csv"), "status": "candidate_not_promoted"}
    write_json(candidate / "manifest.json", manifest)
    return manifest


def process_state(pid):
    return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(") ", 1)[1].split()[0]


def promotion_gate(root, proc_state=process_state):
    hold = read_json(root / "queue/control_hold.json")
    scheduler = read_json(root / "queue/scheduler_pid.json")
    state = read_json(root / "queue/queue_state.json")
    if hold.get("status") != "held" or hold.get("scheduler_pid") != scheduler["pid"]:
        raise ValueError("GT promotion requires the explicit current scheduler control_hold=held")
    if proc_state(scheduler["pid"]) not in {"T", "t"}:
        raise ValueError("Scheduler must actually be stopped (T/t), not merely described as held")
    main = next(job for job in state["jobs"] if job["phase"] == "main")
    if integer(Path(main["exit_file"]).read_text()) != 0:
        raise ValueError("Original main process must exit 0 before GT promotion")
    marker = Path(main["expected_output"])
    proof = read_json(marker)
    if proof.get("status") != "completed" or proof.get("job_id") != main["id"]:
        raise ValueError("Original main validated success marker is required")
    if not read_json(root / "main/validation.json").get("passed"):
        raise ValueError("Original main validation did not pass")
    if any(job["status"] != "pending" for job in state["jobs"] if job["phase"] != "main"):
        raise ValueError("Dependent jobs must remain pending during a GT-only revision")
    return main, hold


def protected_snapshot(root):
    main = root / "main"
    paths = []
    for name in ("trackers", "detections", "visualizations"):
        paths.extend(path for path in (main / name).rglob("*") if path.is_file())
    paths += [main / "phase_results.json", main / "config.json",
              main / "summary/speed.csv", main / "summary/stages.csv"]
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths if path.is_file()}


def assert_preserved(snapshot):
    for name, expected in snapshot.items():
        path = Path(name)
        if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != expected:
            raise ValueError(f"Protected tracking/cache/video/timing artifact changed: {path}")


def count_consistency(root, tracker_names, counts):
    records = []
    for tracker in tracker_names:
        with (root / "main/eval" / tracker / "pedestrian_detailed.csv").open(encoding="utf-8-sig", newline="") as stream:
            detailed = {row["seq"]: row for row in csv.DictReader(stream)}
        for sequence, expected in {**counts, "COMBINED": sum(counts.values())}.items():
            row = detailed[sequence]
            clear = integer(row["CLR_TP"]) + integer(row["CLR_FN"])
            identity = integer(row["IDTP"]) + integer(row["IDFN"])
            if clear != expected or identity != expected:
                raise ValueError(f"Fresh GT count mismatch: {tracker}/{sequence}: {clear}/{identity} != {expected}")
            records.append({"tracker": tracker, "sequence": sequence, "valid_gt": expected,
                            "TP_plus_FN": clear, "IDTP_plus_IDFN": identity})
    return records


class EvaluationOnly:
    """Call existing GT-dependent evaluators; no detector/tracker/renderer calls."""
    def __init__(self, root):
        self.root = root
        sys.path.insert(0, str(root / "source_code"))
        from tools.evaluation.motchallenge_eval import run_motchallenge_eval
        from tools.evaluation import unified_dataset_report as dataset_report
        from tools.evaluation.normalize_comparison_csv import normalize
        from tools.evaluation.msdc_dataset_benchmark import write_diagnostics_for_tracker, _iou
        from tools.experiments import paper_queue
        self.evaluate_mot = run_motchallenge_eval
        self.dataset_report, self.normalize = dataset_report, normalize
        self.diagnostics, self.iou, self.queue = write_diagnostics_for_tracker, _iou, paper_queue

    def run(self, revision, archive):
        root, main = self.root, self.root / "main"
        metadata = read_json(main / "run_metadata.json")
        mapping = metadata["tracker_names"]
        kinds = self.dataset_report.METHODS
        if set(mapping) != set(kinds) or len(mapping) != 4:
            raise ValueError("Expected actual four-method tracker_names metadata")
        inventory = read_json(main / "inventory.json")
        sequences = [row["sequence"] for row in inventory]
        names = [mapping[kind] for kind in kinds]
        metrics = self.evaluate_mot(main / "mot_gt", main / "trackers", main / "eval",
                                    trackers=names, sequences=sequences, print_results=False)
        counts = {seq: gt_statistics(main / "mot_gt" / seq / "gt/gt.txt")["valid_gt_rows"] for seq in sequences}
        count_checks = count_consistency(root, names, counts)
        audits = []
        for item in inventory:
            seq = item["sequence"]
            gt = main / "mot_gt" / seq / "gt/gt.txt"
            for tracker in names:
                self.diagnostics(gt, main / "trackers" / tracker / "data" / f"{seq}.txt", main / "diagnostics" / seq, tracker)
            by_frame = defaultdict(list)
            for row in gt_rows(gt):
                if Decimal(row[6]) != 0:
                    by_frame[integer(row[0])].append(tuple(map(float, row[2:6])))
            exact, best_values = 0, []
            with (main / "detections" / f"{seq}_high_low_detections.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    detection = json.loads(line)
                    boxes = [tuple(float(box[k]) for k in ("x", "y", "w", "h")) for box in detection["low_boxes"]]
                    for box in by_frame.get(detection["frame_id"], []):
                        best_values.append(max((self.iou(box, predicted) for predicted in boxes), default=0))
                        exact += any(all(abs(a-b) <= .02 for a, b in zip(box, predicted)) for predicted in boxes)
            total = counts[seq]
            if len(best_values) != total:
                raise ValueError(f"Detection cache does not cover revised GT frames: {seq}")
            audits.append({"sequence": seq, "gt_boxes": total, "matching_coordinates_within_0.02px": exact,
                           "matched_coordinate_percent": 100 * exact / total,
                           "iou_at_least_0.95_percent": 100 * sum(v >= .95 for v in best_values) / total,
                           "mean_best_iou_percent": 100 * sum(best_values) / total,
                           "gt_revision": revision["revision_id"]})
        write_json(main / "gt_detection_overlap.json", audits)
        performance = self.dataset_report.performance_table({kinds[k]: metrics[mapping[k]] for k in kinds})
        self.dataset_report.csvout(main / "summary/performance.csv", performance)
        report = [
            "# 固定轨迹的GT修订后统一重评", "",
            f'GT版本：{revision["revision_id"]}；精确修正记录：{revision["candidate_dir"]}/change_log.json。',
            f'全部{len(sequences)}条序列、4种方法使用同一修订GT，共{sum(counts.values())}个有效GT框。',
            "仅重新计算评价与GT依赖报告；检测、跟踪、MOT、可视化和原始计时均复用，原始数据集标注未修改。",
            f"旧GT、旧评价和旧报告归档：{archive}。", "",
            "## 标准跨序列汇总", "", self.dataset_report.table(performance), "",
            "原生TrackEval比例为0—1，summary中的展示比例转换为百分数，计数保持原值。",
            "逐序列GT／检测坐标重合审计已按本次GT重新计算，见gt_detection_overlap.json。", ""]
        (main / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
        self.normalize(main)
        self.dataset_report.build_dataset_report(main)
        report_path = main / "REPORT.md"
        text = report_path.read_text(encoding="utf-8")
        inherited_note = "定位与标注来源限制：检测从原视频独立推理，未读取 GT；逐序列 GT/检测坐标重合审计见后文。GT 的制作及人工修订流程尚未核实。若高度重合，LocA/MOTP 的解释存在限制，不能作为关联能力的主要依据；结果限定于所提供 GT。"
        revision_note = (f'本次GT修订依据与逐行变更见 {revision["candidate_dir"]}/change_log.json；'
                         "GT／检测坐标重合审计见 gt_detection_overlap.json。")
        report_path.write_text(text.replace(inherited_note, revision_note), encoding="utf-8")
        self.queue.validate_protocol(main / "eval")
        artifacts = []
        for row in inventory:
            seq, length = row["sequence"], integer(row["frames"])
            cache = main / "detections" / f"{seq}_high_low_detections.jsonl"
            self.queue.frame_lines(cache, length, "frame_id")
            self.queue.timing_check(cache.with_suffix(".timing.json"), length)
            for kind in kinds:
                artifacts.append(self.queue.validate_result(root, "main", mapping[kind], seq, length, kind == "msdc_elt"))
        if len(artifacts) != 36:
            raise ValueError("Main batch revalidation must contain 36 existing results")
        return {"metrics": metrics, "count_checks": count_checks, "artifacts": artifacts,
                "frames_per_method": sum(integer(row["frames"]) for row in inventory)}


def promote(run_root, candidate_dir, backend=None, proc_state=process_state):
    root, candidate = Path(run_root).resolve(), Path(candidate_dir).resolve()
    manifest = read_json(candidate / "manifest.json")
    if manifest.get("status") != "candidate_not_promoted":
        raise ValueError("GT candidate is not active; retracted candidates cannot be promoted")
    if Path(manifest["run_root"]).resolve() != root:
        raise ValueError("Candidate belongs to another RUN")
    main_job, hold = promotion_gate(root, proc_state)
    spec = read_json(root / "experiment_spec.json")
    expected = {row["sequence"] for row in spec["sequences"]}
    if len(expected) != 9 or {r["sequence"] for r in manifest["records"]} != expected:
        raise ValueError("GT revision must retain exactly the nine fixed sequences")
    inventory = read_json(root / "main/inventory.json")
    lengths = {row["sequence"]: integer(row["frames"]) for row in inventory}
    for record in manifest["records"]:
        seq = record["sequence"]
        live, revised = root / "inputs" / seq / "gt/gt.txt", Path(record["revised"])
        if sha(live) != record["original_statistics"]["gt_sha256"] or sha(revised) != record["revised_statistics"]["gt_sha256"]:
            raise ValueError(f"Candidate preimage/revised GT changed: {seq}")
        if record["revised_statistics"]["valid_gt_rows"] == 0:
            raise ValueError(f"Existing per-dataset report requires at least one valid GT for {seq}; review candidate before promotion")
        if any(integer(row[0]) > lengths[seq] for row in gt_rows(revised)):
            raise ValueError(f"Revised GT frame exceeds original EOF: {seq}")
        for target in (live, root / "main/mot_gt" / seq / "gt/gt.txt"):
            if target.is_symlink() or not target.resolve().is_relative_to(root):
                raise ValueError("GT targets must be regular copies inside RUN, never source-data links")
            if sha(target) != record["original_statistics"]["gt_sha256"]:
                raise ValueError(f"Original RUN input/evaluation GT copies differ from candidate preimage: {seq}")
    marker = Path(main_job["expected_output"])
    marker_relative = marker.relative_to(root)
    mutable = ["inputs", "input_inventory.json", "main/mot_gt", "main/eval", "main/summary", "main/diagnostics",
               "main/REPORT.md", "main/inventory.json", "main/gt_detection_overlap.json", "main/validation.json",
               "main/status.json", "main/provenance.json", "main/run_metadata.json", "input_revision.json", str(marker_relative)]
    archive = root / "gt_revision_archives" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive.mkdir(parents=True)
    for name in ("changes.csv", "change_log.json", "manifest.json"):
        shutil.copyfile(candidate / name, archive / ("approved_" + name))
    existed = {}
    for relative in mutable:
        source, target = root / relative, archive / relative
        existed[relative] = source.exists()
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    protected = protected_snapshot(root)
    revision = {"revision_id": manifest["revision_id"], "candidate_dir": str(candidate), "archive": str(archive),
                "promoted_utc": utc(), "changes": manifest["changes"], "source_dataset_gt_modified": False,
                "detection_tracking_render_reexecuted": False, "scheduler_hold_preserved": True,
                "approved_changes_csv": str(archive / "approved_changes.csv"),
                "approved_change_log": str(archive / "approved_change_log.json"),
                "reevaluation_helper": str(Path(__file__).resolve()),
                "evaluation_code_root": str(root / "source_code"),
                "original_main_marker": str(archive / marker_relative), "records": manifest["records"]}
    try:
        for record in manifest["records"]:
            seq = record["sequence"]
            for target in (root / "inputs" / seq / "gt/gt.txt", root / "main/mot_gt" / seq / "gt/gt.txt"):
                shutil.copyfile(record["revised"], target)
        rows = {record["sequence"]: record for record in manifest["records"]}
        for item in inventory:
            item.update(rows[item["sequence"]]["revised_statistics"])
            item.update(gt=str(root / "inputs" / item["sequence"] / "gt/gt.txt"), gt_revision_id=revision["revision_id"])
            item.pop("gt_class_counts", None)
        write_json(root / "main/inventory.json", inventory)
        inputs = read_json(root / "input_inventory.json")
        for item in inputs:
            stats = rows[item["sequence"]]["revised_statistics"]
            item.update(source_gt=item.get("source_gt", item["gt"]),
                        gt=str(root / "inputs" / item["sequence"] / "gt/gt.txt"),
                        gt_rows=stats["gt_rows"], valid_gt_rows=stats["valid_gt_rows"],
                        gt_classes=stats["gt_class_counts"], gt_sha256=stats["gt_sha256"],
                        gt_revision_id=revision["revision_id"])
        write_json(root / "input_inventory.json", inputs)
        # Start fresh official output; old metrics remain in the revision archive.
        evaluation_root = root / "main/eval"
        if evaluation_root.is_symlink() or not evaluation_root.resolve().is_relative_to(root):
            raise ValueError("Evaluation output must be a directory inside RUN")
        shutil.rmtree(evaluation_root)
        proof = (backend or EvaluationOnly(root)).run(revision, archive)
        assert_preserved(protected)
        write_json(root / "input_revision.json", revision)
        for relative in ("main/provenance.json", "main/run_metadata.json"):
            previous = read_json(root / relative)
            previous["input_revision"] = revision
            write_json(root / relative, previous)
        validation = {"passed": True, "validation_scope": "GT-only re-evaluation and revalidation of existing artifacts",
                      "gt_revision": revision["revision_id"], "videos": 9, "methods": list(proof["metrics"]),
                      "gt_identical": True, "gt_identical_scope": "current RUN inputs versus main/mot_gt after revision",
                      "original_dataset_gt_modified": False, "gt_count_id_and_clear_consistent": True,
                      "count_checks": proof["count_checks"], "protected_artifacts_unchanged": True,
                      "detection_tracking_render_reexecuted": False, "old_validation": str(archive / "main/validation.json")}
        write_json(root / "main/validation.json", validation)
        write_json(root / "main/status.json", {"status": "complete_re_evaluated", "videos": 9,
                   "frames_per_method": proof["frames_per_method"], "gt_revision": revision["revision_id"],
                   "detection_tracking_render_reexecuted": False})
        write_json(marker, {"status": "completed", "job_id": main_job["id"], "stage": "main", "sequence": "",
                   "spec": str(root / "experiment_spec.json"), "completed": utc(), "gt_revision": revision["revision_id"],
                   "artifacts": proof["artifacts"], "gt_count_checks": proof["count_checks"],
                   "original_execution_marker": str(archive / marker_relative), "re_evaluation_only": True})
        write_json(archive / "promotion_receipt.json", {"status": "promoted", "revision": revision,
                   "main_exit_code": 0, "hold": hold, "validation": validation})
        return {"status": "promoted", "revision_id": revision["revision_id"], "archive": str(archive)}
    except Exception as error:
        failed = archive / "failed_attempt"
        for relative in mutable:
            target, previous = root / relative, archive / relative
            assert target.resolve().is_relative_to(root), target
            if target.exists():
                destination = failed / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                target.rename(destination)
            if existed[relative]:
                if previous.is_dir():
                    shutil.copytree(previous, target, symlinks=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(previous, target)
        write_json(archive / "promotion_receipt.json", {"status": "rolled_back", "error": repr(error),
                   "traceback": traceback.format_exc(), "scheduler_hold_preserved": True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    first = modes.add_parser("prepare")
    first.add_argument("--run-root", type=Path, required=True)
    first.add_argument("--changes", type=Path, required=True)
    first.add_argument("--candidate-dir", type=Path, required=True)
    second = modes.add_parser("promote")
    second.add_argument("--run-root", type=Path, required=True)
    second.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.run_root, args.changes, args.candidate_dir) if args.mode == "prepare" else promote(args.run_root, args.candidate_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
