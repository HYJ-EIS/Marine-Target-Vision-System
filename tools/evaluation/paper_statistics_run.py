"""Filesystem orchestration for paper_statistics; no inference or tracking."""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import configparser
import csv
import json

import numpy as np

from tools.evaluation.paper_statistics import (
    METRICS, COUNTS, STAGES, StreamingCSV, bootstrap_difference, common_recovery_comparison,
    confirmation_outcome, duplicate_statistics, extract_metrics, identity_statistics,
    internal_statistics, iou_matrix, load_json, ratio, recovery_outcome, reference_availability,
    shared_events, summarize_confirmation, summarize_recovery, write_csv, write_json, complete_recovery,
)
from tools.evaluation.class_aware_mot import validate_gt_labels, validate_mot_class_file
from tools.evaluation.stable_tracking_analysis import read_mot, match_frames


@dataclass(frozen=True)
class Method:
    label: str
    batch: str
    tracker: str
    variant: str
    internal: bool


def methods_for_run(run_root, spec, phase):
    metadata = load_json(run_root/'main/run_metadata.json')
    names = metadata['tracker_names']
    methods = [Method('Full', 'main', names['msdc_elt'], 'msdc_v3', True)]
    methods.extend(Method(label, 'main', names[kind], kind, False) for kind, label in
                   (('bytetrack', 'ByteTrack'), ('official_ocsort', 'OC-SORT'), ('official_botsort', 'BoT-SORT')))
    labels = {row['name']: row['label'] for row in spec['variants']}
    methods.extend(Method(labels[name], 'core', name+'_replay', name, True) for name in spec['core_variants'])
    if phase == 'all':
        methods.extend(Method(labels[name], 'sensitivity', name+'_replay', name, True) for name in spec['sensitivity_variants'])
    if len({method.label for method in methods}) != len(methods):
        raise ValueError('Duplicate method labels in experiment specification')
    return methods


def native_metrics(eval_root, tracker, sequences):
    protocol = load_json(eval_root/'evaluation_protocol.json')
    if protocol.get('class_consistency_required') is not True:
        raise ValueError(f'Class-consistent protocol missing: {eval_root}')
    path = eval_root/tracker/'pedestrian_detailed.csv'
    with path.open(encoding='utf-8-sig', newline='') as fh:
        raw = {row['seq']: row for row in csv.DictReader(fh)}
    expected = set(sequences) | {'COMBINED'}
    if set(raw) != expected:
        raise ValueError(f'Native detailed CSV scope differs from specification: {path}, {set(raw)} != {expected}')
    # These are native proportions, never the *_percent copies.
    for row in raw.values():
        if not 0 <= float(row['IDF1']) <= 1.000001:
            raise ValueError(f'Expected native 0..1 IDF1 in {path}')
    return {sequence: extract_metrics(row) for sequence, row in raw.items()}


def validate_frames(path, length):
    with path.open(encoding='utf-8') as stream:
        count = 0
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if int(row['frame_id']) != count:
                raise ValueError(f'Noncontiguous tracking frame log: {path}')
    if count != length:
        raise ValueError(f'Incomplete tracking frame log: {path}: {count} != {length}')


def timing_row(run_root, method, sequence, length, shared_detection):
    batch = run_root/method.batch
    mot = batch/'trackers'/method.tracker/'data'/f'{sequence}.txt'
    replay = load_json(mot.with_suffix('.timing.json'))
    validate_frames(mot.with_suffix('.frames.jsonl'), length)
    video = batch/'visualizations'/sequence/(method.tracker+'.mp4')
    if not video.is_file():
        raise FileNotFoundError(video)
    render = load_json(video.with_suffix('.timing.json'))
    for label, data in (('shared detection', shared_detection), ('replay', replay), ('render', render)):
        if int(data['processed_frames']) != length:
            raise ValueError(f'{label} frame count differs for {sequence}/{method.label}')
        if any(key not in data['stage_seconds'] for key in STAGES):
            raise ValueError(f'{label} does not contain the six timing stages')
        if any(float(value) < 0 for value in data['stage_seconds'].values()):
            raise ValueError(f'Negative stage timing for {sequence}/{method.label}')
    stages = {key: sum(float(data['stage_seconds'][key]) for data in (shared_detection, replay, render)) for key in STAGES}
    total = sum(float(data['total_time_s']) for data in (shared_detection, replay, render))
    tracker_seconds = float(replay['stage_seconds']['tracker'])
    return {'sequence': sequence, 'method': method.label, 'variant': method.variant, 'processed_frames': length,
            'total_time_s': total, 'mean_latency_ms': total*1000/length,
            'mean_fps': ratio(length, total), 'tracker_time_s': tracker_seconds,
            'tracker_ms_per_frame': tracker_seconds*1000/length, 'tracker_fps': ratio(length, tracker_seconds),
            'frame_source': replay.get('frame_source', 'decode'),
            'pixel_frames_decoded': replay.get('pixel_frames_decoded', length),
            'replay_read_decode_s': float(replay['stage_seconds']['read_decode']),
            'speed_comparison_scope': 'main_comparison' if method.batch == 'main' else 'ablation_execution',
            'shared_detection_time_s': shared_detection['total_time_s'],
            **{key+'_s': value for key, value in stages.items()},
            'other_s': total-sum(stages.values()), 'mot_path': str(mot), 'video_path': str(video),
            'replay_timing_path': str(mot.with_suffix('.timing.json')), 'render_timing_path': str(video.with_suffix('.timing.json'))}


def output_match_rows(gt, pred, matches, sequence, method, length):
    for frame in range(1, length+1):
        gs, ps = gt.get(frame, []), pred.get(frame, [])
        pred_indices = {row[0]: i for i, row in enumerate(ps)}
        matched_public = set()
        overlaps = iou_matrix(gs, ps)
        for gi, (gt_id, _, cls) in enumerate(gs):
            public_id = matches.get((frame, gt_id))
            if public_id is not None:
                matched_public.add(public_id)
            yield {'sequence': sequence, 'method': method, 'frame': frame, 'gt_id': gt_id,
                   'pred_public_id': public_id, 'class': cls,
                   'IoU': float(overlaps[gi, pred_indices[public_id]]) if public_id is not None else None,
                   'matched': public_id is not None, 'row_type': 'matched' if public_id is not None else 'unmatched_gt'}
        for public_id, _, cls in ps:
            if public_id not in matched_public:
                yield {'sequence': sequence, 'method': method, 'frame': frame, 'gt_id': None,
                       'pred_public_id': public_id, 'class': cls, 'IoU': None, 'matched': False, 'row_type': 'unmatched_prediction'}
        if not gs and not ps:
            yield {'sequence': sequence, 'method': method, 'frame': frame, 'gt_id': None,
                   'pred_public_id': None, 'class': None, 'IoU': None, 'matched': False, 'row_type': 'empty_frame'}


def comparison_groups():
    return {
        'main_methods': ['Full', 'ByteTrack', 'OC-SORT', 'BoT-SORT'],
        'R_main': ['Full', '−R'],
        'R_S': ['Full', '−R', 'Full,S=0', '−R,S=0'],
        'Lost_Removed_S0': ['Full,S=0', 'Lost-only,S=0', 'Removed-only,S=0', '−R,S=0'],
    }


def interactions(by_method, sequence):
    groups = {
        'E_R': ['Full', '−E', '−R', '−E−R'],
        'R_S': ['Full', '−R', 'Full,S=0', '−R,S=0'],
        'Lost_Removed_S0': ['Full,S=0', 'Removed-only,S=0', 'Lost-only,S=0', '−R,S=0'],
    }
    fields = list(METRICS) + ['correct_coverage', 'longest_same_id_coverage', 'unmatched_new_public_ids',
                              'extra_public_ids_sum', 'duplicate_gt_frames', 'stable_confirmation_rate',
                              'confirmation_penalized_latency_mean', 'old_id_success_rate', 'recovery_penalized_latency_mean']
    result = []
    for name, methods in groups.items():
        for field in fields:
            values = [by_method[method].get(field) for method in methods]
            value = values[0]-values[1]-values[2]+values[3] if all(v is not None for v in values) else None
            result.append({'sequence': sequence, 'comparison_id': name, 'metric': field,
                           **{f'method_{key}': method for key, method in zip(('11', '01', '10', '00'), methods)},
                           **{f'Y{key}': v for key, v in zip(('11', '01', '10', '00'), values)}, 'interaction': value})
    return result


def choose_cases(candidates):
    result = []
    for module in ('A', 'E', 'R'):
        rows = [row for row in candidates if row['module'] == module]
        used = set()
        for selection, reverse in (('max', True), ('min', False)):
            ordered = sorted(rows, key=lambda row: ((-1 if reverse else 1)*row['correct_match_delta'], row['sequence'], row['start_frame'], row['event_id']))
            if ordered:
                row = ordered[0]
                if row['event_id'] not in used:
                    result.append({**row, 'selection': selection})
                    used.add(row['event_id'])
    return result


def sensitivity_rows(spec, records):
    result = {'distance': [], 'confirmation': []}
    labels = {row['name']: row['label'] for row in spec['variants']}
    for sequence, data in records.items():
        for name, key, values, center, prefix in (
            ('distance', 'association_center_parameter', spec['sensitivity']['association_center_parameter'], 80, 'paper_assoc_d'),
            ('confirmation', 'candidate_only_confirmation_threshold', spec['sensitivity']['candidate_only_confirmation_threshold'], 2.5, 'paper_candidate_t')):
            for value in values:
                variant = 'msdc_v3' if value == center else prefix+(str(int(value)) if name == 'distance' else str(int(value*10)))
                label = 'Full' if variant == 'msdc_v3' else labels[variant]
                result[name].append({**data[label], 'parameter': key, 'parameter_value': value, 'is_full_center': value == center})
    return result


def plot_sensitivity(output, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, rows in tables.items():
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for ax, metric in zip(axes.flat, ('HOTA', 'IDF1', 'FP', 'FN', 'unmatched_new_public_ids_per_1000_frames', 'confirmation_latency_median')):
            pooled = defaultdict(list)
            for number, sequence in enumerate(sorted({row['sequence'] for row in rows}), 1):
                selected = sorted((row for row in rows if row['sequence'] == sequence), key=lambda row: row['parameter_value'])
                valid = [row for row in selected if row.get(metric) is not None]
                ax.plot([row['parameter_value'] for row in valid], [row[metric] for row in valid], alpha=.55, linewidth=1, label=f'S{number:02d}')
                for row in valid:
                    pooled[row['parameter_value']].append(row[metric])
            ax.plot(sorted(pooled), [np.mean(pooled[value]) for value in sorted(pooled)], color='black', linewidth=2.3, label='Macro mean')
            ax.axvline(80 if name == 'distance' else 2.5, color='grey', linestyle='--', linewidth=1)
            ax.set_title(metric)
            ax.set_xlabel('Center distance (px)' if name == 'distance' else 'Candidate score threshold')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=6)
        fig.savefig(output/f'sensitivity_{name}.svg')
        fig.savefig(output/f'sensitivity_{name}.png', dpi=220)
        plt.close(fig)


def export_run(run_root, spec_path, phase='core'):
    if phase not in ('core', 'all'):
        raise ValueError('phase must be core or all')
    run_root, spec_path = Path(run_root).resolve(), Path(spec_path).resolve()
    spec = load_json(spec_path)
    methods = methods_for_run(run_root, spec, phase)
    sequences = [row['sequence'] for row in spec['sequences']]
    expected_key = 'core_runs' if phase == 'core' else 'total_unique_runs'
    if len(methods)*len(sequences) != int(spec['counts'][expected_key]):
        raise ValueError('Specification counts disagree with the experiment matrix')
    inventory = {row['sequence']: row for row in load_json(run_root/'main/inventory.json')}
    if set(inventory) != set(sequences):
        raise ValueError('main/inventory.json must contain exactly the specified sequences')
    output = run_root/'paper_statistics'
    output.mkdir(parents=True, exist_ok=True)
    metric_maps, standard_rows = {}, []
    for method in methods:
        eval_root = run_root/method.batch/('eval' if method.batch == 'main' else 'eval_all')
        metric_maps[method.label] = native_metrics(eval_root, method.tracker, sequences)
        standard_rows.append({'method': method.label, 'variant': method.variant,
                              **metric_maps[method.label]['COMBINED'], 'source_detailed_csv': str(eval_root/method.tracker/'pedestrian_detailed.csv')})
    tables = defaultdict(list)
    all_records = {}
    case_candidates = []
    physical_detection_seconds = 0.0
    measurement = spec['measurement']
    window, stable = int(measurement['window_frames']), int(measurement['stable_correct_id_frames'])
    with ExitStack() as stack:
        match_writer = StreamingCSV(stack, output/'output_matches.csv',
                                   ['sequence', 'method', 'frame', 'gt_id', 'pred_public_id', 'class', 'IoU', 'matched', 'row_type'])
        duplicate_writer = StreamingCSV(stack, output/'duplicate_outputs.csv',
                                       ['sequence', 'method', 'frame', 'gt_id', 'anchor_public_id', 'extra_public_ids', 'extra_ious', 'excess_box_count'])
        for sequence in sequences:
            item = inventory[sequence]
            length = int(item['frames'])
            if length <= 0:
                raise ValueError(f'Expected positive EOF frame count for {sequence}')
            gt_file = run_root/'main/mot_gt'/sequence/'gt/gt.txt'
            validate_gt_labels(gt_file)
            validate_mot_class_file(gt_file)
            gt = read_mot(gt_file, gt=True)
            if any(not 1 <= frame <= length for frame in gt):
                raise ValueError(f'GT outside EOF scope: {gt_file}')
            info = configparser.ConfigParser()
            info.read(gt_file.parent.parent/'seqinfo.ini')
            fps = float(info['Sequence']['frameRate'])
            if int(info['Sequence']['seqLength']) != length or fps <= 0:
                raise ValueError(f'Invalid seqinfo for {sequence}')
            cache = run_root/'main/detections'/f'{sequence}_high_low_detections.jsonl'
            available = reference_availability(cache, gt, length)
            common_confirm, common_recovery = shared_events(gt, available, sequence, window, stable,
                                                         int(measurement['minimum_detection_gap_frames']),
                                                         int(measurement['pre_and_post_gap_detection_frames']), length)
            detection_timing = load_json(cache.with_suffix('.timing.json'))
            physical_detection_seconds += float(detection_timing['total_time_s'])
            records, recovery_by_method, match_by_method, internal_by_method = {}, {}, {}, {}
            for method in methods:
                mot = run_root/method.batch/'trackers'/method.tracker/'data'/f'{sequence}.txt'
                validate_mot_class_file(mot)
                pred = read_mot(mot)
                if any(not 1 <= frame <= length for frame in pred):
                    raise ValueError(f'Prediction outside EOF scope: {mot}')
                matched = match_frames(gt, pred, length, require_class_match=True)
                native = metric_maps[method.label][sequence]
                gt_count, pred_count = sum(map(len, gt.values())), sum(map(len, pred.values()))
                if (native['TP'], native['FN'], native['FP']) != (len(matched), gt_count-len(matched), pred_count-len(matched)):
                    raise ValueError(f'CLEAR correspondence disagrees with official TP/FN/FP: {sequence}/{method.label}')
                for row in output_match_rows(gt, pred, matched, sequence, method.label, length):
                    match_writer.emit(row)
                identities, identity_summary, births = identity_statistics(gt, pred, matched, sequence, method.label)
                duplicate_summary, duplicate_events = duplicate_statistics(gt, pred, length, sequence, method.label, duplicate_writer.emit)
                tables['identity_by_gt'].extend(identities)
                tables['duplicate_events'].extend(duplicate_events)
                confirmation_rows = [{**confirmation_outcome(event, matched, window, stable), 'method': method.label} for event in common_confirm]
                recovery_rows = [{**recovery_outcome(event, matched, births, window, stable,
                                                   int(measurement['pre_and_post_gap_detection_frames'])), 'method': method.label}
                                 for event in common_recovery]
                internal = {}
                if method.internal:
                    log_root = run_root/method.batch/'trackers'/method.tracker/'paper_logs'/sequence
                    overrides = next((item['overrides'] for item in spec['variants'] if item['name'] == method.variant), {})
                    radius = overrides.get('MSDC_ASSOC_CENTER_DIST', 80)
                    bounds = {event['gt_id']: event['t0'] if event['t0'] is not None else event['first_gt_frame'] for event in common_confirm}
                    internal, a_rows, c_rows, r_rows, first_active = internal_statistics(
                        log_root, gt, matched, births, length, sequence, method.label, radius, bounds)
                    tables['association_actions'].extend(a_rows)
                    tables['confirmation_actions'].extend(c_rows)
                    tables['recovery_actions'].extend(r_rows)
                    internal_by_method[method.label] = a_rows
                    for event in confirmation_rows:
                        event['first_internal_active'] = first_active.get(event['gt_id'])
                else:
                    for event in confirmation_rows:
                        event['first_internal_active'] = None
                for event in confirmation_rows:
                    if event.get('latency_frames') is not None:
                        event['latency_seconds'] = event['latency_frames']/fps
                    else:
                        event['latency_seconds'] = None
                tables['confirmation_events'].extend(confirmation_rows)
                tables['recovery_events'].extend({**row, 'comparison_id': 'all_opportunities', 'in_common_pretracked_set': None} for row in recovery_rows)
                confirmation_summary = summarize_confirmation(confirmation_rows)
                recovery_summary = summarize_recovery(recovery_rows)
                row = {'sequence': sequence, 'method': method.label, 'variant': method.variant, 'frames': length, 'fps': fps,
                       **native, **identity_summary, **duplicate_summary, **confirmation_summary, **recovery_summary,
                       **internal, 'unmatched_new_public_ids_per_1000_frames': identity_summary['unmatched_new_public_ids']*1000/length}
                records[method.label] = row
                tables['identity_by_sequence'].append({key: row.get(key) for key in
                    ('sequence', 'method', 'frames', 'unmatched_new_public_ids', 'unmatched_new_public_ids_per_1000_frames',
                     'extra_public_ids_sum', 'extra_public_ids_mean', 'never_correctly_output_gt_count',
                     'duplicate_excess_boxes', 'duplicate_gt_frames', 'duplicate_gt_frame_percent', 'duplicate_events')})
                tables['e_confirmation'].append({'sequence': sequence, 'method': method.label, **confirmation_summary,
                                                **{key: internal.get(key) for key in ('new_identity_confirmations', 'false_confirmations', 'false_confirmation_rate', 'confirmed_never_published')},
                                                **{key: row[key] for key in ('unmatched_new_public_ids', 'unmatched_new_public_ids_per_1000_frames', 'extra_public_ids_sum', 'duplicate_excess_boxes', 'FN')}})
                tables['a_actions'].append({'sequence': sequence, 'method': method.label,
                                           **{key: internal.get(key) for key in ('distance_only_pairs', 'distance_correct', 'distance_wrong', 'distance_unknown', 'distance_precision')}})
                tables['r_recovery'].append({'sequence': sequence, 'method': method.label, 'comparison_id': 'all_opportunities',
                                            **recovery_summary, **{key: value for key, value in internal.items()
                                                                 if 'reuse' in key or 'suppressed' in key}})
                tables['timing_by_sequence'].append(timing_row(run_root, method, sequence, length, detection_timing))
                recovery_by_method[method.label] = recovery_rows
                # Only the four methods used in the fixed A/E/R case comparisons need full maps retained.
                if method.label in ('Full', '−A', '−E', 'Full,S=0', '−R,S=0'):
                    match_by_method[method.label] = matched
            all_records[sequence] = records
            tables['interactions'].extend(interactions(records, sequence))
            for comparison_id, labels in comparison_groups().items():
                summaries, event_rows = common_recovery_comparison({label: recovery_by_method[label] for label in labels}, comparison_id)
                tables['r_recovery'].extend({**row, 'sequence': sequence} for row in summaries)
                tables['recovery_events'].extend(event_rows)
            method_lookup = {method.label: method for method in methods}
            for module, first, second, events in (
                ('A', 'Full', '−A', [{'event_id': f'{sequence}:A:{row["frame"]}:{row["gid"]}', 'gt_id': row['prior_gt'], 't0': row['frame']}
                                    for row in internal_by_method['Full'] if row['prior_gt'] is not None]),
                ('E', 'Full', '−E', [row for row in common_confirm if not row['censored'] and not row['no_detection']]),
                ('R', 'Full,S=0', '−R,S=0', [row for row in common_recovery if complete_recovery(row)])):
                for event in events:
                    start, identity = event['t0'], event['gt_id']
                    if start+window-1 > length or not all(any(row[0] == identity for row in gt.get(frame, [])) for frame in range(start, start+window)):
                        continue
                    counts = [sum((frame, identity) in match_by_method[label] for frame in range(start, start+window)) for label in (first, second)]
                    paths = [run_root/method_lookup[label].batch/'trackers'/method_lookup[label].tracker/'data'/f'{sequence}.txt' for label in (first, second)]
                    case_candidates.append({'sequence': sequence, 'module': module, 'event_id': event['event_id'], 'gt_id': identity,
                                            'method_a': first, 'method_b': second, 'start_frame': start, 'end_frame': start+window-1,
                                            'correct_match_delta': counts[0]-counts[1], 'video_path': item['video'],
                                            'mot_a': str(paths[0]), 'mot_b': str(paths[1])})
    main_labels = ['Full', 'ByteTrack', 'OC-SORT', 'BoT-SORT']
    main_ablation = ['Full', '−A', '−E', '−R', '−E−R']
    recovery_labels = ['Full,S=0', '−R,S=0', 'Lost-only,S=0', 'Removed-only,S=0']
    tables['main_by_sequence'] = [records[label] for records in all_records.values() for label in main_labels]
    tables['ablation_by_sequence'] = [records[label] for records in all_records.values() for label in main_ablation]
    tables['recovery_s0_by_sequence'] = [records[label] for records in all_records.values() for label in recovery_labels]
    for row in standard_rows:
        identity_rows = [identity for identity in tables['identity_by_gt'] if identity['method'] == row['method']]
        total_gt = sum(identity['valid_gt_frames'] for identity in identity_rows)
        row['correct_coverage'] = ratio(row['TP'], total_gt, 100)
        row['longest_same_id_coverage'] = ratio(sum(identity['longest_same_id_frames'] for identity in identity_rows), total_gt, 100)
    tables['standard_aggregate'] = standard_rows
    for method in methods:
        selected = [all_records[sequence][method.label] for sequence in sequences]
        tables['macro_aggregate'].append({'method': method.label, 'sequences': len(selected),
            **{metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS},
            **{field: float(np.mean([row[field] for row in selected if row[field] is not None]))
               if any(row[field] is not None for row in selected) else None
               for field in ('correct_coverage', 'longest_same_id_coverage')}})
    core_labels = main_labels + [method.label for method in methods if method.batch == 'core']
    for control in core_labels:
        if control == 'Full':
            continue
        for metric in ('HOTA', 'IDF1', 'correct_coverage'):
            pairs = [(all_records[seq]['Full'][metric], all_records[seq][control][metric]) for seq in sequences]
            if any(a is None or b is None for a, b in pairs):
                continue
            tables['paired_differences'].extend({'sequence': seq, 'method_a': 'Full', 'method_b': control, 'metric': metric,
                                                'value_a': a, 'value_b': b, 'difference': a-b} for seq, (a, b) in zip(sequences, pairs))
            tables['bootstrap'].append({'method_a': 'Full', 'method_b': control, 'metric': metric,
                                        **bootstrap_difference([a for a, _ in pairs], [b for _, b in pairs],
                                                               int(measurement['bootstrap']['iterations']), int(measurement['bootstrap']['seed']))})
    cases = choose_cases(case_candidates)
    tables['case_selection'] = cases
    write_json(output/'case_selection.json', {'cases': cases, 'window_frames': window,
                                             'rendering': 'Use original video and final MOT, preserve frame numbers and public IDs'})
    sensitivity = None
    if phase == 'all':
        sensitivity = sensitivity_rows(spec, all_records)
        tables['sensitivity_distance'], tables['sensitivity_confirmation'] = sensitivity['distance'], sensitivity['confirmation']
    for name, rows in tables.items():
        write_csv(output/(name+'.csv'), rows)
    if sensitivity is not None:
        plot_sensitivity(output, sensitivity)
        write_json(output/'plot_sequence_labels.json', {f'S{number:02d}': seq for number, seq in enumerate(sorted(sequences), 1)})
    expected = len(sequences)*len(methods)
    if len(tables['timing_by_sequence']) != expected:
        raise AssertionError('Statistics result count differs from the fixed run matrix')
    manifest = {'status': 'complete', 'phase': phase, 'spec': str(spec_path), 'run_root': str(run_root),
                'sequences': len(sequences), 'unique_method_sequence_results': expected,
                'detection_caches': len(sequences), 'shared_detection_physical_seconds': physical_detection_seconds,
                'reference_matching': 'maximum cardinality then total IoU', 'output_matching': 'class-consistent vendored CLEAR',
                'official_counts_verified': True, 'gt_used_only_for_offline_statistics': True,
                'gt_first_public_output_definition': 'First actual same-class CLEAR-supported public output at or after t0 (or first valid GT frame without a reference detection); equals first_correct_output. No future ID ownership is back-projected.',
                'absolute_confirmation_times_scope': 'Full valid GT lifetime at/after t0; H40 limits window success and latency only',
                'replay_performed_by_exporter': False, 'bootstrap_seed': measurement['bootstrap']['seed'],
                'csv_rows': {name: len(rows) for name, rows in tables.items()},
                'case_count': len(cases), 'statistics_dir': str(output)}
    write_json(output/'statistics_manifest.json', manifest)
    return manifest
