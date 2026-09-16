#!/usr/bin/env python3
"""Audit the completed frozen queue and append descriptive hard clDice metrics.

Never trains or selects a checkpoint. Only saved development validation IDs
are read. Original training outputs are not modified. Hard clDice follows
Shit et al., CVPR 2021, https://github.com/jocpae/clDice (2D skeleton overlap).
"""
from __future__ import annotations
import csv
import gc
import hashlib
import json
import math
import signal
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import skimage
from skimage.morphology import skeletonize
import torch
from torch.utils.data import DataLoader

from backbone_runtime import atomic_json, process_matches
from run_backbone_recovery import CANDIDATES, SEEDS, ROOT, DEFAULT_OUTPUT, is_complete
from train_birap_v2 import BiRAPV2, RetinalDevelopmentDataset, binary_dice, case_metrics, runtime_device

OUT = DEFAULT_OUTPUT / 'final_audit'
STOP = False

def load(path):
    return json.loads(path.read_text())

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, float) else True

def cldice(prediction, target):
    prediction, target = prediction.astype(bool), target.astype(bool)
    sp, st = skeletonize(prediction), skeletonize(target)
    if not sp.any() and not st.any():
        return 1.0
    if not sp.any() or not st.any():
        return 0.0
    precision = float(np.logical_and(sp, target).sum() / sp.sum())
    sensitivity = float(np.logical_and(st, prediction).sum() / st.sum())
    return 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0

def metric_test():
    line = np.zeros((9, 9), bool); line[4, 1:8] = True
    other = np.zeros_like(line); other[1, 1:8] = True
    broken = line.copy(); broken[4, 4] = False
    assert cldice(line, line) == 1 and cldice(line, other) == 0
    assert 0 < cldice(line, broken) < 1
    assert cldice(line, np.zeros_like(line)) == 0

def audit_training():
    q = load(DEFAULT_OUTPUT / 'queue_state.json')
    assert q['queue_status'] == 'complete' and len(q['jobs']) == 15
    assert not (DEFAULT_OUTPUT / 'PAUSE').exists()
    assert not process_matches(q['supervisor_pid'], 'run_backbone_recovery.py')
    for name, h in q['source_sha256'].items():
        assert digest(ROOT / 'src' / name) == h, name
    assert digest(ROOT / 'data/rav_geometry_recovery_20260907/audit.json') == q['geometry_audit_sha256']
    report = {'training_sources_match': True, 'geometry_audit_matches': True, 'jobs': [],
              'metric_warnings': []}
    splits = {}
    for key, job in q['jobs'].items():
        out = Path(job['output'])
        assert job['status'] == 'complete' and is_complete(out, 12), key
        assert not process_matches(load(out / 'progress.json')['pid'], str(out))
        history, summary, split = (load(out / n) for n in ('history.json', 'summary.json', 'split.json'))
        assert finite(history) and finite(summary)
        last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert last['epoch'] == 12 and last['history'] == history and last['split'] == split
        assert finite(last['model']) and finite(last['optimizer']) and 'rng' in last
        best = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        assert best['epoch'] == summary['best_epoch'] and finite(best['model'])
        assert all(torch.equal(v, last['best_payload']['model'][k]) for k, v in best['model'].items())
        assert summary['best_epoch'] == max(history, key=lambda r: r['selection_score'])['epoch']
        seed = job['seed']
        if seed in splits:
            assert splits[seed] == split, f'Unpaired split: {key}'
        splits[seed] = split
        assert not set(split['rav_training']) & set(split['rav_validation'])
        assert not set(split['rav_holdout']) & set(split['rav_validation'])
        assert not set(split['fives_training']) & set(split['fives_validation'])
        assert not any(summary['development_boundaries'].values())
        for source, expected in (('rav', 30), ('fives', 120)):
            with (out / f'{source}_validation_metrics.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == len({r['id'] for r in rows}) == expected
            for field, average in summary[f'{source}_validation'].items():
                if field == 'n': continue
                missing = [r['id'] for r in rows if r[field] == '']
                for identifier in missing:
                    assert source == 'fives' and field == 'vessel_dice'
                    from PIL import Image
                    cfg = load(out / 'configuration.json')
                    mask = Path(cfg['fives']) / 'train' / 'Ground truth' / (identifier.removeprefix('FIVES_') + '.png')
                    assert not (np.asarray(Image.open(mask)) > 127).any()
                    report['metric_warnings'].append({'job': key, 'id': identifier, 'field': field,
                        'reason': 'Empty ground truth; original Dice skips undefined empty/empty case.',
                        'original_nonmissing_denominator': len(rows) - len(missing)})
                values = [float(r[field]) for r in rows if r[field] != '']
                assert all(math.isfinite(v) for v in values)
                assert abs(statistics.mean(values) - average) < 1e-10
        report['jobs'].append({'job': key, 'epochs': 12, 'best_epoch': best['epoch'],
                               'best_sha256': digest(out / 'best.pt'),
                               'split_sha256': digest(out / 'split.json')})
        del last, best
    report.update(status='passed_with_metric_warning' if report['metric_warnings'] else 'passed', n=15, total_completed_epochs=180,
                  optimizer_and_rng_present=True, same_seed_splits_match=True,
                  per_image_summary_agreement=True)
    atomic_json(OUT / 'training_integrity.json', report)
    return q

def validation_records(config, split):
    records = []
    assert len(split['rav_validation']) == 30 and len(split['fives_validation']) == 120
    for identifier in split['rav_validation']:
        name = f'RAV_{identifier}'
        records.append({'id': name, 'source': 'rav',
                        'image': Path(config['rav']) / 'preprocessed_rgb' / f'{name}.png',
                        'mask': Path(config['rav_ground_truth']) / f'{name}.png',
                        'logits': Path(config['rav']) / 'logits' / f'{name}.npy'})
    for filename in split['fives_validation']:
        assert Path(filename).name == filename
        records.append({'id': 'FIVES_' + Path(filename).stem, 'source': 'fives',
                        'image': Path(config['fives']) / 'train' / 'Original' / filename,
                        'mask': Path(config['fives']) / 'train' / 'Ground truth' / filename,
                        'logits': None})
    for record in records:
        assert all(path.is_file() for name, path in record.items() if name in ('image', 'mask', 'logits') and path is not None)
    return records

@torch.inference_mode()
def supplement_seed(seed):
    if STOP or (DEFAULT_OUTPUT / 'PAUSE').exists():
        raise SystemExit(75)
    dest = OUT / f'cldice_seed_{seed}.json'
    best_hashes = {name: digest(DEFAULT_OUTPUT / name / f'seed_{seed}' / 'best.pt') for name in CANDIDATES}
    if dest.exists():
        prior = load(dest)
        assert prior['best_checkpoint_sha256'] == best_hashes and prior['script_sha256'] == digest(Path(__file__))
        return
    config = load(DEFAULT_OUTPUT / 'baseline' / f'seed_{seed}' / 'configuration.json')
    split = load(DEFAULT_OUTPUT / 'baseline' / f'seed_{seed}' / 'split.json')
    loader = DataLoader(RetinalDevelopmentDataset(validation_records(config, split), False,
                         config['eval_size'], tuple(config['crop_sizes'])), batch_size=1, num_workers=0)
    device = runtime_device()
    models, originals, output = {}, {}, {name: [] for name in CANDIDATES}
    for name in CANDIDATES:
        path = DEFAULT_OUTPUT / name / f'seed_{seed}'
        assert load(path / 'split.json') == split
        ck = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
        model = BiRAPV2(None, None, config['route_center'], config['route_temperature'],
                       residual_logit=config['residual_logit'], residual_scale=config['residual_scale'], backbone=name)
        model.load_state_dict(ck['model'], strict=True)
        # The complete checkpoint contains the frozen corrector; do not replace it.
        model.mass_available = True
        models[name] = model.to(device).eval()
        originals[name] = {}
        for source in ('rav', 'fives'):
            with (path / f'{source}_validation_metrics.csv').open() as f:
                originals[name].update({r['id']: r for r in csv.DictReader(f)})
        del ck
    started = time.monotonic()
    for step, batch in enumerate(loader, 1):
        if STOP or (DEFAULT_OUTPUT / 'PAUSE').exists():
            raise SystemExit(75)
        identifier, source = batch.pop('id')[0], batch.pop('source')[0]
        batch = {key: value.to(device) for key, value in batch.items()}
        target = batch['target'][0].cpu().numpy()
        truth = batch['vessel'][0, 0].cpu().numpy() > .5
        for name, model in models.items():
            outputs = model(batch, route_enabled=True, identity_enabled=True)
            if source == 'fives':
                occ = outputs['learned_occupancy'][0].cpu().numpy()
                probability = np.concatenate([1 - occ, occ, np.zeros_like(occ), np.zeros_like(occ)], axis=0)
            else:
                probability = outputs['final'][0].cpu().numpy()
            assert np.isfinite(probability).all()
            pred = probability.argmax(0) > 0
            metrics = case_metrics(probability, target, truth)
            old = originals[name][identifier]
            differences = []
            clean_metrics = {}
            for field, value in metrics.items():
                if old[field] == '':
                    assert field == 'vessel_dice' and not truth.any() and not pred.any() and math.isnan(value)
                    clean_metrics[field] = None
                else:
                    assert math.isfinite(value)
                    clean_metrics[field] = value
                    differences.append(abs(value - float(old[field])))
            delta = max(differences, default=0.0)
            if delta > 1e-6:
                raise ValueError(f'Frozen evaluation mismatch: {name} {identifier} {delta}')
            output[name].append({'id': identifier, 'source': source, 'cldice': cldice(pred, truth),
                                 'target_vessel_pixels': int(truth.sum()), 'predicted_vessel_pixels': int(pred.sum()),
                                 'replay_max_metric_delta': delta, **clean_metrics})
            del outputs, probability
        if step % 30 == 0:
            status = {'seed': seed, 'images': step, 'of': 150, 'seconds': time.monotonic() - started}
            atomic_json(OUT / 'supplement_progress.json', status)
            print(json.dumps(status), flush=True)
    result = {'status': 'complete', 'seed': seed, 'best_checkpoint_sha256': best_hashes,
              'script_sha256': digest(Path(__file__)), 'rows': output,
              'thresholds_changed': False, 'post_training_descriptive_only': True,
              'device': str(device), 'torch_version': torch.__version__, 'skimage_version': skimage.__version__,
              'source': 'https://github.com/jocpae/clDice',
              'readout': 'RAV: final four-class argmax > 0; FIVES: learned occupancy argmax (strictly > 0.5)',
              'cldice': 'harmonic mean of skeleton prediction inside truth and skeleton truth inside prediction; skimage 2D skeletonize',
              'evaluation_seconds': time.monotonic() - started}
    atomic_json(dest, result)
    del models, loader
    gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()

def summarize(q):
    results = []
    for seed in SEEDS:
        supplement = load(OUT / f'cldice_seed_{seed}.json')
        for name in CANDIDATES:
            out = DEFAULT_OUTPUT / name / f'seed_{seed}'
            summary, history = load(out / 'summary.json'), load(out / 'history.json')
            row = {'candidate': name, 'seed': seed, 'best_epoch': summary['best_epoch'],
                   'occupancy_parameters': summary['occupancy_parameters'],
                   'trainable_parameters': summary['trainable_parameters'],
                   'training_hours': summary['training_seconds'] / 3600,
                   'selection_score': summary['best_selection_score'],
                   'last_train_loss': history[-1]['train_loss'],
                   'last_selection_score': history[-1]['selection_score'],
                   'last3_selection_gain': history[-1]['selection_score'] - history[-3]['selection_score']}
            for source in ('rav', 'fives'):
                row.update({source + '_' + k: v for k, v in summary[f'{source}_validation'].items() if k != 'n'})
                source_rows = [r for r in supplement['rows'][name] if r['source'] == source]
                values = [r['cldice'] for r in source_rows]
                row[source + '_cldice'] = statistics.mean(values)
                nonempty = [r for r in source_rows if r['target_vessel_pixels'] > 0]
                row[source + '_nonempty_n'] = len(nonempty)
                row[source + '_vessel_dice_nonempty'] = statistics.mean(r['vessel_dice'] for r in nonempty)
                row[source + '_cldice_nonempty'] = statistics.mean(r['cldice'] for r in nonempty)
                row[source + '_empty_target_ids'] = [r['id'] for r in source_rows if not r['target_vessel_pixels']]
            results.append(row)
    fields = ['rav_mean_av_dice', 'rav_vessel_dice', 'rav_cldice', 'fives_vessel_dice', 'fives_cldice',
              'rav_av_ratio_log_error', 'rav_brier', 'rav_bypass', 'selection_score', 'training_hours',
              'fives_vessel_dice_nonempty', 'fives_cldice_nonempty']
    aggregates = []
    for name in CANDIDATES:
        rows = [r for r in results if r['candidate'] == name]
        item = {'candidate': name, 'occupancy_parameters': rows[0]['occupancy_parameters'],
                'trainable_parameters': rows[0]['trainable_parameters'], 'metrics': {}}
        for field in fields:
            values = [r[field] for r in rows]
            delta = [r[field] - next(b[field] for b in results if b['seed'] == r['seed'] and b['candidate'] == 'baseline') for r in rows]
            item['metrics'][field] = {'mean': statistics.mean(values), 'sd': statistics.stdev(values),
                                      'paired_deltas': delta, 'mean_paired_delta': statistics.mean(delta),
                                      'min_paired_delta': min(delta), 'max_paired_delta': max(delta)}
        aggregates.append(item)
    output = {'status': 'complete', 'n_jobs': 15, 'epochs_per_job': 12, 'total_epochs': 180,
              'completed_training_utc': q['completed_at'], 'reported_at_utc': datetime.now(timezone.utc).isoformat(),
              'per_run': results, 'aggregates': aggregates,
              'notes': ['All results are development scores, not independent tests.',
                        'Candidates have different historical pretraining exposure.',
                        'Training hours include concurrent system load and do not establish inference speed.',
                        'clDice is a descriptive supplement on already selected checkpoints; no checkpoint reselection.',
                        'Paired seed differences are descriptive, not independent confidence intervals.',
                        'Original FIVES means have a denominator mismatch for one empty target in seed 20260906.',
                        'Nonempty-target Dice and clDice use exactly the same label-defined subset across all candidates, descriptively without reselection.',
                        'Full-set clDice uses 1 for both-empty and 0 for only-one-empty; nonempty-target sensitivity avoids this convention.']}
    atomic_json(OUT / 'comparison.json', output)
    print(json.dumps(aggregates, indent=2), flush=True)

def main():
    global STOP
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    metric_test()
    queue = audit_training()
    print('Integrity audit passed: 15 jobs / 180 complete epochs.', flush=True)
    for seed in SEEDS:
        supplement_seed(seed)
    summarize(queue)

if __name__ == '__main__':
    main()
