"""Image-only geometry and provenance preflight for a new controlled retraining study."""
import ast
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from rtnls_fundusprep.cfi_bounds import CFIBounds
from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from rtnls_fundusprep.transformation import Interpolation
from backbone_runtime import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/residual_ablation_20260914_v1'
RAW = ROOT / 'data/raw/RAV/data/external/RAV'
CACHE = ROOT / 'data/rav_vascx'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'aligned_masks').mkdir(exist_ok=True)
    meta = pd.read_csv(RAW / 'metadata.csv', dtype=str, usecols=['id', 'device'])
    bounds = pd.read_csv(CACHE / 'bounds_latest_batch.csv').set_index('id')
    rows = []
    for item in meta.itertuples(index=False):
        key = 'RAV_' + item.id
        paths = [RAW / 'dataset/rgb' / (key + '.png'), CACHE / 'preprocessed_rgb' / (key + '.png'),
                 RAW / 'dataset/av_masks' / (key + '.png'), CACHE / 'logits' / (key + '.npy')]
        original, cached, ann = [np.asarray(Image.open(p).convert('RGB')) for p in paths[:3]]
        if key in bounds.index:
            config = ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)', r'\1', bounds.loc[key, 'bounds']))
            field = CFIBounds.from_dict(original, config)
            source = 'saved_bounds'
        else:
            field = get_cfi_bounds(original)
            source = 'image_only_recovered_bounds'
        transform, _ = field.crop(1024)
        reconstructed = transform.warp(original, (1024, 1024))
        assert np.array_equal(reconstructed, cached), 'RGB geometry cannot be replayed: ' + key
        assert ann.shape == original.shape
        aligned = transform.warp(ann, (1024, 1024), mode=Interpolation.NEAREST)
        target = OUT / 'aligned_masks' / (key + '.png')
        Image.fromarray(aligned).save(target)
        z = np.load(paths[3], mmap_mode='r')
        assert z.shape == (1024, 1024, 4) and np.isfinite(z).all()
        old512 = np.asarray(Image.fromarray(ann).resize((512, 512), Image.Resampling.NEAREST))
        new512 = np.asarray(Image.fromarray(aligned).resize((512, 512), Image.Resampling.NEAREST))
        a, b = old512.max(-1) > 0, new512.max(-1) > 0
        rows.append({'id': item.id, 'device': item.device, 'bounds_source': source,
                     'rgb_replayed_exactly': True, 'original_rgb_sha256': sha(paths[0]),
                     'rgb_sha256': sha(paths[1]), 'mask_sha256': sha(paths[2]), 'logits_sha256': sha(paths[3]),
                     'aligned_mask_sha256': sha(target), 'rgb_pixels_sha256': hashlib.sha256(cached.tobytes()).hexdigest(),
                     'mask_changed_fraction_512': float(np.any(old512 != new512, -1).mean()),
                     'vessel_mask_self_dice_512': float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))})
        if len(rows) % 25 == 0:
            print('geometry', len(rows), '/', len(meta), flush=True)
    frame = pd.DataFrame(rows)
    assert not frame.rgb_pixels_sha256.duplicated().any(), 'Duplicate RGB requires grouped split'
    holdout = meta.loc[meta.device == 'TOPCON TRC-50VT - Analog Fundus Camera', 'id'].tolist()
    development = meta.loc[~meta.id.isin(holdout), 'id'].tolist()
    random.Random(20260729).shuffle(development)
    split = {'training': development[20:], 'validation': development[:20], 'historical_camera_holdout': holdout}
    assert len(set(sum(split.values(), []))) == len(meta)
    frame['split'] = frame.id.map({i: name for name, ids in split.items() for i in ids})
    frame.to_csv(OUT / 'input_manifest.csv', index=False)
    atomic_json(OUT / 'split.json', split)
    audit = {'images': len(frame), 'all_rgb_replayed_exactly': True,
             'recovered_bounds_images': frame.loc[frame.bounds_source != 'saved_bounds', 'id'].tolist(),
             'images_with_changed_annotation_geometry': int((frame.mask_changed_fraction_512 > 0).sum()),
             'mean_changed_pixel_fraction_512': float(frame.mask_changed_fraction_512.mean()),
             'mean_original_vs_aligned_vessel_dice_512': float(frame.vessel_mask_self_dice_512.mean()),
             'split_counts': frame.split.value_counts().to_dict(), 'patient_grouping_verified': False,
             'patient_grouping_limitation': 'RAV supplied metadata contain image ID and device but no patient identifier; exact RGB duplicates checked, not patient independence.',
             'legacy_loader_issue': 'train_udct_av_pilot.load_case reads raw masks without applying cached RGB crop. New retraining must use aligned masks.',
             'legacy_mass_training_configuration_available': False,
             'historical_results_modified': False}
    atomic_json(OUT / 'input_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
