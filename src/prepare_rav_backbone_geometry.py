#!/usr/bin/env python3
"""Replay image-only preprocessing on RAV annotations, without label fitting."""
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from rtnls_fundusprep.cfi_bounds import CFIBounds
from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from rtnls_fundusprep.transformation import Interpolation
from rtnls_fundusprep.utils import open_image
from backbone_runtime import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/rav_geometry_recovery_20260907'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    original = ROOT / 'data/raw/RAV/data/external/RAV/dataset'
    cached = ROOT / 'data/rav_vascx'
    with (cached / 'bounds_latest_batch.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    config = {row['id']: ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)', r'\1', row['bounds'])) for row in rows}
    dest = OUTPUT / 'aligned_masks'
    dest.mkdir(parents=True, exist_ok=True)
    audit = []
    sources = sorted((original / 'rgb').glob('*.png'))
    if len(sources) != 206:
        raise ValueError(f'Expected 206 RAV source images, found {len(sources)}')
    for path in sources:
        name = path.stem
        rgb = open_image(path)
        bounds = CFIBounds.from_dict(rgb, config[name]) if name in config else get_cfi_bounds(rgb)
        transform, _ = bounds.crop(1024)
        replay = transform.warp(rgb, (1024, 1024))
        cache_path = cached / 'preprocessed_rgb' / f'{name}.png'
        saved = np.asarray(Image.open(cache_path))
        # A strict image equality check is completed BEFORE opening the label.
        if not np.array_equal(replay, saved):
            raise ValueError(f'Cannot exactly reconstruct image geometry: {name}')
        mask_path = original / 'av_masks' / f'{name}.png'
        mask = np.asarray(Image.open(mask_path).convert('RGB'))
        aligned = transform.warp(mask, (1024, 1024), mode=Interpolation.NEAREST)
        out = dest / f'{name}.png'
        if out.exists():
            if not np.array_equal(np.asarray(Image.open(out)), aligned):
                raise ValueError(f'Existing derived mask differs: {name}')
        else:
            Image.fromarray(aligned.astype(np.uint8)).save(out)
        # No Dice is inspected here; the transform comes only from the image.
        audit.append({'id': name, 'rgb_exact_replay': True,
                      'bounds_source': 'saved' if name in config else 'recovered_from_rgb',
                      'bounds': bounds.to_dict(), 'raw_rgb_sha256': digest(path),
                      'cached_rgb_sha256': digest(cache_path), 'raw_mask_sha256': digest(mask_path),
                      'aligned_mask_sha256': digest(out),
                      'changed_label_pixels': int(np.any(mask != aligned, axis=-1).sum())})
    result = {'status': 'passed', 'n': len(audit), 'exact_rgb_replays': len(audit),
              'image_only_recovered_bounds': sum(r['bounds_source'] == 'recovered_from_rgb' for r in audit),
              'label_interpolation': 'nearest', 'labels_used_to_choose_transform': False,
              'sources_modified': False, 'cases': audit}
    atomic_json(OUTPUT / 'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))

if __name__ == '__main__':
    main()
