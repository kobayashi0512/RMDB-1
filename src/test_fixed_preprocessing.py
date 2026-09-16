#!/usr/bin/env python3
"""Frozen photometric preprocessing trial on already inspected Fundus-AVSeg.

All four arms are fixed before inference. The original crop geometry is shared
with annotations. This is exploratory evaluation, not independent confirmation.
No clinical data or labels are used for preprocessing or parameter selection.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from rtnls_fundusprep.cfi_bounds import CFIBounds
from rtnls_fundusprep.transformation import Interpolation
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from rtnls_inference.ensembles.predict_output import decollate_predict_full
from train_birap_v2 import case_metrics
from train_udct_av_pilot import target_labels, softmax_np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/Fundus_AVSeg/Fundus-AVSeg'
CACHE = ROOT / 'data/fundus_avseg_vascx'
OUT = ROOT / 'artifacts/fixed_preprocessing_trial'
ARMS = ('default', 'illumination', 'clahe', 'denoise_sharpen')
PARAMETERS = {
    'illumination': {'sigma': 51.2, 'gain_bounds': [0.7, 1.4], 'blend': 0.5},
    'clahe': {'lab_l_clip_limit': 1.5, 'tile_grid': [8, 8], 'blend': 0.5},
    'denoise_sharpen': {'bilateral_d': 5, 'sigma_color': 12, 'sigma_space': 2,
                        'unsharp_sigma': 1.0, 'unsharp_amount': 0.25, 'blend': 0.5},
}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def enhance(rgb, arm, valid):
    if arm == 'default':
        return rgb.copy()
    f = rgb.astype(np.float32)
    if arm == 'illumination':
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mask = valid.astype(np.float32)
        den = cv2.GaussianBlur(mask, (0, 0), 51.2)
        illumination = cv2.GaussianBlur(gray * mask, (0, 0), 51.2) / np.maximum(den, 1e-4)
        target = np.median(illumination[valid])
        gain = np.clip(target / np.maximum(illumination, 1), .7, 1.4)
        adjusted = f * gain[..., None]
    elif arm == 'clahe':
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[..., 0])
        adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    elif arm == 'denoise_sharpen':
        smooth = cv2.bilateralFilter(rgb, 5, 12, 2).astype(np.float32)
        adjusted = smooth + .25 * (smooth - cv2.GaussianBlur(smooth, (0, 0), 1.0))
    else:
        raise ValueError(arm)
    result = np.clip(.5 * f + .5 * adjusted, 0, 255).astype(np.uint8)
    result[~valid] = rgb[~valid]
    return result


def ci(values, lower_better=False):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260905)
    means = values[rng.integers(0, len(values), (3000, len(values)))].mean(1)
    return {'delta': float(values.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'n': len(values), 'improved': int(((values < 0) if lower_better else (values > 0)).sum())}


def summarize(ids):
    records = [json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*.json'))]
    frame = pd.DataFrame(records)
    if frame.empty:
        return
    frame.to_csv(OUT/'per_image_metrics.csv', index=False)
    metrics = ['mean_av_dice', 'vessel_dice', 'vessel_precision', 'vessel_recall',
               'thin_centerline_recall', 'brier']
    report = {'complete': len(frame) == len(ids)*len(ARMS), 'images_planned': len(ids),
              'arm_evaluations_completed': len(frame), 'exploratory_only': True,
              'inference': '5 member VascX; 1024 input; 512 tiles; overlap .5; TTA disabled in every arm',
              'evaluation': '768 grid; annotation warped with saved image-only crop',
              'bootstrap_unit': 'image; patient identity unavailable; exploratory intervals',
              'overall': {}, 'strata': {}}
    meta = pd.read_csv(ROOT/'results/birap_v2_fundus_avseg/per_image_metrics_with_metadata.csv')
    meta = meta[['id', 'disease type', 'image quality']].drop_duplicates()
    frame = frame.merge(meta, on='id', validate='many_to_one')
    for level, group in [('all', frame)] + [(str(k), g) for k,g in frame.groupby('image quality')] + [(str(k),g) for k,g in frame.groupby('disease type')]:
        dest = {}
        baseline = group[group.arm == 'default'].set_index('id')
        for arm, g in group.groupby('arm'):
            entry = {'n':len(g), 'means':{m:float(g[m].mean()) for m in metrics}}
            common = g.set_index('id').join(baseline, lsuffix='_arm', rsuffix='_base', how='inner')
            if len(common) and arm != 'default':
                entry['paired_against_default'] = {m:ci(common[m+'_arm']-common[m+'_base'], lower_better=(m=='brier')) for m in metrics}
            dest[arm] = entry
        if level == 'all': report['overall'] = dest
        else: report['strata'][level] = dest
    write_json(OUT/'summary.json', report)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--limit', type=int)
    p.add_argument('--prepare-only', action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'cases').mkdir(exist_ok=True)
    bounds_table = pd.read_csv(CACHE/'bounds_latest_batch.csv').set_index('id')
    ids=sorted(bounds_table.index)
    protocol={'arms':list(ARMS), 'parameters':PARAMETERS, 'image_ids':ids,
              'primary':'paired mean AV Dice vs fresh default',
              'secondary':['vessel precision', 'vessel recall', 'thin centerline recall', 'Brier'],
              'quality_strata':'all images retained; use supplied high/low quality groups',
              'geometry':'reuse original bounds for every arm and warp reference with same transform',
              'selection':'no optimization; report all arms; dataset previously inspected',
              'inference':{'tta':False,'tile':512,'overlap':.5,'size':1024},
              'cimt_access':False,'clinical_outcome_access':False}
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        write_json(OUT/'protocol.json', protocol)
    if args.limit: ids=ids[:args.limit]
    model=None
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    start=time.monotonic()
    for index, ident in enumerate(ids):
        if all((OUT/'cases'/f'{ident}__{arm}.json').exists() for arm in ARMS):
            continue
        rgb=np.asarray(Image.open(RAW/'images'/f'{ident}.png').convert('RGB'))
        bdict=ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)', r'\1', bounds_table.loc[ident,'bounds']))
        transform,bounds=CFIBounds.from_dict(rgb,bdict).crop(1024)
        canonical=transform.warp(rgb,(1024,1024))
        cached_rgb=np.asarray(Image.open(CACHE/'preprocessed_rgb'/f'{ident}.png'))
        assert np.array_equal(canonical,cached_rgb), 'Cached crop cannot be reconstructed'
        entries=[]
        for arm in ARMS:
            if (OUT/'cases'/f'{ident}__{arm}.json').exists(): continue
            rgbpath=OUT/arm/'rgb'/f'{ident}.png'
            cepath=OUT/arm/'ce'/f'{ident}.png'
            rgbpath.parent.mkdir(parents=True,exist_ok=True);cepath.parent.mkdir(parents=True,exist_ok=True)
            if arm=='default':
                rgbpath=CACHE/'preprocessed_rgb'/f'{ident}.png';cepath=CACHE/'preprocessed_ce'/f'{ident}.png'
            elif not rgbpath.exists() or not cepath.exists():
                adjusted=enhance(canonical,arm,bounds.make_binary_mask(.01))
                new_bounds=CFIBounds.from_dict(adjusted,bounds.to_dict())
                Image.fromarray(adjusted).save(rgbpath)
                Image.fromarray(new_bounds.contrast_enhanced_5).save(cepath)
            entries.append({'id':arm,'image':str(rgbpath),'contrast_enhanced':str(cepath)})
        if args.prepare_only: continue
        if model is None:
            model=SegmentationEnsemble.from_torchscript(ROOT/'models/third_party/vascx_weights/av_july24.pt').to(device).eval()
            model.config['inference'].update({'tta':False,'tile_batch_size':2})
        loader=model._make_inference_dataloader({'images':entries},preprocess=False,batch_size=1,num_workers=0,ignore_exceptions=False)
        for batch in loader:
            with torch.inference_mode():
                full=model.predict_step_full(batch)
            for item in decollate_predict_full(full):
                arm=str(item['id'])
                logits=np.asarray(item['logits'],dtype=np.float32)
                assert logits.shape==(1024,1024,4) and np.isfinite(logits).all()
                # Annotation access happens after generation and inference.
                ann=np.asarray(Image.open(RAW/'annotation'/f'{ident}.png').convert('RGB'))
                aligned=transform.warp(ann,(1024,1024),mode=Interpolation.NEAREST)
                ann768=cv2.resize(aligned,(768,768),interpolation=cv2.INTER_NEAREST)
                prob=softmax_np(cv2.resize(logits,(768,768)))
                target=target_labels(ann768); vessel=ann768.max(-1)>0
                scores=case_metrics(np.moveaxis(prob,-1,0),target,vessel)
                pred=prob.argmax(-1)>0;tp=(pred&vessel).sum()
                scores['vessel_precision']=float(tp/max(pred.sum(),1))
                scores['vessel_recall']=float(tp/max(vessel.sum(),1))
                thin=skeletonize(vessel)&(distance_transform_edt(vessel)<=2)
                scores['thin_centerline_recall']=float((pred&thin).sum()/max(thin.sum(),1))
                write_json(OUT/'cases'/f'{ident}__{arm}.json',{'id':ident,'arm':arm,**scores})
                if index<3:
                    maskpath=OUT/arm/'prediction';maskpath.mkdir(parents=True,exist_ok=True)
                    palette=np.array([[0,0,0],[232,92,73],[61,141,219],[180,116,210]],np.uint8)
                    Image.fromarray(palette[prob.argmax(-1)]).save(maskpath/f'{ident}.png')
                print(f'{ident} {arm} completed; elapsed={time.monotonic()-start:.1f}s',flush=True)
            if device.type=='mps':torch.mps.synchronize();torch.mps.empty_cache()
        summarize(ids)
    summarize(ids)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
