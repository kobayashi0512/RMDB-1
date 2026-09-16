#!/usr/bin/env python3
"""Reproduce historical score then replay saved image-only crop on annotation.

Never estimate a transform from labels or predictions. Exact reproduction of
cached RGB is required. Historical scores remain archived for traceability.
"""
import ast
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rtnls_fundusprep.cfi_bounds import CFIBounds
from rtnls_fundusprep.transformation import Interpolation
from train_udct_av_pilot import target_labels, softmax_np, Corrector, hessian_field
from train_birap_v2 import case_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/fundus_coordinate_alignment_audit'


def evaluate(logits, ann):
    probability=softmax_np(logits.copy())
    return case_metrics(np.moveaxis(probability,-1,0),target_labels(ann),ann.max(-1)>0)


def main():
    cv2.setNumThreads(2);torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'data/fundus_avseg_vascx';raw=ROOT/'data/raw/Fundus_AVSeg/Fundus-AVSeg'
    locked=pd.read_csv(ROOT/'results/birap_v2_fundus_avseg/per_image_metrics_with_metadata.csv')
    locked=locked[locked.model=='vascx'].set_index('id')
    corrector=Corrector().eval()
    checkpoint=torch.load(ROOT/'models/baseline/artifacts/udct_av_mass_calibrated/best.pt',map_location='cpu')
    corrector.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    rows=[]
    for b in pd.read_csv(cache/'bounds_latest_batch.csv').itertuples():
        rgb=np.asarray(Image.open(raw/'images'/f'{b.id}.png').convert('RGB'))
        ann=np.asarray(Image.open(raw/'annotation'/f'{b.id}.png').convert('RGB'))
        config=ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)',r'\1',b.bounds))
        transform,bounds=CFIBounds.from_dict(rgb,config).crop(1024)
        canonical=transform.warp(rgb,(1024,1024))
        saved=np.asarray(Image.open(cache/'preprocessed_rgb'/f'{b.id}.png'))
        assert np.array_equal(canonical,saved)
        aligned=transform.warp(ann,(1024,1024),mode=Interpolation.NEAREST)
        logits=np.load(cache/'logits'/f'{b.id}.npy').astype(np.float32)
        # Reproduce the historical evaluation, including double logit resize.
        oldlogits=cv2.resize(cv2.resize(logits,(rgb.shape[1],rgb.shape[0])),(768,768))
        oldann=cv2.resize(ann,(768,768),interpolation=cv2.INTER_NEAREST)
        old=evaluate(oldlogits,oldann)
        assert abs(old['mean_av_dice']-locked.loc[b.id,'mean_av_dice'])<1e-7
        log768=cv2.resize(logits,(768,768));ann768=cv2.resize(aligned,(768,768),interpolation=cv2.INTER_NEAREST)
        aligned_scores=evaluate(log768,ann768)
        rgb768=cv2.resize(canonical,(768,768))
        pr=softmax_np(log768.copy()); ent=-(pr*np.log(np.maximum(pr,1e-8))).sum(-1)/np.log(4)
        features=np.concatenate([np.moveaxis(rgb768.astype(np.float32)/255,-1,0),
                                 np.moveaxis(np.clip(log768,-12,12)/6,-1,0),
                                 ent[None],hessian_field(rgb768)],axis=0).astype(np.float32)
        with torch.inference_mode():
            refined=corrector(torch.from_numpy(features[None]),torch.from_numpy(np.moveaxis(log768,-1,0).copy()[None]))[0].numpy()
        mass=evaluate(np.moveaxis(refined,0,-1),ann768)
        native=evaluate(logits,aligned)
        # Same historical interpolation but correct annotation isolates crop issue.
        matched_interp=evaluate(oldlogits,ann768)
        row={'id':b.id,'radius_raw':config['radius'],
             'rgb_max_abs_reconstruction_error':0,
             'historical_reproduction_abs_error':abs(old['mean_av_dice']-locked.loc[b.id,'mean_av_dice']),
             'disease type':locked.loc[b.id,'disease type'], 'image quality':locked.loc[b.id,'image quality']}
        for name,scores in [('historical',old),('aligned768',aligned_scores),('aligned768_mass',mass),('aligned1024',native),('aligned768_old_interpolation',matched_interp)]:
            row.update({name+'_'+k:v for k,v in scores.items()})
        rows.append(row)
        if len(rows)%10==0:print('coordinate audit',len(rows),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_image.csv',index=False)
    numeric=[k for k in frame if k not in ('id','disease type','image quality')]
    summary={'n':len(frame),'geometry_source':'saved image-only crop; no label alignment fitting',
             'all_cached_rgb_exactly_reproduced':True,
             'historical_mean_av_max_reproduction_error':float(frame.historical_reproduction_abs_error.max()),
             'means':{k:float(frame[k].mean()) for k in numeric},'strata':{}}
    for col in ['image quality','disease type']:
        summary['strata'][col]={str(k):{'n':len(g),**{m:float(g[m].mean()) for m in numeric}} for k,g in frame.groupby(col)}
    delta=frame.aligned768_mass_mean_av_dice-frame.aligned768_mean_av_dice
    rng=np.random.default_rng(20260905)
    boot=delta.to_numpy()[rng.integers(0,len(delta),(3000,len(delta)))].mean(1)
    summary['mass_paired_delta']={'mean':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'unit':'image'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ['n','historical_mean_av_max_reproduction_error','mass_paired_delta']},indent=2))


if __name__=='__main__':main()
