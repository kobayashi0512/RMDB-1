#!/usr/bin/env python3
"""Score frozen VascX structure predictions on FIVES and STARE when labels exist."""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
from PIL import Image

def dice(a,b):
    den=int(a.sum()+b.sum()); return float(2*np.logical_and(a,b).sum()/den) if den else np.nan

def score(pred, target, fov=None):
    if fov is None: fov=np.ones(pred.shape,bool)
    vessel=(pred>0)&fov; truth=(target>0)&fov
    return {"vessel_dice":dice(vessel,truth),"pred_fraction":float(vessel.mean()),"target_fraction":float(truth.mean())}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['fives','stare'],required=True); p.add_argument('--root',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); rows=[]
    if a.dataset=='fives':
        originals=a.root/'test'/'Original'; gt=a.root/'test'/'Ground truth'
        for img in sorted(originals.glob('*.png')):
            target=gt/img.name
            if not target.exists(): continue
            pr=np.asarray(Image.open(a.predictions/'artery_vein'/f'{img.stem}.png').resize(Image.open(target).size,Image.Resampling.NEAREST))
            ta=np.asarray(Image.open(target).convert('L'))>0
            rows.append({'id':img.stem,**score(pr,ta)})
    else:
        labels=a.root/'labels'
        for lab in sorted(labels.glob('*.ah.ppm.gz')):
            stem=lab.name.split('.')[0]; img=a.root/'images'/f'{stem}.ppm'; pred=a.predictions/'artery_vein'/f'{stem}.png'
            if not img.exists() or not pred.exists(): continue
            with gzip.open(lab,'rb') as f: target=np.asarray(Image.open(f))>0
            pr=np.asarray(Image.open(pred).resize(target.shape[::-1],Image.Resampling.NEAREST))
            rows.append({'id':stem,**score(pr,target)})
    import pandas as pd
    frame=pd.DataFrame(rows); a.output.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(a.output,index=False)
    summary={'dataset':a.dataset,'images_scored':int(len(frame)),'vessel_dice_mean':float(frame.vessel_dice.mean()) if len(frame) else None,'vessel_dice_median':float(frame.vessel_dice.median()) if len(frame) else None}
    a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
