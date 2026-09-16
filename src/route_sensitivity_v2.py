#!/usr/bin/env python3
"""Outcome blind route sensitivity on the frozen FIVES/RAV development split."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from train_birap_v2 import (BiRAPV2, RetinalDevelopmentDataset, case_metrics,
                            locate_fives, paired_files, runtime_device)

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--rav',type=Path,required=True); p.add_argument('--rav-ground-truth',type=Path,required=True)
    p.add_argument('--rav-metadata',type=Path,required=True); p.add_argument('--fives',type=Path,required=True)
    p.add_argument('--occupancy-checkpoint',type=Path,required=True); p.add_argument('--mass-checkpoint',type=Path,required=True)
    p.add_argument('--split-json',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); split=json.loads(a.split_json.read_text()); records=[]
    for ident in split['rav_validation']:
        records.append({'id':f'RAV_{ident}','source':'rav','image':a.rav/'preprocessed_rgb'/f'RAV_{ident}.png',
                        'mask':a.rav_ground_truth/f'RAV_{ident}.png','logits':a.rav/'logits'/f'RAV_{ident}.npy'})
    ti,tm,_,_=locate_fives(a.fives); fives={im.name:(im,ma) for im,ma in paired_files(ti,tm)}
    for name in split['fives_validation']:
        im,ma=fives[name]; records.append({'id':f'FIVES_{im.stem}','source':'fives','image':im,'mask':ma,'logits':None})
    loader=DataLoader(RetinalDevelopmentDataset(records,False,512,(384,512,640,768)),batch_size=1,num_workers=0)
    dev=runtime_device(); results=[]
    for center in (0.35,0.55,0.75,0.85):
      for temp in (0.15,):
        model=BiRAPV2(a.occupancy_checkpoint,a.mass_checkpoint,center,temp,residual_logit=True,residual_scale=.1).to(dev)
        ck=torch.load(a.checkpoint,map_location=dev); model.load_state_dict(ck['model']); model.eval(); rows=[]
        with torch.inference_mode():
          for batch in loader:
            ids,sources=batch.pop('id'),batch.pop('source'); batch={k:v.to(dev) for k,v in batch.items()}; out=model(batch,route_enabled=True,identity_enabled=True)
            for i,ident in enumerate(ids):
              prob=out['learned_occupancy'][i].cpu().numpy();
              if sources[i]=='rav': prob=out['final'][i].cpu().numpy()
              else: prob=np.concatenate([1-prob,prob,np.zeros_like(prob),np.zeros_like(prob)],axis=0)
              rows.append({'source':sources[i],**case_metrics(prob,batch['target'][i].cpu().numpy(),batch['vessel'][i,0].cpu().numpy())})
        f=pd.DataFrame(rows); rav=f[f.source.eq('rav')]; fv=f[f.source.eq('fives')]
        score=float(rav.mean_av_dice.mean()+.25*fv.vessel_dice.mean()-.10*rav.av_ratio_log_error.mean())
        results.append({'route_center':center,'route_temperature':temp,'selection_score':score,
                        'rav_mean_av_dice':float(rav.mean_av_dice.mean()),'rav_ratio_error':float(rav.av_ratio_log_error.mean()),
                        'fives_vessel_dice':float(fv.vessel_dice.mean()),'rav_route_mean':float('nan')})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(results,indent=2)); print(json.dumps(sorted(results,key=lambda x:x['selection_score'],reverse=True),indent=2))
if __name__=='__main__': main()
