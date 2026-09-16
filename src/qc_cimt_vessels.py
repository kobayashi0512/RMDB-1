#!/usr/bin/env python3
"""Create blinded qualitative QC panels before extracting CIMT vessel phenotypes."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,default=Path('artifacts/manifest.csv')); p.add_argument('--weights',type=Path,default=Path('artifacts/fives_unet_resnet34/best.pt')); p.add_argument('--output',type=Path,default=Path('artifacts/cimt_vessel_qc')); p.add_argument('--n-eyes',type=int,default=100); p.add_argument('--seed',type=int,default=2026); a=p.parse_args()
    frame=pd.read_csv(a.manifest); choices=[(row.patient_id,side,getattr(row,f'{side}_path')) for row in frame.itertuples() for side in ('left','right')]; random.Random(a.seed).shuffle(choices); choices=choices[:a.n_eyes]
    dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); model=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=1).to(dev); model.load_state_dict(torch.load(a.weights,map_location=dev)); model.eval(); norm=transforms.Normalize((.485,.456,.406),(.229,.224,.225)); a.output.mkdir(parents=True,exist_ok=True); records=[]
    with torch.inference_mode():
        for patient_id,side,path in choices:
            image=Image.open(path).convert('RGB').resize((512,512)); x=norm(transforms.ToTensor()(image)).unsqueeze(0).to(dev); prob=model(x).sigmoid()[0,0].cpu().numpy(); mask=(prob>=.5).astype(np.uint8)*255
            base=np.asarray(image); red=np.zeros_like(base); red[:,:,0]=255; alpha=(mask[...,None]/255*.45); overlay=(base*(1-alpha)+red*alpha).astype(np.uint8)
            stem=f'{patient_id}_{side}'; Image.fromarray(mask).save(a.output/f'{stem}_mask.png'); Image.fromarray(overlay).save(a.output/f'{stem}_overlay.png'); records.append({'patient_id':str(patient_id),'eye':side,'source':path,'mask':f'{stem}_mask.png','overlay':f'{stem}_overlay.png','vessel_fraction':float((mask>0).mean())})
    pd.DataFrame(records).to_csv(a.output/'qc_manifest.csv',index=False); (a.output/'README.txt').write_text('Random blinded sample for manual accept/reject scoring. Assess continuity, thin vessels, optic-disc artefacts, and false-positive background.\n'); print(json.dumps({'n_eyes':len(records),'median_vessel_fraction':float(np.median([x['vessel_fraction'] for x in records]))},indent=2))
if __name__=='__main__': main()
