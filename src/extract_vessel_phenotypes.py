#!/usr/bin/env python3
"""Extract quality-aware bilateral vessel phenotypes from all CIMT fundus images.

Features are deliberately derived from predicted probability maps rather than
treated as ground truth. This preserves segmentation uncertainty for the
downstream risk model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import skeletonize
from torchvision import transforms
from tqdm import tqdm

FEATURES = ['vessel_density','mean_probability','entropy','skeleton_density','branch_density','endpoint_density','component_count','fractal_dimension','sharpness','disc_confidence','roi_vessel_density','roi_skeleton_density','roi_branch_density','roi_mean_width','roi_fractal_dimension']

def fractal_dimension(mask: np.ndarray) -> float:
    sizes = (2, 4, 8, 16, 32, 64)
    counts = []
    for size in sizes:
        h, w = mask.shape; hh, ww = h - h % size, w - w % size
        blocks = mask[:hh,:ww].reshape(hh//size,size,ww//size,size).any(axis=(1,3))
        counts.append(max(int(blocks.sum()), 1))
    return float(np.polyfit(np.log(1/np.asarray(sizes)), np.log(counts), 1)[0])

def one_eye_features(rgb: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    gray = rgb.mean(axis=2) / 255.
    fov = gray > 0.04
    p = np.clip(probability, 1e-5, 1-1e-5)
    mask = (p >= .5) & fov
    # Compute topology on 256px to make the metric stable and tractable.
    mask_small = np.array(Image.fromarray(mask).resize((256,256), Image.NEAREST), dtype=bool, copy=True)
    skeleton = skeletonize(mask_small)
    neighbors = ndi.convolve(skeleton.astype(np.uint8), np.ones((3,3),dtype=np.uint8), mode='constant') - skeleton
    area = float(fov.sum())
    components = label(mask_small, connectivity=2)
    component_sizes = np.bincount(components.ravel())[1:]
    # Disc-referenced annulus (1–3 estimated disc radii). This is an explicit
    # proxy for the standard peripapillary measurement zone; its confidence is
    # retained so incorrect disc localisation is down-weighted downstream.
    red, green, blue = rgb[:,:,0]/255., rgb[:,:,1]/255., rgb[:,:,2]/255.
    disc_score = ndi.gaussian_filter((.55*red + .55*green - .10*blue) * fov, sigma=12)
    cy, cx = np.unravel_index(np.argmax(disc_score), disc_score.shape)
    fov_radius = np.sqrt(fov.sum()/np.pi); disc_radius = max(14., .075*fov_radius)
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]; distance = np.sqrt((yy-cy)**2+(xx-cx)**2)
    roi = (distance >= disc_radius) & (distance <= 3*disc_radius) & fov
    roi_small = np.array(Image.fromarray(roi).resize((256,256), Image.NEAREST),dtype=bool,copy=True)
    width_map = ndi.distance_transform_edt(mask_small) * 2
    roi_skeleton = skeleton & roi_small
    disc_confidence = float((disc_score[cy,cx]-np.median(disc_score[fov]))/(np.std(disc_score[fov])+1e-6))
    return {
        'vessel_density': float(mask.sum()/max(area,1)),
        'mean_probability': float(p[fov].mean()),
        'entropy': float((-(p*np.log(p)+(1-p)*np.log(1-p)))[fov].mean()),
        'skeleton_density': float(skeleton.mean()),
        'branch_density': float((skeleton & (neighbors >= 3)).mean()),
        'endpoint_density': float((skeleton & (neighbors == 1)).mean()),
        'component_count': float((component_sizes >= 8).sum()),
        'fractal_dimension': fractal_dimension(skeleton),
        'sharpness': float(ndi.laplace(gray).var()),
        'disc_confidence': disc_confidence,
        'roi_vessel_density': float((mask & roi).sum()/max(roi.sum(),1)),
        'roi_skeleton_density': float(roi_skeleton.mean()),
        'roi_branch_density': float((roi_skeleton & (neighbors >= 3)).mean()),
        'roi_mean_width': float(width_map[roi_skeleton].mean()) if roi_skeleton.any() else 0.,
        'roi_fractal_dimension': fractal_dimension(roi_skeleton),
    }

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,default=Path('artifacts/manifest.csv')); p.add_argument('--weights',type=Path,default=Path('artifacts/fives_unet_resnet34/best.pt')); p.add_argument('--output',type=Path,default=Path('artifacts/cimt_vessel_phenotypes.csv')); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--limit',type=int,default=None); a=p.parse_args()
    frame=pd.read_csv(a.manifest,dtype={'patient_id':str}); frame=frame if a.limit is None else frame.head(a.limit)
    dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); model=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=1).to(dev); model.load_state_dict(torch.load(a.weights,map_location=dev)); model.eval(); norm=transforms.Normalize((.485,.456,.406),(.229,.224,.225)); rows=[]
    for start in tqdm(range(0,len(frame),a.batch_size),desc='vessel phenotypes'):
        batch=frame.iloc[start:start+a.batch_size]; images=[]; metadata=[]
        for row in batch.itertuples():
            for eye in ('left','right'):
                image=np.asarray(Image.open(getattr(row,f'{eye}_path')).convert('RGB').resize((512,512)),dtype=np.uint8)
                images.append(norm(transforms.ToTensor()(Image.fromarray(image)))); metadata.append((row,eye,image))
        with torch.inference_mode(): probabilities=model(torch.stack(images).to(dev)).sigmoid()[:,0].cpu().numpy()
        per_patient={}
        for (row,eye,image),probability in zip(metadata,probabilities):
            per_patient.setdefault(str(row.patient_id),{'patient_id':str(row.patient_id),'split':row.split,'label':row.label,'cimt_mm':row.cimt_mm,'age':row.age,'sex':row.sex})
            for name,value in one_eye_features(image,probability).items(): per_patient[str(row.patient_id)][f'{eye}_{name}']=value
        for record in per_patient.values():
            for name in FEATURES:
                left,right=record[f'left_{name}'],record[f'right_{name}']; record[f'common_{name}']=(left+right)/2; record[f'asymmetry_{name}']=abs(left-right)
            rows.append(record)
    output=pd.DataFrame(rows).sort_values('patient_id'); a.output.parent.mkdir(parents=True,exist_ok=True); output.to_csv(a.output,index=False)
    print({'patients':len(output),'features':len(FEATURES)*4,'missing':int(output.isna().sum().sum())})
if __name__=='__main__': main()
