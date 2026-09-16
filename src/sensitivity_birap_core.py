#!/usr/bin/env python3
"""Predefined preprocessing/region/subgroup sensitivity checks for core BiRAP signals.

Uses development-set OOF age/sex residuals only; the locked test group is excluded.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

def label(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).resize((256,256),Image.Resampling.NEAREST),dtype=np.uint8)

def fractal(binary: np.ndarray) -> float:
    pts=np.argwhere(binary)
    if len(pts)<20:return np.nan
    sizes=np.array([2,4,8,16,32,64]); counts=[]
    for s in sizes: counts.append(len(np.unique(np.floor(pts/s).astype(int),axis=0)))
    return float(np.polyfit(np.log(1/sizes),np.log(np.maximum(counts,1)),1)[0])

def width(mask: np.ndarray, region: np.ndarray) -> float:
    m=mask&region; sk=skeletonize(m)
    if sk.sum()<10:return np.nan
    return float((2*ndi.distance_transform_edt(m)[sk]).mean())

def perturb(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind=='erode1': return ndi.binary_erosion(mask,structure=np.ones((3,3)))
    if kind=='dilate1': return ndi.binary_dilation(mask,structure=np.ones((3,3)))
    return mask

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--anatomy',type=Path,default=Path('artifacts/vascx_anatomy'))
    ap.add_argument('--cache',type=Path,default=Path('artifacts/bimfo_v2_anatomy_cache/anatomy.csv'))
    ap.add_argument('--residuals',type=Path,default=Path('artifacts/birap_stability/development_oof_residuals.csv'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/birap_sensitivity')); a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(a.cache,dtype={'patient_id':str}); res=pd.read_csv(a.residuals,dtype={'patient_id':str})
    # Only the original development patients are in residuals.
    meta=meta[meta.patient_id.isin(res.patient_id)].copy(); records=[]
    for r in meta.itertuples(index=False):
        av=label(a.anatomy/'artery_vein'/f'{r.id}.png')
        rgb=np.asarray(Image.open(a.anatomy/'preprocessed_rgb'/f'{r.id}.png').resize((256,256),Image.Resampling.BILINEAR),dtype=np.uint8)
        if r.eye=='right': av=np.ascontiguousarray(av[:,::-1]);rgb=np.ascontiguousarray(rgb[:,::-1])
        fov=rgb.max(axis=2)>10; yy,xx=np.indices((256,256)); valid=bool(r.landmark_geometry_valid)
        radius=np.hypot(xx-r.disc_x,yy-r.disc_y)/(r.disc_diameter/2)
        regions={'outer_2to4':fov&(radius>=2)&(radius<4),'outer_1p5to3p5':fov&(radius>=1.5)&(radius<3.5),'outer_2p5to4p5':fov&(radius>=2.5)&(radius<4.5)} if valid else {}
        for mode in ['none','erode1','dilate1']:
            art=perturb(av==1,mode); vein=perturb(av==2,mode)
            ad=float(art[fov].mean());vd=float(vein[fov].mean())
            row={'patient_id':r.patient_id,'eye':r.eye,'mode':mode,'av_density_ratio':ad/max(vd,1e-6),'av_width_ratio':width(art,fov)/max(width(vein,fov),1e-6)}
            for name,region in regions.items(): row[f'vein_{name}_fractal']=fractal(skeletonize(vein&region))
            records.append(row)
    eye=pd.DataFrame(records); joined=[]
    for (pid,mode),g in eye.groupby(['patient_id','mode']):
        if set(g.eye)!={'left','right'}:continue
        row={'patient_id':pid,'mode':mode}
        for col in [x for x in g.columns if x not in {'patient_id','eye','mode'}]: row[col]=g[col].mean()
        joined.append(row)
    patient=pd.DataFrame(joined).merge(res[['patient_id','age','sex','age_sex_oof_residual']],on='patient_id',validate='many_to_one')
    rows=[]
    for mode,part in patient.groupby('mode'):
        for feature in ['av_density_ratio','av_width_ratio','vein_outer_2to4_fractal','vein_outer_1p5to3p5_fractal','vein_outer_2p5to4p5_fractal']:
            if feature not in part:continue
            for subgroup,sub in [('all',part),('female',part[part.sex==0]),('male',part[part.sex==1]),('younger',part[part.age<=part.age.median()]),('older',part[part.age>part.age.median()])]:
                q=sub[[feature,'age_sex_oof_residual']].dropna()
                if len(q)>=100:
                    rho,p=spearmanr(q[feature],q.age_sex_oof_residual);rows.append({'mode':mode,'feature':feature,'subgroup':subgroup,'n':len(q),'rho':rho,'p':p})
    result=pd.DataFrame(rows);result.to_csv(a.output/'sensitivity_associations.csv',index=False)
    summary={'development_only':True,'held_out_test_used':False,'morphology_perturbations':['none','erode1','dilate1'],'outer_annuli_dd':['2-4','1.5-3.5','2.5-4.5'],'subgroups':['sex','median age'],'note':'Sensitivity checks are confirmatory only for direction robustness; no prediction claim is made.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(result.to_string(index=False))
if __name__=='__main__':main()
