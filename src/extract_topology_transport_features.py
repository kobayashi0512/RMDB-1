#!/usr/bin/env python3
"""Outcome-free minimal topology/transport features from bilateral vessel maps.

This is a cheap feasibility gate before implementing differentiable persistent
homology or a trainable optimal-transport network.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance
from skimage.measure import euler_number
from skimage.morphology import skeletonize
from tqdm import tqdm

THRESHOLDS=np.array([.25,.40,.55,.70],dtype=np.float32)

def circular_emd(p: np.ndarray,q: np.ndarray) -> float:
    d=np.cumsum(p-q);return float(np.abs(d-np.median(d)).sum()/len(d))

def eye_features(prob: np.ndarray,av: np.ndarray) -> tuple[dict,np.ndarray,np.ndarray]:
    out={}; series={k:[] for k in ['density','components','largest','euler','skeleton']}
    for t in THRESHOLDS:
        mask=prob>=t; labels,n=ndi.label(mask); sizes=np.bincount(labels.ravel())[1:]
        sk=skeletonize(mask)
        vals={'density':mask.mean(),'components':n/100.0,'largest':(sizes.max()/max(mask.sum(),1)) if len(sizes) else 0.0,'euler':euler_number(mask)/100.0,'skeleton':sk.mean()}
        for k,v in vals.items(): series[k].append(float(v));out[f'{k}_t{int(t*100)}']=float(v)
    for k,v in series.items():
        out[f'{k}_mean']=float(np.mean(v));out[f'{k}_std']=float(np.std(v));out[f'{k}_slope']=float(np.polyfit(THRESHOLDS,v,1)[0])
    # Tangent direction is perpendicular to the probability gradient.
    gy,gx=np.gradient(prob.astype(np.float32));weight=np.hypot(gx,gy)*(prob>=.35)
    angle=np.mod(np.arctan2(gy,gx)+np.pi/2,np.pi);hist,_=np.histogram(angle,bins=12,range=(0,np.pi),weights=weight)
    hist=(hist+1e-6)/(hist.sum()+12e-6)
    sk=skeletonize(prob>=.40);dist=ndi.distance_transform_edt(prob>=.40);width=(2*dist[sk]).astype(np.float32)
    if len(width)<10: width=np.array([0.],dtype=np.float32)
    for i,v in enumerate(hist):out[f'orientation_{i}']=float(v)
    for q in [.1,.25,.5,.75,.9]:out[f'width_q{int(q*100)}']=float(np.quantile(width,q))
    out['artery_density']=float((av==1).mean());out['vein_density']=float((av==2).mean());out['crossing_density']=float((av==3).mean())
    out['av_density_ratio']=out['artery_density']/max(out['vein_density'],1e-6)
    return out,hist,width

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('artifacts/bimfo_v1_1_cache'));p.add_argument('--anatomy-cache',type=Path,default=Path('artifacts/bimfo_v2_anatomy_cache'));p.add_argument('--output',type=Path,default=Path('artifacts/topology_transport_features'));p.add_argument('--limit',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    patients=pd.read_csv(a.cache/'patients.csv',dtype={'patient_id':str});prob=np.load(a.cache/'vessel_probability_maps.npy',mmap_mode='r');labels=np.load(a.anatomy_cache/'anatomy_labels.npy',mmap_mode='r')
    if a.limit: patients=patients.iloc[:a.limit]
    records=[]
    for row in tqdm(patients.itertuples(),total=len(patients),desc='Topology-transport'):
        eyes=[];hists=[];widths=[]
        for eye in range(2):
            f,h,w=eye_features(np.asarray(prob[row.Index,eye],dtype=np.float32),np.asarray(labels[row.Index,eye,0],dtype=np.uint8));eyes.append(f);hists.append(h);widths.append(w)
        rec={'patient_id':str(row.patient_id),'cache_index':int(row.Index)}
        for k in eyes[0]:rec[f'common_{k}']=(eyes[0][k]+eyes[1][k])/2;rec[f'asymmetry_{k}']=abs(eyes[0][k]-eyes[1][k])
        rec['transport_orientation_circular_emd']=circular_emd(hists[0],hists[1]);rec['transport_width_wasserstein']=float(wasserstein_distance(widths[0],widths[1]))
        topo_keys=[k for k in eyes[0] if k.startswith(('components_','largest_','euler_','skeleton_'))]
        rec['transport_topology_l2']=float(np.linalg.norm([eyes[0][k]-eyes[1][k] for k in topo_keys])/np.sqrt(len(topo_keys)))
        records.append(rec)
    out=pd.DataFrame(records);out.to_csv(a.output/'features.csv',index=False)
    metadata={'patients':len(out),'features':len(out.columns)-2,'thresholds':THRESHOLDS.tolist(),'outcome_used':False,'test_used':False,'interpretation':'Multithreshold topology summaries plus bilateral circular-orientation, width, and topology transport distances; feasibility proxy, not full persistent homology.'}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
