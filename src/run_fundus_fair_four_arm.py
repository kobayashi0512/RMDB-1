#!/usr/bin/env python3
"""Fixed exploratory factorial experiment; no training or clinical data access."""
import ast
import hashlib
import json
import re
import subprocess
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
from train_udct_av_pilot import Corrector, softmax_np, hessian_field
from evaluate_fundus_avseg_external import target_labels, score, load_features
from train_birap_v2 import case_metrics
from test_fixed_preprocessing import enhance, write_json

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/Fundus_AVSeg/Fundus-AVSeg'
CACHE=ROOT/'data/fundus_avseg_vascx'
OUT=ROOT/'artifacts/fundus_fair_four_arm'
WEIGHT=ROOT/'models/baseline/artifacts/udct_av_mass_calibrated/best.pt'
ARMS=['vascx','clahe','mass','clahe_mass']
METRICS=['mean_av_dice','artery_dice','vein_dice','vessel_dice','vessel_precision','vessel_recall','thin_centerline_recall','av_ratio_log_error','brier']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def geometry(ident, table):
    raw=np.asarray(Image.open(RAW/'images'/f'{ident}.png').convert('RGB'))
    config=ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)',r'\1',table.loc[ident,'bounds']))
    transform,bounds=CFIBounds.from_dict(raw,config).crop(1024)
    canonical=transform.warp(raw,(1024,1024))
    assert np.array_equal(canonical,np.asarray(Image.open(CACHE/'preprocessed_rgb'/f'{ident}.png')))
    ann=np.asarray(Image.open(RAW/'annotation'/f'{ident}.png').convert('RGB'))
    colours=np.unique(ann.reshape(-1,3),axis=0)
    assert all(tuple(c) in {(0,0,0),(255,0,0),(0,0,255),(0,255,0),(255,255,255)} for c in colours)
    aligned=transform.warp(ann,(1024,1024),mode=Interpolation.NEAREST)
    return canonical,ann,aligned,bounds

def corrected(model,device,rgb,logits):
    # Identical feature definition to the frozen corrector's training loader.
    p=softmax_np(logits)
    ent=-(p*np.log(np.maximum(p,1e-8))).sum(-1)/np.log(4)
    features=np.concatenate([np.moveaxis(rgb.astype(np.float32)/255,-1,0),np.moveaxis(np.clip(logits,-12,12)/6,-1,0),ent[None],hessian_field(rgb)],axis=0).astype(np.float32)
    with torch.inference_mode():
        result=model(torch.from_numpy(features[None]).to(device),torch.from_numpy(np.moveaxis(logits,-1,0).copy()[None]).to(device))[0].softmax(0).cpu().numpy()
    assert np.isfinite(result).all()
    return result

def metrics(p,ann):
    target=target_labels(ann); vessel=ann.max(-1)>0
    result=case_metrics(p,target,vessel)
    original=score(p.argmax(0),target,ann)
    for key in original: assert np.isclose(result[key],original[key],atol=1e-10)
    pred=p.argmax(0)>0;tp=(pred&vessel).sum()
    result['vessel_precision']=float(tp/max(pred.sum(),1))
    result['vessel_recall']=float(tp/max(vessel.sum(),1))
    thin=skeletonize(vessel)&(distance_transform_edt(vessel)<=2)
    result['thin_centerline_recall']=float((pred&thin).sum()/max(thin.sum(),1))
    return result

def paired(delta,metric):
    d=np.asarray(delta,float);rng=np.random.default_rng(20260905)
    b=d[rng.integers(0,len(d),(5000,len(d)))].mean(1)
    return {'delta':float(d.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'n':len(d),'improved':int(((d<0) if metric in ('brier','av_ratio_log_error') else (d>0)).sum())}

def summarize(ids):
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*.json'))]
    if not rows:return
    frame=pd.DataFrame(rows)
    meta=pd.read_csv(ROOT/'results/birap_v2_fundus_avseg/per_image_metrics_with_metadata.csv')[['id','image quality','disease type']].drop_duplicates()
    frame=frame.merge(meta,on='id',validate='many_to_one')
    frame.to_csv(OUT/'per_image_metrics.csv',index=False)
    report={'complete':len(frame)==4*len(ids),'images':frame.id.nunique(),'arm_evaluations':len(frame),'exploratory':True,'overall':{},'contrasts':{},'strata':{}}
    for arm,g in frame.groupby('arm'):report['overall'][arm]={m:float(g[m].mean()) for m in METRICS}
    for col in ('image quality','disease type'):
        report['strata'][col]={str(k):{'n':g.id.nunique(),'arms':{a:{m:float(x[m].mean()) for m in METRICS} for a,x in g.groupby('arm')}} for k,g in frame.groupby(col)}
    for new,base in [('mass','vascx'),('clahe','vascx'),('clahe_mass','clahe'),('clahe_mass','mass'),('clahe_mass','vascx')]:
        report['contrasts'][new+'_minus_'+base]={}
        for m in METRICS:
            p=frame.pivot(index='id',columns='arm',values=m)
            if new in p and base in p:
                d=(p[new]-p[base]).dropna()
                if len(d):report['contrasts'][new+'_minus_'+base][m]=paired(d,m)
    write_json(OUT/'summary.json',report)

def legacy_audit(ids,table,model,device):
    rows=[]
    for i,ident in enumerate(ids):
        canonical,ann,aligned,_=geometry(ident,table)
        features,base,p=load_features(CACHE/'preprocessed_rgb'/f'{ident}.png',CACHE/'logits'/f'{ident}.npy',512)
        with torch.inference_mode():q=model(features[None].to(device),base[None].to(device))[0].softmax(0).cpu().numpy()
        # Confirm our array feature loader exactly matches the historical one.
        rgb512=np.asarray(Image.fromarray(canonical).resize((512,512),Image.Resampling.BILINEAR))
        check=corrected(model,device,rgb512,np.moveaxis(base.numpy(),0,-1).copy())
        assert np.allclose(q,check,atol=1e-6)
        for mode,mask in [('legacy_unregistered',ann),('aligned',aligned)]:
            mask=np.asarray(Image.fromarray(mask).resize((512,512),Image.Resampling.NEAREST))
            for arm,prob in [('vascx',p),('mass',q)]:rows.append({'id':ident,'geometry':mode,'arm':arm,**metrics(prob,mask)})
        if (i+1)%10==0:print('legacy audit',i+1,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'legacy_reproduction.csv',index=False)
    means=frame.groupby(['geometry','arm'])[METRICS].mean()
    report={'size':512,'means':{'/'.join(k):r.to_dict() for k,r in means.iterrows()},'all_100_rgb_crops_reconstructed_exactly':True,'label_decoder':'official green crossing; white uncertain','legacy_invalid_reason':'raw annotation not cropped to prediction coordinates','original_paper_rounded':{'vascx':.5545,'mass':.5635}}
    report['reproduces_paper_to_4_decimals']={a:round(float(means.loc[('legacy_unregistered',a),'mean_av_dice']),4)==v for a,v in report['original_paper_rounded'].items()}
    write_json(OUT/'legacy_reproduction.json',report)
    print(json.dumps(report),flush=True)

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    OUT.mkdir(exist_ok=True,parents=True)
    for d in ('cases','logits','probabilities','predictions'): (OUT/d).mkdir(exist_ok=True)
    table=pd.read_csv(CACHE/'bounds_latest_batch.csv').set_index('id');ids=sorted(table.index)
    assert len(ids)==100
    assert target_labels(np.array([[(0,0,0),(255,0,0),(0,0,255),(0,255,0),(255,255,255)]],np.uint8)).tolist()==[[0,1,2,3,-100]]
    protocol={'ids':ids,'arms':ARMS,'size':768,'inference':{'canonical_size':1024,'tta':False,'tile_batch_size':2,'weights':str(ROOT/'models/third_party/vascx_weights/av_july24.pt')},'mass_weight_sha256':sha(WEIGHT),'clahe':{'clip_limit':1.5,'tile_grid':[8,8],'blend':.5},'label_convention':'red artery; blue vein; green crossing contributes to both AV classes; white ignored only for identity, retained for occupancy','label_source':'https://www.nature.com/articles/s41597-025-05381-2','main_incremental_contrast':'clahe_mass minus clahe','bootstrap':{'unit':'image (dataset publication reports 100 different patients)','replicates':5000,'seed':20260905},'brier_definition':'mean across four classes and all non-uncertain pixels, not sum across classes','thin_definition':'GT vessel skeleton and distance transform radius <=2 at 768; recall only, not specificity','av_error_definition':'absolute log difference of hard artery/vein pixel-count ratios, +1 smoothing, crossings in both, uncertain ignored','selection':'fixed comparisons; CLAHE selected following exploratory earlier arms; not independent confirmation','no_CIMT_or_clinical_access':True}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:write_json(OUT/'protocol.json',protocol)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    corrector=Corrector().to(device).eval();ckpt=torch.load(WEIGHT,map_location=device);corrector.load_state_dict(ckpt['model'])
    if not (OUT/'legacy_reproduction.json').exists():legacy_audit(ids,table,corrector,device)
    model=SegmentationEnsemble.from_torchscript(ROOT/'models/third_party/vascx_weights/av_july24.pt').to(device).eval()
    model.config['inference'].update({'tta':False,'tile_batch_size':2})
    start=time.monotonic()
    for i,ident in enumerate(ids):
        if all((OUT/'cases'/f'{ident}__{a}.json').exists() for a in ARMS):continue
        canonical,_,aligned,bounds=geometry(ident,table)
        ann768=cv2.resize(aligned,(768,768),interpolation=cv2.INTER_NEAREST)
        for input_arm,output_arms in [('default',('vascx','mass')),('clahe',('clahe','clahe_mass'))]:
            if all((OUT/'cases'/f'{ident}__{a}.json').exists() for a in output_arms):continue
            if input_arm=='default':rgbpath=CACHE/'preprocessed_rgb'/f'{ident}.png';cepath=CACHE/'preprocessed_ce'/f'{ident}.png'
            else:
                rgbpath=ROOT/'artifacts/fixed_preprocessing_trial/clahe/rgb'/f'{ident}.png';cepath=ROOT/'artifacts/fixed_preprocessing_trial/clahe/ce'/f'{ident}.png'
                assert np.array_equal(np.asarray(Image.open(rgbpath)),enhance(canonical,'clahe',bounds.make_binary_mask(.01)))
            lp=OUT/'logits'/f'{ident}__{input_arm}.npy'
            if lp.exists():logits=np.load(lp)
            else:
                loader=model._make_inference_dataloader({'images':[{'id':ident,'image':str(rgbpath),'contrast_enhanced':str(cepath)}]},preprocess=False,batch_size=1,num_workers=0,ignore_exceptions=False)
                with torch.inference_mode():
                    output=[x for batch in loader for x in decollate_predict_full(model.predict_step_full(batch))]
                assert len(output)==1
                native=np.asarray(output[0]['logits'],np.float32)
                assert native.shape==(1024,1024,4) and np.isfinite(native).all()
                logits=cv2.resize(native,(768,768));np.save(lp,logits)
            rgb768=np.asarray(Image.open(rgbpath).convert('RGB').resize((768,768),Image.Resampling.BILINEAR))
            probs=[np.moveaxis(softmax_np(logits),-1,0),corrected(corrector,device,rgb768,logits)]
            for arm,prob in zip(output_arms,probs):
                scores=metrics(prob,ann768)
                np.save(OUT/'probabilities'/f'{ident}__{arm}.npy',prob)
                palette=np.array([[0,0,0],[255,0,0],[0,0,255],[0,255,0]],np.uint8)
                Image.fromarray(palette[prob.argmax(0)]).save(OUT/'predictions'/f'{ident}__{arm}.png')
                write_json(OUT/'cases'/f'{ident}__{arm}.json',{'id':ident,'arm':arm,**scores})
            if device.type=='mps':torch.mps.synchronize();torch.mps.empty_cache()
        print(f'{i+1}/100 all four arms completed; {time.monotonic()-start:.1f}s',flush=True)
        if (i+1)%10==0:summarize(ids)
    summarize(ids)
    provenance={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),WEIGHT,ROOT/'src/evaluate_fundus_avseg_external.py',ROOT/'src/train_udct_av_pilot.py',CACHE/'bounds_latest_batch.csv',*sorted((RAW/'images').glob('*.png')),*sorted((RAW/'annotation').glob('*.png'))]}
    write_json(OUT/'sha256_manifest.json',provenance)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
