"""One bounded exploratory gate and an image-wise update-budget-matched control."""
import argparse
import json
import time
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import readout_mechanism_experiment as base
from extract_mbrset_repeatability_features import readout_features
fc=base.fc;tr=base.tr
SOURCE=base.OUT
OUT=fc.ROOT/'artifacts/readout_confidence_20260914_v1'
ARMS=['raw','joint','confidence','budget_matched']

def prepare():
    m=pd.read_csv(fc.OUT/'manifest.csv',dtype=str)
    p={'status':'Exploratory follow-up; historical development and test already inspected',
       'basis':'Phase-one log-ratio decomposition establishes additional paired reweighting fluctuations; entropy gating is an unproven candidate, not a confirmed mechanism',
       'candidate':'g=1-H(q)/log(3); v=s+g*(o-s); original raw identity q retained',
       'control':'alpha_image=sum_FOV(g*abs(o-s))/sum_FOV(abs(o-s)); v=s+alpha_image*(o-s)',
       'matched_quantity':'Exact FOV integrated absolute occupancy update, not mean gate or target A/V ratio',
       'no_fitting':True,'no_test_selection':True,'no_ratio_restoration':True,
       'arms':ARMS,'primary_comparator':'budget_matched','reporting':'all12 endpoints; adverse results retained; compare structure and between-eye variability',
       'bootstrap':{'n':5000,'seed':20260914},
       'stop':'Single gate vs matched/simple baselines; no output-driven further gate search',
       'hashes':{str(x):fc.sha(x) for x in [Path(__file__),Path(base.__file__),SOURCE/'protocol.json',SOURCE/'repeat_per_image.csv',SOURCE/'fundus_per_image.csv']}}
    dest=OUT/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==p
    else:base.save(dest,p)
    return m,p

def variants(raw,o,fov):
    s=raw[1:].sum(0);q=raw[1:]/np.maximum(s,1e-7)
    h=-(q*np.log(np.maximum(q,1e-12))).sum(0)/np.log(3.)
    gate=np.clip(1-h,0,1);d=o-s
    alpha=float((gate*abs(d))[fov].sum(dtype=np.float64)/max(abs(d)[fov].sum(dtype=np.float64),1e-12))
    vals={'raw':s,'joint':o,'confidence':s+gate*d,'budget_matched':s+alpha*d}
    probs={a:raw if a=='raw' else np.concatenate([(1-v)[None],v[None]*q],0).astype(np.float32) for a,v in vals.items()}
    budgets={a:float(abs(v-s)[fov].sum(dtype=np.float64)) for a,v in vals.items()}
    assert abs(budgets['confidence']-budgets['budget_matched'])<max(1e-3,1e-6*budgets['confidence'])
    return probs,{'alpha_image':alpha,'update_budgets':budgets}

def extract(which):
    m,_=prepare();torch.set_num_threads(4);cv2.setNumThreads(2);start=time.monotonic()
    if which=='repeat':
        table=pd.read_csv(tr.CACHE/'bounds_latest_batch.csv',dtype={'id':str}).set_index('id')
        items=list(m.iterrows());old=pd.read_csv(SOURCE/'repeat_per_image.csv').set_index(['id','arm'])
    else:
        table=pd.read_csv(fc.FCACHE/'bounds_latest_batch.csv').set_index('id');items=list(enumerate(sorted(table.index)))
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');net=fc.load_occupancy(device)
        old=pd.read_csv(SOURCE/'fundus_per_image.csv').set_index(['id','arm'])
    for i,(_,r) in enumerate(items):
        ident=r.id if which=='repeat' else r;dest=OUT/(which+'_cases')/f'{ident}.json'
        if dest.exists():continue
        if which=='repeat':
            r=r.copy()
            if not Path(r['path']).exists():r['path']=str(fc.ROOT/r['path'].split('/rmdb_v2_experiments/',1)[1])
            rgb,z,o,paths=tr.load_inputs(r,table);fov=rgb.max(2)>10
            metadata={k:r[k] for k in ['id','patient_id','eye','view','split']}
        else:
            canonical,_,ann,_=fc.geometry(ident,table);rgb=np.asarray(Image.fromarray(canonical).resize((512,512),Image.Resampling.BILINEAR));fov=rgb.max(2)>10
            truth=(cv2.resize(ann,(512,512),interpolation=cv2.INTER_NEAREST).max(2)>0)&fov
            z=cv2.resize(np.load(fc.FCACHE/'logits'/f'{ident}.npy').astype(np.float32),(512,512),interpolation=cv2.INTER_LINEAR)
            x=(np.moveaxis(rgb.astype(np.float32)/255,-1,0)-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
            with torch.inference_mode():o=net(torch.from_numpy(x[None]).to(device)).sigmoid()[0,0].cpu().numpy().astype(np.float16).astype(np.float32)
            ref=fc.reference_features(truth,fov);metadata={'id':ident}
        raw=np.moveaxis(fc.softmax_np(z),-1,0);probs,audit=variants(raw,o,fov);rows=[]
        for arm,prob in probs.items():
            fc.check_prob(prob);values=readout_features(prob,fov,'')
            if which=='fundus':values={**fc.retention(fc.binary_mask(prob,fov),truth),**{'abs_error_'+k:abs(values[k]-ref[k]) for k in fc.STRUCTURE}}
            if arm in ['raw','joint']:assert max(abs(v-old.loc[(ident,arm),k]) for k,v in values.items())<1e-5
            rows.append({**metadata,'arm':arm,**values})
        base.save(dest,{'rows':rows,'audit':audit})
        if (i+1)%(50 if which=='repeat' else 10)==0:print(f'{which} {i+1}/{len(items)} {time.monotonic()-start:.1f}s',flush=True)
    data=[json.loads(x.read_text()) for x in sorted((OUT/(which+'_cases')).glob('*.json'))]
    assert len(data)==len(items)
    pd.DataFrame([r for x in data for r in x['rows']]).to_csv(OUT/(which+'_per_image.csv'),index=False)
    base.save(OUT/(which+'_complete.json'),{'images':len(data),'max_budget_difference':max(abs(x['audit']['update_budgets']['confidence']-x['audit']['update_budgets']['budget_matched']) for x in data)})

def analyze():
    # Shared analyzer's mechanism summary is explicitly the phase-one raw/joint
    # decomposition, not a decomposition of the new gated variants.
    shutil.copy2(SOURCE/'mechanism_per_image.csv',OUT/'mechanism_per_image.csv')
    base.save(OUT/'mechanism_source.json',{'source':str(SOURCE/'mechanism_per_image.csv'),'scope':'Raw versus joint only; copied for shared analyzer, not gated-mechanism evidence'})
    base.OUT=OUT;base.ARMS=ARMS;base.protocol=prepare
    base.analyze()

def test():
    rng=np.random.default_rng(9);r=rng.random((4,512,512),dtype=np.float32);r/=r.sum(0);o=rng.random((512,512),dtype=np.float32);f=np.ones((512,512),bool)
    probs,a=variants(r,o,f)
    for p in probs.values():fc.check_prob(p)
    assert np.array_equal(probs['raw'],r)
    print('PASS: probability validity and exact absolute-update budget match',a['alpha_image'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','test','repeat','fundus','analyze']);a=p.parse_args()
    if a.stage=='test':test()
    elif a.stage=='prepare':prepare()
    elif a.stage=='analyze':analyze()
    else:extract(a.stage)
