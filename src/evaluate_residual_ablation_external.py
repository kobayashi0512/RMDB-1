"""Frozen 21-checkpoint external structure/repeatability extraction; no selection."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import final_measurement_controls as fc
import three_readout_repeatability as tr
from train_controlled_residual_ablation import ARMS, SEEDS, OUT as TRAIN, model_for, apply_mask, feature_tensor
from extract_mbrset_repeatability_features import readout_features

OUT = fc.ROOT / 'artifacts/residual_external_20260914_v1'
LABELS = ['raw', 'legacy_mass'] + [f'{a}__{s}' for a in ARMS for s in SEEDS]
BATCH = 4

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tr.write(path, value)

def prepare():
    assert json.loads((TRAIN/'checkpoint_audit.json').read_text())['all_21_verified']
    m = pd.read_csv(fc.OUT/'manifest.csv', dtype=str)
    assert len(m)==1960 and m.groupby('patient_id').split.nunique().max()==1
    fids = sorted(pd.read_csv(fc.FCACHE/'bounds_latest_batch.csv').id.tolist())
    weights = {f'{a}__{s}': TRAIN/'runs'/a/str(s)/'last.pt' for a in ARMS for s in SEEDS}
    for label, path in weights.items():
        c=json.loads((path.parent/'complete.json').read_text())
        assert c['checkpoint_sha256']==fc.sha(path)
    files=[Path(__file__),fc.ROOT/'src/train_controlled_residual_ablation.py',
           fc.ROOT/'src/train_udct_av_pilot.py',fc.ROOT/'src/extract_mbrset_repeatability_features.py',
           fc.ROOT/'src/final_measurement_controls.py',fc.ROOT/'src/three_readout_repeatability.py',
           fc.OUT/'manifest.csv',fc.OUT/'fundus_per_image.csv',tr.OUT/'per_image.csv',TRAIN/'training_protocol.json']
    p={'purpose':'Evaluate ALL21 final checkpoints without further fitting or selection',
       'labels':LABELS,'arms':ARMS,'seeds':SEEDS,'fundus_ids':fids,'repeat_ids':m.id.tolist(),
       'checkpoint_hashes':{k:fc.sha(v) for k,v in weights.items()},
       'source_hashes':{str(v):fc.sha(v) for v in files},'batch_size':BATCH,'precision':'float32',
       'geometry':'Exactly replayed original canonical RGB;512 PIL bilinear RGB,cv2 linear logit resize THEN softmax;RGBmax>10 FOV;256 skeleton;vessel marginal>=0.5',
       'structural_endpoints':list(fc.retention(np.eye(4,dtype=bool),np.eye(4,dtype=bool)))+['abs_error_'+k for k in fc.STRUCTURE],
       'repeat_endpoints':tr.FEATURES,'controls':'raw recomputed and replayed; legacy_mass reuses audited historical same-operator per-image features and structure rows',
       'statistics':'5000 paired patient-cluster bootstrap for repeats; paired image bootstrap for Fundus; each seed separate then mean of metric estimates, NOT probability ensemble; all arms/seed results retained',
       'comparators':'each checkpoint vs raw and legacy mass; within-seed ablations vs full; seed-mean paired intervals also resample seeds',
       'limitations':['Both external datasets previously inspected; exploratory only','mBRSET public mirror224 pixels, not highresolution confirmation','New default-loss retraining is not exact legacy mass recipe','no clinical outcomes or gate success claim','separate cohorts cannot establish within-patient accuracy-repeatability relation'],
       'stop_rule':'Complete all fixed arms once; no ranking-based model replacement or extra training'}
    path=OUT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==p,'Frozen protocol drift'
    else: save(path,p)
    return m,fids,weights

def models(weights,device):
    result=[]
    for label,path in weights.items():
        arm=label.rsplit('__',1)[0];net=model_for(arm)
        net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        result.append((label,arm,net.to(device).eval()))
    return result

def load_repeat(row,table):
    row=row.copy()
    if not Path(row['path']).exists():
        suffix=row['path'].split('/rmdb_v2_experiments/',1)[1];row['path']=str(fc.ROOT/suffix)
    rgb,z,_,paths=tr.load_inputs(row,table)
    return {'meta':{k:row[k] for k in ['id','patient_id','eye','view','split']},'rgb':rgb,'z':z,
            'input_hashes':{str(p):fc.sha(p) for p in paths}}

def load_fundus(key,table):
    rgb,_,ann,_=fc.geometry(key,table)
    rgb=np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR))
    zpath=fc.FCACHE/'logits'/(key+'.npy')
    z=cv2.resize(np.load(zpath).astype(np.float32),(512,512),interpolation=cv2.INTER_LINEAR)
    truth=(cv2.resize(ann,(512,512),interpolation=cv2.INTER_NEAREST).max(2)>0)&(rgb.max(2)>10)
    return {'meta':{'id':key},'rgb':rgb,'z':z,'truth':truth,
            'input_hashes':{str(p):fc.sha(p) for p in [zpath,fc.FCACHE/'preprocessed_rgb'/(key+'.png'),fc.FRAW/'annotation'/(key+'.png')]}}

def values(prob,case,stage):
    fc.check_prob(prob);fov=case['rgb'].max(2)>10
    v=readout_features(prob,fov,'')
    if stage=='repeat':return v
    met=fc.retention(fc.binary_mask(prob,fov),case['truth'])
    return {**met,**{'abs_error_'+k:abs(v[k]-case['ref'][k]) for k in fc.STRUCTURE}}

def extract(stage,limit=0):
    m,fids,weights=prepare();torch.set_num_threads(2);cv2.setNumThreads(1)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    nets=models(weights,device)
    old=pd.read_csv(tr.OUT/'per_image.csv',dtype={'id':str}).set_index('id') if stage=='repeat' else pd.read_csv(fc.OUT/'fundus_per_image.csv').set_index(['id','arm'])
    table=pd.read_csv((tr.CACHE if stage=='repeat' else fc.FCACHE)/'bounds_latest_batch.csv',dtype={'id':str}).set_index('id')
    records=[r for _,r in m.iterrows()] if stage=='repeat' else fids
    ids=m.id.tolist() if stage=='repeat' else fids
    selected=records[:limit] if limit else records
    missing=[r for r in selected if not (OUT/(stage+'_cases')/((r['id'] if stage=='repeat' else r)+'.json')).exists()]
    start=time.monotonic();done=0
    for pos in range(0,len(missing),BATCH):
        cases=[load_repeat(r,table) if stage=='repeat' else load_fundus(r,table) for r in missing[pos:pos+BATCH]]
        for c in cases:
            if stage=='fundus':c['ref']=fc.reference_features(c['truth'],c['rgb'].max(2)>10)
            c['rows']=[];key=c['meta']['id'];raw=np.moveaxis(fc.softmax_np(c['z']),-1,0)
            rawvalues=values(raw,c,stage)
            if stage=='repeat':
                err=max(abs(v-old.loc[key,'raw_'+k]) for k,v in rawvalues.items())
                historical={k:float(old.loc[key,'mass_'+k]) for k in tr.FEATURES}
            else:
                err=max(abs(v-old.loc[(key,'raw'),k]) for k,v in rawvalues.items())
                historical={k:float(old.loc[(key,'mass'),k]) for k in rawvalues}
            assert err<1e-6,(stage,key,err)
            c['raw_replay_error']=err
            c['rows']=[{**c['meta'],'label':'raw',**rawvalues},{**c['meta'],'label':'legacy_mass',**historical}]
        x=torch.stack([feature_tensor(c['rgb'],c['z']) for c in cases]).to(device)
        z=torch.stack([torch.from_numpy(c['z']).permute(2,0,1) for c in cases]).to(device)
        for label,arm,net in nets:
            with torch.inference_mode():prob=net(apply_mask(x,arm),z).softmax(1).cpu().numpy()
            for i,c in enumerate(cases):c['rows'].append({**c['meta'],'label':label,**values(prob[i],c,stage)})
        for c in cases:
            assert len(c['rows'])==23
            save(OUT/(stage+'_cases')/(c['meta']['id']+'.json'),{'rows':c['rows'],'raw_replay_error':c['raw_replay_error'],'input_hashes':c['input_hashes']})
        done+=len(cases)
        elapsed=time.monotonic()-start
        save(OUT/(stage+'_progress.json'),{'new_images':done,'requested_missing':len(missing),'seconds':elapsed,'seconds_per_image':elapsed/done})
        print(stage,done,'/',len(missing),'new images',round(elapsed,1),'seconds',flush=True)
    files=[OUT/(stage+'_cases')/(key+'.json') for key in ids]
    if all(p.exists() for p in files):
        allcases=[json.loads(p.read_text()) for p in files]
        frame=pd.DataFrame([r for c in allcases for r in c['rows']]);frame.to_csv(OUT/(stage+'_per_image.csv'),index=False)
        save(OUT/(stage+'_complete.json'),{'images':len(ids),'rows':len(frame),'new_model_inferences':len(ids)*21,'legacy_control_reused':True,'max_raw_replay_error':max(c['raw_replay_error'] for c in allcases)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fundus','repeat']);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:extract(a.stage,a.limit)
