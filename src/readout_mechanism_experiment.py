"""Versioned exploratory readout mechanism study; no clinical outcomes or training."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import final_measurement_controls as fc
import three_readout_repeatability as tr
from extract_mbrset_repeatability_features import readout_features

OUT=fc.ROOT/'artifacts/readout_mechanism_20260914_v2'
ARMS=['raw','mix25','mix50','mix75','joint']
ALPHA=[0.,.25,.5,.75,1.]

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tr.write(path,obj)

def protocol():
    m=pd.read_csv(fc.OUT/'manifest.csv',dtype=str)
    assert m.groupby('patient_id').split.nunique().max()==1
    p={'status':'Exploratory follow-up; both historical splits already inspected',
       'arms':dict(zip(ARMS,ALPHA)), 'intervention':'o_alpha=(1-alpha)*raw_vessel+alpha*FIVES; original conditional identity fixed',
       'hypotheses':['Occupancy reweighting changes aggregate A/V despite fixed pixel identity',
                     'Partial mixing may moderate the observed trade-off; no winner presumed'],
       'mechanism':'R_new-R_raw=(deltaA-R_raw*deltaV)/V_new, crossing contributes to both masses',
       'strata':'Raw vessel probability: <0.1, [0.1,0.9), >=0.9; descriptive, not clinical calibre',
       'selection':'No best-alpha selection on test; report all five arms on both splits and structural benchmark',
       'endpoints':tr.FEATURES,'controls':'Historical frozen raw/mass/joint/smooth/cleanup retained in separate source tables',
       'bootstrap':{'n':5000,'seed':20260914,'unit':'patient for repeats; image for Fundus','interval':'percentile95; exploratory unadjusted'},
       'reference':'Fundus binary structure only; no truth for soft A/V mass ratio',
       'no_training':True,'no_clinical_outcomes':True,'no_cross_camera_confirmation':True,
       'source_hashes':{str(x):fc.sha(x) for x in [fc.OUT/'manifest.csv',tr.OUT/'per_image.csv',fc.OUT/'fundus_per_image.csv',fc.OCC_WEIGHT,Path(__file__),fc.ROOT/'src/extract_mbrset_repeatability_features.py']}}
    dest=OUT/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==p,'Protocol/source drift: use a new version'
    else:save(dest,p)
    return m,p

def arms(raw,o):
    s=raw[1:].sum(0);q=raw[1:]/np.maximum(s,1e-7)
    for name,a in zip(ARMS,ALPHA):
        v=(1-a)*s+a*o
        yield name,raw if a==0 else np.concatenate([(1-v)[None],v[None]*q],0)

def mechanism(raw,o,fov):
    r=raw.astype(np.float64);s=r[1:].sum(0);q=r[1:]/np.maximum(s,1e-7)
    aa=q[0]+q[2];vv=q[1]+q[2];d=o.astype(np.float64)-s
    A=(s*aa)[fov].mean();V=(s*vv)[fov].mean()
    da=(d*aa)[fov].mean();dv=(d*vv)[fov].mean();R=A/V
    exact=(da-R*dv)/(V+dv);actual=(A+da)/(V+dv)-R
    assert abs(exact-actual)<1e-10
    h=-(q*np.log(np.maximum(q,1e-12))).sum(0)/np.log(3)
    result={'raw_ratio_unclipped':R,'joint_ratio_unclipped':(A+da)/(V+dv),
            'ratio_change':actual,'predicted_ratio_change':exact,'identity_error':float(abs(q.sum(0)-1)[fov].max()),
            'delta_artery_mass':da,'delta_vein_mass':dv,
            'log_artery_change':np.log((A+da)/A),'log_vein_change':np.log((V+dv)/V),
            'occupancy_absolute_update':float(abs(d)[fov].mean()),
            'identity_entropy_update_weighted':float((abs(d)*h)[fov].sum()/max(abs(d)[fov].sum(),1e-12))}
    contribution=d*(aa-R*vv)/(V+dv)
    for name,mask in [('low_support',s<.1),('transition',(s>=.1)&(s<.9)),('core',s>=.9),('uncertain_identity',h>=.5),('confident_identity',h<.5)]:
        result[name+'_ratio_contribution']=float(contribution[fov&mask].sum()/fov.sum())
        result[name+'_absolute_update']=float(abs(d)[fov&mask].sum()/fov.sum())
    return result

def repeat_extract(limit):
    m,p=protocol();cv2.setNumThreads(2)
    table=pd.read_csv(tr.CACHE/'bounds_latest_batch.csv',dtype={'id':str}).set_index('id')
    old=pd.read_csv(tr.OUT/'per_image.csv').set_index('id');start=time.monotonic()
    for i,(_,r) in enumerate(m.iterrows()):
        if limit and i>=limit:break
        dest=OUT/'repeat_cases'/f'{r.id}.json'
        if dest.exists():continue
        r=r.copy()
        if not Path(r['path']).exists():
            suffix=r['path'].split('/rmdb_v2_experiments/',1)[1]
            r['path']=str(fc.ROOT/suffix)
        rgb,z,o,paths=tr.load_inputs(r,table);fov=rgb.max(2)>10
        raw=np.moveaxis(fc.softmax_np(z),-1,0)
        rows=[];maxerr=0.
        for arm,prob in arms(raw,o):
            fc.check_prob(prob);values=readout_features(prob,fov,'')
            if arm in ['raw','joint']:
                err=max(abs(v-old.loc[r.id,arm+'_'+k]) for k,v in values.items())
                assert err<1e-6,(r.id,arm,err);maxerr=max(maxerr,err)
            rows.append({'id':r.id,'patient_id':r.patient_id,'eye':r.eye,'view':r['view'],'split':r.split,'arm':arm,**values})
        save(dest,{'rows':rows,'mechanism':{'id':r.id,'split':r.split,'patient_id':r.patient_id,'eye':r.eye,'view':r['view'],**mechanism(raw,o,fov)},
                   'raw_joint_replay_error':maxerr,'input_sha256':{str(x):fc.sha(x) for x in paths}})
        if (i+1)%50==0:print(f'Repeats {i+1}/{len(m)} {time.monotonic()-start:.1f}s',flush=True)
    cases=[OUT/'repeat_cases'/f'{i}.json' for i in m.id]
    if all(x.exists() for x in cases):
        data=[json.loads(x.read_text()) for x in cases]
        pd.DataFrame([r for x in data for r in x['rows']]).to_csv(OUT/'repeat_per_image.csv',index=False)
        pd.DataFrame([x['mechanism'] for x in data]).to_csv(OUT/'mechanism_per_image.csv',index=False)
        save(OUT/'repeat_complete.json',{'images':len(m),'evaluations':len(m)*len(ARMS),'max_replay_error':max(x['raw_joint_replay_error'] for x in data)})

def fundus_extract(limit):
    protocol();torch.set_num_threads(4);cv2.setNumThreads(2)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net=fc.load_occupancy(device);table=pd.read_csv(fc.FCACHE/'bounds_latest_batch.csv').set_index('id')
    old=pd.read_csv(fc.OUT/'fundus_per_image.csv').set_index(['id','arm'])
    ids=sorted(table.index);start=time.monotonic()
    for i,ident in enumerate(ids):
        if limit and i>=limit:break
        dest=OUT/'fundus_cases'/f'{ident}.json'
        if dest.exists():continue
        canonical,_,ann,_=fc.geometry(ident,table)
        rgb=np.asarray(Image.fromarray(canonical).resize((512,512),Image.Resampling.BILINEAR));fov=rgb.max(2)>10
        truth=(cv2.resize(ann,(512,512),interpolation=cv2.INTER_NEAREST).max(2)>0)&fov
        z=cv2.resize(np.load(fc.FCACHE/'logits'/f'{ident}.npy').astype(np.float32),(512,512),interpolation=cv2.INTER_LINEAR)
        raw=np.moveaxis(fc.softmax_np(z),-1,0)
        x=(np.moveaxis(rgb.astype(np.float32)/255,-1,0)-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
        with torch.inference_mode():o=net(torch.from_numpy(x[None]).to(device)).sigmoid()[0,0].cpu().numpy().astype(np.float16).astype(np.float32)
        ref=fc.reference_features(truth,fov);rows=[]
        for arm,prob in arms(raw,o):
            fc.check_prob(prob);v=readout_features(prob,fov,'');met=fc.retention(fc.binary_mask(prob,fov),truth)
            if arm in ['raw','joint']:
                assert max(abs(met[k]-old.loc[(ident,arm),k]) for k in met)<1e-5,(ident,arm)
            rows.append({'id':ident,'arm':arm,**met,**{'abs_error_'+k:abs(v[k]-ref[k]) for k in fc.STRUCTURE}})
        save(dest,{'rows':rows,'mechanism':{'id':ident,**mechanism(raw,o,fov)}})
        if (i+1)%10==0:print(f'Fundus {i+1}/{len(ids)} {time.monotonic()-start:.1f}s',flush=True)
    cases=[OUT/'fundus_cases'/f'{i}.json' for i in ids]
    if all(x.exists() for x in cases):
        data=[json.loads(x.read_text()) for x in cases]
        pd.DataFrame([r for x in data for r in x['rows']]).to_csv(OUT/'fundus_per_image.csv',index=False)
        save(OUT/'fundus_complete.json',{'images':len(ids),'evaluations':len(ids)*len(ARMS)})

def analyze():
    _,p=protocol();f=pd.read_csv(OUT/'repeat_per_image.csv',dtype={'patient_id':str});est=[];contr=[]
    for split,g in f.groupby('split'):
        for feature in tr.FEATURES:
            w=g.pivot(index=['patient_id','eye','view','id'],columns='arm',values=feature)[ARMS].reset_index()
            assert np.isfinite(w[ARMS].to_numpy()).all(),feature
            rows=[]
            for (pid,eye),v in w.groupby(['patient_id','eye']):
                if len(v)<2:continue
                a=v.sort_values(['view','id'])[ARMS].to_numpy();mu=a.mean(0);d=a-mu
                rows.append((pid,len(a),mu,(d*d).sum(0),abs(d).mean(0),a[1]-a[0]))
            counts=np.array([x[1] for x in rows],float);means=np.stack([x[2] for x in rows]);ss=np.stack([x[3] for x in rows]);mad=np.stack([x[4] for x in rows]);diff=np.stack([x[5] for x in rows])
            u,ix=np.unique([x[0] for x in rows],return_inverse=True)
            rng=np.random.default_rng(p['bootstrap']['seed']);weights=rng.multinomial(len(u),np.ones(len(u))/len(u),size=p['bootstrap']['n'])[:,ix].astype(float)
            pt=tr.estimate(np.ones((1,len(rows))),counts,means,ss,mad,2*mad)
            bt=tr.estimate(weights,counts,means,ss,mad,2*mad)
            for j,arm in enumerate(ARMS):
                for met in ['icc','within_eye_mad','within_eye_sd','between_eye_sd']:
                    lo,hi=np.quantile(bt[met][:,j],[.025,.975]);est.append({'split':split,'feature':feature,'arm':arm,'metric':met,'estimate':pt[met][0,j],'ci_low':lo,'ci_high':hi,'eyes':len(rows),'patients':len(u)})
                    if j:
                        delta=bt[met][:,j]-bt[met][:,0];lo,hi=np.quantile(delta,[.025,.975]);contr.append({'split':split,'feature':feature,'arm':arm,'reference':'raw','metric':met,'delta':pt[met][0,j]-pt[met][0,0],'ci_low':lo,'ci_high':hi})
                for met,value in [('pair_bias',diff[:,j].mean()),('loa_low',diff[:,j].mean()-1.96*diff[:,j].std(ddof=1)),('loa_high',diff[:,j].mean()+1.96*diff[:,j].std(ddof=1)),('eye_mean_sd',means[:,j].std(ddof=1)),('eye_mean_iqr',np.subtract(*np.quantile(means[:,j],[.75,.25])))]:
                    est.append({'split':split,'feature':feature,'arm':arm,'metric':met,'estimate':value,'eyes':len(rows),'patients':len(u)})
            print(f'Statistics {split} {feature}',flush=True)
    pd.DataFrame(est).to_csv(OUT/'repeat_estimates.csv',index=False);pd.DataFrame(contr).to_csv(OUT/'repeat_contrasts.csv',index=False)
    f=pd.read_csv(OUT/'fundus_per_image.csv');rows=[]
    for met in f.columns.drop(['id','arm']):
        w=f.pivot(index='id',columns='arm',values=met)[ARMS].to_numpy();rng=np.random.default_rng(p['bootstrap']['seed']);boot=w[rng.integers(0,len(w),(5000,len(w)))].mean(1)
        for j,a in enumerate(ARMS):
            lo,hi=np.quantile(boot[:,j],[.025,.975]);dl,dh=np.quantile(boot[:,j]-boot[:,0],[.025,.975]);rows.append({'metric':met,'arm':a,'estimate':w[:,j].mean(),'ci_low':lo,'ci_high':hi,'delta_raw':(w[:,j]-w[:,0]).mean(),'delta_ci_low':dl,'delta_ci_high':dh})
    pd.DataFrame(rows).to_csv(OUT/'fundus_estimates.csv',index=False)
    m=pd.read_csv(OUT/'mechanism_per_image.csv');save(OUT/'mechanism_summary.json',{'formula_max_error':float(abs(m.ratio_change-m.predicted_ratio_change).max()),'descriptive_means_by_split':m.groupby('split').mean(numeric_only=True).to_dict(),'interpretation':'Algebraic identity verified, NOT proof of causal source of repeat error; strata descriptive; no clinical accuracy truth'})
    save(OUT/'analysis_complete.json',{'bootstrap':5000,'selection_performed':False,'new_independent_confirmation':False})

def tests():
    rng=np.random.default_rng(12);p=rng.random((4,512,512),dtype=np.float32);p/=p.sum(0);o=rng.random((512,512),dtype=np.float32);fov=np.ones((512,512),bool)
    v=dict(arms(p,o));assert np.array_equal(v['raw'],p)
    for x in v.values():fc.check_prob(x)
    m=mechanism(p,o,fov);assert abs(m['ratio_change']-m['predicted_ratio_change'])<1e-10
    assert abs(sum(m[x+'_ratio_contribution'] for x in ['low_support','transition','core'])-m['ratio_change'])<1e-10
    identical=dict(arms(p,p[1:].sum(0)))
    assert max(float(abs(x-p).max()) for x in identical.values())<3e-7
    print('PASS: probability normalization, raw identity, no-change control, exact ratio decomposition, stratum additivity')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['test','prepare','repeat','fundus','analyze']);a.add_argument('--limit',type=int,default=0);args=a.parse_args()
    if args.stage=='test':tests()
    elif args.stage=='prepare':protocol()
    elif args.stage=='repeat':repeat_extract(args.limit)
    elif args.stage=='fundus':fundus_extract(args.limit)
    else:analyze()
