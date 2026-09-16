"""Frozen, outcome-blind three-readout phenotype repeatability comparison.

Recompute all readouts under the frozen corrector's logit-resize convention.
No optimizer, model selection, clinical label access, or CIMT data access.
"""
import argparse
import ast
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rtnls_fundusprep.cfi_bounds import CFIBounds
from run_fundus_fair_four_arm import ROOT, WEIGHT, corrected, Corrector, softmax_np, sha
from extract_mbrset_repeatability_features import readout_features

OUT=ROOT/'artifacts/three_readout_repeatability'
CACHE=ROOT/'artifacts/mbrset_public_vascx'
OCC=ROOT/'artifacts/mbrset_public_occupancy'
SOURCE=ROOT/'artifacts/mbrset_repeatability_split/per_image.csv'
ARMS=['raw','mass','joint']
FEATURES=['vessel_mass','artery_mass','vein_mass','av_ratio','vessel_density',
          'skeleton_density','branch_density','endpoint_density','component_count',
          'fractal_dimension','occupancy_entropy','identity_entropy']
ALLOWED=['id','patient_id','eye','view','split','file','path']


def write(path,obj):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(obj),indent=2,allow_nan=False))
    tmp.replace(path)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    frame=pd.read_csv(SOURCE,usecols=ALLOWED,dtype=str)
    assert len(frame)==4884 and frame.id.nunique()==4884
    assert frame.groupby('patient_id').split.nunique().max()==1
    frame=frame[frame.split.isin(['development','test'])].sort_values(['split','patient_id','eye','view','id'])
    assert len(frame)==1960 and frame.file.nunique()==1960
    protocol={'purpose':'same images, same phenotype operators: raw VascX versus frozen Mass UDCT versus occupancy-conditioned VascX',
              'exploratory':True,'selection':'existing development/test splits already inspected; no new independent test claim',
              'primary_reporting_split':'test','other_split':'development, descriptive only; NOT used to choose models or gates',
              'ids':frame.id.tolist(),'arms':ARMS,'features':FEATURES,
              'contrasts':['mass_minus_raw','joint_minus_raw','mass_minus_joint'],
              'all_features_reported':True,'threshold':.5,'size':512,'skeleton_grid':256,
              'probability_rule':'cv2 INTER_LINEAR resize native float32 logits to512, THEN softmax. All arms share this base.',
              'mass':'original frozen two-step Corrector, original 11-channel features, no fitting',
              'joint':'[1-O, O*P(A,V,C|vessel)] from raw VascX; no Mass UDCT in this arm',
              'phenotype_operator':'unchanged readout_features from extract_mbrset_repeatability_features.py',
              'legacy_bridge':'legacy raw/joint first softmax then resized probability; never concatenate with new Mass as a fair comparison',
              'geometry':'all RGB replayed from source via saved image-only bounds to1024; RGB512 PIL bilinear; same RGBmax>10 field across arms',
              'occupancy':'existing512 cached occupancy; frozen model replay checks on six evenly spaced images, independent of labels',
              'stats':{'unit':'eye for point estimates; patient for resampling, both eyes and all views retained together',
                       'repeats_only':True,'missing':'feature-wise common finite image rows across ALL three arms; keep eyes with>=2 such rows',
                       'icc':'one-way random effects ICC(1,1), unequal-count effective k; negative ICC allowed',
                       'within_eye_mad':'mean absolute deviation from own-eye mean, then average across repeated eyes; NOT ground-truth error',
                       'within_eye_sd':'sqrt pooled within-eye mean square',
                       'between_eye_sd':'sqrt max((MSbetween-MSwithin)/effective_k,0); not necessarily biological variability',
                       'bootstrap_repetitions':10000,'seed':20260908,'interval':'paired patient-cluster percentile95%; exploratory, no multiplicity correction',
                       'singletons':'excluded from ALL repeatability point estimates and resampling'},
              'limitations':['224px public mirror; upsampling does not create anatomical detail','no phenotype reference standard in mBRSET',
                             'ICC alone can rise because between-eye variation increases','different from original manuscript nine-phenotype construction; no four-of-nine/six-of-nine substitution'],
              'stop_rule':'complete fixed comparisons once; no retraining, no output-driven feature selection or threshold tuning',
              'source_manifest_sha256':sha(SOURCE),'mass_weight_sha256':sha(WEIGHT),
              'occupancy_weight_sha256':sha(ROOT/'models/baseline/artifacts/fives_unet_resnet34/best.pt'),
              'phenotype_source_sha256':sha(ROOT/'src/extract_mbrset_repeatability_features.py'),
              'clinical_columns_loaded_by_pipeline':[],'CIMT_used':False}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:write(OUT/'protocol.json',protocol)
    frame.to_csv(OUT/'manifest.csv',index=False)
    print('Fixed protocol:1960 images,12 phenotypes,3 arms; no training',flush=True)


def load_inputs(row,table):
    key=Path(row['file']).stem
    rgbpath=CACHE/'preprocessed_rgb'/f'{key}.png'
    rgb_native=np.asarray(Image.open(rgbpath).convert('RGB'))
    original=np.asarray(Image.open(row['path']).convert('RGB'))
    b=ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)',r'\1',table.loc[key,'bounds']))
    transform,_=CFIBounds.from_dict(original,b).crop(1024)
    assert np.array_equal(transform.warp(original,(1024,1024)),rgb_native),row['id']
    rgb=np.asarray(Image.fromarray(rgb_native).resize((512,512),Image.Resampling.BILINEAR))
    lp=CACHE/'logits'/f'{key}.npy';op=OCC/f'{key}.npy'
    logits=np.load(lp).astype(np.float32);occ=np.load(op).astype(np.float32)
    assert logits.shape==(1024,1024,4) and np.isfinite(logits).all()
    assert occ.shape==(512,512) and np.isfinite(occ).all() and occ.min()>=0 and occ.max()<=1
    return rgb,cv2.resize(logits,(512,512),interpolation=cv2.INTER_LINEAR),occ,[rgbpath,lp,op,Path(row['path'])]


def occupancy_replay(frame,device):
    path=OUT/'occupancy_replay.json'
    if path.exists():return
    import segmentation_models_pytorch as smp
    from torchvision import transforms
    net=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=1).to(device).eval()
    ck=torch.load(ROOT/'models/baseline/artifacts/fives_unet_resnet34/best.pt',map_location=device)
    net.load_state_dict(ck['model'] if 'model' in ck else ck)
    norm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    rows=[]
    for i in np.linspace(0,len(frame)-1,6,dtype=int):
        row=frame.iloc[i];key=Path(row['file']).stem
        x=norm(Image.open(CACHE/'preprocessed_rgb'/f'{key}.png').convert('RGB').resize((512,512),Image.Resampling.BILINEAR))[None].to(device)
        old=np.load(OCC/f'{key}.npy').astype(np.float32)
        with torch.inference_mode():p=net(x).sigmoid()[0,0].cpu().numpy()
        mae=float(abs(p-old).mean());maximum=float(abs(p-old).max())
        assert mae<5e-5 and maximum<1e-3,(row['id'],mae,maximum)
        rows.append({'id':row['id'],'mean_abs_error':mae,'max_abs_error':maximum,'fp16_fraction_exact':float((p.astype(np.float16)==old).mean())})
    write(path,{'n':6,'image_selection':'evenly spaced manifest indices before outcomes','tolerance':'allows float16 cached quantization','results':rows})
    del net
    if device.type=='mps':torch.mps.empty_cache()


def extract():
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(WEIGHT)==p['mass_weight_sha256']
    assert sha(ROOT/'src/extract_mbrset_repeatability_features.py')==p['phenotype_source_sha256']
    frame=pd.read_csv(OUT/'manifest.csv',dtype=str)
    table=pd.read_csv(CACHE/'bounds_latest_batch.csv',dtype={'id':str}).set_index('id')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    occupancy_replay(frame,device)
    model=Corrector().to(device).eval();model.load_state_dict(torch.load(WEIGHT,map_location=device)['model'])
    started=time.monotonic();completed=0
    for _,row in frame.iterrows():
        dest=OUT/'cases'/f'{row["id"]}.json'
        if dest.exists():continue
        rgb,z,occ,paths=load_inputs(row,table)
        raw=np.moveaxis(softmax_np(z),-1,0)
        mass=corrected(model,device,rgb,z)
        conditional=raw[1:]/np.maximum(raw[1:].sum(0,keepdims=True),1e-7)
        joint=np.concatenate([1-occ[None],occ[None]*conditional],0)
        result={k:row[k] for k in ['id','patient_id','eye','view','split']}
        maxerr=0.
        for arm,prob in zip(ARMS,[raw,mass,joint]):
            assert prob.shape==(4,512,512) and np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
            err=float(np.abs(prob.sum(0)-1).max());assert err<2e-6;maxerr=max(maxerr,err)
            result.update(readout_features(prob,rgb.max(2)>10,arm+'_'))
        result['audit']={'exact_RGB_geometry_replay':True,'max_probability_sum_error':maxerr,
                         'input_sha256':{str(v.relative_to(ROOT)):sha(v) for v in paths}}
        write(dest,result);completed+=1
        if completed%50==0:
            elapsed=time.monotonic()-started
            print(f'extracted {len(list((OUT/"cases").glob("*.json")))}/{len(frame)};{elapsed:.1f}s; {elapsed/completed:.3f}s/new image',flush=True)
    records=[json.loads((OUT/'cases'/f'{i}.json').read_text()) for i in p['ids']]
    data=pd.DataFrame([{k:v for k,v in r.items() if k!='audit'} for r in records])
    assert len(data)==1960 and data.id.nunique()==1960
    data.to_csv(OUT/'per_image.csv',index=False)
    write(OUT/'extraction_complete.json',{'images':1960,'probability_evaluations':5880,'all_RGB_exact':True,
                                        'max_probability_sum_error':max(r['audit']['max_probability_sum_error'] for r in records),
                                        'missing_feature_values':{c:int(data[c].isna().sum()) for c in data if any(c.startswith(a+'_') for a in ARMS)},
                                        'seconds_this_run':time.monotonic()-started,'source_sha256':sha(Path(__file__))})
    print('EXTRACTION COMPLETE',flush=True)


def eye_stats(frame,feature):
    cols=[a+'_'+feature for a in ARMS]
    f=frame[['patient_id','eye',*cols]].replace([np.inf,-np.inf],np.nan).dropna()
    rows=[]
    for (pid,eye),g in f.groupby(['patient_id','eye'],sort=True):
        if len(g)<2:continue
        x=g[cols].to_numpy(float);mu=x.mean(0);d=x-mu
        pair=np.abs(x[:,None,:]-x[None,:,:])[np.triu_indices(len(x),1)].mean(0)
        rows.append((pid,eye,len(x),mu,np.square(d).sum(0),np.abs(d).mean(0),pair))
    return rows


def estimate(weights,counts,means,ss,mad,pair):
    """A bootstrap patient copy keeps its eyes as separate subject copies."""
    N=weights@counts;G=weights.sum(1);N2=weights@(counts**2)
    weighted_mean=weights@(counts[:,None]*means)
    B=(weights@(counts[:,None]*means**2)-weighted_mean**2/N[:,None])/(G[:,None]-1)
    W=(weights@ss)/(N-G)[:,None]
    k=(N-N2/N)/(G-1)
    den=B+(k[:,None]-1)*W
    icc=np.divide(B-W,den,out=np.full_like(den,np.nan),where=den>0)
    return {'icc':icc,'within_eye_mad':(weights@mad)/G[:,None],
            'mean_pairwise_abs_diff':(weights@pair)/G[:,None],
            'within_eye_sd':np.sqrt(np.maximum(W,0)),
            'between_eye_sd':np.sqrt(np.maximum((B-W)/k[:,None],0)),
            'between_ms':B,'within_ms':W}


def analyze():
    p=json.loads((OUT/'protocol.json').read_text());f=pd.read_csv(OUT/'per_image.csv',dtype={'patient_id':str})
    assert len(f)==len(p['ids']) and set(f.id)==set(p['ids'])
    assert f.groupby('patient_id').split.nunique().max()==1
    estimates=[];contrasts=[];inventory={}
    for split,g in f.groupby('split',sort=True):
        sizes=g.groupby(['patient_id','eye']).size()
        inventory[split]={'images':len(g),'patients':g.patient_id.nunique(),'repeated_eyes':int((sizes>=2).sum()),
                          'singleton_eyes_excluded':int((sizes==1).sum())}
        for feature in FEATURES:
            rows=eye_stats(g,feature);counts=np.array([r[2] for r in rows],float)
            arrays=[np.stack([r[j] for r in rows]) for j in [3,4,5,6]]
            pids=np.array([r[0] for r in rows]);unique,ix=np.unique(pids,return_inverse=True)
            point=estimate(np.ones((1,len(rows))),counts,*arrays)
            rng=np.random.default_rng(p['stats']['seed'])
            multiplicities=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=p['stats']['bootstrap_repetitions'])
            weights=multiplicities[:,ix].astype(float)
            boot=estimate(weights,counts,*arrays)
            for j,arm in enumerate(ARMS):
                for metric,v in point.items():
                    ci=np.nanquantile(boot[metric][:,j],[.025,.975])
                    estimates.append({'split':split,'feature':feature,'arm':arm,'metric':metric,'estimate':float(v[0,j]),
                                      'ci_low':float(ci[0]),'ci_high':float(ci[1]),'eyes':len(rows),'patients':len(unique)})
            for a,b in [('mass','raw'),('joint','raw'),('mass','joint')]:
                ai,bi=ARMS.index(a),ARMS.index(b)
                for metric,v in point.items():
                    bd=boot[metric][:,ai]-boot[metric][:,bi];ci=np.nanquantile(bd,[.025,.975])
                    contrasts.append({'split':split,'feature':feature,'contrast':a+'_minus_'+b,'metric':metric,
                                      'delta':float(v[0,ai]-v[0,bi]),'ci_low':float(ci[0]),'ci_high':float(ci[1]),
                                      'eyes':len(rows),'patients':len(unique)})
                reduction=100*(1-point['within_eye_mad'][0,ai]/point['within_eye_mad'][0,bi])
                bd=100*(1-boot['within_eye_mad'][:,ai]/boot['within_eye_mad'][:,bi]);ci=np.nanquantile(bd,[.025,.975])
                contrasts.append({'split':split,'feature':feature,'contrast':a+'_minus_'+b,'metric':'MAD_reduction_percent',
                                  'delta':float(reduction),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'eyes':len(rows),'patients':len(unique)})
            print(f'analyzed {split}/{feature}: {len(rows)} repeated eyes, {len(unique)} patient clusters',flush=True)
    pd.DataFrame(estimates).to_csv(OUT/'estimates.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'paired_patient_contrasts.csv',index=False)
    write(OUT/'analysis_complete.json',{'splits':inventory,'bootstrap':p['stats'],'all_12_features_reported':True,
                                      'source_sha256':sha(Path(__file__))})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','extract','analyze']);args=parser.parse_args()
    {'prepare':prepare,'extract':extract,'analyze':analyze}[args.phase]()
