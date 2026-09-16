"""Final bounded readout comparison: reference error, repeats, structure retention.

No training, outcome loading, hyperparameter search or CIMT access.
All five arms and all defined metrics are retained, including adverse results.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from skimage.morphology import skeletonize

from run_fundus_fair_four_arm import ROOT, CACHE as FCACHE, RAW as FRAW, WEIGHT, geometry, corrected, Corrector, softmax_np, sha
from three_readout_repeatability import OUT as OLD, CACHE as MCACHE, FEATURES, load_inputs, write, estimate
from extract_mbrset_repeatability_features import readout_features

OUT=ROOT/'artifacts/final_measurement_controls'
ARMS=['raw','mass','joint','smooth','cleanup']
STRUCTURE=['vessel_density','skeleton_density','branch_density','endpoint_density','component_count','fractal_dimension']
PAIRS=[('mass','raw'),('joint','raw'),('smooth','raw'),('cleanup','raw'),
       ('mass','smooth'),('mass','cleanup'),('joint','smooth'),('joint','cleanup')]
OCC_WEIGHT=ROOT/'models/baseline/artifacts/fives_unet_resnet34/best.pt'
METRICS=['vessel_dice','vessel_precision','vessel_recall','thin_centerline_recall','cldice','predicted_support_fraction']


def simple_readouts(raw,fov):
    """Fixed image-grid operations; do not use reference masks or clinical labels."""
    smooth=gaussian_filter(raw,sigma=(0,1.,1.),mode='reflect',truncate=4.)
    smooth=smooth/np.maximum(smooth.sum(0,keepdims=True),1e-12)
    vessel=(raw[1:].sum(0)>=.5)&fov
    components,n=label(vessel,structure=np.ones((3,3),int))
    sizes=np.bincount(components.ravel(),minlength=n+1)
    remove=(components>0)&(sizes[components]<32)
    cleanup=raw.copy();cleanup[:,remove]=0.;cleanup[0,remove]=1.
    return {'smooth':smooth.astype(np.float32),'cleanup':cleanup}


def binary_mask(prob,fov):
    # Match the legacy phenotype helper's numerical clipping exactly.
    return (np.clip(prob.astype(np.float32),1e-7,1.)[1:].sum(0)>=.5)&fov


def retention(mask,truth):
    pred_n=int(mask.sum());true_n=int(truth.sum());tp=int((mask&truth).sum())
    ps=skeletonize(mask);ts=skeletonize(truth)
    precision=tp/pred_n if pred_n else (1. if not true_n else 0.)
    recall=tp/true_n if true_n else 1.
    tprec=float((ps&truth).sum()/ps.sum()) if ps.any() else (1. if not ts.any() else 0.)
    tsens=float((ts&mask).sum()/ts.sum()) if ts.any() else 1.
    thin=ts&(distance_transform_edt(truth)<=2.)
    return {'vessel_dice':2*tp/(pred_n+true_n) if pred_n+true_n else 1.,
            'vessel_precision':precision,'vessel_recall':recall,
            'thin_centerline_recall':float((thin&mask).sum()/thin.sum()) if thin.any() else float('nan'),
            'cldice':2*tprec*tsens/(tprec+tsens) if tprec+tsens else 0.}


def reference_features(truth,fov):
    p=np.zeros((4,*truth.shape),np.float32);p[0]=~truth;p[1]=truth
    all_values=readout_features(p,fov,'')
    return {k:all_values[k] for k in STRUCTURE}


def check_prob(p):
    assert p.shape==(4,512,512) and np.isfinite(p).all()
    assert p.min()>=0 and p.max()<=1.000001
    error=float(abs(p.sum(0)-1).max());assert error<2e-6
    return error


def prepare():
    for d in ['', 'fundus_cases','mbrset_cases']: (OUT/d).mkdir(parents=True,exist_ok=True)
    ids=sorted(pd.read_csv(FCACHE/'bounds_latest_batch.csv').id.tolist());assert len(ids)==100
    m=pd.read_csv(OLD/'manifest.csv',dtype=str)
    assert len(m)==1960 and set(m.split)=={'development','test'}
    assert set(m.columns)=={'id','patient_id','eye','view','split','file','path'}
    protocol={'purpose':'test reference measurement error, same-eye repeatability, and structural loss under five frozen readouts',
      'exploratory':True,'fundus_ids':ids,'mbrset_ids':m.id.tolist(),'arms':ARMS,'contrasts':[a+'_minus_'+b for a,b in PAIRS],
      'size':512,'skeleton_grid':256,'vessel_threshold':.5,
      'inputs':'native frozen VascX logits resized with cv2 linear to512, then softmax; canonical RGB PIL bilinear512; FOV RGBmax>10',
      'raw':'unmodified VascX probabilities', 'mass':'original frozen two-step Corrector; no retraining',
      'joint':'frozen FIVES occupancy times ORIGINAL VascX conditional identity, NOT corrected identity',
      'smooth':{'method':'Gaussian filter each raw probability channel, then normalize','sigma_pixels':1.,'mode':'reflect','truncate':4.},
      'cleanup':{'method':'remove8-connected components smaller than32 pixels in raw vessel>=0.5 mask inside FOV; removed pixels become background1, other probabilities unchanged','area_pixels':32},
      'parameter_basis':'two transparent fixed comparators chosen before this run, NOT optimized; no claim these span all denoisers or are best tuned baselines',
      'accuracy_features':STRUCTURE,'repeatability_features':FEATURES,
      'reference':'same phenotype operator on geometrically aligned manual binary vessel mask; all nonblack labels including unknown identity count as vessels',
      'reference_limit':'mask-derived reference, not individually annotated anatomical branch count; no reference truth for soft masses or entropies',
      'retention':'same512 vessel marginal threshold>=0.5 for all arms; NOT four-class argmax or old768 mean AV Dice',
      'thin':'reference skeleton distance-transform radius<=2 pixels at512; descriptive image-grid definition, not clinical caliber',
      'repeatability':'reuse audited raw/mass/joint features only after exact convention replay; compute smooth/cleanup on same1960 images; common finite rows for ALL5, repeated eyes only',
      'bootstrap':{'reference_unit':'Fundus image,100different patients reported by source','reference_n':5000,
                   'repeat_unit':'patient, retaining both eyes and all views','repeat_n':10000,'seed':20260905,'ci':'percentile95, exploratory, no multiplicity correction'},
      'reporting':'all six structure errors and all12 repeatability features; no posthoc selection or global best-arm claim',
      'limits':['Fundus already inspected, not new independent confirmation','mBRSET224 public mirror, no highresolution recovery','two cohorts support separate measurements, not subject-level accuracy-repeatability association'],
      'stop':'complete fixed run once and report; adverse results do not trigger tuning or another branch',
      'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [WEIGHT,OCC_WEIGHT,OLD/'manifest.csv',OLD/'per_image.csv',ROOT/'src/extract_mbrset_repeatability_features.py',ROOT/'src/three_readout_repeatability.py']},
      'no_CIMT_or_clinical_outcomes':True}
    dest=OUT/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==protocol
    else:write(dest,protocol)
    m.to_csv(OUT/'manifest.csv',index=False)
    print('FROZEN: five arms;100 annotated images;1960 repeatability images;no training',flush=True)


def verify_sources():
    p=json.loads((OUT/'protocol.json').read_text())
    for name,h in p['source_hashes'].items():assert sha(ROOT/name)==h,name
    return p


def load_occupancy(device):
    import segmentation_models_pytorch as smp
    net=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=1).to(device).eval()
    ck=torch.load(OCC_WEIGHT,map_location=device);net.load_state_dict(ck['model'] if 'model' in ck else ck)
    return net


def fundus():
    protocol=verify_sources();torch.set_num_threads(4);cv2.setNumThreads(2)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net=Corrector().to(device).eval();net.load_state_dict(torch.load(WEIGHT,map_location=device)['model'])
    occnet=load_occupancy(device);table=pd.read_csv(FCACHE/'bounds_latest_batch.csv').set_index('id')
    started=time.monotonic()
    for i,ident in enumerate(protocol['fundus_ids']):
        dest=OUT/'fundus_cases'/f'{ident}.json'
        if dest.exists():continue
        canonical,_,aligned,_=geometry(ident,table)
        rgb=np.asarray(Image.fromarray(canonical).resize((512,512),Image.Resampling.BILINEAR));fov=rgb.max(2)>10
        ann=cv2.resize(aligned,(512,512),interpolation=cv2.INTER_NEAREST);truth=(ann.max(2)>0)&fov
        assert truth.any() and fov.any()
        z=cv2.resize(np.load(FCACHE/'logits'/f'{ident}.npy').astype(np.float32),(512,512),interpolation=cv2.INTER_LINEAR)
        raw=np.moveaxis(softmax_np(z),-1,0);mass=corrected(net,device,rgb,z)
        x=(np.moveaxis(rgb.astype(np.float32)/255,-1,0)-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
        with torch.inference_mode():o=occnet(torch.from_numpy(x[None]).to(device)).sigmoid()[0,0].cpu().numpy()
        # Same cached-occupancy precision as mBRSET.
        o=o.astype(np.float16).astype(np.float32)
        q=raw[1:]/np.maximum(raw[1:].sum(0,keepdims=True),1e-7)
        probs={'raw':raw,'mass':mass,'joint':np.concatenate([1-o[None],o[None]*q],0),**simple_readouts(raw,fov)}
        ref=reference_features(truth,fov);rows=[];maxerr=0.
        for a,p in probs.items():
            maxerr=max(maxerr,check_prob(p));feat=readout_features(p,fov,'');mask=binary_mask(p,fov)
            row={'id':ident,'arm':a,**retention(mask,truth),'predicted_support_fraction':float(mask.sum()/fov.sum())}
            for k in STRUCTURE:
                row['reference_'+k]=ref[k];row['value_'+k]=feat[k]
                row['error_'+k]=feat[k]-ref[k];row['abs_error_'+k]=abs(feat[k]-ref[k])
            rows.append(row)
        paths=[FCACHE/'preprocessed_rgb'/f'{ident}.png',FCACHE/'logits'/f'{ident}.npy',FRAW/'images'/f'{ident}.png',FRAW/'annotation'/f'{ident}.png']
        write(dest,{'rows':rows,'geometry_exact':True,'probability_sum_error':maxerr,'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
        if (i+1)%10==0:print(f'Fundus {i+1}/100;{time.monotonic()-started:.1f}s',flush=True)
    rows=[r for ident in protocol['fundus_ids'] for r in json.loads((OUT/'fundus_cases'/f'{ident}.json').read_text())['rows']]
    f=pd.DataFrame(rows);assert len(f)==500 and not f.duplicated(['id','arm']).any()
    f.to_csv(OUT/'fundus_per_image.csv',index=False)
    write(OUT/'fundus_complete.json',{'images':100,'arm_evaluations':500,'geometry_exact_all':True,'seconds':time.monotonic()-started})


def repeats():
    protocol=verify_sources();cv2.setNumThreads(2)
    m=pd.read_csv(OUT/'manifest.csv',dtype=str)
    old=pd.read_csv(OLD/'per_image.csv',dtype={'patient_id':str}).set_index('id')
    table=pd.read_csv(MCACHE/'bounds_latest_batch.csv',dtype={'id':str}).set_index('id')
    started=time.monotonic()
    for i,r in m.iterrows():
        dest=OUT/'mbrset_cases'/f'{r.id}.json'
        if dest.exists():continue
        rgb,z,_,paths=load_inputs(r,table);fov=rgb.max(2)>10
        raw=np.moveaxis(softmax_np(z),-1,0);replay=readout_features(raw,fov,'raw_')
        maxdiff=max(abs(v-old.loc[r.id,k]) for k,v in replay.items());assert maxdiff<1e-6,(r.id,maxdiff)
        result={k:r[k] for k in ['id','patient_id','eye','view','split']}
        for a in ['raw','mass','joint']:
            for k in FEATURES:result[a+'_'+k]=float(old.loc[r.id,a+'_'+k])
        error=0.
        for a,p in simple_readouts(raw,fov).items():
            error=max(error,check_prob(p));result.update(readout_features(p,fov,a+'_'))
        # Existing frozen outputs and geometric inputs are anchored by protocol hashes;
        # per-image logit and RGB hashes anchor new control calculations.
        result['audit']={'raw_replay_max_error':float(maxdiff),'geometry_exact':True,'probability_sum_error':error,
                         'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
        write(dest,result)
        if (i+1)%100==0:print(f'mBRSET {i+1}/1960;{time.monotonic()-started:.1f}s',flush=True)
    records=[json.loads((OUT/'mbrset_cases'/f'{ident}.json').read_text()) for ident in protocol['mbrset_ids']]
    f=pd.DataFrame([{k:v for k,v in r.items() if k!='audit'} for r in records])
    assert len(f)==1960 and f.id.nunique()==1960
    f.to_csv(OUT/'mbrset_per_image.csv',index=False)
    write(OUT/'repeats_complete.json',{'images':1960,'new_control_probability_evaluations':3920,'all5_arm_feature_rows':9800,
          'max_raw_feature_replay_error':max(r['audit']['raw_replay_max_error'] for r in records),'seconds':time.monotonic()-started})


def rows_for_feature(g,feature):
    cols=[a+'_'+feature for a in ARMS]
    f=g[['patient_id','eye',*cols]].replace([np.inf,-np.inf],np.nan).dropna()
    return [(pid,eye,x[cols].to_numpy(float)) for (pid,eye),x in f.groupby(['patient_id','eye']) if len(x)>=2]


def analyze():
    protocol=verify_sources();rng=np.random.default_rng(protocol['bootstrap']['seed'])
    f=pd.read_csv(OUT/'fundus_per_image.csv');points=[];deltas=[]
    mets=METRICS+['abs_error_'+k for k in STRUCTURE]+['error_'+k for k in STRUCTURE]
    indices=rng.integers(0,100,(5000,100))
    for met in mets:
        w=f.pivot(index='id',columns='arm',values=met)[ARMS];good=np.isfinite(w).all(1);v=w[good].to_numpy()
        # Fail closed instead of silently bootstrapping incompatible rows.
        assert len(v)==100,(met,len(v))
        point=v.mean(0);boot=v[indices].mean(1)
        for j,a in enumerate(ARMS):
            lo,hi=np.quantile(boot[:,j],[.025,.975]);points.append({'metric':met,'arm':a,'estimate':point[j],'ci_low':lo,'ci_high':hi,'n':100})
        for a,b in PAIRS:
            j,k=ARMS.index(a),ARMS.index(b);lo,hi=np.quantile(boot[:,j]-boot[:,k],[.025,.975])
            deltas.append({'metric':met,'contrast':a+'_minus_'+b,'delta':point[j]-point[k],'ci_low':lo,'ci_high':hi,'n':100})
    pd.DataFrame(points).to_csv(OUT/'reference_estimates.csv',index=False);pd.DataFrame(deltas).to_csv(OUT/'reference_contrasts.csv',index=False)
    f=pd.read_csv(OUT/'mbrset_per_image.csv',dtype={'patient_id':str});points=[];deltas=[];counts_report={}
    assert f.groupby('patient_id').split.nunique().max()==1
    for split,g in f.groupby('split'):
        for feature in FEATURES:
            rows=rows_for_feature(g,feature);n=np.array([len(r[2]) for r in rows],float)
            mu=np.stack([r[2].mean(0) for r in rows]);ss=np.stack([((r[2]-r[2].mean(0))**2).sum(0) for r in rows])
            mad=np.stack([abs(r[2]-r[2].mean(0)).mean(0) for r in rows]);pair=np.stack([abs(r[2][:,None]-r[2][None,:])[np.triu_indices(len(r[2]),1)].mean(0) for r in rows])
            pids,ix=np.unique([r[0] for r in rows],return_inverse=True)
            rng=np.random.default_rng(protocol['bootstrap']['seed']);weights=rng.multinomial(len(pids),np.full(len(pids),1/len(pids)),size=10000)[:,ix].astype(float)
            point=estimate(np.ones((1,len(rows))),n,mu,ss,mad,pair);boot=estimate(weights,n,mu,ss,mad,pair)
            counts_report[split+'/'+feature]={'eyes':len(rows),'patients':len(pids),'images':int(n.sum())}
            for met,v in point.items():
                assert np.isfinite(v).all() and np.isfinite(boot[met]).all(),(split,feature,met)
                for j,a in enumerate(ARMS):
                    lo,hi=np.quantile(boot[met][:,j],[.025,.975]);points.append({'split':split,'feature':feature,'arm':a,'metric':met,'estimate':v[0,j],'ci_low':lo,'ci_high':hi,'eyes':len(rows),'patients':len(pids)})
                for a,b in PAIRS:
                    j,k=ARMS.index(a),ARMS.index(b);lo,hi=np.quantile(boot[met][:,j]-boot[met][:,k],[.025,.975])
                    deltas.append({'split':split,'feature':feature,'contrast':a+'_minus_'+b,'metric':met,'delta':v[0,j]-v[0,k],'ci_low':lo,'ci_high':hi,'eyes':len(rows),'patients':len(pids)})
            for a,b in PAIRS:
                j,k=ARMS.index(a),ARMS.index(b);v=100*(1-point['within_eye_mad'][0,j]/point['within_eye_mad'][0,k]);bs=100*(1-boot['within_eye_mad'][:,j]/boot['within_eye_mad'][:,k]);assert np.isfinite(bs).all()
                lo,hi=np.quantile(bs,[.025,.975]);deltas.append({'split':split,'feature':feature,'contrast':a+'_minus_'+b,'metric':'MAD_reduction_percent','delta':v,'ci_low':lo,'ci_high':hi,'eyes':len(rows),'patients':len(pids)})
            print('statistics',split,feature,flush=True)
    pd.DataFrame(points).to_csv(OUT/'repeat_estimates.csv',index=False);pd.DataFrame(deltas).to_csv(OUT/'repeat_contrasts.csv',index=False)
    write(OUT/'analysis_complete.json',{'repeat_samples':counts_report,'reference_images':100,'all_metrics_finite':True,'all_contrasts_reported':True})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','fundus','repeats','analyze']);args=ap.parse_args()
    globals()[args.phase]()
