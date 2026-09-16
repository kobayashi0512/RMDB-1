"""All-seed, paired external measurement analysis; never select a model."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import three_readout_repeatability as tr
import final_measurement_controls as fc
from evaluate_residual_ablation_external import OUT, LABELS, ARMS, SEEDS, save

NBOOT=5000
SEED=20260919
GROUPS=['raw','legacy_mass',*ARMS]
METRICS=['icc','within_eye_mad','within_eye_sd','between_eye_sd','pair_bias','loa_low','loa_high','eye_mean_sd','eye_mean_iqr']

def prepare():
    p={'source_sha256':fc.sha(Path(__file__)),'extraction_protocol_sha256':fc.sha(OUT/'protocol.json'),
       'bootstrap_replicates':NBOOT,'seed':SEED,'groups':GROUPS,'repeat_metrics':METRICS,
       'bootstrap':'Patient cluster (all repeated eyes and all views) crossed with seeds; same patient and seed resamples for paired contrasts. Fundus uses paired images crossed with seeds.',
       'seed_mean':'Mean of the metric estimates from three separately evaluated checkpoints, not probability-ensemble performance',
       'loa':'First two manifest-ordered views; LoA=bias+-1.96 sample SD of eye pair differences; direction not known acquisition time',
       'mad':'Mean absolute deviation from each own-eye mean, then mean over eyes, NOT full pair absolute difference',
       'sem':'Pooled within-eye SD; no clinical-tolerance interpretation',
       'eye_mean_iqr':'Point estimate only; no bootstrap interval reported',
       'comparisons':'All seven arms versus raw and legacy_mass; six variants versus full paired by seed; legacy_mass versus raw',
       'selection':'None; report both historical subsets, all endpoints and all seeds. No clinical gate PASS claim.',
       'missing':'Fail on any missing or nonfinite value; no post-hoc dropping',
       'inference':'Unadjusted exploratory intervals; absent significance does not establish equivalence'}
    path=OUT/'analysis_protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:save(path,p)

def indices(group):
    return [LABELS.index(group)] if group in ['raw','legacy_mass'] else [LABELS.index(f'{group}__{s}') for s in SEEDS]

def collapse(point,boot,draws,group):
    ix=indices(group);value=float(point[ix].mean())
    if len(ix)==1:b=boot[:,ix[0]]
    else:b=boot[np.arange(len(boot))[:,None],np.asarray(ix)[draws]].mean(1)
    return value,b

PAIRS=[(a,r) for a in ARMS for r in ['raw','legacy_mass']]+[(a,'full') for a in ARMS if a!='full']+[('legacy_mass','raw')]

def summarize_metric(point,boot,draws,context,metric,seed_rows,group_rows,contrast_rows):
    assert np.isfinite(point).all() and np.isfinite(boot).all(),(context,metric)
    for j,label in enumerate(LABELS):
        lo,hi=np.quantile(boot[:,j],[.025,.975])
        seed_rows.append({**context,'label':label,'metric':metric,'estimate':point[j],'ci_low':lo,'ci_high':hi})
    vals={}
    for g in GROUPS:
        val,b=collapse(point,boot,draws,g);vals[g]=(val,b)
        lo,hi=np.quantile(b,[.025,.975]);ix=indices(g)
        group_rows.append({**context,'arm':g,'metric':metric,'estimate':val,'ci_low':lo,'ci_high':hi,
                           'seed_estimate_sd':np.std(point[ix],ddof=1) if len(ix)>1 else 0.})
    for a,r in PAIRS:
        av,ab=vals[a];rv,rb=vals[r];lo,hi=np.quantile(ab-rb,[.025,.975])
        contrast_rows.append({**context,'arm':a,'reference':r,'metric':metric,'delta':av-rv,'ci_low':lo,'ci_high':hi})

def repeat():
    prepare();assert (OUT/'repeat_complete.json').exists()
    f=pd.read_csv(OUT/'repeat_per_image.csv',dtype={'patient_id':str,'id':str,'view':str,'eye':str})
    assert len(f)==1960*23 and not f.duplicated(['id','label']).any()
    seed_rows=[];group_rows=[];contrast_rows=[]
    for split,g in f.groupby('split'):
        for feature in tr.FEATURES:
            wide=g.pivot(index=['patient_id','eye','view','id'],columns='label',values=feature)[LABELS].reset_index()
            assert np.isfinite(wide[LABELS].to_numpy()).all()
            rows=[]
            for (pid,eye),v in wide.groupby(['patient_id','eye']):
                if len(v)<2:continue
                a=v.sort_values(['view','id'])[LABELS].to_numpy();mu=a.mean(0);d=a-mu
                pair=np.abs(a[:,None,:]-a[None,:,:])[np.triu_indices(len(a),1)].mean(0)
                rows.append((pid,len(a),mu,(d*d).sum(0),abs(d).mean(0),pair,a[1]-a[0]))
            counts=np.array([r[1] for r in rows],float)
            means=np.stack([r[2] for r in rows]);ss=np.stack([r[3] for r in rows]);mad=np.stack([r[4] for r in rows]);pair=np.stack([r[5] for r in rows]);diff=np.stack([r[6] for r in rows])
            patients,ix=np.unique([r[0] for r in rows],return_inverse=True)
            rng=np.random.default_rng(SEED)
            w=rng.multinomial(len(patients),np.ones(len(patients))/len(patients),size=NBOOT)[:,ix].astype(float)
            draws=rng.integers(0,3,(NBOOT,3));one=np.ones((1,len(rows)))
            pt=tr.estimate(one,counts,means,ss,mad,pair);bt=tr.estimate(w,counts,means,ss,mad,pair)
            for weights,target in [(one,pt),(w,bt)]:
                total=weights.sum(1)[:,None];bias=weights@diff/total
                sd=np.sqrt(np.maximum((weights@(diff*diff)-total*bias*bias)/(total-1),0))
                em=weights@means/total
                esd=np.sqrt(np.maximum((weights@(means*means)-total*em*em)/(total-1),0))
                target.update({'pair_bias':bias,'loa_low':bias-1.96*sd,'loa_high':bias+1.96*sd,'eye_mean_sd':esd})
            context={'split':split,'feature':feature,'eyes':len(rows),'patients':len(patients)}
            for metric in METRICS[:-1]:
                summarize_metric(pt[metric][0],bt[metric],draws,context,metric,seed_rows,group_rows,contrast_rows)
            iqr=np.subtract(*np.quantile(means,[.75,.25],axis=0))
            for j,label in enumerate(LABELS):seed_rows.append({**context,'label':label,'metric':'eye_mean_iqr','estimate':iqr[j]})
            for group in GROUPS:group_rows.append({**context,'arm':group,'metric':'eye_mean_iqr','estimate':iqr[indices(group)].mean()})
            print('stats',split,feature,flush=True)
    pd.DataFrame(seed_rows).to_csv(OUT/'repeat_seed_estimates.csv',index=False)
    pd.DataFrame(group_rows).to_csv(OUT/'repeat_group_estimates.csv',index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT/'repeat_group_contrasts.csv',index=False)
    save(OUT/'repeat_analysis_complete.json',{'all_12_features':True,'seed_estimates':len(seed_rows),'group_estimates':len(group_rows),'contrasts':len(contrast_rows)})

def fundus():
    prepare();assert (OUT/'fundus_complete.json').exists()
    f=pd.read_csv(OUT/'fundus_per_image.csv');assert len(f)==100*23 and not f.duplicated(['id','label']).any()
    seed_rows=[];group_rows=[];contrast_rows=[]
    rng=np.random.default_rng(SEED);draw_images=rng.integers(0,100,(NBOOT,100));draw_seeds=rng.integers(0,3,(NBOOT,3))
    for metric in f.columns.drop(['id','label']):
        a=f.pivot(index='id',columns='label',values=metric)[LABELS].to_numpy()
        assert a.shape==(100,23) and np.isfinite(a).all()
        summarize_metric(a.mean(0),a[draw_images].mean(1),draw_seeds,{'images':100},metric,seed_rows,group_rows,contrast_rows)
    pd.DataFrame(seed_rows).to_csv(OUT/'fundus_seed_estimates.csv',index=False)
    pd.DataFrame(group_rows).to_csv(OUT/'fundus_group_estimates.csv',index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT/'fundus_group_contrasts.csv',index=False)
    save(OUT/'fundus_analysis_complete.json',{'all_11_metrics':True,'seed_estimates':len(seed_rows),'group_estimates':len(group_rows),'contrasts':len(contrast_rows)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','repeat','fundus']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='repeat':repeat()
    else:fundus()
