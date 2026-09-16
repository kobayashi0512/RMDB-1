"""Independent point-estimate and frozen-input integrity audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_residual_ablation_external import OUT, LABELS, TRAIN, ARMS, SEEDS
from analyze_residual_ablation_external import indices
import final_measurement_controls as fc
import three_readout_repeatability as tr

def preflight():
    files=[fc.ROOT/'src/run_fundus_fair_four_arm.py',fc.ROOT/'src/backbone_runtime.py',
           fc.FCACHE/'bounds_latest_batch.csv',tr.CACHE/'bounds_latest_batch.csv',
           fc.WEIGHT,TRAIN/'input_and_compute_audit.json']
    p={'additional_source_hashes':{str(f):fc.sha(f) for f in files},
       'scope':'Supplemental provenance only, no parameter or endpoint changes'}
    path=OUT/'supplemental_provenance.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:tr.write(path,p)
    # Paired seed resampling must give exactly zero for identical arms.
    from analyze_residual_ablation_external import collapse
    point=np.ones(23);boot=np.ones((100,23));draws=np.random.default_rng(9).integers(0,3,(100,3))
    a,ab=collapse(point,boot,draws,'full');b,bb=collapse(point,boot,draws,'no_rgb')
    assert a-b==0 and not (ab-bb).any()
    point[indices('full')]=[1,2,3];point[indices('no_rgb')]=[2,3,4]
    boot[:]=point
    a,ab=collapse(point,boot,draws,'full');b,bb=collapse(point,boot,draws,'no_rgb')
    assert b-a==1 and np.allclose(bb-ab,1)
    tr.write(OUT/'statistics_smoke.json',{'paired_identical_arms_zero':True,'all_three_seeds_retained':True,
                                        'paired_seed_constant_difference_preserved':True})

def run():
    preflight()
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,h in protocol['source_hashes'].items():assert fc.sha(Path(path))==h,path
    for label,h in protocol['checkpoint_hashes'].items():
        arm,seed=label.rsplit('__',1);assert fc.sha(TRAIN/'runs'/arm/seed/'last.pt')==h
    for path,h in json.loads((OUT/'supplemental_provenance.json').read_text())['additional_source_hashes'].items():assert fc.sha(Path(path))==h
    checked=0;maximum=0.
    for stage,n in [('repeat',1960),('fundus',100)]:
        f=pd.read_csv(OUT/(stage+'_per_image.csv'),dtype={'id':str,'patient_id':str,'eye':str,'view':str})
        assert len(f)==n*23 and f.id.nunique()==n and (f.groupby('id').label.nunique()==23).all()
        assert not f.duplicated(['id','label']).any()
        if stage=='fundus':
            estimates=pd.read_csv(OUT/'fundus_seed_estimates.csv').set_index(['label','metric'])
            for label,g in f.groupby('label'):
                for metric in f.columns.drop(['id','label']):
                    value=float(np.sum(g[metric].to_numpy())/len(g));got=estimates.loc[(label,metric),'estimate']
                    error=abs(got-value)/max(1,abs(value));assert error<1e-10
                    checked+=1;maximum=max(maximum,error)
            continue
        assert f.groupby('patient_id').split.nunique().max()==1
        estimates=pd.read_csv(OUT/'repeat_seed_estimates.csv').set_index(['split','feature','label','metric'])
        for (split,label),g in f.groupby(['split','label']):
            for feature in tr.FEATURES:
                arrays=[x.sort_values(['view','id'])[feature].to_numpy() for _,x in g.groupby(['patient_id','eye']) if len(x)>=2]
                assert all(np.isfinite(x).all() for x in arrays)
                n=np.array([len(x) for x in arrays]);mu=np.array([x.mean() for x in arrays]);N=n.sum();G=len(n)
                grand=np.concatenate(arrays).mean();B=np.sum(n*(mu-grand)**2)/(G-1)
                W=sum(np.sum((x-x.mean())**2) for x in arrays)/(N-G);k=(N-(n*n).sum()/N)/(G-1)
                diff=np.array([x[1]-x[0] for x in arrays])
                values={'icc':(B-W)/(B+(k-1)*W),'within_eye_mad':np.mean([abs(x-x.mean()).mean() for x in arrays]),
                        'within_eye_sd':np.sqrt(W),'between_eye_sd':np.sqrt(max((B-W)/k,0)),
                        'pair_bias':diff.mean(),'loa_low':diff.mean()-1.96*diff.std(ddof=1),'loa_high':diff.mean()+1.96*diff.std(ddof=1),
                        'eye_mean_sd':mu.std(ddof=1),'eye_mean_iqr':np.subtract(*np.quantile(mu,[.75,.25]))}
                for metric,value in values.items():
                    got=estimates.loc[(split,feature,label,metric),'estimate'];error=abs(got-value)/max(1,abs(value))
                    assert error<1e-10,(split,label,feature,metric,error);checked+=1;maximum=max(maximum,error)
    # The old controls are only replay references, never used to pick weights.
    current=pd.read_csv(OUT/'repeat_seed_estimates.csv');current=current[current.label.isin(['raw','legacy_mass'])].copy()
    current['arm']=current.label.replace({'legacy_mass':'mass'})
    old=pd.read_csv(fc.OUT/'repeat_estimates.csv')
    merged=current.merge(old,on=['split','feature','arm','metric'],suffixes=('_new','_old'))
    assert len(merged)>0
    historical=float(abs(merged.estimate_new-merged.estimate_old).max());assert historical<1e-8
    tr.write(OUT/'audit.json',{'independent_point_checks':checked,'max_scaled_error':maximum,
              'historical_point_overlap':len(merged),'historical_max_absolute_error':historical,
              'all_weights_and_sources_verified':True,'all_images_arms_seeds_present':True,
              'patient_split_overlap':False,'bootstrap_intervals_not_independently_recomputed':True,
              'not_independent_confirmation':True})
    print(checked,maximum,historical)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','run']);a=p.parse_args()
    if a.stage=='preflight':preflight()
    else:run()
