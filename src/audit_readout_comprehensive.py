"""Independent point-estimate replay and provenance audit for the new experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from consolidate_readout_experiments import OUT,fc,tr,ARMS

def run():
    f=pd.read_csv(OUT/'repeat_per_image.csv',dtype={'patient_id':str})
    estimates=pd.read_csv(OUT/'repeat_estimates.csv').set_index(['split','feature','arm','metric'])
    maximum=0.;checks=0
    for (split,arm),g in f.groupby(['split','arm']):
        for feat in tr.FEATURES:
            v=[x.sort_values(['view','id'])[feat].to_numpy() for _,x in g.groupby(['patient_id','eye']) if len(x)>=2]
            sizes=np.array([len(x) for x in v]);means=np.array([x.mean() for x in v]);N=sizes.sum();G=len(v)
            grand=np.concatenate(v).mean();B=np.sum(sizes*(means-grand)**2)/(G-1)
            W=sum(np.sum((x-x.mean())**2) for x in v)/(N-G);k=(N-(sizes*sizes).sum()/N)/(G-1)
            diffs=np.array([x[1]-x[0] for x in v]);pts={'icc':(B-W)/(B+(k-1)*W),'within_eye_mad':np.mean([abs(x-x.mean()).mean() for x in v]),
                'within_eye_sd':np.sqrt(W),'between_eye_sd':np.sqrt(max((B-W)/k,0)),
                'pair_bias':diffs.mean(),'loa_low':diffs.mean()-1.96*diffs.std(ddof=1),'loa_high':diffs.mean()+1.96*diffs.std(ddof=1),
                'eye_mean_sd':means.std(ddof=1),'eye_mean_iqr':np.subtract(*np.quantile(means,[.75,.25]))}
            for met,val in pts.items():
                got=estimates.loc[(split,feat,arm,met),'estimate'];err=abs(got-val)/max(1.,abs(val));assert err<1e-10,(split,feat,arm,met,err);maximum=max(maximum,err);checks+=1
    original=pd.read_csv(fc.OUT/'repeat_estimates.csv')
    # Compare point estimates only: new CIs intentionally use a new seed and 5,000
    # resamples, and are kept separate from historical 10,000-resample intervals.
    overlap=original.merge(estimates.reset_index(),on=['split','feature','arm','metric'],suffixes=('_old','_new'))
    assert len(overlap)>0
    historical_error=float(abs(overlap.estimate_old-overlap.estimate_new).max());assert historical_error<1e-8
    for folder in ['readout_mechanism_20260914_v2','readout_confidence_20260914_v1']:
        p=json.loads((fc.ROOT/'artifacts'/folder/'protocol.json').read_text())
        for path,h in p.get('source_hashes',p.get('hashes',{})).items():assert fc.sha(Path(path))==h,path
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,h in protocol['sources'].items():assert fc.sha(Path(path))==h,path
    report={'point_estimate_checks':checks,'max_scaled_error':maximum,'historical_point_rows':len(overlap),'historical_max_absolute_error':historical_error,
        'all_source_hashes_verified':True,'all_arms_per_image':bool((f.groupby('id').arm.nunique()==len(ARMS)).all()),
        'same_patient_cross_split':bool((f.groupby('patient_id').split.nunique()>1).any()),
        'bootstrap_intervals':'Exploratory percentile, no multiplicity correction; not acceptance gates',
        'not_a_full_submission_audit':True}
    tr.write(OUT/'audit.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':run()
