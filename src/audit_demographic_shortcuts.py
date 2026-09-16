#!/usr/bin/env python3
"""Audit age/sex shortcuts on the locked validation cohort only.

The official test split is never read. Cases and controls are optimally paired
within sex by age, then filtered by a predeclared maximum age difference.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def matched_ids(frame: pd.DataFrame, caliper: float) -> list[str]:
    keep=[]
    for sex in sorted(frame.sex.unique()):
        cases=frame[(frame.sex==sex)&(frame.label==1)].reset_index(drop=True)
        controls=frame[(frame.sex==sex)&(frame.label==0)].reset_index(drop=True)
        cost=np.abs(cases.age.to_numpy()[:,None]-controls.age.to_numpy()[None,:])
        ri,ci=linear_sum_assignment(cost)
        for i,j in zip(ri,ci):
            if cost[i,j] <= caliper: keep.extend([str(cases.loc[i,'patient_id']),str(controls.loc[j,'patient_id'])])
    return keep

def bootstrap_auc(y: np.ndarray, score: np.ndarray, repeats: int, seed: int) -> list[float]:
    rng=np.random.default_rng(seed); vals=[]; n=len(y)
    for _ in range(repeats):
        idx=rng.integers(0,n,n)
        if len(np.unique(y[idx]))==2: vals.append(roc_auc_score(y[idx],score[idx]))
    return [float(x) for x in np.quantile(vals,[.025,.975])]

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('artifacts/manifest.csv'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'));p.add_argument('--output',type=Path,default=Path('artifacts/demographic_shortcut_audit'))
    p.add_argument('--caliper',type=float,default=2.0);p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=20260728);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(a.manifest,dtype={'patient_id':str});train=manifest[manifest.split=='train'].copy();val=manifest[manifest.split=='val'].copy()
    models={}
    for name,features in [('age_only',['age']),('sex_only',['sex']),('age_sex',['age','sex'])]:
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=a.seed)).fit(train[features],train.label)
        models[name]=pd.Series(m.predict_proba(val[features])[:,1],index=val.patient_id)
    paths={
      'bilateral_image':a.artifacts/'bilateral_vit_small_patch14_dinov2_lvd142m/validation_report/predictions.csv',
      'image_clinical':a.artifacts/'multimodal_vit_small_patch14_dinov2_lvd142m/validation_report/predictions.csv',
      'clinical_residual':a.artifacts/'clinical_residual_vit_small_patch14_dinov2_lvd142m/validation_report/predictions.csv',
      'clinical_residual_debiased':a.artifacts/'clinical_residual_debiased_vit_small_patch14_dinov2_lvd142m/validation_report/predictions.csv',
    }
    for name,path in paths.items():
        d=pd.read_csv(path,dtype={'patient_id':str});models[name]=d.set_index('patient_id').probability
    ids=matched_ids(val,a.caliper);matched=val[val.patient_id.isin(ids)].copy()
    rows=[]
    for cohort_name,cohort in [('full_validation',val),('age_sex_matched',matched)]:
        y=cohort.label.to_numpy()
        for name,scores in models.items():
            score=scores.reindex(cohort.patient_id).to_numpy()
            rows.append({'cohort':cohort_name,'model':name,'n':len(cohort),'positives':int(y.sum()),'age_case_mean':float(cohort.loc[cohort.label==1,'age'].mean()),'age_control_mean':float(cohort.loc[cohort.label==0,'age'].mean()),'auroc':float(roc_auc_score(y,score)),'auroc_ci95':json.dumps(bootstrap_auc(y,score,a.bootstrap,a.seed)),'auprc':float(average_precision_score(y,score))})
    result=pd.DataFrame(rows);result.to_csv(a.output/'metrics.csv',index=False)
    summary={'validation_only':True,'held_out_test_used':False,'caliper_years':a.caliper,'full_n':len(val),'matched_n':len(matched),'matched_pairs':len(matched)//2,'train_prevalence':float(train.label.mean()),'validation_prevalence':float(val.label.mean())}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(result.to_string(index=False))
if __name__=='__main__':main()
