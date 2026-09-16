#!/usr/bin/env python3
"""Leakage-safe age-conditioned CIMT baselines using frozen bilateral features.

All labels are constructed in pre-existing outer manifests using outer-training
data only. Hyperparameters are selected only inside each outer training fold.
The official validation/test groups are not read.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def metrics(d: pd.DataFrame) -> dict:
    y=d.accelerated_cimt.to_numpy(); delta=d.delta_cimt_mm.to_numpy()
    return {'n':len(d),'prevalence':float(y.mean()),'auroc':float(roc_auc_score(y,d.probability)),'auprc':float(average_precision_score(y,d.probability)),'delta_mae':float(mean_absolute_error(delta,d.prediction_delta_mm)),'delta_r2':float(r2_score(delta,d.prediction_delta_mm)),'cimt_mae':float(mean_absolute_error(d.cimt_mm,d.expected_cimt_mm+d.prediction_delta_mm))}

def design(global_features: np.ndarray, frame: pd.DataFrame, mode: str) -> np.ndarray:
    idx=frame.cache_index.to_numpy(int); x=np.asarray(global_features[idx],dtype=np.float32)
    left,right=x[:,0],x[:,1]; image={'left':left,'bilateral':np.concatenate([(left+right)/2,np.abs(left-right)],axis=1)}
    if mode=='clinical': return frame[['age','sex']].to_numpy(float)
    if mode=='left': return image['left']
    if mode=='bilateral': return image['bilateral']
    return np.concatenate([image['bilateral'],frame[['age','sex']].to_numpy(float)],axis=1)

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,default=Path('artifacts/bimfo_v1_1_cache/global_features.npy'))
    p.add_argument('--manifests',type=Path,default=Path('artifacts/bimfo_v2_outer_cv/manifests'));p.add_argument('--output',type=Path,default=Path('artifacts/conditioned_baselines'));p.add_argument('--seed',type=int,default=20260728);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    features=np.load(a.features,mmap_mode='r'); predictions=[]; fold_metrics=[]
    for fold in range(1,6):
        m=pd.read_csv(a.manifests/f'fold_{fold}.csv',dtype={'patient_id':str});train=m[m.split=='train'].copy();val=m[m.split=='val'].copy()
        for mode in ['clinical','left','bilateral','bilateral_clinical']:
            xt,xv=design(features,train,mode),design(features,val,mode)
            clf=make_pipeline(StandardScaler(),LogisticRegressionCV(Cs=np.logspace(-4,1,8),cv=4,scoring='roc_auc',class_weight='balanced',max_iter=3000,random_state=a.seed+fold,n_jobs=-1)).fit(xt,train.accelerated_cimt)
            reg=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,13))).fit(xt,train.delta_cimt_mm)
            out=val[['patient_id','age','sex','accelerated_cimt','delta_cimt_mm','expected_cimt_mm','cimt_mm']].copy();out['fold']=fold;out['model']=mode;out['probability']=clf.predict_proba(xv)[:,1];out['prediction_delta_mm']=reg.predict(xv);predictions.append(out);fold_metrics.append({'fold':fold,'model':mode,**metrics(out)})
    pred=pd.concat(predictions,ignore_index=True);pred.to_csv(a.output/'oof_predictions.csv',index=False);fold_table=pd.DataFrame(fold_metrics);fold_table.to_csv(a.output/'fold_metrics.csv',index=False)
    overall=[]
    for model,d in pred.groupby('model'):
        row={'model':model,**metrics(d)}
        for label,lo,hi in [('age_lt40',-np.inf,40),('age_40_49',40,50),('age_50_59',50,60),('age_ge60',60,np.inf)]:
            q=d[(d.age>=lo)&(d.age<hi)]
            row[f'{label}_n']=len(q);row[f'{label}_auroc']=float(roc_auc_score(q.accelerated_cimt,q.probability)) if q.accelerated_cimt.nunique()==2 else np.nan
        overall.append(row)
    overall=pd.DataFrame(overall);overall.to_csv(a.output/'overall_metrics.csv',index=False)
    summary={'development_patients':int(pred.patient_id.nunique()),'outer_folds':5,'models':overall.model.tolist(),'label':'outer-train age/sex-conditioned upper-quartile CIMT residual','hyperparameter_selection':'inner CV only','official_validation_used':False,'official_test_used':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(overall.to_string(index=False));print('\nFold metrics\n',fold_table.to_string(index=False))
if __name__=='__main__':main()
