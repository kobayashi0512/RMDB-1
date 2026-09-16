#!/usr/bin/env python3
"""Nested patient-level evaluation of BiRAP beyond age and sex (development only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import (average_precision_score, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p); order = np.argsort(p); ranked = p[order] * n / np.arange(1, n + 1)
    out = np.empty(n); out[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return out


def select_features(data: pd.DataFrame, features: list[str], seed: int) -> list[str]:
    """Selection happens using only an outer-training fold via its inner OOF residuals."""
    splitter = StratifiedKFold(4, shuffle=True, random_state=seed)
    residual = np.full(len(data), np.nan); fold = np.full(len(data), -1)
    for k, (tr, va) in enumerate(splitter.split(data, data.accelerated_cimt)):
        m = LinearRegression().fit(data.iloc[tr][["age", "sex"]], data.iloc[tr].cimt_mm)
        residual[va] = data.iloc[va].cimt_mm - m.predict(data.iloc[va][["age", "sex"]])
        fold[va] = k
    rows=[]
    for feature in features:
        valid = np.isfinite(data[feature].to_numpy())
        if valid.sum() < 500 or data.loc[valid, feature].nunique() < 20: continue
        rhos=[]
        for k in range(4):
            ix=valid & (fold==k); rhos.append(float(spearmanr(data.loc[ix,feature], residual[ix]).statistic))
        rho,p=spearmanr(data.loc[valid,feature], residual[valid])
        rows.append((feature,float(rho),float(p),max((np.array(rhos)>0).sum(),(np.array(rhos)<0).sum())))
    table=pd.DataFrame(rows,columns=['feature','rho','p','same'])
    if table.empty: return []
    table['q']=bh(table.p.to_numpy())
    return table.loc[(table.same>=3)&(table.q<0.10),'feature'].tolist()


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, repeats: int, seed: int) -> tuple[float,float,float]:
    """CI for MAE(b)-MAE(a): negative favours BiRAP."""
    rng=np.random.default_rng(seed); n=len(y); values=[]
    for _ in range(repeats):
        ix=rng.integers(0,n,n)
        values.append(mean_absolute_error(y[ix],b[ix])-mean_absolute_error(y[ix],a[ix]))
    return float(np.mean(values)), *[float(x) for x in np.quantile(values,[.025,.975])]


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--phenotypes',type=Path,default=Path('artifacts/birap_phenotypes/patient_level.csv'))
    parser.add_argument('--patients',type=Path,default=Path('artifacts/bimfo_v1_1_cache/patients.csv'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/birap_nested'))
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20260728)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(args.phenotypes,dtype={'patient_id':str})
    c=pd.read_csv(args.patients,dtype={'patient_id':str})
    data=c.loc[c.group==1,['patient_id','age','sex','cimt_mm','accelerated_cimt']].merge(p,on='patient_id',validate='one_to_one')
    features=[x for x in p.columns if x.startswith(('common_','discordance_')) and not x.endswith(('fov_fraction','landmark_valid'))]
    splitter=StratifiedKFold(5,shuffle=True,random_state=args.seed)
    records=[]; selections=[]
    for outer,(tr,va) in enumerate(splitter.split(data,data.accelerated_cimt),1):
        train,val=data.iloc[tr].copy(),data.iloc[va].copy()
        chosen=select_features(train,features,args.seed+outer)
        selections.append({'fold':outer,'selected_features':chosen,'n_selected':len(chosen)})
        xbase=['age','sex']
        base_reg=LinearRegression().fit(train[xbase],train.cimt_mm)
        base_prob=LogisticRegressionCV(Cs=10,cv=4,scoring='roc_auc',max_iter=2000,random_state=args.seed).fit(train[xbase],train.accelerated_cimt)
        if chosen:
            aug_reg=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),RidgeCV(alphas=np.logspace(-3,4,15))).fit(train[xbase+chosen],train.cimt_mm)
            aug_prob=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegressionCV(Cs=10,cv=4,scoring='roc_auc',max_iter=2000,random_state=args.seed)).fit(train[xbase+chosen],train.accelerated_cimt)
            pred_aug=aug_reg.predict(val[xbase+chosen]); prob_aug=aug_prob.predict_proba(val[xbase+chosen])[:,1]
        else:
            pred_aug=base_reg.predict(val[xbase]); prob_aug=base_prob.predict_proba(val[xbase])[:,1]
        for local,i in enumerate(val.index):
            records.append({'patient_id':data.loc[i,'patient_id'],'outer_fold':outer,'cimt_mm':data.loc[i,'cimt_mm'],'accelerated_label':data.loc[i,'accelerated_cimt'],
                            'base_cimt':base_reg.predict(val.loc[[i],xbase])[0],'birap_cimt':pred_aug[local],
                            'base_probability':base_prob.predict_proba(val.loc[[i],xbase])[0,1],'birap_probability':prob_aug[local],
                            'n_selected':len(chosen)})
    out=pd.DataFrame(records); out.to_csv(args.output/'oof_predictions.csv',index=False)
    y=out.cimt_mm.to_numpy(); label=out.accelerated_label.to_numpy()
    metrics={
        'n':len(out),'development_only':True,'held_out_test_used':False,
        'baseline_mae':mean_absolute_error(y,out.base_cimt),'birap_mae':mean_absolute_error(y,out.birap_cimt),
        'baseline_r2':r2_score(y,out.base_cimt),'birap_r2':r2_score(y,out.birap_cimt),
        'baseline_auroc':roc_auc_score(label,out.base_probability),'birap_auroc':roc_auc_score(label,out.birap_probability),
        'baseline_auprc':average_precision_score(label,out.base_probability),'birap_auprc':average_precision_score(label,out.birap_probability),
        'prevalence':float(label.mean()),
        'mae_difference_birap_minus_baseline_bootstrap':paired_bootstrap(y,out.base_cimt.to_numpy(),out.birap_cimt.to_numpy(),args.bootstrap,args.seed),
        'selection_by_fold':selections,
    }
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (args.output/'selection.json').write_text(json.dumps(selections,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
