#!/usr/bin/env python3
"""Outcome-permutation and random-eye-pairing checks for stable BiRAP signals."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

def oof_residual(d: pd.DataFrame, seed: int) -> np.ndarray:
    out=np.full(len(d),np.nan); skf=StratifiedKFold(5,shuffle=True,random_state=seed)
    for tr,va in skf.split(d,d.accelerated_cimt):
        m=LinearRegression().fit(d.iloc[tr][['age','sex']],d.iloc[tr].cimt_mm)
        out[va]=d.iloc[va].cimt_mm-m.predict(d.iloc[va][['age','sex']])
    return out

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--eye',type=Path,default=Path('artifacts/birap_phenotypes/eye_level.csv'))
    ap.add_argument('--patient',type=Path,default=Path('artifacts/birap_phenotypes/patient_level.csv'))
    ap.add_argument('--stability',type=Path,default=Path('artifacts/birap_stability/phenotype_stability.csv'))
    ap.add_argument('--cohort',type=Path,default=Path('artifacts/bimfo_v1_1_cache/patients.csv'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/birap_negative_controls'))
    ap.add_argument('--permutations',type=int,default=1000); ap.add_argument('--seed',type=int,default=20260728)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(a.seed)
    cohort=pd.read_csv(a.cohort,dtype={'patient_id':str}); cohort=cohort[cohort.group==1][['patient_id','age','sex','cimt_mm','accelerated_cimt']]
    patient=pd.read_csv(a.patient,dtype={'patient_id':str}); d=cohort.merge(patient,on='patient_id',validate='one_to_one')
    d['residual']=oof_residual(d,a.seed)
    stable=pd.read_csv(a.stability).query('stable')['feature'].tolist()
    eye=pd.read_csv(a.eye,dtype={'patient_id':str}); eye=eye[eye.patient_id.isin(d.patient_id) & (eye.measurement_reliability==1)]
    left=eye[eye.eye=='left'].set_index('patient_id'); right=eye[eye.eye=='right'].set_index('patient_id')
    ids=d.patient_id.tolist(); d=d.set_index('patient_id').loc[ids].reset_index()
    rows=[]
    for feature in stable:
        eye_feature=feature.removeprefix('common_')
        # Only common-state features can be checked with random bilateral re-pairing.
        if eye_feature not in left.columns or eye_feature not in right.columns: continue
        lv=left.reindex(ids)[eye_feature].to_numpy(float); rv=right.reindex(ids)[eye_feature].to_numpy(float); y=d.residual.to_numpy(float)
        valid=np.isfinite(lv)&np.isfinite(rv)&np.isfinite(y)
        actual=float(spearmanr((lv[valid]+rv[valid])/2,y[valid]).statistic)
        shuffled=[]; outcome_perm=[]
        for _ in range(a.permutations):
            shuffled.append(float(spearmanr((lv[valid]+rng.permutation(rv[valid]))/2,y[valid]).statistic))
            outcome_perm.append(float(spearmanr((lv[valid]+rv[valid])/2,rng.permutation(y[valid])).statistic))
        shuffled=np.asarray(shuffled); outcome_perm=np.asarray(outcome_perm)
        rows.append({'feature':feature,'n':int(valid.sum()),'actual_rho':actual,
                     'random_eye_pair_mean_rho':float(shuffled.mean()),'random_eye_pair_ci_low':float(np.quantile(shuffled,.025)),
                     'random_eye_pair_ci_high':float(np.quantile(shuffled,.975)),
                     'actual_stronger_than_random_pairing_p':float((np.abs(shuffled)>=abs(actual)).mean()),
                     'outcome_permutation_p':float((np.abs(outcome_perm)>=abs(actual)).mean())})
    result=pd.DataFrame(rows); result.to_csv(a.output/'negative_controls.csv',index=False)
    summary={'development_only':True,'held_out_test_used':False,'stable_features_checked':len(result),'permutations':a.permutations,
             'interpretation':'Random pairing asks whether true bilateral averaging is stronger than averaging unrelated eyes; outcome permutation asks whether the observed feature-residual association exceeds chance for that prespecified stable feature.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); print(result.to_string(index=False))
if __name__=='__main__': main()
