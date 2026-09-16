#!/usr/bin/env python3
"""Age-conditioned, bilateral vessel-phenotype residual baseline.

This is the first falsifiable ACBVR experiment: explicit vessel phenotypes are
allowed to improve age/sex risk only if they generalize on the fixed validation
cohort.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from reporting import write_report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--phenotypes',type=Path,default=Path('artifacts/cimt_vessel_phenotypes.csv')); p.add_argument('--output',type=Path,default=Path('artifacts/acbvr_phenotype_residual')); p.add_argument('--bootstrap-repetitions',type=int,default=1000); a=p.parse_args()
    frame=pd.read_csv(a.phenotypes,dtype={'patient_id':str}); phenotype=[c for c in frame if c.startswith('common_') or c.startswith('asymmetry_')]; clinical=['age','sex']; train=frame.query("split == 'train'").copy(); val=frame.query("split == 'val'").copy()
    # Regularization limits overfitting from correlated morphology features.
    classifier=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=5000,class_weight='balanced',random_state=2026))])
    regressor=Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=20.0))])
    features=clinical+phenotype; classifier.fit(train[features],train.label); regressor.fit(train[features],train.cimt_mm)
    output=val[['patient_id','age','sex','label','cimt_mm']].copy(); output['probability']=classifier.predict_proba(val[features])[:,1]; output['prediction_mm']=regressor.predict(val[features]); output['age_band']=pd.cut(output.age,[-float('inf'),39,49,59,float('inf')],labels=['<40','40-49','50-59','≥60'])
    a.output.mkdir(parents=True,exist_ok=True); report=write_report(output,a.output,bootstrap_repetitions=a.bootstrap_repetitions); (a.output/'feature_list.json').write_text(json.dumps({'clinical':clinical,'phenotype':phenotype},indent=2)); print(json.dumps({k:v['point'] for k,v in report['metrics'].items()},indent=2))
if __name__=='__main__': main()
