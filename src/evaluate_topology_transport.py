#!/usr/bin/env python3
"""Nested outer-CV evaluation of topology/transport feasibility features."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV,RidgeCV
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error,r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def score(d):
 y=d.accelerated_cimt.to_numpy();delta=d.delta_cimt_mm.to_numpy();return {'n':len(d),'auroc':roc_auc_score(y,d.probability),'auprc':average_precision_score(y,d.probability),'delta_mae':mean_absolute_error(delta,d.prediction_delta_mm),'delta_r2':r2_score(delta,d.prediction_delta_mm)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,default=Path('artifacts/topology_transport_features/features.csv'));p.add_argument('--global-features',type=Path,default=Path('artifacts/bimfo_v1_1_cache/global_features.npy'));p.add_argument('--manifests',type=Path,default=Path('artifacts/bimfo_v2_outer_cv/manifests'));p.add_argument('--output',type=Path,default=Path('artifacts/topology_transport_cv'));p.add_argument('--seed',type=int,default=20260728);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 ph=pd.read_csv(a.features,dtype={'patient_id':str}).set_index('patient_id');cols=[c for c in ph if c!='cache_index'];gf=np.load(a.global_features,mmap_mode='r');outputs=[];folds=[]
 for fold in range(1,6):
  m=pd.read_csv(a.manifests/f'fold_{fold}.csv',dtype={'patient_id':str});tr=m[m.split=='train'].copy();va=m[m.split=='val'].copy()
  def X(d,mode):
   math=ph.reindex(d.patient_id)[cols].to_numpy(float);g=np.asarray(gf[d.cache_index.to_numpy(int)],dtype=np.float32);bilat=np.concatenate([(g[:,0]+g[:,1])/2,np.abs(g[:,0]-g[:,1])],1);clinical=d[['age','sex']].to_numpy(float)
   return {'math':math,'math_clinical':np.concatenate([math,clinical],1),'dino_math':np.concatenate([bilat,math],1)}[mode]
  for mode in ['math','math_clinical','dino_math']:
   xt,xv=X(tr,mode),X(va,mode);clf=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegressionCV(Cs=np.logspace(-4,1,8),cv=4,scoring='roc_auc',class_weight='balanced',max_iter=4000,random_state=a.seed+fold,n_jobs=-1)).fit(xt,tr.accelerated_cimt);reg=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,13))).fit(xt,tr.delta_cimt_mm)
   o=va[['patient_id','age','sex','accelerated_cimt','delta_cimt_mm']].copy();o['fold']=fold;o['model']=mode;o['probability']=clf.predict_proba(xv)[:,1];o['prediction_delta_mm']=reg.predict(xv);outputs.append(o);folds.append({'fold':fold,'model':mode,**score(o)})
 out=pd.concat(outputs);out.to_csv(a.output/'oof_predictions.csv',index=False);ft=pd.DataFrame(folds);ft.to_csv(a.output/'fold_metrics.csv',index=False);overall=pd.DataFrame([{'model':k,**score(v)} for k,v in out.groupby('model')]);overall.to_csv(a.output/'overall_metrics.csv',index=False)
 summary={'development_patients':int(out.patient_id.nunique()),'outer_folds':5,'outcome_free_feature_extraction':True,'inner_cv_only':True,'official_validation_used':False,'official_test_used':False,'full_persistent_homology_claimed':False};(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(overall.to_string(index=False));print(ft.to_string(index=False))
if __name__=='__main__':main()
