#!/usr/bin/env python3
"""Cross-fold stability audit for outcome-free topology/transport features."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr

def bh(p):
 """Benjamini-Hochberg correction over finite p-values only."""
 p=np.asarray(p,dtype=float);q=np.full(p.shape,np.nan,dtype=float);valid=np.isfinite(p)
 pv=p[valid];n=len(pv)
 if n==0:return q
 o=np.argsort(pv);r=pv[o]*n/np.arange(1,n+1);qv=np.empty(n);qv[o]=np.clip(np.minimum.accumulate(r[::-1])[::-1],0,1);q[valid]=qv
 return q

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,default=Path('artifacts/topology_transport_features/features.csv'));ap.add_argument('--manifests',type=Path,default=Path('artifacts/bimfo_v2_outer_cv/manifests'));ap.add_argument('--output',type=Path,default=Path('artifacts/topology_transport_stability'));a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 x=pd.read_csv(a.features,dtype={'patient_id':str});cols=[c for c in x if c not in ['patient_id','cache_index']];parts=[]
 for fold in range(1,6):
  m=pd.read_csv(a.manifests/f'fold_{fold}.csv',dtype={'patient_id':str}).query("split=='val'")[['patient_id','delta_cimt_mm','accelerated_cimt']];m['fold']=fold;parts.append(m)
 d=pd.concat(parts).merge(x,on='patient_id',validate='one_to_one');rows=[]
 for c in cols:
  rho,p=spearmanr(d[c],d.delta_cimt_mm)
  fr=[]
  for _,q in d.groupby('fold'):
   fold_rho=float(spearmanr(q[c],q.delta_cimt_mm).statistic)
   fr.append(fold_rho if np.isfinite(fold_rho) else None)
  finite_fr=np.asarray([v for v in fr if v is not None],dtype=float);s=np.sign(finite_fr)
  rows.append({'feature':c,'rho':rho,'p':p,'fold_rhos':json.dumps(fr),'finite_folds':int(len(finite_fr)),'same_sign_folds':int(max((s>0).sum(),(s<0).sum())) if len(s) else 0})
 out=pd.DataFrame(rows).sort_values('p');out['fdr_q']=bh(out.p);out['stable_4of5_fdr10']=(out.same_sign_folds>=4)&(out.fdr_q<.10);out.to_csv(a.output/'stability.csv',index=False)
 summary={'patients':len(d),'features':len(cols),'stable_features':int(out.stable_4of5_fdr10.sum()),'official_validation_used':False,'official_test_used':False};(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(out.head(20).to_string(index=False))
if __name__=='__main__':main()
