"""Join all exploratory arms with source replay checks, then paired contrasts."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import readout_mechanism_experiment as base
import readout_confidence_experiment as gate
fc=base.fc;tr=base.tr
OUT=fc.ROOT/'artifacts/readout_comprehensive_20260914_v1'
ARMS=['raw','mass','smooth','cleanup','mix25','mix50','mix75','joint','confidence','budget_matched']

def build():
    OUT.mkdir(exist_ok=True)
    old=pd.read_csv(fc.OUT/'mbrset_per_image.csv',dtype={'patient_id':str})
    meta=['id','patient_id','eye','view','split'];frames=[]
    for a in ['mass','smooth','cleanup']:
        f=old[meta+[a+'_'+k for k in tr.FEATURES]].rename(columns={a+'_'+k:k for k in tr.FEATURES});f['arm']=a;frames.append(f)
    f1=pd.read_csv(base.OUT/'repeat_per_image.csv',dtype={'patient_id':str})
    f2=pd.read_csv(gate.OUT/'repeat_per_image.csv',dtype={'patient_id':str})
    w1=f1.set_index(['id','arm']);w2=f2.set_index(['id','arm'])
    common=w1.index.intersection(w2.index)
    assert abs(w1.loc[common,tr.FEATURES]-w2.loc[common,tr.FEATURES]).to_numpy().max()<1e-6
    for a in ['raw','joint']:
        f=f1[f1.arm==a].set_index('id');r=old.set_index('id');assert max(float(abs(f[k]-r[a+'_'+k]).max()) for k in tr.FEATURES)<1e-6
    f=pd.concat([f1,f2[f2.arm.isin(['confidence','budget_matched'])],*frames],ignore_index=True)
    assert len(f)==19600 and not f.duplicated(['id','arm']).any()
    assert f.groupby('patient_id').split.nunique().max()==1
    f.to_csv(OUT/'repeat_per_image.csv',index=False)
    s1=pd.read_csv(base.OUT/'fundus_per_image.csv');s2=pd.read_csv(gate.OUT/'fundus_per_image.csv');old_s=pd.read_csv(fc.OUT/'fundus_per_image.csv')
    common_columns=s1.columns
    s=pd.concat([s1,s2[s2.arm.isin(['confidence','budget_matched'])][common_columns],old_s[old_s.arm.isin(['mass','smooth','cleanup'])][common_columns]],ignore_index=True)
    assert len(s)==1000 and not s.duplicated(['id','arm']).any();s.to_csv(OUT/'fundus_per_image.csv',index=False)
    shutil.copy2(base.OUT/'mechanism_per_image.csv',OUT/'mechanism_per_image.csv')
    for name in ['mechanism_pairs.csv','mechanism_pair_estimates.csv','mechanism_pair_notes.json']:
        shutil.copy2(base.OUT/name,OUT/name)
    protocol={'arms':ARMS,'status':'All exploratory, no independent confirmation','selection':False,'bootstrap':{'n':5000,'seed':20260914},
        'primary_new_comparison':'confidence versus budget_matched','mechanism_source':'phase-one raw/joint only',
        'sources':{str(x):fc.sha(x) for x in [base.OUT/'protocol.json',gate.OUT/'protocol.json',Path(__file__),fc.OUT/'mbrset_per_image.csv',fc.OUT/'fundus_per_image.csv',base.OUT/'repeat_per_image.csv',gate.OUT/'repeat_per_image.csv']}}
    base.save(OUT/'protocol.json',protocol)
    return protocol

def paired():
    pairs=[('confidence','budget_matched'),('confidence','joint'),('confidence','mix25'),('confidence','mix50')]
    frame=pd.read_csv(OUT/'repeat_per_image.csv',dtype={'patient_id':str});records=[]
    for split,g in frame.groupby('split'):
        for feature in tr.FEATURES:
            w=g.pivot(index=['patient_id','eye','view','id'],columns='arm',values=feature)[ARMS].reset_index();rows=[]
            for (pid,eye),v in w.groupby(['patient_id','eye']):
                if len(v)<2:continue
                x=v[ARMS].to_numpy();mu=x.mean(0);d=x-mu;rows.append((pid,len(x),mu,(d*d).sum(0),abs(d).mean(0)))
            n=np.array([r[1] for r in rows],float);mu,ss,mad=[np.stack([r[j] for r in rows]) for j in [2,3,4]]
            u,ix=np.unique([r[0] for r in rows],return_inverse=True);rng=np.random.default_rng(20260914)
            weights=rng.multinomial(len(u),np.ones(len(u))/len(u),size=5000)[:,ix].astype(float)
            pt=tr.estimate(np.ones((1,len(rows))),n,mu,ss,mad,2*mad);bt=tr.estimate(weights,n,mu,ss,mad,2*mad)
            for a,b in pairs:
                j,k=ARMS.index(a),ARMS.index(b)
                for met in ['icc','within_eye_mad','within_eye_sd','between_eye_sd']:
                    delta=bt[met][:,j]-bt[met][:,k];lo,hi=np.quantile(delta,[.025,.975])
                    records.append({'split':split,'feature':feature,'arm':a,'reference':b,'metric':met,'delta':pt[met][0,j]-pt[met][0,k],'ci_low':lo,'ci_high':hi})
    pd.DataFrame(records).to_csv(OUT/'new_method_paired_contrasts.csv',index=False)
    s=pd.read_csv(OUT/'fundus_per_image.csv');records=[]
    for met in s.columns.drop(['id','arm']):
        w=s.pivot(index='id',columns='arm',values=met)[ARMS].to_numpy();rng=np.random.default_rng(20260914);bt=w[rng.integers(0,len(w),(5000,len(w)))].mean(1)
        for a,b in pairs:
            j,k=ARMS.index(a),ARMS.index(b);lo,hi=np.quantile(bt[:,j]-bt[:,k],[.025,.975]);records.append({'metric':met,'arm':a,'reference':b,'delta':(w[:,j]-w[:,k]).mean(),'ci_low':lo,'ci_high':hi})
    pd.DataFrame(records).to_csv(OUT/'new_method_structural_contrasts.csv',index=False)

if __name__=='__main__':
    p=build();base.OUT=OUT;base.ARMS=ARMS;base.protocol=lambda:(None,p);base.analyze();paired()
    base.save(OUT/'complete.json',{'repeat_images':1960,'repeat_arm_evaluations':19600,'structural_images':100,'structural_arm_evaluations':1000,
         'candidate_claim':'No winner certified; all contrasts retained','independent_confirmation':False,'training_performed':False})
    print('COMPLETE: ten arms, paired comparisons, all endpoints retained')
