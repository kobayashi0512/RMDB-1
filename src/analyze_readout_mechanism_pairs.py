"""Exact paired log-ratio error decomposition, with patient cluster intervals."""
import json
import numpy as np
import pandas as pd
from readout_mechanism_experiment import OUT,save

def run():
    m=pd.read_csv(OUT/'mechanism_per_image.csv',dtype={'patient_id':str})
    rows=[]
    for (split,pid,eye),g in m.groupby(['split','patient_id','eye']):
        if len(g)<2:continue
        g=g.sort_values(['view','id']);a,b=g.iloc[0],g.iloc[1]
        e=np.log(b.raw_ratio_unclipped/a.raw_ratio_unclipped)
        da=b.log_artery_change-a.log_artery_change;dv=b.log_vein_change-a.log_vein_change
        observed=np.log(b.joint_ratio_unclipped/a.joint_ratio_unclipped)
        assert abs(observed-(e+da-dv))<1e-10
        contribution_a=2*e*da+da*da-da*dv
        contribution_v=-2*e*dv+dv*dv-da*dv
        delta_mse=observed**2-e**2
        assert abs(delta_mse-contribution_a-contribution_v)<1e-10
        rows.append({'split':split,'patient_id':pid,'eye':eye,'raw_log_pair_difference':e,'joint_log_pair_difference':observed,
            'artery_reweight_pair_difference':da,'vein_reweight_pair_difference':dv,
            'delta_squared_log_pair_error':delta_mse,'artery_symmetric_allocation':contribution_a,'vein_symmetric_allocation':contribution_v,
            'raw_squared_log_pair_error':e*e,'joint_squared_log_pair_error':observed*observed,
            'raw_absolute_log_pair_error':abs(e),'joint_absolute_log_pair_error':abs(observed),
            'amplification_cross_term':2*e*(da-dv),'new_reweight_difference_squared':(da-dv)**2})
    pairs=pd.DataFrame(rows);pairs.to_csv(OUT/'mechanism_pairs.csv',index=False)
    results=[]
    for split,g in pairs.groupby('split'):
        u,ix=np.unique(g.patient_id,return_inverse=True);rng=np.random.default_rng(20260914)
        w=rng.multinomial(len(u),np.ones(len(u))/len(u),size=5000)[:,ix]
        for key in g.columns.drop(['split','patient_id','eye']):
            v=g[key].to_numpy();bt=w@v/w.sum(1);lo,hi=np.quantile(bt,[.025,.975])
            results.append({'split':split,'metric':key,'estimate':v.mean(),'ci_low':lo,'ci_high':hi,'eyes':len(g),'patients':len(u)})
    pd.DataFrame(results).to_csv(OUT/'mechanism_pair_estimates.csv',index=False)
    save(OUT/'mechanism_pair_notes.json',{'exact_identity_verified':True,'allocation':'Symmetric equal allocation of artery-vein interaction; accounting decomposition, not causal attribution',
         'scale':'Natural log ratio; secondary exploratory scale, not the historical raw-ratio MAD',
         'view_order':'First two manifest views; not verified acquisition time','bootstrap':5000,'clinical_truth':False})
    print(pd.DataFrame(results).query("metric in ['delta_squared_log_pair_error','artery_symmetric_allocation','vein_symmetric_allocation','amplification_cross_term','new_reweight_difference_squared']").to_string(index=False))

if __name__=='__main__':run()
