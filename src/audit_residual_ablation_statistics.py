"""Independent arithmetic audit of all reported point estimates."""
import math
import pandas as pd
from prepare_residual_ablation import OUT
from backbone_runtime import atomic_json

def avg(values):
    values=list(values)
    return math.fsum(values)/len(values)

def main():
    data=pd.read_csv(OUT/'all_final_per_image.csv',dtype={'id':str})
    summary=pd.read_csv(OUT/'final_summary.csv'); contrasts=pd.read_csv(OUT/'paired_contrasts.csv')
    errors=[]
    for row in summary.itertuples():
        part=data[(data.split==row.split)&(data.arm==row.arm)]
        means=[avg(group[row.metric]) for _,group in part.groupby('seed')]
        mean=avg(means);sd=math.sqrt(math.fsum((v-mean)**2 for v in means)/(len(means)-1))
        errors += [abs(mean-row.mean),abs(sd-row.seed_mean_sd)]
    for row in contrasts.itertuples():
        a=data[(data.split==row.split)&(data.arm==row.arm)].set_index(['seed','id'])[row.metric].to_dict()
        b=data[(data.split==row.split)&(data.arm==row.reference)].set_index(['seed','id'])[row.metric].to_dict()
        assert a.keys()==b.keys()
        errors.append(abs(avg(a[k]-b[k] for k in a)-row.delta))
    assert len(errors)==290 and max(errors)<1e-12
    atomic_json(OUT/'statistics_point_audit.json',{'independent_point_checks':len(errors),'max_absolute_error':max(errors),'bootstrap_intervals_not_independently_recomputed':True})
    print(len(errors),max(errors))

if __name__=='__main__':main()
