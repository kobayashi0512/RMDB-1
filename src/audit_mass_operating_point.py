"""Independent count-based audit, including AP from sorted tied scores."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from run_fundus_fair_four_arm import ROOT, OUT as FOUR, geometry, sha, RAW

OUT=ROOT/'artifacts/mass_operating_point_diagnosis'


def exact_ap_and_frontier(y,s):
    ix=np.argsort(-s.ravel(),kind='stable')
    scores=s.ravel()[ix]; truth=y.ravel()[ix]
    ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1]
    hits=np.cumsum(truth,dtype=np.int64)[ends]
    recall=hits/truth.sum();precision=hits/(ends+1)
    ap=float(np.sum(np.diff(np.r_[0,recall])*precision))
    return ap,precision,recall


def independent_metrics(pred,t,v,sk,thin,topology=False):
    support=pred>0;tp=np.count_nonzero(support&v)
    vp=np.count_nonzero(support);vt=np.count_nonzero(v)
    valid=t>=0
    a=((pred==1)|(pred==3))&valid;b=((pred==2)|(pred==3))&valid
    ga=(t==1)|(t==3);gb=(t==2)|(t==3)
    dice_a=2*np.count_nonzero(a&ga)/max(np.count_nonzero(a)+np.count_nonzero(ga),1)
    dice_b=2*np.count_nonzero(b&gb)/max(np.count_nonzero(b)+np.count_nonzero(gb),1)
    values={'mean_av_dice':(dice_a+dice_b)/2,
            'vessel_dice':2*tp/max(vp+vt,1),'vessel_precision':tp/vp if vp else 1.,
            'vessel_recall':tp/max(vt,1),'thin_recall':np.count_nonzero(support&thin)/max(np.count_nonzero(thin),1),
            'av_ratio_log_error':abs(np.log((np.count_nonzero(a)+1)/(np.count_nonzero(b)+1))-np.log((np.count_nonzero(ga)+1)/(np.count_nonzero(gb)+1))),
            'false_positive_pixels':np.count_nonzero(support&~v)}
    if topology:
        psk=skeletonize(support)
        x=np.count_nonzero(psk&v)/max(np.count_nonzero(psk),1)
        z=np.count_nonzero(sk&support)/max(np.count_nonzero(sk),1)
        values['cldice']=2*x*z/max(x+z,1e-15)
    return values


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text());assert summary['complete']
    frozen=json.loads((OUT/'frozen_baseline.json').read_text())
    assert frozen['protocol_sha256']==sha(OUT/'protocol.json')
    assert frozen['source_sha256']==sha(ROOT/'src/diagnose_mass_operating_point.py')
    dev=pd.read_csv(OUT/'development_bias_grid.csv')
    assert dev.id.nunique()==30 and len(dev)==30*61
    means=dev.groupby('bias').mean_av_dice.mean()
    best=sorted(means.index,key=lambda b:(-means.loc[b],abs(b),b))[0]
    assert best==frozen['bias']
    expected=json.loads((FOUR/'sha256_manifest.json').read_text())
    table=pd.read_csv(ROOT/'data/fundus_avseg_vascx/bounds_latest_batch.csv').set_index('id')
    max_ap_error=max_count_error=0.;checked=0;mask_checks=0
    for i,ident in enumerate(protocol['fundus_ids']):
        for directory in ['images','annotation']:
            path=RAW/directory/f'{ident}.png'
            assert sha(path)==expected[str(path.relative_to(ROOT))]
        _,_,aligned,_=geometry(ident,table)
        ann=cv2.resize(aligned,(768,768),interpolation=cv2.INTER_NEAREST)
        # Independent explicit Fundus palette, deliberately not importing target_labels.
        t=np.zeros((768,768),int)
        for rgb,label in [((255,0,0),1),((0,0,255),2),((0,255,0),3),((255,255,255),-100)]:
            t[np.all(ann==rgb,axis=-1)]=label
        v=np.any(ann>0,axis=-1);sk=skeletonize(v);thin=sk&(distance_transform_edt(v)<=2)
        for arm in protocol['arms']:
            row=json.loads((OUT/'cases'/f'{ident}__{arm}.json').read_text())
            path=FOUR/'probabilities'/f'{ident}__{arm}.npy'
            assert sha(path)==row['source_sha256']
            p=np.load(path)
            scores=np.log(np.clip(p[1:].max(0),1e-30,None))-np.log(np.clip(p[0],1e-30,None))
            ap,precision,recall=exact_ap_and_frontier(v,scores)
            max_ap_error=max(max_ap_error,abs(ap-row['ranking']['decision_ap']))
            checks=[(p.argmax(0),row['default'],True)]
            if arm=='vascx':
                # Equivalent direct shifted-logit argmax, not classifier helper.
                logits=np.log(np.maximum(p,1e-30));logits[1:]+=frozen['bias']
                checks.append((logits.argmax(0),row['frozen_bias'],True))
            for pt in protocol['precision_targets']:
                r=row['oracle_matched_precision'][f'{pt:.2f}']
                feasible=recall[precision>=pt]
                assert bool(len(feasible))==r['attainable']
                if not r['attainable']:continue
                assert abs(feasible.max()-r['vessel_recall'])<1e-12
                pred=p[1:].argmax(0)+1;pred[scores<r['oracle_threshold']]=0
                checks.append((pred,r,pt==.9))
            for pred,r,topology in checks:
                for m,value in independent_metrics(pred,t,v,sk,thin,topology).items():
                    err=abs(value-r[m]);max_count_error=max(max_count_error,err)
                    assert err<1e-10,(ident,arm,m,value,r[m])
                mask_checks+=1
            curve=np.load(OUT/'curves'/f'{ident}__{arm}.npz')['precision_envelope']
            assert np.all((curve>=0)&(curve<=1)) and np.all(np.diff(curve)<=1e-12)
            checked+=1
        if (i+1)%10==0:print(f'Independent audit {i+1}/100',flush=True)
    assert max_ap_error<1e-10
    # Recompute the primary paired interval independently from saved per-image rows.
    df=pd.read_csv(OUT/'ranking_per_image.csv')
    assert len(df)==400 and not df.duplicated(['id','arm']).any()
    pivot=df.pivot(index='id',columns='arm',values='decision_ap')
    delta=(pivot['mass']-pivot['vascx']).to_numpy()
    rng=np.random.default_rng(protocol['bootstrap']['seed'])
    draws=rng.integers(0,100,(5000,100));ci=np.quantile(delta[draws].mean(1),[.025,.975])
    np.testing.assert_allclose(ci,summary['contrasts']['mass_minus_vascx']['decision_ap']['ci95'],atol=1e-12)
    audit={'complete':True,'probability_outputs':checked,'operating_point_mask_checks':mask_checks,
           'max_independent_AP_error':max_ap_error,'max_independent_metric_error':max_count_error,
           'all_raw_RGB_and_annotation_hashes_match_prior_manifest':True,
           'all_matched_precision_points_maximize_recall_subject_to_precision':True,
           'development_bias_selection_reproduced':True,'primary_bootstrap_reproduced':True,
           'source_sha256':sha(Path(__file__))}
    (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
