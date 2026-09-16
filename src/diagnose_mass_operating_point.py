"""Fixed, exploratory diagnosis of ranking versus operating-point gains.

No training. RAV development labels select ONE global foreground bias for
VascX. Fundus labels only evaluate the frozen rule and descriptive PR oracles.
"""
import argparse
import ast
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.metrics import average_precision_score, precision_recall_curve
from rtnls_fundusprep.cfi_bounds import CFIBounds
from rtnls_fundusprep.transformation import Interpolation

from run_fundus_fair_four_arm import ROOT, OUT as FOUR, geometry, sha
from evaluate_fundus_avseg_external import target_labels as fundus_target
from train_udct_av_pilot import target_labels as rav_target

OUT = ROOT / 'artifacts/mass_operating_point_diagnosis'
ARMS = ['vascx', 'clahe', 'mass', 'clahe_mass']
PRECISIONS = [.85, .90, .95]
GRID = np.linspace(0, 1, 1001)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['cases', 'curves']:
        (OUT / name).mkdir(exist_ok=True)
    split = json.loads((ROOT/'results/birap_v2_residual_full_512/split.json').read_text())
    protocol = {
        'status': 'exploratory diagnostic, NOT independent confirmation',
        'fundus_ids': json.loads((FOUR/'protocol.json').read_text())['ids'],
        'development_ids': split['rav_validation'],
        'development_source': 'existing RAV development validation, not an untouched test',
        'development_predictions': 'structure_restoration_pilot/evaluation/probabilities/*__undegraded__input.npy',
        'size': 768, 'arms': ARMS,
        'primary_contrast': 'mass minus vascx',
        'primary_endpoint': 'image macro average precision of max foreground versus background decision margin',
        'decision_score': 'log(max(P_A,P_V,P_C)) - log(P_background); clip probabilities at 1e-30',
        'primary_reason': 'a shared constant foreground logit bias cannot change this ranking or AP',
        'secondary_ranking_endpoints': ['vessel marginal AP using 1-P0', 'mean A/V marginal AP using PA+PC and PV+PC'],
        'simple_baseline': {'bias_grid': np.round(np.arange(-3, 3.001, .1), 6).tolist(),
                            'selection': 'max mean AV Dice across 30 RAV development images; ties prefer smallest absolute bias then smaller bias',
                            'rule': 'add the SAME bias to A,V,C logits; unchanged foreground winning identity; foreground iff margin+bias>0'},
        'frozen_rule_scope': 'only VascX gets the development-selected simple baseline; Mass standard operating point is not retuned',
        'precision_targets': PRECISIONS,
        'matched_precision': 'per image largest attainable recall at precision>=target across all empirical thresholds; oracle descriptive ONLY, not deployable; report actual precision and unattainable count',
        'thin_definition': 'GT vessel skeleton, Euclidean distance-transform radius<=2 pixels at768; sensitivity endpoint',
        'cldice': 'harmonic mean of predicted skeleton precision within GT vessel and GT skeleton recall within predicted vessel; no topology guarantee',
        'AV_error': 'absolute log difference of hard A/V pixel ratios with +1 smoothing; not clinical caliber AVR',
        'labels': 'Fundus green crossing/white unknown; RAV redblue crossing/green unknown; unknown retained for occupancy, excluded for identity',
        'geometry': 'saved image-only canonical transform on RGB and annotation, nearest neighbor labels; exact RGB replay required',
        'PR_display': '1001 recall-grid interpolated precision envelope per image, macro mean; exact un-interpolated AP reported separately',
        'bootstrap': {'replicates': 5000, 'seed': 20260907, 'unit': 'paired image (Fundus publication reports100patients)', 'multiple_testing': 'not adjusted; exploratory'},
        'interpretation': 'AP increase shows ranking change beyond constant bias, not automatically useful effect size, topology, novelty, or external validity; null CI is inconclusive, not equivalence',
        'stop_rule': 'one fixed diagnostic; no subsequent threshold/model search on Fundus and no new training',
        'no_CIMT_or_clinical_access': True,
    }
    path = OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write(path, protocol)
    print('Protocol fixed', flush=True)


def margin(p):
    return np.log(np.maximum(p[1:].max(0), 1e-30)) - np.log(np.maximum(p[0], 1e-30))


def classify(p, bias=0.):
    labels = p[1:].argmax(0).astype(np.uint8)+1
    labels[margin(p)+bias <= 0] = 0
    return labels


def label_metrics(pred, target, vessel, skeleton, thin, topology=True):
    vp = pred > 0
    tp = int((vp & vessel).sum())
    pn, tn = int(vp.sum()), int(vessel.sum())
    precision = tp/pn if pn else 1.
    recall = tp/tn if tn else 0.
    valid = target != -100
    pa = ((pred == 1) | (pred == 3)) & valid
    pv = ((pred == 2) | (pred == 3)) & valid
    ta = (target == 1) | (target == 3)
    tv = (target == 2) | (target == 3)
    da = 2*(pa & ta).sum()/max(pa.sum()+ta.sum(), 1)
    dv = 2*(pv & tv).sum()/max(pv.sum()+tv.sum(), 1)
    result = dict(mean_av_dice=float((da+dv)/2), vessel_dice=2*tp/max(pn+tn,1),
                  vessel_precision=precision, vessel_recall=recall,
                  thin_recall=float((vp & thin).sum()/max(thin.sum(),1)),
                  av_ratio_log_error=float(abs(np.log((pa.sum()+1)/(pv.sum()+1))-np.log((ta.sum()+1)/(tv.sum()+1)))),
                  false_positive_pixels=int((vp & ~vessel).sum()))
    if topology:
        ps = skeletonize(vp)
        tprec = float((ps & vessel).sum()/max(ps.sum(), 1))
        tsens = float((skeleton & vp).sum()/max(skeleton.sum(), 1))
        result.update(cldice=2*tprec*tsens/max(tprec+tsens,1e-15), skeleton_precision=tprec, skeleton_recall=tsens)
    return result


def rav_geometry(ident, table):
    raw = ROOT/'data/raw/RAV/data/external/RAV/dataset'
    key = 'RAV_'+ident
    rgb = np.asarray(Image.open(raw/'rgb'/f'{key}.png').convert('RGB'))
    config = ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)',r'\1',table.loc[key,'bounds']))
    transform,_ = CFIBounds.from_dict(rgb,config).crop(1024)
    assert np.array_equal(transform.warp(rgb,(1024,1024)), np.asarray(Image.open(ROOT/'data/rav_vascx/preprocessed_rgb'/f'{key}.png')))
    ann = np.asarray(Image.open(raw/'av_masks'/f'{key}.png').convert('RGB'))
    ann = cv2.resize(transform.warp(ann,(1024,1024),mode=Interpolation.NEAREST),(768,768),interpolation=cv2.INTER_NEAREST)
    return ann


def bias_dice_grid(p, target, biases):
    """Exact cumulative counts; equal to direct masks for every bias."""
    valid = (target != -100).ravel()
    score = margin(p).ravel()[valid]
    ident = (p[1:].argmax(0)+1).ravel()[valid]
    t = target.ravel()[valid]
    order = np.argsort(score)
    score, ident, t = score[order], ident[order], t[order]
    k = np.searchsorted(score, -np.asarray(biases), side='right')
    dices = []
    for label in (1,2):
        pred = (ident == label) | (ident == 3)
        truth = (t == label) | (t == 3)
        n = np.r_[0,np.cumsum(pred)]
        hit = np.r_[0,np.cumsum(pred & truth)]
        dices.append(2*(hit[-1]-hit[k])/np.maximum(n[-1]-n[k]+truth.sum(),1))
    return (dices[0]+dices[1])/2


def develop():
    pr = json.loads((OUT/'protocol.json').read_text())
    if (OUT/'frozen_baseline.json').exists():
        print('Frozen baseline already exists; no reselection', flush=True)
        return
    table = pd.read_csv(ROOT/'data/rav_vascx/bounds_latest_batch.csv').set_index('id')
    rows, manifest = [], {}
    biases = pr['simple_baseline']['bias_grid']
    for i, ident in enumerate(pr['development_ids']):
        ann = rav_geometry(ident, table)
        path = ROOT/'artifacts/structure_restoration_pilot/evaluation/probabilities'/f'{ident}__undegraded__input.npy'
        p = np.load(path)
        assert p.shape == (4,768,768) and np.isfinite(p).all()
        assert np.array_equal(classify(p),p.argmax(0))
        values = bias_dice_grid(p,rav_target(ann),biases)
        rows.extend({'id':ident,'bias':b,'mean_av_dice':float(d)} for b,d in zip(biases,values))
        manifest[str(path.relative_to(ROOT))] = sha(path)
        print(f'RAV baseline development {i+1}/30',flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'development_bias_grid.csv',index=False)
    means = df.groupby('bias').mean_av_dice.mean()
    best = sorted(means.index, key=lambda b: (-means.loc[b],abs(b),b))[0]
    write(OUT/'frozen_baseline.json', {'bias':float(best),'threshold_margin':float(-best),
                                    'development_mean_av_dice':float(means.loc[best]),'development_default_av_dice':float(means.loc[0.]),
                                    'selection_data':'RAV existing development only', 'n':30,
                                    'protocol_sha256':sha(OUT/'protocol.json'), 'source_sha256':sha(Path(__file__)),
                                    'input_sha256':manifest})
    print('Frozen shared foreground bias:',best,flush=True)


def pr_data(truth, scores):
    y, s = truth.ravel(), scores.ravel()
    prec, rec, thresholds = precision_recall_curve(y,s,drop_intermediate=False)
    ap = float(average_precision_score(y,s))
    # Standard interpolated precision display. AP above is NOT area of this envelope.
    envelope = np.maximum.accumulate(prec)
    idx = np.searchsorted(-rec, -GRID, side='right')-1
    display = envelope[np.maximum(idx,0)]
    matches = {}
    for pt in PRECISIONS:
        feasible = np.flatnonzero(prec[:-1] >= pt)
        if len(feasible):
            k = feasible[0]  # thresholds ascend, recall non-increasing
            matches[pt] = (float(thresholds[k]),float(prec[k]),float(rec[k]))
        else:
            matches[pt] = None
    return ap, display, matches


def evaluate():
    pr = json.loads((OUT/'protocol.json').read_text())
    frozen = json.loads((OUT/'frozen_baseline.json').read_text())
    assert frozen['protocol_sha256']==sha(OUT/'protocol.json')
    bias = frozen['bias']
    table = pd.read_csv(ROOT/'data/fundus_avseg_vascx/bounds_latest_batch.csv').set_index('id')
    input_hashes = json.loads((FOUR/'output_sha256.json').read_text())
    old = pd.read_csv(FOUR/'per_image_metrics.csv').set_index(['id','arm'])
    for i,ident in enumerate(pr['fundus_ids']):
        if all((OUT/'cases'/f'{ident}__{a}.json').exists() for a in ARMS):
            continue
        _,_,aligned,_=geometry(ident,table)
        ann = cv2.resize(aligned,(768,768),interpolation=cv2.INTER_NEAREST)
        target = fundus_target(ann)
        vessel = ann.max(-1)>0
        valid = target != -100
        sk = skeletonize(vessel)
        thin = sk & (distance_transform_edt(vessel)<=2)
        for arm in ARMS:
            path = FOUR/'probabilities'/f'{ident}__{arm}.npy'
            assert sha(path)==input_hashes[str(path.relative_to(FOUR))]
            p = np.load(path)
            assert np.isfinite(p).all() and p.shape==(4,768,768)
            assert p.min()>=0 and p.max()<=1 and np.max(np.abs(p.sum(0)-1))<2e-6
            pred = classify(p)
            assert np.array_equal(pred,p.argmax(0))
            default = label_metrics(pred,target,vessel,sk,thin)
            assert abs(default['mean_av_dice']-old.loc[(ident,arm),'mean_av_dice'])<1e-10
            score = margin(p)
            ap,display,matches = pr_data(vessel,score)
            av_aps = []
            for lab in (1,2):
                truth = ((target==lab)|(target==3))[valid]
                av_aps.append(float(average_precision_score(truth,(p[lab]+p[3])[valid])))
            ranking = {'decision_ap':ap,'vessel_marginal_ap':float(average_precision_score(vessel.ravel(),(1-p[0]).ravel())),
                       'mean_av_marginal_ap':float(np.mean(av_aps))}
            matched = {}
            fg = p[1:].argmax(0).astype(np.uint8)+1
            for pt,match in matches.items():
                key=f'{pt:.2f}'
                if match is None:
                    matched[key]={'attainable':False}
                else:
                    threshold,expected_precision,expected_recall=match
                    q=fg.copy();q[score<threshold]=0
                    m=label_metrics(q,target,vessel,sk,thin)
                    assert abs(m['vessel_precision']-expected_precision)<1e-10
                    assert abs(m['vessel_recall']-expected_recall)<1e-10
                    matched[key]={'attainable':True,'oracle_threshold':threshold,**m}
            result={'id':ident,'arm':arm,'default':default,'ranking':ranking,'oracle_matched_precision':matched,
                    'source_sha256':sha(path)}
            if arm=='vascx':
                result['frozen_bias']=label_metrics(classify(p,bias),target,vessel,sk,thin)
            write(OUT/'cases'/f'{ident}__{arm}.json',result)
            np.savez_compressed(OUT/'curves'/f'{ident}__{arm}.npz',recall=GRID,precision_envelope=display)
        print(f'Fundus {i+1}/100: 4 arms, ranking + fixed bias + precision-matched oracle',flush=True)
    summarize()


def paired(values):
    values=np.asarray(values,float)
    values=values[np.isfinite(values)]
    if not len(values):return {'n':0,'delta':None,'ci95':None}
    rng=np.random.default_rng(20260907)
    boots=values[rng.integers(0,len(values),(5000,len(values)))].mean(1)
    return {'n':len(values),'delta':float(values.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist()}


def summarize():
    records=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*.json'))]
    assert len(records)==400
    rankings=[];points=[]
    for r in records:
        common={'id':r['id'],'arm':r['arm']}
        rankings.append({**common,**r['ranking']})
        points.append({**common,'point':'default',**r['default']})
        if 'frozen_bias' in r:
            points.append({**common,'point':'RAV_frozen_bias',**r['frozen_bias']})
        for p,m in r['oracle_matched_precision'].items():
            points.append({**common,'point':'oracle_P'+p,**m})
    ranks=pd.DataFrame(rankings);pts=pd.DataFrame(points)
    ranks.to_csv(OUT/'ranking_per_image.csv',index=False)
    pts.to_csv(OUT/'operating_points_per_image.csv',index=False)
    means_rank=ranks.groupby('arm').mean(numeric_only=True).to_dict('index')
    mets=['mean_av_dice','vessel_precision','vessel_recall','thin_recall','cldice','av_ratio_log_error','false_positive_pixels']
    means_pts={'/'.join(k):{m:float(g[m].mean()) for m in mets} for k,g in pts.groupby(['arm','point'])}
    contrasts={}
    for a,b in [('mass','vascx'),('clahe','vascx'),('clahe_mass','clahe')]:
        name=a+'_minus_'+b;contrasts[name]={}
        for m in ['decision_ap','vessel_marginal_ap','mean_av_marginal_ap']:
            p=ranks.pivot(index='id',columns='arm',values=m)
            contrasts[name][m]=paired(p[a]-p[b])
        for point in ['default']+['oracle_P'+f'{p:.2f}' for p in PRECISIONS]:
            for m in mets:
                p=pts[pts.point==point].pivot(index='id',columns='arm',values=m)
                contrasts[name][point+'/'+m]=paired(p[a]-p[b])
    for a,point in [('mass','default'),('vascx','default'),('clahe','default')]:
        name=a+'_default_minus_vascx_RAV_frozen_bias';contrasts[name]={}
        for m in mets:
            first=pts[(pts.arm==a)&(pts.point==point)].set_index('id')[m]
            second=pts[(pts.arm=='vascx')&(pts.point=='RAV_frozen_bias')].set_index('id')[m]
            contrasts[name][m]=paired(first-second)
    missing={f'{a}/P{p:.2f}':sum(not r['oracle_matched_precision'][f'{p:.2f}']['attainable'] for r in records if r['arm']==a) for a in ARMS for p in PRECISIONS}
    result={'complete':True,'images':100,'arms':4,'frozen_baseline':json.loads((OUT/'frozen_baseline.json').read_text()),
            'ranking_means':means_rank,'operating_point_means':means_pts,'contrasts':contrasts,'oracle_unattainable':missing,
            'limitations':['repeatedly inspected Fundus; exploratory','oracle matched thresholds are NOT deployable',
                           'bootstrap conditional on fixed RAV bias and existing checkpoints; no model-selection uncertainty',
                           'RAV reused development, patient independence not established','no independent clinical confirmation'],
            'all_old_default_Dice_reproduced':True}
    write(OUT/'summary.json',result)
    print(json.dumps({'primary':contrasts['mass_minus_vascx']['decision_ap'],
                      'fixed_simple_baseline_comparison':contrasts['mass_default_minus_vascx_RAV_frozen_bias']['mean_av_dice']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','develop','evaluate','summarize'])
    args=parser.parse_args();cv2.setNumThreads(2)
    {'prepare':prepare,'develop':develop,'evaluate':evaluate,'summarize':summarize}[args.phase]()
