"""Independent AV Dice calculation and checkpoint/duplicate audit after completion."""
import ast
import json
import re
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rtnls_fundusprep.cfi_bounds import CFIBounds
from rtnls_fundusprep.transformation import Interpolation
from structure_restoration_pilot import ROOT,OUT,sha,write

assert (OUT/'complete.json').exists(), 'Full fixed experiment must finish first'
p=json.loads((OUT/'protocol.json').read_text());h=json.loads((OUT/'history.json').read_text())
orig=json.loads((OUT/'input_sha256.json').read_text())
train_hash={orig[k]['image'] for k in p['training_ids']};val_hash={orig[k]['image'] for k in p['validation_ids']}
assert not train_hash&val_hash, 'Exact image file duplicate across development split'
cache=ROOT/'data/rav_vascx';raw=ROOT/'data/raw/RAV/data/external/RAV/dataset'
bounds=pd.read_csv(cache/'bounds_latest_batch.csv').set_index('id')
frame=pd.read_csv(OUT/'evaluation/per_image.csv',dtype={'id':str}).set_index(['id','condition','arm'])
assert len(frame)==240 and not frame.index.duplicated().any()
max_error=0.;hashes={}
for ident in p['rav_evaluation_ids']:
    name='RAV_'+ident;rgb=np.asarray(Image.open(raw/'rgb'/f'{name}.png').convert('RGB'))
    ann=np.asarray(Image.open(raw/'av_masks'/f'{name}.png').convert('RGB'))
    b=ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)',r'\1',bounds.loc[name,'bounds']))
    t,_=CFIBounds.from_dict(rgb,b).crop(1024)
    ann=cv2.resize(t.warp(ann,(1024,1024),mode=Interpolation.NEAREST),(768,768),interpolation=cv2.INTER_NEAREST)
    valid=ann[:,:,1]==0;ga=(ann[:,:,0]>0)&valid;gv=(ann[:,:,2]>0)&valid
    for condition in ('undegraded','moderate'):
        for arm in ('input','clahe','reconstruction','structure'):
            path=OUT/'evaluation/probabilities'/f'{ident}__{condition}__{arm}.npy'
            prob=np.load(path);assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
            assert np.allclose(prob.sum(0),1,atol=2e-6)
            pred=prob.argmax(0);pa=((pred==1)|(pred==3))&valid;pv=((pred==2)|(pred==3))&valid
            dice=.5*(2*(pa&ga).sum()/(pa.sum()+ga.sum())+2*(pv&gv).sum()/(pv.sum()+gv.sum()))
            err=abs(dice-frame.loc[(ident,condition,arm),'mean_av_dice']);assert err<1e-7
            max_error=max(max_error,float(err));hashes[str(path.relative_to(OUT))]=sha(path)
    print('audit',ident,flush=True)
selected={}
for arm in ('reconstruction','structure'):
    checkpoint=torch.load(OUT/f'{arm}_best.pt',map_location='cpu')
    best=min(h,key=lambda row:row['validation'][arm]['selection_score'])
    assert checkpoint['epoch']==best['epoch']
    selected[arm]={'epoch':checkpoint['epoch'],'best_at_budget_boundary':checkpoint['epoch']==20}
report={'evaluations':240,'all_probabilities_valid':True,'independent_AV_Dice_max_abs_error':max_error,'exact_training_validation_file_hash_overlap':0,'patient_independence_not_established':True,'checkpoint_selection_reproduced':selected,'no_new_fit_or_threshold_selection':True}
write(OUT/'independent_audit.json',report);write(OUT/'output_sha256.json',hashes)
print(json.dumps(report,indent=2))
