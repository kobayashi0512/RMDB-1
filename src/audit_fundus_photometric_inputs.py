"""Outcome-blind integrity audit of reused photometric inputs, no annotations."""
import ast
import json
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from rtnls_fundusprep.cfi_bounds import CFIBounds
from run_fundus_fair_four_arm import ROOT, RAW, CACHE, OUT, sha
from test_fixed_preprocessing import enhance

cv2.setNumThreads(2)
rows=[]
for entry in pd.read_csv(CACHE/'bounds_latest_batch.csv').itertuples():
    raw=np.asarray(Image.open(RAW/'images'/f'{entry.id}.png').convert('RGB'))
    config=ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)',r'\1',entry.bounds))
    transform,bounds=CFIBounds.from_dict(raw,config).crop(1024)
    rgb=transform.warp(raw,(1024,1024))
    enhanced=enhance(rgb,'clahe',bounds.make_binary_mask(.01))
    rgbpath=ROOT/'artifacts/fixed_preprocessing_trial/clahe/rgb'/f'{entry.id}.png'
    cepath=ROOT/'artifacts/fixed_preprocessing_trial/clahe/ce'/f'{entry.id}.png'
    assert np.array_equal(enhanced,np.asarray(Image.open(rgbpath)))
    ce=CFIBounds.from_dict(enhanced,bounds.to_dict()).contrast_enhanced_5
    assert np.array_equal(ce,np.asarray(Image.open(cepath))),entry.id
    rows.append({'id':entry.id,'clahe_rgb_sha256':sha(rgbpath),'clahe_ce_sha256':sha(cepath),'default_rgb_sha256':sha(CACHE/'preprocessed_rgb'/f'{entry.id}.png'),'default_ce_sha256':sha(CACHE/'preprocessed_ce'/f'{entry.id}.png')})
    if len(rows)%10==0:print(len(rows),'input pairs exact',flush=True)
assert len(rows)==100
pd.DataFrame(rows).to_csv(OUT/'photometric_input_sha256.csv',index=False)
(OUT/'photometric_input_audit.json').write_text(json.dumps({'n':100,'clahe_rgb_exact':True,'clahe_contrast_enhanced_exact':True,'no_annotation_access':True},indent=2))
print('PASS',flush=True)
