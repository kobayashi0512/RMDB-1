#!/usr/bin/env python3
"""Deterministic source-image comparison; no generated medical content."""
import json
import ast
import re
import cv2
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from rtnls_fundusprep.cfi_bounds import CFIBounds
from rtnls_fundusprep.transformation import Interpolation
from train_udct_av_pilot import target_labels

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/fixed_preprocessing_trial'
IDENT='001_G'  # first lexical image, fixed before viewing trial results
ARMS=['default','illumination','clahe','denoise_sharpen']
LABELS=['Default input','Illumination correction','Local contrast (CLAHE)','Denoising + sharpening']
brief={'figure_goal':'Inspect the appearance of four fixed preprocessing arms',
       'paper_claim':'Visual comparison only; diagnostic value is quantified separately',
       'figure_type':'real image comparison','mode':'plot',
       'panels':['full canonical RGB images','fixed 384 x 384 central crop'],
       'must_keep_labels':LABELS, 'data':{'id':IDENT,'crop_xyxy':[320,320,704,704]},
       'style_constraints':['real dataset images only','identical crop','white background','readable sans serif'],
       'output_formats':['png','pdf'],
       'verification_checklist':['same source image in every column','identical geometric crop','no synthetic anatomy','all input paths recorded']}
fig,axes=plt.subplots(2,4,figsize=(15,8.2),layout='constrained')
paths=[]
for j,(arm,label) in enumerate(zip(ARMS,LABELS)):
    path=(ROOT/'data/fundus_avseg_vascx/preprocessed_rgb'/f'{IDENT}.png') if arm=='default' else OUT/arm/'rgb'/f'{IDENT}.png'
    rgb=np.asarray(Image.open(path).convert('RGB'));assert rgb.shape==(1024,1024,3)
    paths.append(str(path))
    axes[0,j].imshow(rgb);axes[0,j].set_title(label,fontsize=15,fontweight='bold')
    axes[1,j].imshow(rgb[320:704,320:704]);axes[1,j].set_title('Same fixed central crop',fontsize=12,color='#525b64')
    for ax in axes[:,j]:ax.set_axis_off()
fig.suptitle('Fundus-AVSeg 001_G | Actual input images and fixed crops',fontsize=18)
fig.savefig(OUT/'input_comparison.png',dpi=180,facecolor='white')
fig.savefig(OUT/'input_comparison.pdf',facecolor='white')
brief['data']['paths']=paths
(OUT/'figure_brief.json').write_text(json.dumps(brief,indent=2))
plt.close(fig)

fig,axes=plt.subplots(1,5,figsize=(19,4.5),layout='constrained')
for j,(arm,label) in enumerate(zip(ARMS,LABELS)):
    mask=Image.open(OUT/arm/'prediction'/f'{IDENT}.png')
    result=json.loads((OUT/'cases'/f'{IDENT}__{arm}.json').read_text())
    axes[j].imshow(mask)
    axes[j].set_title(f"{label}\nAV Dice {result['mean_av_dice']:.4f}",fontsize=13)
bounds=pd.read_csv(ROOT/'data/fundus_avseg_vascx/bounds_latest_batch.csv').set_index('id')
config=ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)',r'\1',bounds.loc[IDENT,'bounds']))
raw=ROOT/'data/raw/Fundus_AVSeg/Fundus-AVSeg'
rgb=np.asarray(Image.open(raw/'images'/f'{IDENT}.png').convert('RGB'))
ann=np.asarray(Image.open(raw/'annotation'/f'{IDENT}.png').convert('RGB'))
transform,_=CFIBounds.from_dict(rgb,config).crop(1024)
gt=target_labels(cv2.resize(transform.warp(ann,(1024,1024),mode=Interpolation.NEAREST),(768,768),interpolation=cv2.INTER_NEAREST))
palette=np.array([[0,0,0],[232,92,73],[61,141,219],[180,116,210]],np.uint8)
display=palette[np.maximum(gt,0)];display[gt==-100]=[210,210,210]
axes[4].imshow(display);axes[4].set_title('Aligned reference\nGray = ignored AV labels',fontsize=13)
for ax in axes:ax.set_axis_off()
fig.suptitle('Fundus-AVSeg 001_G | Frozen model predictions and original annotation',fontsize=17)
fig.savefig(OUT/'prediction_comparison.png',dpi=180,facecolor='white')
fig.savefig(OUT/'prediction_comparison.pdf',facecolor='white')
plt.close(fig)
