"""Exact/near-image duplicate screening, NOT proof of patient independence."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from evaluate_residual_ablation_external import OUT, TRAIN
import final_measurement_controls as fc
import three_readout_repeatability as tr

def fingerprint(path):
    rgb=np.asarray(Image.open(path).convert('RGB').resize((512,512),Image.Resampling.BILINEAR))
    exact=hashlib.sha256(rgb.tobytes()).hexdigest()
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    low=cv2.dct(cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].ravel()
    bits=low>np.median(low[1:]);bits[0]=False
    return exact,int.from_bytes(np.packbits(bits).tobytes(),'big')

def main():
    train=pd.read_csv(TRAIN/'input_manifest.csv',dtype={'id':str}).query('split=="training"')
    refs=[]
    for ident in train.id:
        path=fc.ROOT/'data/rav_vascx/preprocessed_rgb'/('RAV_'+ident+'.png')
        exact,ph=fingerprint(path);refs.append((ident,exact,ph,str(path)))
    manifest=pd.read_csv(fc.OUT/'manifest.csv',dtype=str)
    entries=[('mBRSET',r.id,tr.CACHE/'preprocessed_rgb'/(Path(r['file']).stem+'.png')) for _,r in manifest.iterrows()]
    entries += [('Fundus',p.stem,p) for p in sorted((fc.FCACHE/'preprocessed_rgb').glob('*.png'))]
    assert len(entries)==2060 and len(refs)==179
    rows=[]
    for cohort,key,path in entries:
        exact,ph=fingerprint(path)
        distances=[(ph^r[2]).bit_count() for r in refs];j=int(np.argmin(distances))
        rows.append({'cohort':cohort,'id':key,'path':str(path),'nearest_training_id':refs[j][0],
                     'nearest_training_path':refs[j][3],'min_phash_distance':distances[j],
                     'exact_resized_RGB_match':any(exact==r[1] for r in refs)})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'training_external_overlap_screen.csv',index=False)
    flagged=frame[(frame.min_phash_distance<=4)|frame.exact_resized_RGB_match]
    tr.write(OUT/'overlap_screen.json',{'training_images':179,'external_images':2060,
             'exact_resized_rgb_matches':int(frame.exact_resized_RGB_match.sum()),
             'phash_screen_threshold':4,'flagged_for_manual_review':flagged.to_dict('records'),
             'minimum_distance_by_cohort':frame.groupby('cohort').min_phash_distance.min().to_dict(),
             'limitations':'Screening only; perceptual hashes can miss transformed duplicates. No patient identifiers for RAV; absence of matches does not prove patient independence or audit VascX pretraining overlap.'})
    print(json.dumps({'exact':int(frame.exact_resized_RGB_match.sum()),'flagged':len(flagged),'minimum_distance':frame.groupby('cohort').min_phash_distance.min().to_dict()},indent=2))

if __name__=='__main__':main()
