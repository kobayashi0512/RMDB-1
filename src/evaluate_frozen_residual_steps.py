"""Fixed-checkpoint one/two-step truncation sensitivity, separate from retraining."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import final_measurement_controls as fc
from train_controlled_residual_ablation import feature_tensor
from train_udct_av_pilot import Corrector
from backbone_runtime import atomic_json

OUT = fc.ROOT / 'artifacts/frozen_residual_steps_20260914_v1'

def main():
    OUT.mkdir(exist_ok=True); (OUT / 'cases').mkdir(exist_ok=True)
    torch.set_num_threads(2); cv2.setNumThreads(1)
    table = pd.read_csv(fc.FCACHE / 'bounds_latest_batch.csv').set_index('id')
    ids = sorted(table.index); assert len(ids) == 100
    checkpoint = fc.ROOT / 'models/baseline/artifacts/udct_av_mass_calibrated/best.pt'
    oldpath = fc.OUT / 'fundus_per_image.csv'
    protocol = {'arms': ['raw', 'step1', 'step2'], 'ids': ids,
                'intervention': 'Same frozen two-step-trained checkpoint, inspect first and second residual update. Step1 is inference truncation, NOT a separately trained one-step model.',
                'exploratory': True, 'no_training': True, 'no_parameter_selection': True,
                'geometry': 'Existing exactly replayed Fundus canonical crop and nearest-neighbor labels; common512FOV and original binary probability-sum threshold0.5',
                'metrics': 'All existing retention metrics; structural-only evidence, not phenotype repeatability',
                'bootstrap': {'unit': 'image', 'replicates': 5000, 'seed': 20260918, 'interval': 'unadjusted percentile95'},
                'source_hashes': {str(p): fc.sha(p) for p in [Path(__file__), checkpoint, oldpath, fc.ROOT / 'src/final_measurement_controls.py', fc.ROOT / 'src/train_controlled_residual_ablation.py']}}
    if (OUT / 'protocol.json').exists(): assert json.loads((OUT / 'protocol.json').read_text()) == protocol
    else: atomic_json(OUT / 'protocol.json', protocol)
    net = Corrector().eval()
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
    old = pd.read_csv(oldpath).set_index(['id', 'arm'])
    start = time.monotonic()
    for i, key in enumerate(ids):
        dest = OUT / 'cases' / (key + '.json')
        if dest.exists(): continue
        canonical, _, ann, _ = fc.geometry(key, table)
        rgb = np.asarray(Image.fromarray(canonical).resize((512,512), Image.Resampling.BILINEAR))
        fov = rgb.max(2) > 10
        truth = (cv2.resize(ann, (512,512), interpolation=cv2.INTER_NEAREST).max(2) > 0) & fov
        z = cv2.resize(np.load(fc.FCACHE / 'logits' / (key + '.npy')).astype(np.float32), (512,512), interpolation=cv2.INTER_LINEAR)
        raw = np.moveaxis(fc.softmax_np(z), -1, 0)
        x = feature_tensor(rgb, z)[None]
        base = torch.from_numpy(z).permute(2,0,1)[None]
        with torch.inference_mode():
            z1 = base + net.step(x, base.softmax(1))
            z2 = z1 + net.step(x, z1.softmax(1))
            p1 = z1.softmax(1)[0].numpy(); p2 = z2.softmax(1)[0].numpy()
        rows = []; replay = 0.
        for arm, prob in [('raw',raw), ('step1',p1), ('step2',p2)]:
            met = fc.retention(fc.binary_mask(prob, fov), truth)
            if arm in ['raw','step2']:
                legacy = 'raw' if arm == 'raw' else 'mass'
                err = max(abs(v - old.loc[(key,legacy),m]) for m,v in met.items())
                assert err < 1e-4, (key, arm, err)
                replay = max(replay, err)
            rows.append({'id': key, 'arm': arm, **met})
        atomic_json(dest, {'rows': rows, 'max_replay_error': replay})
        if (i+1) % 10 == 0: print('frozen steps', i+1, '/100', round(time.monotonic()-start,1), flush=True)
    cases = [json.loads((OUT/'cases'/(key+'.json')).read_text()) for key in ids]
    frame = pd.DataFrame([r for case in cases for r in case['rows']])
    frame.to_csv(OUT/'per_image.csv',index=False)
    metrics = [c for c in frame if c not in ['id','arm']]
    frame.groupby('arm')[metrics].mean().to_csv(OUT/'means.csv')
    rows=[]; rng=np.random.default_rng(20260918); sample=rng.integers(0,100,(5000,100))
    for metric in metrics:
        wide=frame.pivot(index='id',columns='arm',values=metric).loc[ids]
        for arm, ref in [('step1','raw'),('step2','raw'),('step2','step1')]:
            d=(wide[arm]-wide[ref]).to_numpy(); lo,hi=np.quantile(d[sample].mean(1),[.025,.975])
            rows.append({'metric':metric,'arm':arm,'reference':ref,'delta':d.mean(),'ci_low':lo,'ci_high':hi})
    pd.DataFrame(rows).to_csv(OUT/'paired_contrasts.csv',index=False)
    atomic_json(OUT/'complete.json',{'images':100,'arms':3,'max_replay_error':max(c['max_replay_error'] for c in cases),'same_checkpoint_truncation_only':True,'not_training_ablation':True})
    print(frame.groupby('arm')[metrics].mean().to_string())

if __name__ == '__main__': main()
