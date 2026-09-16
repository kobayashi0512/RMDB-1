"""Bounded new-protocol RAV input/iteration ablations, not legacy mass-weight reproduction."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader

from train_udct_av_pilot import Corrector, hessian_field, softmax_np, target_labels, loss_fn, metrics
from backbone_runtime import atomic_checkpoint, atomic_json
from prepare_residual_ablation import ROOT, OUT, CACHE, sha

ARMS = ['full', 'no_rgb', 'no_logit_input', 'no_entropy_input', 'no_hessian_input',
        'one_step', 'plain_conv']
SEEDS = [20260914, 20260915, 20260916]
EPOCHS = 3
MASKS = {'no_rgb': (0, 3), 'no_logit_input': (3, 7), 'no_entropy_input': (7, 8),
         'no_hessian_input': (8, 11)}

class PlainConv(nn.Module):
    """Single-resolution residual convolution stack, shared across two updates."""
    def __init__(self, width):
        super().__init__()
        layers = []
        for inp in [15, width, width, width]:
            layers += [nn.Conv2d(inp, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU()]
        self.net = nn.Sequential(*layers, nn.Conv2d(width, 4, 1))
    def forward(self, x, z):
        for _ in range(2):
            z = z + self.net(torch.cat([x, z.softmax(1)], 1))
        return z

def count(model):
    return sum(p.numel() for p in model.parameters())

def model_for(arm):
    if arm != 'plain_conv':
        return Corrector(iterations=1 if arm == 'one_step' else 2)
    reference = count(Corrector())
    width = min(range(8, 161, 8), key=lambda w: abs(count(PlainConv(w)) - reference))
    return PlainConv(width)

def feature_tensor(rgb, logits):
    probability = softmax_np(logits)
    entropy = -(probability * np.log(np.maximum(probability, 1e-8))).sum(-1) / np.log(4)
    return torch.from_numpy(np.concatenate([
        np.moveaxis(rgb.astype(np.float32) / 255, -1, 0),
        np.moveaxis(np.clip(logits, -12, 12) / 6, -1, 0),
        entropy[None].astype(np.float32), hessian_field(rgb)], axis=0))

class Cases:
    def __init__(self):
        self.cache = {}
    def get(self, key):
        if key not in self.cache:
            name = 'RAV_' + key
            rgb = np.asarray(Image.open(CACHE / 'preprocessed_rgb' / (name + '.png')).convert('RGB'))
            ann = np.asarray(Image.open(OUT / 'aligned_masks' / (name + '.png')).convert('RGB'))
            z = np.load(CACHE / 'logits' / (name + '.npy'))
            self.cache[key] = (rgb, z, ann)
        return self.cache[key]

class Crops(Dataset):
    def __init__(self, cases, ids, seed, epoch):
        self.cases, self.ids, self.seed, self.epoch = cases, ids, seed, epoch
    def __len__(self):
        return len(self.ids) * 2
    def __getitem__(self, index):
        key = self.ids[index % len(self.ids)]
        rgb, z, ann = self.cases.get(key)
        rng = random.Random(f'{self.seed}/{self.epoch}/{index}')
        y, x = rng.randrange(769), rng.randrange(769)
        rgb, z, ann = rgb[y:y+256, x:x+256], z[y:y+256, x:x+256].astype(np.float32), ann[y:y+256, x:x+256]
        return feature_tensor(rgb, z), torch.from_numpy(z.copy()).permute(2, 0, 1), torch.from_numpy(target_labels(ann))

def apply_mask(x, arm):
    if arm not in MASKS:
        return x
    y = x.clone()
    lo, hi = MASKS[arm]
    y[:, lo:hi] = 0
    return y

def prepare_protocol():
    audit = json.loads((OUT / 'input_audit.json').read_text())
    assert audit['all_rgb_replayed_exactly']
    parameters = {a: count(model_for(a)) for a in ARMS}
    assert abs(parameters['plain_conv'] / parameters['full'] - 1) < .10
    protocol = {
        'purpose': 'New controlled retraining to test feature inputs and shared residual iterations; no claim of exact legacy mass-corrector reproduction',
        'exploratory': True, 'arms': ARMS, 'training_seeds': SEEDS, 'epochs': EPOCHS,
        'split_sha256': sha(OUT / 'split.json'), 'manifest_sha256': sha(OUT / 'input_manifest.csv'),
        'source_sha256': {p.name: sha(p) for p in [Path(__file__), ROOT / 'src/train_udct_av_pilot.py', ROOT / 'src/prepare_residual_ablation.py']},
        'parameters': parameters, 'batch_size': 4, 'crop': 256, 'training_repeats_per_image': 2,
        'optimizer': 'AdamW lr=0.0002 weight_decay=0.0001; gradient clipping=5; no schedule',
        'loss': 'Existing loss_fn default: CE + 0.5 Dice + 0.1 directional + 0.1 vessel BCE + 0.01 identity; av_mass_weight=0 and vessel_mass_weight=0 explicitly',
        'loss_features': 'All loss terms receive intact fixed features for every arm. Input ablations remove network input only, not associated regularizers.',
        'legacy_limitation': 'Historical mass-calibration weights and split file are unavailable. This study cannot isolate the historical checkpoint training recipe.',
        'training_budget': 'Same optimizer updates, examples and crop schedule across arms. one_step uses less compute; plain_conv has matched parameters within 10%, not matched FLOPs.',
        'checkpoint': 'Final epoch only; validation logged, no best-epoch or arm selection',
        'splits': 'Fixed image split reconstructed once using legacy code seed 20260729: 179 train,20 validation,7 historical analogue-camera holdout. Patient independence unverified.',
        'evaluation': '512 image and logit resize then softmax; aligned nearest-neighbor annotation; same convention for all arms',
        'endpoints': 'All per-image artery/vein/mean A-V Dice,type accuracy,hard A-V log-ratio error; no repeatability claim from RAV',
        'stop_rule': 'Complete three seeds and all seven arms once. No holdout-driven hyperparameter changes. Only smoke runtime/safety fixes before protocol freeze.',
        'external_data': 'No Fundus, mBRSET or clinical outcomes used for fitting or model selection',
        'insufficient_for': ['convergence claim after three epochs', 'clinical translation', 'independent confirmation', 'full repeatability validation of new checkpoints']}
    p = OUT / 'training_protocol.json'
    if p.exists():
        assert json.loads(p.read_text()) == protocol, 'Frozen protocol changed; create a new version'
    else:
        atomic_json(p, protocol)
    return protocol

@torch.no_grad()
def evaluate(net, arm, cases, ids, device):
    net.eval()
    rows = []
    for key in ids:
        rgb, z, ann = cases.get(key)
        rgb = np.asarray(Image.fromarray(rgb).resize((512, 512), Image.Resampling.BILINEAR))
        z = cv2.resize(z.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR)
        ann = np.asarray(Image.fromarray(ann).resize((512, 512), Image.Resampling.NEAREST))
        x = feature_tensor(rgb, z)[None].to(device)
        base = torch.from_numpy(z).permute(2, 0, 1)[None].to(device)
        pred = net(apply_mask(x, arm), base).argmax(1)[0].cpu().numpy()
        rows.append({'id': key, **metrics(pred, target_labels(ann))})
    return pd.DataFrame(rows)

def smoke():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cases = Cases()
    ids = json.loads((OUT / 'split.json').read_text())['training']
    data = Crops(cases, ids, SEEDS[0], 1)
    batch = next(iter(DataLoader(data, batch_size=4)))
    x, z, y = [t.to(device) for t in batch]
    records = []
    for arm in ARMS:
        torch.manual_seed(SEEDS[0])
        model = model_for(arm).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        times = []
        for _ in range(3):
            start = time.monotonic()
            opt.zero_grad(set_to_none=True)
            pred = model(apply_mask(x, arm), z)
            loss, _ = loss_fn(pred, z, x, y, av_mass_weight=0., vessel_mass_weight=0.)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            if device.type == 'mps': torch.mps.synchronize()
            times.append(time.monotonic() - start)
        records.append({'arm': arm, 'parameters': count(model), 'seconds_per_batch_after_warmup': float(np.mean(times[1:])), 'finite_loss_and_gradients': True})
        del model, opt, pred, loss
        if device.type == 'mps': torch.mps.empty_cache()
        print(records[-1], flush=True)
    atomic_json(OUT / 'training_smoke.json', {'device': str(device), 'runs': records, 'not_scientific_results': True})

def train_all():
    protocol = prepare_protocol()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cases = Cases()
    split = json.loads((OUT / 'split.json').read_text())
    for seed in SEEDS:
        for arm in ARMS:
            dest = OUT / 'runs' / arm / str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'complete.json').exists(): continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            net = model_for(arm).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
            history = []
            first = 1
            if (dest / 'last.pt').exists():
                ck = torch.load(dest / 'last.pt', map_location='cpu', weights_only=False)
                net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
                for state in opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and k != 'step': state[k] = v.to(device)
                first = ck['epoch'] + 1; history = ck['history']
            for epoch in range(first, EPOCHS + 1):
                net.train(); start = time.monotonic(); losses = []
                loader = DataLoader(Crops(cases, split['training'], seed, epoch), batch_size=4,
                                    shuffle=True, generator=torch.Generator().manual_seed(seed + epoch), num_workers=0)
                for step, batch in enumerate(loader, 1):
                    x, z, y = [t.to(device) for t in batch]
                    opt.zero_grad(set_to_none=True)
                    pred = net(apply_mask(x, arm), z)
                    loss, _ = loss_fn(pred, z, x, y, av_mass_weight=0., vessel_mass_weight=0.)
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); opt.step()
                    losses.append(loss.item())
                    if step % 15 == 0:
                        atomic_json(OUT / 'progress.json', {'seed': seed, 'arm': arm, 'epoch': epoch, 'step': step, 'steps_per_epoch': len(loader)})
                val = evaluate(net, arm, cases, split['validation'], device)
                val.to_csv(dest / f'validation_epoch_{epoch}.csv', index=False)
                record = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_mean_av_dice': float(val.mean_av_dice.mean()), 'seconds': time.monotonic() - start, 'optimizer_steps': len(loader)}
                history.append(record)
                atomic_checkpoint(dest / 'last.pt', {'model': net.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'history': history})
                atomic_json(dest / 'history.json', history)
                print(json.dumps({'seed': seed, 'arm': arm, **record}), flush=True)
            test = evaluate(net, arm, cases, split['historical_camera_holdout'], device)
            test.to_csv(dest / 'historical_camera_holdout.csv', index=False)
            atomic_json(dest / 'complete.json', {'arm': arm, 'seed': seed, 'epochs': EPOCHS, 'checkpoint_sha256': sha(dest / 'last.pt'), 'protocol_sha256': sha(OUT / 'training_protocol.json'), 'exploratory': True})
            del net, opt
            if device.type == 'mps': torch.mps.empty_cache()
    atomic_json(OUT / 'training_complete.json', {'runs': len(SEEDS) * len(ARMS), 'protocol_sha256': sha(OUT / 'training_protocol.json')})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke', 'prepare', 'train'])
    args = parser.parse_args()
    if args.stage == 'smoke': smoke()
    elif args.stage == 'prepare': print(json.dumps(prepare_protocol(), indent=2))
    else: train_all()
