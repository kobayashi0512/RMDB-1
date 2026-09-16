"""Unit checks, not measurements of scientific performance."""
import hashlib
import json
import torch
from train_controlled_residual_ablation import ARMS, MASKS, Cases, Crops, apply_mask, count, model_for, OUT
from backbone_runtime import atomic_json

def state_hash(net):
    return hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in net.state_dict().values())).hexdigest()

def main():
    x = torch.randn(2, 11, 32, 32)
    original = x.clone()
    for arm, (lo, hi) in MASKS.items():
        masked = apply_mask(x, arm)
        assert not masked[:, lo:hi].any()
        assert torch.equal(masked[:, :lo], x[:, :lo])
        assert torch.equal(masked[:, hi:], x[:, hi:])
        assert torch.equal(x, original)
    hashes = []
    for arm in ARMS:
        torch.manual_seed(20260914)
        net = model_for(arm)
        if arm != 'plain_conv': hashes.append(state_hash(net))
        with torch.no_grad():
            z = torch.randn(2, 4, 32, 32)
            y = net(apply_mask(x, arm), z)
        assert y.shape == z.shape and torch.isfinite(y).all()
    assert len(set(hashes)) == 1
    ids = json.loads((OUT / 'split.json').read_text())['training']
    ds = Crops(Cases(), ids, 20260914, 1)
    a, b = ds[0], ds[0]
    assert all(torch.equal(i, j) for i, j in zip(a, b))
    report = {'seven_forward_shapes_valid': True, 'input_masks_only_target_channels': True,
              'loss_features_unmodified': True, 'paired_initial_weights_identical': True,
              'crop_schedule_deterministic': True,
              'plain_to_reference_parameter_ratio': count(model_for('plain_conv')) / count(model_for('full'))}
    atomic_json(OUT / 'unit_checks.json', report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
