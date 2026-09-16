"""Independent source-file verification and convolution-MAC accounting."""
import json
import pandas as pd
import torch
from torch import nn
from prepare_residual_ablation import OUT, RAW, CACHE, sha
from train_controlled_residual_ablation import ARMS, model_for, count
from backbone_runtime import atomic_json

def main():
    frame = pd.read_csv(OUT / 'input_manifest.csv', dtype={'id':str})
    checked = 0
    for row in frame.itertuples():
        name = 'RAV_' + row.id
        items = [('original_rgb_sha256', RAW/'dataset/rgb'/(name+'.png')),
                 ('rgb_sha256',CACHE/'preprocessed_rgb'/(name+'.png')),
                 ('mask_sha256',RAW/'dataset/av_masks'/(name+'.png')),
                 ('logits_sha256',CACHE/'logits'/(name+'.npy')),
                 ('aligned_mask_sha256',OUT/'aligned_masks'/(name+'.png'))]
        for column, path in items:
            assert sha(path) == getattr(row,column), str(path)
            checked += 1
    records=[]
    for arm in ARMS:
        model=model_for(arm).eval(); costs=[]
        def hook(module, inputs, output):
            costs.append(output.numel() * (module.in_channels//module.groups) * module.kernel_size[0] * module.kernel_size[1])
        handles=[m.register_forward_hook(hook) for m in model.modules() if isinstance(m,nn.Conv2d)]
        with torch.inference_mode(): model(torch.zeros(1,11,32,32),torch.zeros(1,4,32,32))
        for handle in handles: handle.remove()
        records.append({'arm':arm,'parameters':count(model),'conv_MACs_per_256_crop':sum(costs)*64})
    atomic_json(OUT/'input_and_compute_audit.json',{'input_files_verified':checked,'images':len(frame),
                'conv_MACs_only_excludes_norm_activations_resize_softmax':True,'arms':records,
                'not_FLOP_matched':True,'not_wall_clock_benchmark':True})
    print(json.dumps({'verified_input_files':checked,'compute':records},indent=2))

if __name__ == '__main__': main()
