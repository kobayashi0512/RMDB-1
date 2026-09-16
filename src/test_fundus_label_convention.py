"""Regression checks: Fundus-specific decoding must not change RAV training."""
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from evaluate_fundus_avseg_external import target_labels as fundus
from train_udct_av_pilot import target_labels as rav, softmax_np
from train_birap_v2 import RetinalDevelopmentDataset

colours=np.array([[(0,0,0),(255,0,0),(0,0,255),(0,255,0),(255,255,255),(255,0,255)]],np.uint8)
assert fundus(colours[:,:5]).tolist()==[[0,1,2,3,-100]]
assert rav(colours).tolist()==[[0,1,2,-100,-100,3]]
logits=np.array([[[2.,3.,4.,1.]]],np.float32);before=logits.copy()
softmax_np(logits)
assert np.array_equal(logits,before), 'softmax must not mutate its input features'
with tempfile.TemporaryDirectory(prefix='fundus-label-test-') as d:
    root=Path(d);mask=np.tile(colours[:,:5],(10,2,1))
    Image.fromarray(np.full((10,10,3),120,np.uint8)).save(root/'rgb.png')
    Image.fromarray(mask).save(root/'mask.png')
    np.save(root/'logits.npy',np.zeros((10,10,4),np.float32))
    record={'id':'test','source':'rav','image':root/'rgb.png','mask':root/'mask.png','logits':root/'logits.npy'}
    for convention,decoder in [('rav',rav),('fundus_avseg',fundus)]:
        record['label_convention']=convention
        result=RetinalDevelopmentDataset([record.copy()],False,10,(10,))[0]
        assert np.array_equal(result['target'].numpy(),decoder(mask))
print('PASS: official colours, RAV isolation, dataset loader, no softmax side effects')
