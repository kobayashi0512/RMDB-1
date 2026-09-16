"""Engineering regression tests, independent of downstream comparison results."""
import numpy as np
import torch
from structure_restoration_pilot import Restorer,degrade,losses,gradients,Patches,OUT
torch.set_num_threads(2)
torch.manual_seed(100)
m=Restorer();x=torch.rand(2,3,64,64);x[:,:,:8,:]=0
assert torch.equal(m(x),x), 'identity initialization'
with torch.no_grad():m.head.bias.fill_(5)
y=m(x)
assert float((y-x).abs().max())<=.250001 and torch.equal(y[:,:,:8,:],x[:,:,:8,:])
mask=torch.zeros(2,1,64,64);mask[:,:,20:24,:]=1
loss,mae,edge=losses(y,x,mask,True);assert torch.isfinite(loss)
loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
z=torch.ones(1,3,32,32)*.5
loss,_,_=losses(z,z,torch.zeros(1,1,32,32),True);assert float(loss)==0
rgb=np.uint8(np.random.default_rng(3).uniform(0,255,(64,64,3)));rgb[:8]=0
a=degrade(rgb,np.random.default_rng(11),'moderate');b=degrade(rgb,np.random.default_rng(11),'moderate')
assert np.array_equal(a,b) and a.shape==rgb.shape and np.array_equal(a[:8],rgb[:8])
import json
p=json.loads((OUT/'protocol.json').read_text());assert not(set(p['training_ids'])&set(p['validation_ids']))
d=Patches(p['training_ids'],True);u=d[0];v=d[0]
assert all(torch.equal(a,b) for a,b in zip(u,v)), 'identical degradation/crop for both arms'
print('PASS: identity initialization, residual bound, background bypass, finite structural gradients, deterministic degradation/crops, split isolation')
