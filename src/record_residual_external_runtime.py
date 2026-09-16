"""Record relevant runtime versions, without collecting account or host identifiers."""
import platform
import importlib.metadata
import torch
from evaluate_residual_ablation_external import OUT,save

def main():
    packages=['torch','numpy','pandas','scipy','scikit-image','Pillow','opencv-python','rtnls-fundusprep']
    versions={}
    for p in packages:
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]='package distribution metadata unavailable'
    save(OUT/'runtime.json',{'python':platform.python_version(),'platform':platform.platform(),
         'packages':versions,'device':'mps' if torch.backends.mps.is_available() else 'cpu',
         'evaluation_precision':'float32','neural_batch_size':4,
         'bitwise_cross_hardware_reproducibility_claimed':False})

if __name__=='__main__':main()
