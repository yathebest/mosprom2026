"""Install the CUDA-12/cuDNN-9 ORT profile locally without changing global ORT."""
from pathlib import Path
import subprocess
import sys

if __name__=='__main__':
    import torch
    if not torch.cuda.is_available() or not str(torch.version.cuda).startswith('12.'):
        raise SystemExit('This optional profile requires an existing CUDA 12 PyTorch installation')
    target=Path(__file__).resolve().parents[1]/'data/cache/ort-gpu-cu12'
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--progress-bar','off',
                    '--target',str(target),'--no-deps','onnxruntime-gpu==1.20.2'],check=True)
    print('Installed isolated ORT CUDA 12 profile:',target)
