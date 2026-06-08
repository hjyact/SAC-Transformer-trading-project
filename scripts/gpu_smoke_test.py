import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import configure_runtime

info = configure_runtime(
    device=os.getenv("SAC_DEVICE", "auto"),
    cpu_workers=int(os.getenv("SAC_CPU_WORKERS", "0") or "0") or None,
)

import torch

print(f"device={info['device']}")
print(f"cpu_workers={info['cpu_workers']}")
print(f"torch={torch.__version__}")
print(f"torch_threads={torch.get_num_threads()}")
print(f"cuda_available={torch.cuda.is_available()}")

if torch.cuda.is_available():
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
    print(f"cuda_matmul_ok={tuple(y.shape)}")
else:
    x = torch.randn(2048, 2048)
    y = x @ x
    print(f"cpu_matmul_ok={tuple(y.shape)}")
