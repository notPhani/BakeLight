import torch
import os
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
EXPIREMENTAL = False
EPSILON = 1e-6
if EXPIREMENTAL:
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = 1
else:
    if "PYTORCH_NO_CUDA_MEMORY_CACHING" in os.environ:
        del os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"]
