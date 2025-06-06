import torch
from BakeLight.config import DEVICE

def dot(v1: torch.Tensor, v2: torch.Tensor, device=DEVICE) -> torch.Tensor:
    return torch.sum(v1*v2, dim=-1, device=DEVICE)
