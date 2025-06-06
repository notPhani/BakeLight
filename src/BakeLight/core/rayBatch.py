import torch
from BakeLight.config import DEVICE

class RayBatch:
    def __init__(self, origins, directions, device=DEVICE):
        self.origins = torch.as_tensor(origins, device=device, dtype=torch.float32)
        self.directions = torch.as_tensor(directions, device=device, dtype=torch.float32)
        self.directions = self.directions / (torch.norm(self.directions, dim=-1, keepdim=True) + 1e-8)
        if self.origins.shape[-1] != 3 or self.directions.shape[-1] != 3:
            raise ValueError("Ray origins/directions must be 3D vectors")
        if self.origins.shape[0] != self.directions.shape[0]:
            raise ValueError("Origins and directions must have the same batch size")

    def at(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute points along rays at parameter t.
        t can be a scalar or a tensor of shape (N,) matching the batch size.
        """
        t = t.to(self.origins.device)
        if t.ndim == 0:
            t = t.expand(self.origins.shape[0])
        return self.origins + self.directions * t.unsqueeze(-1)

    def __repr__(self):
        return f"Origins Shape : {self.origins.shape}\nDirections Shape: {self.directions.shape}\n"
