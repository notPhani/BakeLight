import torch
from BakeLight.config import DEVICE

class RayBatch:
    def __init__(self, origins, directions, device=DEVICE):
        self.origins = torch.as_tensor(origins, device=device, dtype=torch.float32)
        self.directions = torch.as_tensor(directions, device=device, dtype=torch.float32)
        self.directions = self.directions / (torch.norm(self.directions, dim=-1, keepdim=True) + 1e-8)
        if self.origins.shape[-1] != 3 or self.directions.shape[-1] != 3:
            raise Exception("Ray origins/directions must be 3D vectors")
    def at(self,t: torch.Tensor) -> torch.Tensor:
        return self.origins + self.directions*(t.unsqueeze(-1).to(DEVICE))
    def __repr__(self):
        return f"Origins Shape : {self.origins.shape} \n Directions Shape: {self.directions.shape}\n"
    
