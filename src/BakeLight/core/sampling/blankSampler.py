import torch
from BakeLight.config import DEVICE

class blankSampler:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.num_samples = width*height
    def generate_jitter(self):
        zero_grid = torch.zeros([self.width, self.height,2], dtype=torch.float32, device=DEVICE)
        return zero_grid
    