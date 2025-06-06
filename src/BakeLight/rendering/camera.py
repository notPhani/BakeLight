import torch
from BakeLight.config import DEVICE
from BakeLight.core.rayBatch import RayBatch

class Camera:
    def __init__(self, look_from, look_at, vfov, aspect_ratio, aperture=0.1, focus_dist=None):
        self.origin = torch.tensor(look_from, dtype=torch.float32, device=DEVICE)
        self.vfov = torch.deg2rad(torch.tensor(vfov))
        self.aspect_ratio = aspect_ratio
        self.aperture = aperture
        
        w = (self.origin - torch.tensor(look_at, device=DEVICE)).normalize()
        u = torch.cross(torch.tensor([0,1,0], device=DEVICE), w).normalize()
        v = torch.cross(w, u)
        
        if focus_dist is None:
            focus_dist = torch.norm(self.origin - torch.tensor(look_at, device=DEVICE))
        
        self.horizontal = focus_dist * 2 * torch.tan(self.vfov/2) * aspect_ratio * u
        self.vertical = focus_dist * 2 * torch.tan(self.vfov/2) * v
        self.lower_left = self.origin - self.horizontal/2 - self.vertical/2 - focus_dist*w
        self.lens_radius = aperture / 2

    def generateRays(self, width, height, jitter_strength, samples=1) -> RayBatch:
        """Batch ray generation with depth of field"""
        u = torch.linspace(0, 1, width, device=DEVICE)
        v = torch.linspace(0, 1, height, device=DEVICE)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
        
        # Jitter within pixel
        jitter_u = (grid_u + jitter_strength*torch.rand_like(grid_u)/width).reshape(-1)
        jitter_v = (grid_v + jitter_strength*torch.rand_like(grid_v)/height).reshape(-1)
        
        # Depth of field sampling
        rand_disk = self.lens_radius * self._random_in_unit_disk(jitter_u.shape[0])
        origins = self.origin + rand_disk[:,0,None]*u + rand_disk[:,1,None]*v
        
        directions = (self.lower_left + jitter_u[:,None]*self.horizontal + 
                     jitter_v[:,None]*self.vertical - origins)
        
        return RayBatch(origins, directions.normalize())

    def _random_in_unit_disk(self, n):
        """Vectorized version without infinite loop"""
        phi = 2 * torch.pi * torch.rand(n, device=DEVICE)
        r = torch.sqrt(torch.rand(n, device=DEVICE))
        return torch.stack([r*torch.cos(phi), r*torch.sin(phi)], -1)
