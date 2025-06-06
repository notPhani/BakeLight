import torch
from BakeLight.config import DEVICE, EPSILON
from BakeLight.core.rayBatch import RayBatch
from BakeLight.core.hitInfo import HitInfo

class Sphere:
    def __init__(self, centers, radii, materials, device=DEVICE):
        self.centers = torch.as_tensor(centers, dtype=torch.float32, device=device)
        self.radii = torch.as_tensor(radii, dtype=torch.float32, device=device)
        self.materials = materials
        
        if self.centers.ndim != 2 or self.centers.shape[-1] != 3:
            raise ValueError("centers must be (N,3)")
        if self.radii.shape != (self.centers.shape[0],):
            raise ValueError("radii must match sphere count")
        if len(materials) != self.centers.shape[0]:
            raise ValueError("Material count ≠ sphere count")

    def intersect(self, ray_batch: RayBatch) -> HitInfo:
        B = ray_batch.origins.shape[0]
        N = self.centers.shape[0]
        
        oc = ray_batch.origins[:, None] - self.centers[None]
        a = torch.sum(ray_batch.directions ** 2, dim=-1)
        b = 2 * torch.einsum('bni,bi->bn', oc, ray_batch.directions)
        c = torch.einsum('bni,bni->bn', oc, oc) - (self.radii**2)[None]
        
        discriminant = b**2 - 4 * a[:, None] * c
        hit_mask = discriminant > EPSILON
        sqrt_disc = torch.sqrt(discriminant.clamp(min=0))
        
        q = -0.5 * (b + torch.sign(b) * sqrt_disc)
        t1 = q / (a[:, None] + EPSILON)
        t2 = c / (q + EPSILON)
        
        t_vals = torch.where(
            (t1 > EPSILON) & (t1 < t2), 
            t1, 
            torch.where(t2 > EPSILON, t2, torch.inf)
        )
        
        hit_t = torch.full((B, N), float('inf'), device=self.centers.device)
        hit_t[hit_mask] = t_vals[hit_mask]
        
        closest_t, closest_idx = torch.min(hit_t, dim=1)
        valid_hits = closest_t < torch.inf
        
        hit_points = ray_batch.at(closest_t)
        hit_normals = torch.zeros_like(hit_points)
        hit_normals[valid_hits] = (
            hit_points[valid_hits] - 
            self.centers[closest_idx][valid_hits]
        ) / self.radii[closest_idx][valid_hits, None]
        
        return HitInfo(
            t=closest_t,
            hit_mask=valid_hits,
            material_idx=closest_idx,
            normal=hit_normals
        )
