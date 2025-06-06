import torch
from BakeLight.config import DEVICE
from BakeLight.core.rayBatch import RayBatch
from BakeLight.core.hitInfo import HitInfo

class Sphere:
    def __init__(self, centers, radii, materials, device=DEVICE):
        self.centers =  torch.as_tensor(centers, dtype=torch.float32, device= device)
        self.radii = torch.as_tensor(radii, dtype=torch.float32, device= device)
        self.materials = materials

    def intersect(self,ray_batch: RayBatch)->HitInfo:
        B = ray_batch.origins.shape[0]
        N = self.centers.shape[0]
        oc = ray_batch.origins[:, None] - self.centers[None]  # (B, N, 3)
        a = torch.einsum('bi,bi->b', ray_batch.directions, ray_batch.directions)  # (B,)
        b = 2 * torch.einsum('bni,bi->bn', oc, ray_batch.directions)  # (B, N)
        c = torch.einsum('bni,bni->bn', oc, oc) - (self.radii**2)[None]  # (B, N)

        discriminant = b**2 - 4 * a[:, None] * c
        hit_mask = discriminant > 1e-6  # (B, N)
        sqrt_disc = torch.sqrt(discriminant[hit_mask])
        q = -0.5 * (b[hit_mask] + torch.sign(b[hit_mask]) * sqrt_disc)
        a_expanded = a.unsqueeze(1).expand(-1, N)  # (B,) → (B, N)
        t1 = q / a_expanded[hit_mask]  # Now works with (B, N) mask

        t2 = c[hit_mask] / q
        
        # Find smallest positive t
        t_vals = torch.where((t1 > 1e-6) & (t1 < t2), t1, t2)
        t_vals = torch.where(t_vals < 1e-6, float('inf'), t_vals)

        # Build hit info
        hit_t = torch.full((B, N), float('inf'), device='cuda')
        hit_t[hit_mask] = t_vals
        
        # Find closest hit per ray
        closest_t, closest_idx = torch.min(hit_t, dim=1)  # (B,)
        
        # Calculate hit normals
        hit_points = ray_batch.at(closest_t)  # (B, 3)
        hit_normals = (hit_points - self.centers[closest_idx]) / self.radii[closest_idx][:, None]
        
        return HitInfo(
            t=closest_t,
            material_idx=closest_idx,
            normal=hit_normals
        )
