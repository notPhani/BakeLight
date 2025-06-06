import torch

class Material:
    def __init__(self, color, roughness=0.5, metallic=0.0, ior=1.5, emission=None):
        self.color = torch.tensor(color, dtype=torch.float32)
        self.roughness = float(roughness)
        self.metallic = float(metallic)
        self.ior = float(ior)
        self.emission = torch.tensor(emission, dtype=torch.float32) if emission is not None else None

    def __repr__(self):
        return (f"Material(color={self.color.tolist()}, roughness={self.roughness}, "
                f"metallic={self.metallic}, ior={self.ior}, emission={self.emission})")

    def scatter(self, ray_in, hit_point, normal, front_face=True):
        """
        Returns (scattered_ray_direction, attenuation, emission)
        """
        if self.emission is not None:
            return None, None, self.emission, hit_point

        # Lambertian (diffuse)
        if self.metallic < 0.1 and self.ior == 1.0:
            scatter_dir = normal + torch.randn_like(normal)
            scatter_dir = scatter_dir / (torch.norm(scatter_dir, dim=-1, keepdim=True) + 1e-8)
            attenuation = self.color
            return scatter_dir, attenuation, None, hit_point

        # Metal (specular)
        if self.metallic >= 0.1 and self.ior == 1.0:
            reflected = reflect(ray_in, normal)
            fuzz = self.roughness * torch.randn_like(reflected)
            scatter_dir = reflected + fuzz
            scatter_dir = scatter_dir / (torch.norm(scatter_dir, dim=-1, keepdim=True) + 1e-8)
            attenuation = self.color
            return scatter_dir, attenuation, None, hit_point

        # Dielectric (glass)
        if self.ior != 1.0:
            refraction_ratio = (1.0 / self.ior) if front_face else self.ior
            unit_direction = ray_in / (torch.norm(ray_in, dim=-1, keepdim=True) + 1e-8)
            cos_theta = torch.clamp((-unit_direction * normal).sum(-1, keepdim=True), max=1.0)
            sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
            cannot_refract = (refraction_ratio * sin_theta > 1.0).squeeze(-1)
            reflect_prob = schlick(cos_theta, refraction_ratio)
            reflect_mask = (torch.rand_like(reflect_prob) < reflect_prob).squeeze(-1) | cannot_refract

            reflected = reflect(unit_direction, normal)
            refracted = refract(unit_direction, normal, refraction_ratio)
            scatter_dir = torch.where(reflect_mask.unsqueeze(-1), reflected, refracted)
            attenuation = torch.ones_like(self.color)
            return scatter_dir, attenuation, None, hit_point

        # Default: diffuse
        scatter_dir = normal + torch.randn_like(normal)
        scatter_dir = scatter_dir / (torch.norm(scatter_dir, dim=-1, keepdim=True) + 1e-8)
        attenuation = self.color

        return scatter_dir, attenuation, None, hit_point

# --- Helper functions ---
def reflect(v, n):
    return v - 2 * (v * n).sum(-1, keepdim=True) * n

def refract(uv, n, etai_over_etat):
    cos_theta = (-uv * n).sum(-1, keepdim=True)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -torch.sqrt(torch.abs(1.0 - (r_out_perp ** 2).sum(-1, keepdim=True))) * n
    return r_out_perp + r_out_parallel

def schlick(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 ** 2
    return r0 + (1 - r0) * (1 - cosine) ** 5

