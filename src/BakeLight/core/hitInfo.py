
class HitInfo:
    def __init__(self, t, material_idx, normal):
        self.t = t            # (B,) hit distances
        self.material_idx = material_idx  # (B,) material indices
        self.normal = normal  # (B, 3) surface normals

    def __repr__(self):
        return f"t: {self.t}\n material_idx: {self.material_idx}\n normal: {self.normal}"