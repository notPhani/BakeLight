import torch

class Material:
    def __init__(self, color, roughness, metallic, specularity, em_strength, em_color, ir):
        self.color = torch.tensor(color, dtype=torch.float32)
        self.roughness = float(roughness)
        self.metallic = float(metallic)
        self.specularity = float(specularity)
        self.em_strength = float(em_strength)
        self.em_color = torch.tensor(em_color, dtype=torch.float32)
        self.ir = float(ir) 

    def __repr__(self):
        return (
            f"Material(color={self.color.tolist()},\n roughness={self.roughness},\n "
            f"metallic={self.metallic},\n specularity={self.specularity},\n "
            f"em_strength={self.em_strength},\n em_color={self.em_color.tolist()},\n "
            f"ir={self.ir})")
    
    

