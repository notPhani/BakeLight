import torch
from BakeLight.config import DEVICE
from BakeLight.core.rayBatch import RayBatch
from BakeLight.core.hitInfo import HitInfo
class Scene:
    def __init__(self, objects, lights, background_color = torch.tensor([0.0,0.0,0.0],device=DEVICE)):
        self.background_color = background_color
        self.objects = objects
        self.lights = lights
    def interact(self, rayBatch:RayBatch)->HitInfo:
        return self.objects.intersect(rayBatch)
    