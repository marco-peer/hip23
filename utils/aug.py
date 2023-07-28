import random

import kornia
import torch

from PIL import Image

class RandomApply(torch.nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class Resize:

    def __init__(self, size):
        self.max_size = size
    
        
    def __call__(self, img):
        aspect_ratio = img.height / img.width 

        if img.height >= img.width:
            new_height = self.max_size
            new_width = int(self.max_size / aspect_ratio)
        else:
            new_width = self.max_size
            new_height = int(self.max_size * aspect_ratio)

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

class Erosion:

    def __call__(self, tensor):
        return kornia.morphology.erosion(tensor, torch.rand(3,3).round().cuda())


class Dilation:
        
    def __call__(self, tensor):
        return kornia.morphology.dilation(tensor, torch.rand(3,3).round().cuda())