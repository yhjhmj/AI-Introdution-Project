import torch.nn as nn

from .discriminator import NLayerDiscriminator
from .generator import ResnetGenerator


class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G_AB = ResnetGenerator()
        self.G_BA = ResnetGenerator()
        self.D_A = NLayerDiscriminator()
        self.D_B = NLayerDiscriminator()
