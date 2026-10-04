"""Small, randomly initialized U-Net. No pretrained components."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing), nn.ReLU(inplace=True),
        )


class CustomUNet(nn.Module):
    def __init__(self, base=8):
        super().__init__()
        if base < 4 or base % 4:
            raise ValueError('base must be a positive multiple of four')
        self.e1 = Block(3, base)
        self.e2 = Block(base, base*2)
        self.e3 = Block(base*2, base*4)
        self.bottleneck = Block(base*4, base*8)
        self.d3 = Block(base*12, base*4)
        self.d2 = Block(base*6, base*2)
        self.d1 = Block(base*3, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def join(x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                                       align_corners=False), skip], dim=1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        z = self.d3(self.join(z, c))
        z = self.d2(self.join(z, b))
        return self.head(self.d1(self.join(z, a)))
