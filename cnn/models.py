"""Convolutional networks for CIFAR-10 (64 channels, stride-2 stem, D blocks of 3x3 convolutions, no BatchNorm).

    plain    h <- ReLU(conv(Q(h)))
    plaind   plain, Dirac (identity) initialisation of the block kernels
    plainln  h <- ReLU(Norm(conv(Q(h))))                    post-norm plain
    res      h <- h + s * conv(Q(ReLU(h))),  s = 1/sqrt(D)
    resc     res with fixed s = 1/sqrt(8)
    res1     res with s = 1
    preln    h <- h + conv(Q(ReLU(Norm(h))))                pre-norm residual
Norm is LayerNorm over channels (GroupNorm with one group) and stays at full precision. Q marks the quantizer
sites: every conv/linear weight and every activation entering a conv/linear after the stem.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision.common import Sites

ARCHS = ("plain", "plaind", "plainln", "res", "resc", "res1", "preln")


class CNN(nn.Module):
    def __init__(self, arch, D, C=64):
        super().__init__()
        assert arch in ARCHS, arch
        self.arch, self.D = arch, D
        if arch == "res":
            self.s = 1 / math.sqrt(D)
        elif arch in ("res1", "preln"):
            self.s = 1.0
        else:
            self.s = 1 / math.sqrt(8)
        self.stem = nn.Conv2d(3, C, 3, stride=2, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
        if arch in ("preln", "plainln"):
            self.norms = nn.ModuleList([nn.GroupNorm(1, C) for _ in range(D)])
            if arch == "preln":
                self.final_norm = nn.GroupNorm(1, C)
        if arch == "plaind":
            for m in self.convs:
                nn.init.dirac_(m.weight)
                m.weight.data += 0.01 * torch.randn_like(m.weight)
                nn.init.zeros_(m.bias)
        self.head = nn.Linear(C, 10)
        self.sites = Sites()

    @property
    def noise(self):
        return self.sites.noise

    @noise.setter
    def noise(self, value):
        self.sites.noise = value

    def conv(self, m, a, b):
        return F.conv2d(a, self.sites.weight(m.weight, b), m.bias, m.stride, m.padding)

    def block(self, i, h, b):
        m, Q = self.convs[i], self.sites.act
        if self.arch == "plainln":
            return F.relu(self.norms[i](self.conv(m, Q(h, b), b)))
        if self.arch.startswith("plain"):
            return F.relu(self.conv(m, Q(h, b), b))
        r = F.relu(self.norms[i](h)) if self.arch == "preln" else F.relu(h)
        return h + self.s * self.conv(m, Q(r, b), b)

    def stem_out(self, x, b):
        h = self.conv(self.stem, x, b)
        return F.relu(h) if self.arch.startswith("plain") else h

    def head_out(self, h, b):
        a = h if self.arch.startswith("plain") else F.relu(self.final_norm(h) if self.arch == "preln" else h)
        a = self.sites.act(a, b).mean((2, 3))
        return F.linear(a, self.sites.weight(self.head.weight, b), self.head.bias)

    def forward(self, x, b=32):
        self.sites.reset()
        h = self.stem_out(x, b)
        for i in range(self.D):
            h = self.block(i, h, b)
        return self.head_out(h, b)
