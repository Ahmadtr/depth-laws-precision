"""Vision Transformers for CIFAR-10 (patch size 4, width 128, four heads, MLP ratio 2).

    pre    h <- h + Attn(LN(h));  h <- h + MLP(LN(h))            pre-LN
    premu  pre-LN with branch scale s = 1/sqrt(D)
    post   h <- LN(h + Attn(h));  h <- LN(h + MLP(h))            post-LN
Quantizer sites: the weight of every linear layer (patch embedding, qkv, projection, fc1, fc2, head) and the
activation entering every linear layer (signed quantizer). Softmax and LayerNorm stay at full precision.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision.common import Sites

ARCHS = ("pre", "premu", "post")


class ViT(nn.Module):
    def __init__(self, arch, D, dim=128, heads=4, mlp=2):
        super().__init__()
        assert arch in ARCHS, arch
        self.arch, self.D, self.h = arch, D, heads
        self.s = 1 / math.sqrt(D) if arch == "premu" else 1.0
        self.embed = nn.Conv2d(3, dim, 4, stride=4)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.qkv = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(D)])
        self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(D)])
        self.fc1 = nn.ModuleList([nn.Linear(dim, mlp * dim) for _ in range(D)])
        self.fc2 = nn.ModuleList([nn.Linear(mlp * dim, dim) for _ in range(D)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(D)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(D)])
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
        self.sites = Sites(signed=True)

    @property
    def noise(self):
        return self.sites.noise

    @noise.setter
    def noise(self, value):
        self.sites.noise = value

    def lin(self, m, x, b):
        return F.linear(self.sites.act(x, b), self.sites.weight(m.weight, b), m.bias)

    def attn(self, i, x, b):
        B, T, C = x.shape
        q, k, v = self.lin(self.qkv[i], x, b).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, C)
        return self.lin(self.proj[i], a, b)

    def mlp(self, i, x, b):
        return self.lin(self.fc2[i], F.gelu(self.lin(self.fc1[i], x, b)), b)

    def forward(self, x, b=32):
        self.sites.reset()
        h = F.conv2d(x, self.sites.weight(self.embed.weight, b), self.embed.bias, stride=4)
        h = h.flatten(2).transpose(1, 2) + self.pos
        for i in range(self.D):
            if self.arch == "post":
                h = self.ln1[i](h + self.attn(i, h, b))
                h = self.ln2[i](h + self.mlp(i, h, b))
            else:
                h = h + self.s * self.attn(i, self.ln1[i](h), b)
                h = h + self.s * self.mlp(i, self.ln2[i](h), b)
        return self.lin(self.head, self.lnf(h).mean(1), b)
