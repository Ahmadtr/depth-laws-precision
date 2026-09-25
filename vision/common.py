"""CIFAR-10 data and quantizers shared by the CNN and ViT experiments (PyTorch)."""
import math
import os

import torch
import torch.nn.functional as F

import paths

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BITS = [1, 2, 3, 4, 5, 6, 8]
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_cifar():
    """Return (Xtr, Ytr, Xte, Yte) on the device, normalised per channel."""
    d = torch.load(os.path.join(paths.DATA, "cifar10.pt"), weights_only=False)
    prep = lambda x: ((x.float() / 255 - MEAN) / STD).to(DEV)
    return prep(d["train"][0]), d["train"][1].to(DEV), prep(d["test"][0]), d["test"][1].to(DEV)


def augment(x):
    """Per-image random horizontal flip and random 4-pixel crop with reflect padding."""
    N = x.shape[0]
    flip = torch.rand(N, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (N,), device=x.device)
    j = torch.randint(0, 9, (N,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar).view(N, 1, 32, 1).expand(N, 3, 32, 32)
    cols = (j[:, None] + ar).view(N, 1, 1, 32).expand(N, 3, 32, 32)
    n = torch.arange(N, device=x.device).view(N, 1, 1, 1)
    c = torch.arange(3, device=x.device).view(1, 3, 1, 1)
    return xp[n, c, rows, cols]


def quantize_weight(W, b):
    """Symmetric per-tensor LSQ-style quantizer, step 2*mean|W|/sqrt(q_p)."""
    if b >= 32:
        return W
    if b == 1:
        return torch.sign(W) * W.abs().mean()
    qp = 2 ** (b - 1) - 1
    s = 2 * W.abs().mean() / math.sqrt(qp)
    return torch.clamp(torch.round(W / s), -qp, qp) * s


def quantize_weight_minmax(W, b):
    """Symmetric per-tensor min-max quantizer, step max|W|/q_p."""
    if b >= 32:
        return W
    if b == 1:
        return torch.sign(W) * W.abs().mean()
    qp = 2 ** (b - 1) - 1
    s = W.abs().max() / qp
    return torch.round(W / s) * s


def quantize_unsigned(h, b, mean_pos=None):
    """Unsigned LSQ-style quantizer for post-ReLU activations; step 2*mean(h>0)/sqrt(q_p)."""
    if b >= 32:
        return h
    qp = 2 ** b - 1
    if mean_pos is None:
        pos = h[h > 0]
        s = (2 * pos.mean() / math.sqrt(qp)) if pos.numel() else torch.tensor(1.0, device=h.device)
    else:
        s = max(2 * mean_pos / math.sqrt(qp), 1e-12)
    return torch.clamp(torch.round(h / s), 0, qp) * s


def quantize_unsigned_minmax(h, b, max_value):
    """Unsigned min-max quantizer with a calibrated range [0, max_value]."""
    if b >= 32:
        return h
    qp = 2 ** b - 1
    s = max(max_value, 1e-12) / qp
    return torch.clamp(torch.round(h / s), 0, qp) * s


def quantize_signed(h, b, mean_abs=None):
    """Signed LSQ-style quantizer for transformer activations; step 2*mean|h|/sqrt(q_p)."""
    if b >= 32:
        return h
    m = h.abs().mean() if mean_abs is None else mean_abs
    if b == 1:
        return torch.sign(h) * m
    qp = 2 ** (b - 1) - 1
    s = max(float(2 * m / math.sqrt(qp)), 1e-12)
    return torch.clamp(torch.round(h / s), -qp, qp) * s


class Sites:
    """Perturbation policy applied at every quantizer site (weights and activations entering a conv/linear).

    noise > 0      relative Gaussian noise x + noise * rms(x) * N(0, I); drawn from `generator` when set
    quantizer      "lsq" (default) or "minmax"; activation steps are dynamic unless calibrated
    mode="calib"   record activation statistics per site and leave activations unchanged
    errs / stats   optional lists that collect relative quantization errors / activation moments
    """

    def __init__(self, signed=False):
        self.signed = signed
        self.noise, self.generator = 0.0, None
        self.quantizer, self.scales, self.mode = "lsq", None, None
        self.errs, self.stats, self.idx = None, None, 0

    def reset(self):
        self.idx = 0

    def _noisy(self, x):
        if self.generator is None:
            z = torch.randn_like(x)
        else:
            z = torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)
        return x + z * self.noise * x.detach().pow(2).mean().sqrt()

    def _record(self, x, xq):
        if self.errs is not None:
            self.errs.append(((xq - x).norm() / (x.norm() + 1e-12)).item())
        return xq

    def weight(self, W, b):
        if self.noise > 0:
            return self._record(W, self._noisy(W))
        q = quantize_weight_minmax if self.quantizer == "minmax" else quantize_weight
        return self._record(W, q(W, b))

    def act(self, h, b):
        i = self.idx
        self.idx += 1
        if self.stats is not None:
            x = h.detach().flatten().double()
            m, sd = x.mean(), x.std()
            self.stats.append((i, (((x - m) ** 4).mean() / sd ** 4 - 3).item()))
            return h
        if self.noise > 0:
            return self._noisy(h)
        if self.mode == "calib":
            self._calibrate(i, h)
            return h
        if b >= 32:
            return h
        s = None if self.scales is None else self.scales[i]
        if self.quantizer == "minmax":
            return self._record(h, quantize_unsigned_minmax(h, b, s))
        if self.signed:
            return self._record(h, quantize_signed(h, b, s))
        return self._record(h, quantize_unsigned(h, b, s))

    def _calibrate(self, i, h):
        if self.quantizer == "minmax":
            self.scales[i] = max(self.scales.get(i, 0.0), h.max().item())
        elif self.signed:
            self.scales.setdefault(i, []).append(h.detach().abs().mean().item())
        else:
            pos = h[h > 0]
            self.scales.setdefault(i, []).append(pos.mean().item() if pos.numel() else 0.0)


@torch.no_grad()
def calibrate(net, Xtr, quantizer="lsq", n=5000, bs=1000, weight_bits=8):
    """Fix every activation step from n training images (mean statistic for LSQ-style, max for min-max)."""
    S = net.sites
    S.quantizer, S.mode, S.scales = quantizer, "calib", {}
    for k in range(0, n, bs):
        net(Xtr[k:k + bs], weight_bits)
    if quantizer != "minmax":
        S.scales = {i: sum(v) / len(v) for i, v in S.scales.items()}
    S.mode = None


def uncalibrate(net):
    net.sites.quantizer, net.sites.scales, net.sites.mode = "lsq", None, None


@torch.no_grad()
def logits(net, X, n=None, bs=1000):
    n = len(X) if n is None else n
    return torch.cat([net(X[k:k + bs]) for k in range(0, n, bs)])


@torch.no_grad()
def accuracy(net, X, Y, b=32, bs=1000):
    net.eval()
    return sum((net(X[k:k + bs], b).argmax(1) == Y[k:k + bs]).sum().item() for k in range(0, len(X), bs)) / len(X)


@torch.no_grad()
def quant_error(net, X, b, n=500):
    """Mean relative quantization error over all sites at b bits."""
    net.sites.errs = []
    net(X[:n], b)
    e = sum(net.sites.errs) / len(net.sites.errs)
    net.sites.errs = None
    return e
