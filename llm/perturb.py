"""Perturbation and quantization of the linear layers inside the blocks of a pretrained causal language model.

Sites are the weight and the input of every linear layer in the transformer blocks (GPT-2 Conv1D, GPT-NeoX
Linear). Embeddings and the output head are excluded (GPT-2 ties them). Activation perturbations use forward
pre-hooks; weight perturbations overwrite the weights in place, and restore() puts the originals back.
With per_channel=True, noise scales and quantizer steps are computed per output channel (weights) and per
feature (activations) instead of per tensor.
"""
import math

import torch


def quantize_weight(W, b, axis=None, per_channel=False):
    qp = 2 ** (b - 1) - 1
    m = W.abs().mean(dim=axis, keepdim=True) if (per_channel and axis is not None) else W.abs().mean()
    s = 2 * m / math.sqrt(qp)
    return torch.clamp(torch.round(W / s), -qp, qp) * s


def quantize_signed(x, b, mean_abs):
    qp = 2 ** (b - 1) - 1
    s = torch.clamp(torch.as_tensor(2 * mean_abs / math.sqrt(qp), device=x.device, dtype=x.dtype), min=1e-12)
    return torch.clamp(torch.round(x / s), -qp, qp) * s


class Perturber:
    """mode: None | "noise" | "calib" | "quant"."""

    def __init__(self, model, per_channel=False, generator=None):
        self.mods = [m for n, m in model.named_modules() if type(m).__name__ in ("Linear", "Conv1D")
                     and n not in ("lm_head", "embed_out") and ("h." in n or "layers." in n)]
        self.W0 = [m.weight.data.clone() for m in self.mods]
        self.axis = [0 if type(m).__name__ == "Conv1D" else 1 for m in self.mods]
        self.pc, self.gen = per_channel, generator
        self.mode, self.eta, self.b = None, 0.0, 32
        self.cal = [[] for _ in self.mods]
        self.scales, self.errs = None, None
        for i, m in enumerate(self.mods):
            m.register_forward_pre_hook(self._hook(i))

    def randn(self, x):
        if self.gen is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, generator=self.gen, device=x.device, dtype=x.dtype)

    def rms(self, W, axis):
        return W.pow(2).mean(dim=axis, keepdim=True).sqrt() if self.pc else W.pow(2).mean().sqrt()

    def _hook(self, i):
        def f(mod, inp):
            x = inp[0]
            red = tuple(range(x.dim() - 1))
            if self.mode == "noise":
                r = x.pow(2).mean(dim=red, keepdim=True).sqrt() if self.pc else x.pow(2).mean().sqrt()
                return (x + self.randn(x) * self.eta * r,) + inp[1:]
            if self.mode == "calib":
                self.cal[i].append(x.abs().mean(dim=red) if self.pc else x.abs().mean().item())
                return None
            if self.mode == "quant":
                xq = quantize_signed(x, self.b, self.scales[i])
                if self.errs is not None:
                    self.errs.append(((xq - x).norm() / (x.norm() + 1e-12)).item())
                return (xq,) + inp[1:]
            return None
        return f

    def set_weights(self, fn):
        for m, W, ax in zip(self.mods, self.W0, self.axis):
            m.weight.data.copy_(fn(W, ax))

    def restore(self):
        self.set_weights(lambda W, ax: W)
        self.mode = None

    def add_noise(self, eta):
        self.set_weights(lambda W, ax: W + self.randn(W) * eta * self.rms(W, ax))
        self.mode, self.eta = "noise", eta

    def quantize(self, b):
        self.set_weights(lambda W, ax: quantize_weight(W, b, ax, self.pc))
        self.mode, self.b = "quant", b

    def finish_calibration(self):
        self.scales = [torch.stack(v).mean(0) if self.pc else sum(v) / len(v) for v in self.cal]
        self.mode = None
