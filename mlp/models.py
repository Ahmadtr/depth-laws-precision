"""MLPs with quantization-aware forward passes.

Architectures (D weight matrices: 784 -> W, D-2 hidden W -> W, W -> 10):
    plain  h_{l+1} = ReLU(Q(h_l) Q(W_l) + b_l)
    res2   h_{l+1} = h_l + s * (Q(ReLU(h_l)) Q(W_l) + b_l),  s = 1/sqrt(D)   (pre-activation, linear branch)
    res2c  as res2 with fixed s = 1/sqrt(8)                                 (scale control)
Q quantizes every weight matrix and every activation entering a matrix multiplication (except the input).
Training uses the straight-through estimator with clipping masks.
"""
import numpy as np

from .common import FULL_PRECISION, Adam, quantize_unsigned, quantize_weight, relative_noise, softmax_grad

ARCHS = ("plain", "res2", "res2c")


class MLP:
    epochs, batch_size, lr = 3, 128, 1e-3

    def __init__(self, arch, depth, seed, width=32):
        assert arch in ARCHS, arch
        rng = np.random.default_rng(seed)
        self.arch, self.D = arch, depth
        self.s = 1 / np.sqrt(8) if arch == "res2c" else 1 / np.sqrt(depth)
        dims = [784] + [width] * (depth - 1) + [10]
        self.W = [rng.standard_normal((a, c)).astype(np.float32) * np.sqrt(2 / a) for a, c in zip(dims[:-1], dims[1:])]
        self.B = [np.zeros(c, np.float32) for c in dims[1:]]
        self.opt = Adam(self.W + self.B, self.lr)
        self.act_scales = None          # calibrated activation steps (site index -> mean positive value)
        self.noise, self.rng = 0.0, None

    def _act(self, h, b, site):
        if self.noise > 0:
            return relative_noise(h, self.noise, self.rng), np.ones_like(h)
        scale = None if self.act_scales is None else self.act_scales[site]
        return quantize_unsigned(h, b, scale)

    def _weights(self, b):
        if self.noise > 0:
            return [relative_noise(W, self.noise, self.rng) for W in self.W]
        return [quantize_weight(W, b) for W in self.W]

    def forward(self, X, b=FULL_PRECISION):
        """Returns (hidden states, cache, effective weights); hidden states[-1] are the logits."""
        return self._forward_plain(X, b) if self.arch == "plain" else self._forward_residual(X, b)

    def _forward_plain(self, X, b):
        Q = self._weights(b)
        z = X @ Q[0] + self.B[0]
        hs, cache = [X, np.maximum(z, 0)], [(X, z, None)]
        for l in range(1, self.D):
            a, mk = self._act(hs[-1], b, l - 1)
            z = a @ Q[l] + self.B[l]
            cache.append((a, z, mk))
            hs.append(z if l == self.D - 1 else np.maximum(z, 0))
        return hs, cache, Q

    def _forward_residual(self, X, b):
        Q = self._weights(b)
        hs, cache = [X, X @ Q[0] + self.B[0]], [(X, None, None)]
        for l in range(1, self.D):
            h = hs[-1]
            a, mk = self._act(np.maximum(h, 0), b, l - 1)
            z = a @ Q[l] + self.B[l]
            cache.append((a, h, mk))
            hs.append(z if l == self.D - 1 else h + self.s * z)
        return hs, cache, Q

    def step(self, X, y, b):
        hs, cache, Q = self.forward(X, b)
        g = softmax_grad(hs[-1], y)
        D = self.D
        gW, gB = [None] * D, [None] * D
        a, h, mk = cache[D - 1]
        gW[D - 1], gB[D - 1] = a.T @ g, g.sum(0)
        if self.arch == "plain":
            gh = (g @ Q[D - 1].T) * mk
            for l in range(D - 2, 0, -1):
                a, z, mk = cache[l]
                gz = gh * (z > 0)
                gW[l], gB[l] = a.T @ gz, gz.sum(0)
                gh = (gz @ Q[l].T) * mk
            a, z, _ = cache[0]
            gz = gh * (z > 0)
            gW[0], gB[0] = a.T @ gz, gz.sum(0)
        else:
            gh = (g @ Q[D - 1].T) * mk * (h > 0)
            for l in range(D - 2, 0, -1):
                a, h, mk = cache[l]
                gz = self.s * gh
                gW[l], gB[l] = a.T @ gz, gz.sum(0)
                gh = gh + (gz @ Q[l].T) * mk * (h > 0)
            gW[0], gB[0] = X.T @ gh, gh.sum(0)
        self.opt.step(gW + gB)

    def fit(self, X, y, b=FULL_PRECISION, seed=0):
        """Train with b-bit quantization (b=32: full precision) or with self.noise > 0 (noise-aware training)."""
        rng = np.random.default_rng(seed + 7)
        for _ in range(self.epochs):
            idx = rng.permutation(len(X))
            for k in range(0, len(idx), self.batch_size):
                j = idx[k:k + self.batch_size]
                self.step(X[j], y[j], b)
        return self

    def logits(self, X, b=FULL_PRECISION):
        return self.forward(X, b)[0][-1]

    def acc(self, X, y, b=FULL_PRECISION):
        return float((self.logits(X, b).argmax(1) == y).mean())

    def calibrate(self, X, n=5000, batch=1000):
        """Fix every activation step from the mean positive activation over n training images."""
        sums = {}
        self.act_scales = None
        for k in range(0, n, batch):
            for site, a in enumerate(self.site_inputs(X[k:k + batch])):
                pos = a[a > 0]
                sums.setdefault(site, []).append(pos.mean() if pos.size else 0.0)
        self.act_scales = {i: float(np.mean(v)) for i, v in sums.items()}

    def site_inputs(self, X, weight_bits=8):
        """Activations entering each quantized matmul, with 8-bit weights and unquantized activations."""
        W, self.W = self.W, [quantize_weight(w, weight_bits) for w in self.W]
        try:
            hs = self.forward(X)[0]
        finally:
            self.W = W
        if self.arch == "plain":
            return hs[1:-1]
        return [np.maximum(h, 0) for h in hs[1:-1]]
