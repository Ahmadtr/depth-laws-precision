"""Data loading, uniform quantizers and optimizer shared by the MLP experiments (NumPy, CPU)."""
import gzip
import os
import time

import numpy as np

import paths

N_TRAIN = 20000
BITS = [1, 2, 3, 4, 5, 6, 8]
FULL_PRECISION = 32


def _read(path, offset):
    return np.frombuffer(gzip.open(path).read(), np.uint8, offset=offset)


def load(dataset):
    """Return (Xtr, ytr, Xte, yte) for 'fashion' or 'mnist'; the first 20k training images are used."""
    d = os.path.join(paths.DATA, dataset)
    Xtr = _read(os.path.join(d, "train-images.gz"), 16).reshape(-1, 784).astype(np.float32) / 255
    ytr = _read(os.path.join(d, "train-labels.gz"), 8).astype(int)
    Xte = _read(os.path.join(d, "test-images.gz"), 16).reshape(-1, 784).astype(np.float32) / 255
    yte = _read(os.path.join(d, "test-labels.gz"), 8).astype(int)
    return Xtr[:N_TRAIN], ytr[:N_TRAIN], Xte, yte


def quantize_weight(W, b):
    """Symmetric per-tensor uniform quantizer with LSQ-style step 2*mean|W|/sqrt(q_p)."""
    if b >= FULL_PRECISION:
        return W
    if b == 1:
        return np.sign(W) * np.mean(np.abs(W))
    qp = 2 ** (b - 1) - 1
    s = 2 * np.mean(np.abs(W)) / np.sqrt(qp)
    return np.clip(np.round(W / s), -qp, qp) * s


def quantize_unsigned(h, b, mean_pos=None):
    """Unsigned quantizer for post-ReLU activations. Returns (quantized, straight-through mask).

    The step is 2*mean(h>0)/sqrt(q_p); mean_pos fixes it to a calibrated value instead of the batch value."""
    if b >= FULL_PRECISION:
        return h, np.ones_like(h)
    qp = 2 ** b - 1
    if mean_pos is None:
        pos = h[h > 0]
        s = (2 * pos.mean() / np.sqrt(qp)) if pos.size else 1.0
    else:
        s = max(2 * mean_pos / np.sqrt(qp), 1e-12)
    return np.clip(np.round(h / s), 0, qp) * s, (h <= qp * s).astype(h.dtype)


def relative_noise(x, eta, rng):
    """x + eta * rms(x) * N(0, I)."""
    if eta == 0:
        return x
    return x + rng.standard_normal(x.shape).astype(x.dtype) * eta * np.sqrt(np.mean(x * x) + 1e-12)


def softmax_grad(z, y):
    """Gradient of the mean cross-entropy with respect to the logits."""
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    p[np.arange(len(y)), y] -= 1
    return p / len(y)


class Adam:
    def __init__(self, params, lr=1e-3):
        self.params, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for n, (p, g) in enumerate(zip(self.params, grads)):
            self.m[n] = 0.9 * self.m[n] + 0.1 * g
            self.v[n] = 0.999 * self.v[n] + 0.001 * g * g
            p -= self.lr * (self.m[n] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[n] / (1 - 0.999 ** self.t)) + 1e-8)


def write_line(path, line):
    """Append one line; a lock file keeps lines intact when many processes write to the same file."""
    lock = path + ".lock"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.02)
    try:
        with open(path, "a") as f:
            f.write(line + "\n")
    finally:
        os.close(fd)
        os.remove(lock)
    print(line, flush=True)
