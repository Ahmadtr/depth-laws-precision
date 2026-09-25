"""Plain MLP trained with or without relative Gaussian noise on every weight matrix and hidden activation.

Used for the plain-MLP noise floors (PTQ and QAT) and the per-layer sensitivity map."""
import numpy as np

from .common import Adam, relative_noise, softmax_grad

EPOCHS, BATCH_SIZE, LR = 3, 128, 1e-3


def init(depth, width, rng):
    dims = [784] + [width] * (depth - 1) + [10]
    Ws = [rng.standard_normal((a, c)).astype(np.float32) * np.sqrt(2 / a) for a, c in zip(dims[:-1], dims[1:])]
    Bs = [np.zeros(c, np.float32) for c in dims[1:]]
    return Ws, Bs


def forward(X, Ws, Bs, eta, rng):
    hs, masks = [X], []
    for i, (W, b) in enumerate(zip(Ws, Bs)):
        z = hs[-1] @ relative_noise(W, eta, rng) + b
        if i < len(Ws) - 1:
            hs.append(relative_noise(np.maximum(z, 0), eta, rng))
            masks.append(z > 0)
        else:
            hs.append(z)
    return hs, masks


def train(Xtr, ytr, depth, eta, seed, width=32, epochs=EPOCHS, label_smoothing=0.0):
    """Returns (weights, biases). eta = 0 gives full-precision training."""
    rng = np.random.default_rng(seed)
    Ws, Bs = init(depth, width, rng)
    opt = Adam(Ws + Bs, LR)
    for _ in range(epochs):
        idx = rng.permutation(len(Xtr))
        for k in range(0, len(idx), BATCH_SIZE):
            j = idx[k:k + BATCH_SIZE]
            Wn = [relative_noise(W, eta, rng) for W in Ws]
            hs, masks = [Xtr[j]], []
            for i, (W, b) in enumerate(zip(Wn, Bs)):
                z = hs[-1] @ W + b
                if i < len(Wn) - 1:
                    hs.append(relative_noise(np.maximum(z, 0), eta, rng))
                    masks.append(z > 0)
                else:
                    hs.append(z)
            g = softmax_grad(hs[-1], ytr[j])
            if label_smoothing:
                g += label_smoothing * (np.eye(10, dtype=g.dtype)[ytr[j]] - 0.1) / len(j)
            gW, gB = [None] * len(Ws), [None] * len(Ws)
            for i in range(len(Ws) - 1, -1, -1):
                gW[i], gB[i] = hs[i].T @ g, g.sum(0)
                if i:
                    g = (g @ Wn[i].T) * masks[i - 1]
            opt.step(gW + gB)
    return Ws, Bs


def noisy_accuracy(X, y, Ws, Bs, eta, seed, draws=3):
    rng = np.random.default_rng(seed + 999)
    return float(np.mean([(forward(X, Ws, Bs, eta, rng)[0][-1].argmax(1) == y).mean() for _ in range(draws)]))
