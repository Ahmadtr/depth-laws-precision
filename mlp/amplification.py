"""Small-noise amplification G of full-precision MLPs.

Relative Gaussian noise eta is injected at every quantizer site (all weight matrices and every activation
entering a matrix multiplication). Two definitions are reported, each averaged over 50 draws:
    G_logit = E||z_eta - z|| / ||z|| / eta
    G_kl    = sqrt(2 E KL(p || p_eta)) / eta         (predictive amplification, used in the paper)
at eta in {0.001, 0.005, 0.01}; their ratio across eta checks the first-order (linear) regime.
Also reported: the median normalised margin M = median (z_(1) - z_(2)) / ||z|| over the test set.

Example:
    python -m mlp.amplification --arch plain --depths 4 8 16 --seeds 0 10
Output: results/mlp/amplification.txt
"""
import argparse

import numpy as np

import paths
from .common import load, write_line
from .models import ARCHS, MLP

ETAS = (0.001, 0.005, 0.01)
DRAWS = 50


def log_softmax(z):
    z = z.astype(np.float64)
    y = z - z.max(1, keepdims=True)
    return y - np.log(np.exp(y).sum(1, keepdims=True))


def amplification(net, X, seed):
    z0 = net.logits(X)
    lp0 = log_softmax(z0)
    rng = np.random.default_rng(seed + 5)
    G, K = {}, {}
    for eta in ETAS:
        net.noise, net.rng = eta, rng
        zs = [net.logits(X) for _ in range(DRAWS)]
        net.noise = 0.0
        G[eta] = np.mean([np.linalg.norm(z - z0) / np.linalg.norm(z0) for z in zs]) / eta
        K[eta] = np.sqrt(2 * np.mean([(np.exp(lp0) * (lp0 - log_softmax(z))).sum(1).mean() for z in zs])) / eta
    return G, K


def median_margin(net, X):
    z = net.logits(X)
    top = np.sort(z, axis=1)
    return float(np.median((top[:, -1] - top[:, -2]) / (np.linalg.norm(z, axis=1) + 1e-12)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--depths", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 1), metavar=("FIRST", "STOP"))
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", default="mlp/amplification.txt")
    args = ap.parse_args()
    Xtr, ytr, Xte, yte = load("fashion")
    for depth in args.depths:
        for seed in range(*args.seeds):
            net = MLP(args.arch, depth, seed, args.width).fit(Xtr, ytr, seed=seed)
            G, K = amplification(net, Xte[:2000], seed)
            write_line(paths.result(args.out),
                       f"arch={args.arch} dataset=fashion width={args.width} depth={depth} seed={seed} "
                       f"fp={net.acc(Xte, yte):.4f} M={median_margin(net, Xte):.4f} "
                       + " ".join(f"G_logit@{e}={G[e]:.4f}" for e in ETAS) + " "
                       + " ".join(f"G_kl@{e}={K[e]:.4f}" for e in ETAS))


if __name__ == "__main__":
    main()
