"""Noise floors of MLPs: test accuracy under relative Gaussian noise at every quantizer site.

    mode=infer  full-precision training, noise at test time only        (PTQ analogue)
    mode=train  the same noise during training and at test time        (QAT analogue)

Example:
    python -m mlp.noise_floor --arch plain --mode infer --depths 4 8 16 --seeds 0 10
Output: results/mlp/noise_floor.txt, one line per (arch, dataset, width, mode, depth, seed).
"""
import argparse
import time

import numpy as np

import paths
from . import plain_noise
from .common import load, write_line
from .models import ARCHS, MLP

FINE = np.round(np.logspace(-2, 0.3, 12), 4)          # plain MLP, Fashion-MNIST, width 32
COARSE = np.round(np.logspace(-1.7, 0, 8), 4)         # plain MLP, other datasets and widths
RESIDUAL = np.round(np.logspace(-1.5, 0.5, 10), 4)    # residual MLPs


def noise_grid(arch, dataset, width):
    if arch != "plain":
        return RESIDUAL
    return FINE if (dataset == "fashion" and width == 32) else COARSE


def residual_curve(arch, depth, mode, seed, width, data, etas):
    Xtr, ytr, Xte, yte = data

    def run(eta_train):
        net = MLP(arch, depth, seed, width)
        net.noise, net.rng = eta_train, np.random.default_rng(seed + 11)
        return net.fit(Xtr, ytr, seed=seed)

    def test(net, eta):
        net.noise, net.rng = eta, np.random.default_rng(seed + 99)
        a = np.mean([(net.logits(Xte).argmax(1) == yte).mean() for _ in range(3)])
        net.noise = 0.0
        return float(a)

    if mode == "infer":
        net = run(0.0)
        return [test(net, e) for e in etas]
    return [test(run(e), e) for e in etas]


def plain_curve(depth, mode, seed, width, data, etas):
    Xtr, ytr, Xte, yte = data
    if mode == "infer":
        Ws, Bs = plain_noise.train(Xtr, ytr, depth, 0.0, seed, width)
        return [plain_noise.noisy_accuracy(Xte, yte, Ws, Bs, e, seed) for e in etas]
    curve = []
    for e in etas:
        Ws, Bs = plain_noise.train(Xtr, ytr, depth, e, seed, width)
        curve.append(plain_noise.noisy_accuracy(Xte, yte, Ws, Bs, e, seed))
    return curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--mode", choices=("infer", "train"), required=True)
    ap.add_argument("--depths", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 1), metavar=("FIRST", "STOP"))
    ap.add_argument("--dataset", choices=("fashion", "mnist"), default="fashion")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", default="mlp/noise_floor.txt")
    args = ap.parse_args()
    data = load(args.dataset)
    etas = noise_grid(args.arch, args.dataset, args.width)
    for depth in args.depths:
        for seed in range(*args.seeds):
            t0 = time.time()
            if args.arch == "plain":
                curve = plain_curve(depth, args.mode, seed, args.width, data, etas)
            else:
                curve = residual_curve(args.arch, depth, args.mode, seed, args.width, data, etas)
            write_line(paths.result(args.out),
                       f"arch={args.arch} dataset={args.dataset} width={args.width} mode={args.mode} depth={depth} "
                       f"seed={seed} curve=" + ",".join(f"{e}:{a:.4f}" for e, a in zip(etas, curve))
                       + f" time={time.time() - t0:.0f}")


if __name__ == "__main__":
    main()
