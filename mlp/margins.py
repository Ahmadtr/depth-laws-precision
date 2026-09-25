"""Decision-margin control: does changing the margin distribution move the noise floor?

Plain MLPs are trained with label smoothing (ls) or symmetric label noise (lnoise). For each condition
we report the median normalised margin of the full-precision network on the test set,
    m(x) = (z_(1) - z_(2)) / rms(z),
where z_(1), z_(2) are the two largest logits and rms(z) is taken over all test logits, the fraction
of test points with m < 0.25, and the accuracy-versus-noise curve
(mode=infer: noise at test time only; mode=train: noise-aware training).

Example:
    python -m mlp.margins --mode infer --depths 8 16 --seeds 0 3
Output: results/mlp/margins_<mode>.txt
"""
import argparse

import numpy as np

import paths
from . import plain_noise
from .common import load, write_line
from .noise_floor import COARSE

CONDITIONS = ((0.0, 0.0), (0.1, 0.0), (0.3, 0.0), (0.5, 0.0), (0.0, 0.2), (0.0, 0.4))


def corrupt_labels(y, fraction, seed):
    """Replace a fraction of the labels by a uniformly chosen different class."""
    if fraction == 0:
        return y
    rng = np.random.default_rng(seed + 500)
    y = y.copy()
    idx = rng.choice(len(y), int(round(fraction * len(y))), replace=False)
    y[idx] = (y[idx] + rng.integers(1, 10, len(idx))) % 10
    return y


def normalised_margins(X, Ws, Bs):
    z = plain_noise.forward(X, Ws, Bs, 0.0, None)[0][-1]
    top = np.sort(z, axis=1)
    return (top[:, -1] - top[:, -2]) / np.sqrt(np.mean(z * z))


def run(depth, ls, lnoise, mode, seed, data, etas):
    Xtr, ytr, Xte, yte = data
    ytr = corrupt_labels(ytr, lnoise, seed)
    ref = plain_noise.train(Xtr, ytr, depth, 0.0, seed, label_smoothing=ls)
    m = normalised_margins(Xte, *ref)
    if mode == "infer":
        curve = [plain_noise.noisy_accuracy(Xte, yte, *ref, e, seed) for e in etas]
    else:
        curve = []
        for e in etas:
            Ws, Bs = plain_noise.train(Xtr, ytr, depth, e, seed, label_smoothing=ls)
            curve.append(plain_noise.noisy_accuracy(Xte, yte, Ws, Bs, e, seed))
    return float(np.median(m)), float((m < 0.25).mean()), curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("infer", "train"), required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=(8, 16))
    ap.add_argument("--seeds", type=int, nargs=2, default=None, metavar=("FIRST", "STOP"),
                    help="default: 0 3 for infer, 0 2 for train")
    args = ap.parse_args()
    seeds = args.seeds or ((0, 3) if args.mode == "infer" else (0, 2))
    data = load("fashion")
    out = paths.result(f"mlp/margins_{args.mode}.txt")
    for depth in args.depths:
        for ls, lnoise in CONDITIONS:
            for seed in range(*seeds):
                med, low, curve = run(depth, ls, lnoise, args.mode, seed, data, COARSE)
                write_line(out, f"depth={depth} ls={ls} lnoise={lnoise} seed={seed} med_nmargin={med:.4f} "
                                f"frac_nm<0.25={low:.4f} " + " ".join(f"{e}:{a:.4f}" for e, a in zip(COARSE, curve)))


if __name__ == "__main__":
    main()
