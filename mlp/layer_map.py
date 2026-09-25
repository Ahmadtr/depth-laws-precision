"""Per-layer sensitivity map of plain MLPs trained without (PTQ) and with (QAT) noise.

For each layer l, small relative noise (eta = 0.01, 50 draws) is injected only into its weight matrix and the
activation entering it, giving the layer sensitivity q_l = E||dz||^2 / (eta^2 ||z||^2), with G^2 = sum_l q_l.
The QAT network is trained with noise equal to the PTQ noise floor of the same depth.

Example:
    python -m mlp.layer_map --depths 4 8 16 24 --seeds 0 5
Output: results/mlp/layer_map.txt
"""
import argparse

import numpy as np

import paths
from . import plain_noise
from .common import load, write_line

PTQ_NOISE_FLOOR = {4: 0.274, 8: 0.159, 16: 0.107, 24: 0.077}     # Fashion-MNIST, width 32, 10 seeds


def layer_sensitivity(Ws, Bs, X, layer, rng, eta=0.01, draws=50):
    def forward(noisy_layer):
        h = X
        for i, (W, b) in enumerate(zip(Ws, Bs)):
            if i == noisy_layer:
                W = W + rng.standard_normal(W.shape).astype(W.dtype) * eta * np.sqrt(np.mean(W * W))
                if i > 0:
                    h = h + rng.standard_normal(h.shape).astype(h.dtype) * eta * np.sqrt(np.mean(h * h))
            z = h @ W + b
            h = np.maximum(z, 0) if i < len(Ws) - 1 else z
        return h

    z0 = forward(-1)
    return float(np.mean([np.sum((forward(layer) - z0) ** 2) for _ in range(draws)]) / (eta ** 2 * np.sum(z0 * z0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", type=int, nargs="+", default=sorted(PTQ_NOISE_FLOOR))
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 5), metavar=("FIRST", "STOP"))
    ap.add_argument("--out", default="mlp/layer_map.txt")
    args = ap.parse_args()
    Xtr, ytr, Xte, _ = load("fashion")
    for depth in args.depths:
        for seed in range(*args.seeds):
            rng = np.random.default_rng(100 + seed)
            for mode, eta in (("ptq", 0.0), ("qat", PTQ_NOISE_FLOOR[depth])):
                Ws, Bs = plain_noise.train(Xtr, ytr, depth, eta, seed)
                q = [layer_sensitivity(Ws, Bs, Xte[:2000], l, rng) for l in range(depth)]
                write_line(paths.result(args.out),
                           f"arch=plain mode={mode} depth={depth} seed={seed} eta_train={eta} G2={sum(q):.4f} "
                           "q=" + ",".join(f"{x:.4f}" for x in q))


if __name__ == "__main__":
    main()
