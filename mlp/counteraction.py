"""Cross-layer error decomposition at a fixed bit-width.

At every layer l the quantized hidden state differs from the full-precision one by e_l = P_l + N_l:
    P_l = f_l(h~_{l-1}) - f_l(h_{l-1})        inherited error, propagated by the full-precision layer
    N_l = f~_l(h~_{l-1}) - f_l(h~_{l-1})      new error, added by quantizing layer l
Reported (averaged over layers 1..D-1 on 2000 test images):
    cancel = -2<P,N>/|N|^2     fraction of the new error's energy cancelled by the inherited error
    gain   = |P_l| / |e_{l-1}|  per-layer gain of the inherited error
and final_rel_err = |e_D| / |h_D| at the logits and the quantized test accuracy.

Example:
    python -m mlp.counteraction --arch plain --depths 8 16 32 --train fp qat
Output: results/mlp/counteraction.txt
"""
import argparse

import numpy as np

import paths
from .common import load, quantize_unsigned, quantize_weight, write_line
from .models import ARCHS, MLP

N_EVAL = 2000


def layer(net, l, h, bits):
    """Layer l of the network, quantized when bits < 32."""
    W = quantize_weight(net.W[l], bits)
    last = l == net.D - 1
    if l == 0:
        z = h @ W + net.B[0]
        return np.maximum(z, 0) if net.arch == "plain" else z
    a = h if net.arch == "plain" else np.maximum(h, 0)
    a = quantize_unsigned(a, bits)[0]
    z = a @ W + net.B[l]
    if net.arch == "plain":
        return z if last else np.maximum(z, 0)
    return z if last else h + net.s * z


def decompose(net, X, bits):
    h, hq = X, X
    cancel, gain, e_prev = [], [], None
    for l in range(net.D):
        f_fp = layer(net, l, h, 32)
        f_inh = layer(net, l, hq, 32)
        f_q = layer(net, l, hq, bits)
        P, N = f_inh - f_fp, f_q - f_inh
        if l > 0:
            cancel.append(-2 * np.sum(P * N) / (np.sum(N * N) + 1e-12))
            gain.append(np.linalg.norm(P) / (np.linalg.norm(e_prev) + 1e-12))
        h, hq, e_prev = f_fp, f_q, f_q - f_fp
    return float(np.mean(cancel)), float(np.mean(gain)), float(np.linalg.norm(e_prev) / np.linalg.norm(h))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--depths", type=int, nargs="+", required=True)
    ap.add_argument("--train", nargs="+", choices=("fp", "qat"), default=("fp", "qat"))
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 3), metavar=("FIRST", "STOP"))
    args = ap.parse_args()
    Xtr, ytr, Xte, yte = load("fashion")
    for depth in args.depths:
        for mode in args.train:
            for seed in range(*args.seeds):
                net = MLP(args.arch, depth, seed).fit(Xtr, ytr, b=32 if mode == "fp" else args.bits, seed=seed)
                cancel, gain, err = decompose(net, Xte[:N_EVAL], args.bits)
                write_line(paths.result("mlp/counteraction.txt"),
                           f"{args.arch} depth={depth} train={mode} bits={args.bits} seed={seed} cancel={cancel:.4f} "
                           f"gain={gain:.4f} final_rel_err={err:.4f} acc={net.acc(Xte, yte, args.bits):.4f}")


if __name__ == "__main__":
    main()
