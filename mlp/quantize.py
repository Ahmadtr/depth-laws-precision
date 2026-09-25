"""Post-training integer quantization of MLPs (bit floors).

Train at full precision, then quantize every weight matrix and every activation entering a matrix
multiplication to b bits. Activation steps are either computed per batch ("dyn") or calibrated once on 5,000
training images ("cal"). The relative quantization error eta_q(b) of the calibrated quantizer is also recorded.

Example:
    python -m mlp.quantize --arch plain --depths 4 8 16 24 32 --seeds 0 10
Output: results/mlp/quantize.txt
"""
import argparse

import numpy as np

import paths
from .common import BITS, load, quantize_unsigned, quantize_weight, write_line
from .models import ARCHS, MLP


def relative_error(net, X, b):
    """Mean relative error of the b-bit quantizer over all weight tensors and activation sites."""
    errs = [np.linalg.norm(quantize_weight(W, b) - W) / np.linalg.norm(W) for W in net.W]
    for site, a in enumerate(net.site_inputs(X, weight_bits=32)):
        q, _ = quantize_unsigned(a, b, net.act_scales[site])
        errs.append(np.linalg.norm(q - a) / (np.linalg.norm(a) + 1e-12))
    return float(np.mean(errs))


def ptq(arch, depth, seed, width, data):
    Xtr, ytr, Xte, yte = data
    net = MLP(arch, depth, seed, width).fit(Xtr, ytr, seed=seed)
    fp = net.acc(Xte, yte)
    dyn = [net.acc(Xte, yte, b) for b in BITS]
    net.calibrate(Xtr)
    cal = [net.acc(Xte, yte, b) for b in BITS]
    eta = [relative_error(net, Xte[:1000], b) for b in BITS]
    return (f"fp={fp:.4f} dyn=" + ",".join(f"{b}:{a:.4f}" for b, a in zip(BITS, dyn))
            + " cal=" + ",".join(f"{b}:{a:.4f}" for b, a in zip(BITS, cal))
            + " eta_q=" + ",".join(f"{b}:{e:.4f}" for b, e in zip(BITS, eta)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--depths", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 1), metavar=("FIRST", "STOP"))
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", default="mlp/quantize.txt")
    args = ap.parse_args()
    data = load("fashion")
    for depth in args.depths:
        for seed in range(*args.seeds):
            write_line(paths.result(args.out),
                       f"arch={args.arch} dataset=fashion width={args.width} mode=ptq depth={depth} seed={seed} "
                       + ptq(args.arch, depth, seed, args.width, data))


if __name__ == "__main__":
    main()
