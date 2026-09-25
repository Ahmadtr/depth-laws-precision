"""Cross-layer error decomposition of the saved CNNs at a fixed bit-width (calibrated LSQ-style PTQ).

Same definition as mlp/counteraction.py. Layers are the stem, the D blocks and the head. At layer l
    P_l = f_l(h~) - f_l(h)        inherited error, propagated by the full-precision layer
    N_l = f~_l(h~) - f_l(h~)      new error, added by quantizing layer l
cancel = -2<P,N>/|N|^2 and gain = |P_l|/|e_{l-1}|, averaged over the layers after the stem (sums over 2000
test images), final_rel_err = |e| / |h| at the logits, and the quantized test accuracy.

Example:
    python -m cnn.counteraction --bits 6
Output: results/cnn/counteraction.txt
"""
import argparse
import glob
import math
import os
import re

import torch

import paths
from vision.common import DEV, accuracy, calibrate, load_cifar
from .models import CNN


def layer(net, l, h, b):
    if l == 0:
        return net.stem_out(h, b)
    if l == net.D + 1:
        net.sites.idx = net.D
        return net.head_out(h, b)
    net.sites.idx = l - 1
    return net.block(l - 1, h, b)


@torch.no_grad()
def decompose(net, X, b, n=2000, bs=500):
    L = net.D + 2
    pn, nn_, pp, ee = [0.0] * L, [0.0] * L, [0.0] * L, [0.0] * L
    out_e = out_h = 0.0
    for k in range(0, n, bs):
        h = hq = X[k:k + bs]
        for l in range(L):
            f_fp, f_inh, f_q = layer(net, l, h, 32), layer(net, l, hq, 32), layer(net, l, hq, b)
            P, N = f_inh - f_fp, f_q - f_inh
            pn[l] += (P * N).sum().item()
            nn_[l] += (N * N).sum().item()
            pp[l] += (P * P).sum().item()
            h, hq = f_fp, f_q
            ee[l] += ((hq - h) ** 2).sum().item()
        out_e += ((hq - h) ** 2).sum().item()
        out_h += (h ** 2).sum().item()
    cancel = [-2 * pn[l] / (nn_[l] + 1e-12) for l in range(1, L)]
    gain = [math.sqrt(pp[l] / (ee[l - 1] + 1e-12)) for l in range(1, L)]
    return sum(cancel) / len(cancel), sum(gain) / len(gain), math.sqrt(out_e / out_h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    Xtr, _, Xte, Yte = load_cifar()
    out = paths.result("cnn/counteraction.txt")
    for f in sorted(glob.glob(os.path.join(paths.MODELS, f"*_e{args.epochs}.pt"))):
        m = re.match(rf"(\w+?)_d(\d+)_s(\d+)_e{args.epochs}$", os.path.basename(f)[:-3])
        if m is None or os.path.basename(f).startswith("vit_"):
            continue
        arch, depth, seed = m.group(1), int(m.group(2)), int(m.group(3))
        net = CNN(arch, depth).to(DEV)
        net.load_state_dict(torch.load(f, map_location=DEV, weights_only=False))
        net.eval()
        calibrate(net, Xtr)
        cancel, gain, err = decompose(net, Xte, args.bits)
        line = (f"cnn_{arch} depth={depth} seed={seed} train=fp bits={args.bits} cancel={cancel:.4f} "
                f"gain={gain:.4f} final_rel_err={err:.4f} acc={accuracy(net, Xte, Yte, args.bits):.4f}")
        with open(out, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)


if __name__ == "__main__":
    main()
