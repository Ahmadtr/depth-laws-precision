"""Measure every saved CNN or ViT checkpoint (inference only).

For each model:
    fp           full-precision test accuracy
    M            median normalised margin (z_(1) - z_(2)) / ||z|| over the test set
    G_logit, G_kl  amplification at eta in {0.001, 0.005, 0.01}, 50 draws on 2000 test images:
                 G_logit = sqrt(E ||dz||^2 / ||z||^2) / eta,  G_kl = sqrt(2 E KL(p || p_eta)) / eta
    eta_c        Gaussian-noise floor: noise level at which accuracy on 2000 test images falls to (fp + 0.1) / 2
                 (14 log-spaced levels from 0.02 to 2, three draws each, log-linear interpolation)
    lsq          calibrated LSQ-style PTQ: accuracy and relative quantization error at 1..8 bits
    minmax       calibrated min-max PTQ (CNNs only)
    kurt         mean excess kurtosis of the activations entering the quantizers (CNNs only)
Noise draws use a generator seeded per model, so repeated runs give identical numbers.

Example:
    python -m vision.measure --family cnn
    python -m vision.measure --family vit --pattern "vit_pre_*"
Output: results/{cnn,vit}/measure.txt (one line per checkpoint; checkpoints already listed are skipped)
"""
import argparse
import glob
import math
import os
import re

import torch

import paths
from vision.common import BITS, DEV, accuracy, calibrate, load_cifar, logits, quant_error, uncalibrate

ETAS_G = (0.001, 0.005, 0.01)
ETAS_FLOOR = [0.02 * (100 ** (i / 13)) for i in range(14)]


def build(family, arch, depth):
    if family == "cnn":
        from cnn.models import CNN
        return CNN(arch, depth).to(DEV)
    from vit.models import ViT
    return ViT(arch, depth).to(DEV)


@torch.no_grad()
def margin(net, X):
    z = logits(net, X)
    t = z.topk(2, 1).values
    return ((t[:, 0] - t[:, 1]) / z.norm(dim=1)).median().item()


@torch.no_grad()
def amplification(net, X, eta, draws=50, n=2000):
    z0 = logits(net, X, n).double()
    lp0 = torch.log_softmax(z0, 1)
    a = kl = 0.0
    net.noise = eta
    for _ in range(draws):
        z = logits(net, X, n).double()
        a += ((z - z0).norm() / z0.norm()).item() ** 2
        kl += (lp0.exp() * (lp0 - torch.log_softmax(z, 1))).sum(1).mean().item()
    net.noise = 0.0
    return math.sqrt(a / draws) / eta, math.sqrt(2 * kl / draws) / eta


@torch.no_grad()
def noise_floor(net, X, Y, fp, n=2000, draws=3):
    accs = []
    for e in ETAS_FLOOR:
        net.noise = e
        accs.append(sum((logits(net, X, n).argmax(1) == Y[:n]).float().mean().item() for _ in range(draws)) / draws)
    net.noise = 0.0
    mid = (fp + 0.1) / 2
    for (e0, a0), (e1, a1) in zip(zip(ETAS_FLOOR, accs), zip(ETAS_FLOOR[1:], accs[1:])):
        if a0 >= mid > a1:
            return math.exp(math.log(e0) + (mid - a0) / (a1 - a0) * (math.log(e1) - math.log(e0)))
    return float("nan")


def ptq(net, Xtr, Xte, Yte, quantizer, weight_bits):
    calibrate(net, Xtr, quantizer, weight_bits=weight_bits)
    cells = ",".join(f"{b}:{accuracy(net, Xte, Yte, b):.4f}:{quant_error(net, Xte, b):.4f}" for b in BITS)
    uncalibrate(net)
    return cells


@torch.no_grad()
def kurtosis(net, X, n=2000, bs=500):
    net.sites.stats = []
    for k in range(0, n, bs):
        net(X[k:k + bs])
    per_site = {}
    for i, v in net.sites.stats:
        per_site.setdefault(i, []).append(v)
    net.sites.stats = None
    return sum(sum(v) / len(v) for v in per_site.values()) / len(per_site)


def measure(family, path, data):
    Xtr, Ytr, Xte, Yte = data
    name = os.path.basename(path)[:-3]
    arch, depth, seed, epochs = re.match(r"(?:vit_)?(\w+?)_d(\d+)_s(\d+)_e(\d+)$", name).groups()
    net = build(family, arch, int(depth))
    net.load_state_dict(torch.load(path, map_location=DEV, weights_only=False))
    net.eval()
    net.sites.generator = torch.Generator(device=DEV).manual_seed(1000 + int(seed))
    fp = accuracy(net, Xte, Yte)
    G = {e: amplification(net, Xte, e) for e in ETAS_G}
    fields = [f"{family}_{arch}", f"depth={depth}", f"seed={seed}", f"epochs={epochs}", f"fp={fp:.4f}",
              f"M={margin(net, Xte):.4f}"]
    fields += [f"G_logit@{e}={G[e][0]:.4f}" for e in ETAS_G] + [f"G_kl@{e}={G[e][1]:.4f}" for e in ETAS_G]
    fields.append(f"eta_c={noise_floor(net, Xte, Yte, fp):.4f}")
    fields.append("lsq=" + ptq(net, Xtr, Xte, Yte, "lsq", 8 if family == "cnn" else 32))
    if family == "cnn":
        fields.append("minmax=" + ptq(net, Xtr, Xte, Yte, "minmax", 8))
        fields.append(f"kurt={kurtosis(net, Xte):.3f}")
    return " ".join(fields)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=("cnn", "vit"), required=True)
    ap.add_argument("--pattern", default=None, help="glob on checkpoint names, e.g. 'res1_d64_*'")
    args = ap.parse_args()
    pattern = args.pattern or ("vit_*" if args.family == "vit" else "*")
    files = sorted(glob.glob(os.path.join(paths.MODELS, pattern + ".pt")))
    files = [f for f in files if os.path.basename(f).startswith("vit_") == (args.family == "vit")
             and "_qat" not in os.path.basename(f)]
    out = paths.result(f"{args.family}/measure.txt")
    done = set()
    if os.path.exists(out):
        done = {(ln.split()[0], ln.split()[1], ln.split()[2], ln.split()[3]) for ln in open(out) if ln.strip()}
    data = load_cifar()
    for f in files:
        m = re.match(r"(?:vit_)?(\w+?)_d(\d+)_s(\d+)_e(\d+)$", os.path.basename(f)[:-3])
        key = (f"{args.family}_{m.group(1)}", f"depth={m.group(2)}", f"seed={m.group(3)}", f"epochs={m.group(4)}")
        if key in done:
            continue
        line = measure(args.family, f, data)
        with open(out, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)


if __name__ == "__main__":
    main()
