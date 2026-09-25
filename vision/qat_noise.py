"""Noise-aware training (the QAT analogue) against post-training noise, for CNNs and ViTs.

    infer  the full-precision checkpoint (trained first if missing), tested with relative Gaussian noise eta
    train  a network trained with noise eta at every site, tested at the same eta (checkpoint *_qat{eta}.pt)
Test accuracy uses 5000 test images and three noise draws.

Examples:
    python -m vision.qat_noise --family cnn --arch res1 --depths 8 16 32 64 --seeds 0 3
    python -m vision.qat_noise --family cnn --arch res1 --depths 8 64 --seeds 0 2 --epochs 30
    python -m vision.qat_noise --family vit --arch pre --depths 2 4 8 16 --seeds 0 3
Output: results/{cnn,vit}/qat_noise.txt
"""
import argparse
import os
import time

import torch

import paths
from vision.common import DEV, load_cifar

ETAS = {("cnn", 15): [0.1, 0.18, 0.33, 0.6, 1.1, 2.0], ("cnn", 30): [0.18, 0.33, 0.6, 1.1],
        ("vit", 30): [0.15, 0.3, 0.5, 0.8, 1.3]}


@torch.no_grad()
def noisy_accuracy(net, X, Y, eta, seed, draws=3, n=5000, bs=1000):
    net.eval()
    net.noise, net.sites.generator = eta, torch.Generator(device=DEV).manual_seed(2000 + seed)
    c = 0
    for _ in range(draws):
        c += sum((net(X[k:k + bs]).argmax(1) == Y[k:k + bs]).sum().item() for k in range(0, n, bs))
    net.noise, net.sites.generator = 0.0, None
    return c / (draws * n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=("cnn", "vit"), required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--depths", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 3), metavar=("FIRST", "STOP"))
    ap.add_argument("--epochs", type=int, default=None, help="default: 15 (CNN), 30 (ViT)")
    ap.add_argument("--etas", type=float, nargs="+", default=None)
    args = ap.parse_args()
    if args.family == "cnn":
        from cnn.models import CNN as Model
        from cnn.train import checkpoint, train_and_save
    else:
        from vit.models import ViT as Model
        from vit.train import checkpoint, train_and_save
    epochs = args.epochs or (15 if args.family == "cnn" else 30)
    etas = args.etas or ETAS[(args.family, epochs)]
    data = load_cifar()
    Xte, Yte = data[2], data[3]
    out = paths.result(f"{args.family}/qat_noise.txt")
    name = f"{args.family}_{args.arch}"
    for depth in args.depths:
        for seed in range(*args.seeds):
            path = checkpoint(args.arch, depth, seed, epochs)
            if os.path.exists(path):
                net = Model(args.arch, depth).to(DEV)
                net.load_state_dict(torch.load(path, map_location=DEV, weights_only=False))
            else:
                net = train_and_save(args.arch, depth, seed, epochs, data)
                torch.save(net.state_dict(), path)
            cells = " ".join(f"{e}:{noisy_accuracy(net, Xte, Yte, e, seed):.4f}" for e in etas)
            line = f"{name} infer depth={depth} seed={seed} epochs={epochs} {cells}"
            with open(out, "a") as fh:
                fh.write(line + "\n")
            print(line, flush=True)
            cells = []
            for e in etas:
                t0 = time.time()
                net = train_and_save(args.arch, depth, seed, epochs, data, noise=e)
                torch.save(net.state_dict(), path[:-3] + f"_qat{e}.pt")
                cells.append(f"{e}:{noisy_accuracy(net, Xte, Yte, e, seed):.4f}")
                print(f"  trained with eta={e} in {time.time() - t0:.0f}s -> {cells[-1]}", flush=True)
            line = f"{name} train depth={depth} seed={seed} epochs={epochs} " + " ".join(cells)
            with open(out, "a") as fh:
                fh.write(line + "\n")
            print(line, flush=True)


if __name__ == "__main__":
    main()
