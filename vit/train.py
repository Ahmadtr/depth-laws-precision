"""Train CIFAR-10 Vision Transformers and save checkpoints.

AdamW (lr 1e-3, weight decay 0.05), two warm-up epochs then cosine decay, batch 256, gradient clipping at 1,
flip and crop augmentation.

Example:
    python -m vit.train --arch pre --depths 2 4 8 16 32 --seeds 0 3 --epochs 30
Checkpoints: <DL_MODELS>/vit_{arch}_d{D}_s{seed}_e{epochs}.pt
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

import paths
from vision.common import DEV, accuracy, augment, load_cifar
from .models import ARCHS, ViT


def train(net, X, Y, epochs, seed, bs=256, lr=1e-3, wd=0.05, warm=2):
    g = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=wd)
    spe = (len(X) + bs - 1) // bs
    total = epochs * spe
    schedule = lambda t: min(1.0, (t + 1) / (warm * spe)) * 0.5 * (1 + math.cos(math.pi * min(t, total) / total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X), device=DEV, generator=g)
        for k in range(0, len(X), bs):
            j = perm[k:k + bs]
            loss = F.cross_entropy(net(augment(X[j])), Y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    return net


def checkpoint(arch, depth, seed, epochs):
    return os.path.join(paths.MODELS, f"vit_{arch}_d{depth}_s{seed}_e{epochs}.pt")


def train_and_save(arch, depth, seed, epochs, data, noise=0.0):
    Xtr, Ytr, _, _ = data
    torch.manual_seed(seed)
    net = ViT(arch, depth).to(DEV)
    net.noise = noise
    return train(net, Xtr, Ytr, epochs, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--depths", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 3), metavar=("FIRST", "STOP"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--overwrite", action="store_true", help="retrain even if the checkpoint exists")
    args = ap.parse_args()
    data = load_cifar()
    for depth in args.depths:
        for seed in range(*args.seeds):
            path = checkpoint(args.arch, depth, seed, args.epochs)
            if os.path.exists(path) and not args.overwrite:
                continue
            t0 = time.time()
            net = train_and_save(args.arch, depth, seed, args.epochs, data)
            torch.save(net.state_dict(), path)
            print(f"vit_{args.arch} depth={depth} seed={seed} epochs={args.epochs} "
                  f"fp={accuracy(net, data[2], data[3]):.4f} train_s={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
