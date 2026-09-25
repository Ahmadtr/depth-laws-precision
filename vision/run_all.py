"""Run the full CNN and ViT grid of the paper on one GPU, in order: training, measurement, noise-aware training,
error decomposition. Existing checkpoints are reused, and measured checkpoints are skipped.

    python -m vision.run_all              run everything (several GPU-days, dominated by noise-aware training)
    python -m vision.run_all --list       print the commands only
    python -m vision.run_all --only train measure
"""
import argparse
import subprocess
import sys

CNN_DEPTHS = {"plain": (4, 8, 16), "res": (4, 8, 16, 32, 64), "resc": (4, 8, 16, 32, 64), "res1": (8, 16, 32, 64),
              "plaind": (8, 16, 32, 64), "preln": (4, 8, 16, 32, 64), "plainln": (4, 8, 16, 32, 64)}
VIT_DEPTHS = (2, 4, 8, 16, 32)


def commands():
    py = [sys.executable, "-m"]
    out = []
    for arch, depths in CNN_DEPTHS.items():
        out.append(("train", py + ["cnn.train", "--arch", arch, "--depths", *map(str, depths), "--seeds", "0", "3"]))
    out.append(("train", py + ["cnn.train", "--arch", "res1", "--depths", "8", "64", "--seeds", "0", "2", "--epochs", "30"]))
    for arch in ("pre", "premu", "post"):
        out.append(("train", py + ["vit.train", "--arch", arch, "--depths", *map(str, VIT_DEPTHS), "--seeds", "0", "3"]))
    out.append(("measure", py + ["vision.measure", "--family", "cnn"]))
    out.append(("measure", py + ["vision.measure", "--family", "vit"]))
    out.append(("counteraction", py + ["cnn.counteraction", "--bits", "6"]))
    out.append(("qat", py + ["vision.qat_noise", "--family", "cnn", "--arch", "res1", "--depths", "8", "16", "32", "64"]))
    out.append(("qat", py + ["vision.qat_noise", "--family", "cnn", "--arch", "res1", "--depths", "8", "64",
                             "--seeds", "0", "2", "--epochs", "30"]))
    out.append(("qat", py + ["vision.qat_noise", "--family", "vit", "--arch", "pre", "--depths", "2", "4", "8", "16"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=("train", "measure", "counteraction", "qat"), default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    todo = [c for g, c in commands() if args.only is None or g in args.only]
    for c in todo:
        print(" ".join(c[1:]), flush=True)
        if not args.list:
            subprocess.run(c, check=True)


if __name__ == "__main__":
    main()
