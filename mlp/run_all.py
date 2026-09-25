"""Run the full MLP experiment grid of the paper, one process per (experiment, depth, seed).

    python -m mlp.run_all --jobs 8            run everything
    python -m mlp.run_all --only noise_floor  run one group
    python -m mlp.run_all --list              print the commands only
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

PLAIN = (4, 6, 8, 12, 16, 24)
RESIDUAL = (8, 16, 32, 64)
SEEDS = range(10)


def jobs():
    out = []

    def add(group, module, depths, seeds, *extra):
        for d in depths:
            for s in seeds:
                out.append((group, [sys.executable, "-m", f"mlp.{module}", "--depths", str(d),
                                    "--seeds", str(s), str(s + 1), *extra]))

    for mode in ("infer", "train"):
        add("noise_floor", "noise_floor", PLAIN, SEEDS, "--arch", "plain", "--mode", mode)
        add("noise_floor", "noise_floor", PLAIN, SEEDS, "--arch", "plain", "--mode", mode, "--dataset", "mnist")
        for arch in ("res2", "res2c"):
            add("noise_floor", "noise_floor", RESIDUAL, SEEDS, "--arch", arch, "--mode", mode)
        add("noise_floor", "noise_floor", RESIDUAL, SEEDS, "--arch", "res2c", "--mode", mode, "--dataset", "mnist")
    for width in (128, 512):
        add("width", "noise_floor", PLAIN, range(5), "--arch", "plain", "--mode", "infer", "--width", str(width))
        add("width", "amplification", (4, 8, 16, 32), range(5), "--arch", "plain", "--width", str(width))
        add("width", "amplification", RESIDUAL, range(5), "--arch", "res2c", "--width", str(width))
    add("amplification", "amplification", PLAIN + (32,), SEEDS, "--arch", "plain")
    for arch in ("res2", "res2c"):
        add("amplification", "amplification", RESIDUAL, SEEDS, "--arch", arch)
    add("quantize", "quantize", (4, 8, 16, 24, 32), SEEDS, "--arch", "plain")
    for arch in ("res2", "res2c"):
        add("quantize", "quantize", RESIDUAL, SEEDS, "--arch", arch)
    add("layer_map", "layer_map", (4, 8, 16, 24), range(5))
    add("margins", "margins", (8, 16), range(3), "--mode", "infer")
    add("margins", "margins", (8, 16), range(2), "--mode", "train")
    add("counteraction", "counteraction", (8, 16, 32), range(3), "--arch", "plain")
    for arch in ("res2", "res2c"):
        add("counteraction", "counteraction", RESIDUAL, range(3), "--arch", arch)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    todo = [cmd for group, cmd in jobs() if args.only is None or group in args.only]
    if args.list:
        print("\n".join(" ".join(c[1:]) for c in todo))
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print(f"{len(todo)} runs on {args.jobs} workers", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(lambda c: subprocess.call(c, env=env), todo))
    failed = [" ".join(c[1:]) for c, r in zip(todo, codes) if r]
    print(f"done, {len(failed)} failed" + "".join("\n  " + f for f in failed))


if __name__ == "__main__":
    main()
