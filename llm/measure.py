"""Precision floor and amplification of a pretrained language model (inference only).

Data: WikiText-103 validation, 512-token sequences (32 for accuracy, 8 for amplification, 4 for calibration).
    fp        next-token top-1 accuracy
    G_raw, G_cen, G_kl   amplification at eta in {0.002, 0.005, 0.01}, 30 draws: relative change of the logits,
              of the centred logits, and sqrt(2 E KL(p || p_eta)) / eta of the predictive distribution
    eta_c     noise level at which accuracy halves (14 log-spaced levels from 0.01 to 2, two draws each)
    b_c       bit-width at which calibrated LSQ-style PTQ accuracy halves; q lists accuracy and eta_q per bit

Example:
    python -m llm.measure gpt2
    python -m llm.measure gpt2-xl --g-batch 1 --g-cpu        (lower GPU memory)
    python -m llm.measure pythia-410m --per-channel
Output: results/llm/measure.txt
"""
import argparse
import math
import os
import time

import numpy as np
import torch

import paths
from .perturb import Perturber, quantize_weight

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ, N_EVAL, N_G, N_CAL = 512, 32, 8, 4
BITS = [3, 4, 5, 6, 8, 10, 12]
ETAS = list(np.round(np.logspace(-2, 0.3, 14), 4))
ETAS_G = (0.002, 0.005, 0.01)


def _transformers():
    """Import transformers; disable its optional scikit-learn integration if that installation is broken."""
    try:
        import sklearn.metrics  # noqa: F401
    except Exception:
        import transformers.utils as tu
        import transformers.utils.import_utils as iu
        iu.is_sklearn_available = tu.is_sklearn_available = (lambda: False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM, AutoTokenizer


def load_tokens(tok):
    import pyarrow.parquet as pq
    path = os.path.join(paths.HF, "wikitext", "wikitext-103-raw-v1", "validation-00000-of-00001.parquet")
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = tok(text[:2_000_000], return_tensors="pt").input_ids[0]
    n = N_EVAL + N_G + N_CAL
    ids = ids[:n * SEQ].view(n, SEQ)
    return ids[:N_EVAL].to(DEV), ids[N_EVAL:N_EVAL + N_G].to(DEV), ids[N_EVAL + N_G:].to(DEV)


@torch.no_grad()
def accuracy(model, X, bs=4):
    c = n = 0
    for k in range(0, len(X), bs):
        x = X[k:k + bs]
        lg = model(x).logits[:, :-1]
        c += (lg.argmax(-1) == x[:, 1:]).sum().item()
        n += x[:, 1:].numel()
    return c / n


@torch.no_grad()
def amplification(model, P, X, eta, draws=30, bs=2, offload=False):
    raw = cen = kl = den_r = den_c = 0.0
    ntok = 0
    keep = (lambda v: v.cpu()) if offload else (lambda v: v)
    clean = []
    for k in range(0, len(X), bs):
        z = model(X[k:k + bs]).logits.float()
        clean.append(tuple(keep(v) for v in (z, z - z.mean(-1, keepdim=True), torch.log_softmax(z, -1))))
    for _ in range(draws):
        P.add_noise(eta)
        for i, k in enumerate(range(0, len(X), bs)):
            z0, c0, lp0 = (v.to(DEV) for v in clean[i])
            z = model(X[k:k + bs]).logits.float()
            c = z - z.mean(-1, keepdim=True)
            raw += (z - z0).pow(2).sum().item()
            den_r += z0.pow(2).sum().item()
            cen += (c - c0).pow(2).sum().item()
            den_c += c0.pow(2).sum().item()
            kl += (lp0.exp() * (lp0 - torch.log_softmax(z, -1))).sum().item()
            ntok += z.shape[0] * z.shape[1]
        P.restore()
    return math.sqrt(raw / den_r) / eta, math.sqrt(cen / den_c) / eta, math.sqrt(2 * kl / ntok) / eta


@torch.no_grad()
def noisy_accuracy(model, P, X, eta, draws=2):
    a = []
    for _ in range(draws):
        P.add_noise(eta)
        a.append(accuracy(model, X))
        P.restore()
    return float(np.mean(a))


def crossing(xs, ys, mid, log=True):
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= mid > y1:
            f = (y0 - mid) / (y0 - y1)
            return math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0))) if log else x0 + f * (x1 - x0)
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="directory name under <DL_DATA>/hf/models, e.g. gpt2 or pythia-1b")
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--g-batch", type=int, default=2)
    ap.add_argument("--g-cpu", action="store_true", help="keep clean logits on the CPU while measuring G")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    AutoModelForCausalLM, AutoTokenizer = _transformers()
    path = os.path.join(paths.HF, "models", args.model)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEV).eval()
    cfg = model.config
    depth = getattr(cfg, "n_layer", None) or cfg.num_hidden_layers
    width = getattr(cfg, "n_embd", None) or cfg.hidden_size
    Xe, Xg, Xc = load_tokens(tok)
    P = Perturber(model, args.per_channel, torch.Generator(device=DEV).manual_seed(args.seed))
    fp = accuracy(model, Xe)
    G = {e: amplification(model, P, Xg, e, bs=args.g_batch, offload=args.g_cpu) for e in ETAS_G}
    accs = []
    for e in ETAS:
        accs.append(noisy_accuracy(model, P, Xe, e))
        if accs[-1] < 0.25 * fp:
            break
    eta_c = crossing(ETAS[:len(accs)], accs, fp / 2)
    P.mode = "calib"
    with torch.no_grad():
        for k in range(len(Xc)):
            model(Xc[k:k + 1])
    P.finish_calibration()
    cells = []
    for b in sorted(BITS, reverse=True):
        werr = [((quantize_weight(W, b, ax, P.pc) - W).norm() / W.norm()).item() for W, ax in zip(P.W0, P.axis)]
        P.quantize(b)
        P.errs = []
        a = accuracy(model, Xe)
        cells.append((b, a, float(np.mean(werr + P.errs))))
        P.errs = None
        P.restore()
    b_c = crossing([c[0] for c in cells], [c[1] for c in cells], fp / 2, log=False)
    fields = [f"llm_{args.model}", f"mode={'pc' if args.per_channel else 'pt'}", f"depth={depth}", f"width={width}",
              f"fp={fp:.4f}"]
    for name, j in (("raw", 0), ("cen", 1), ("kl", 2)):
        fields += [f"G_{name}@{e}={G[e][j]:.4f}" for e in ETAS_G]
    fields += [f"eta_c={eta_c:.4f}", f"b_c={b_c:.3f}",
               "q=" + ",".join(f"{b}:{a:.4f}:{e:.4f}" for b, a, e in sorted(cells)),
               "noise=" + ",".join(f"{x}:{y:.4f}" for x, y in zip(ETAS, accs)), f"t={time.time() - t0:.0f}"]
    line = " ".join(fields)
    with open(paths.result("llm/measure.txt"), "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
