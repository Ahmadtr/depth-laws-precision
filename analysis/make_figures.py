"""Build every figure, table and number reported in the paper from the result files.

    python -m analysis.make_figures
Outputs in figures/: fig_*.pdf, tab_*.tex and numbers.json (all numbers quoted in the text).
"""
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

import paths
from .common import (bit_floor, cells, eta_at_bits, eta_c, gamma, group, linfit, loglog, pairs, records)

OUT = os.path.join(paths.REPO, "figures")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
NUM = {}

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 6.4, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 4.3,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL = {"mlp_plain": "#1f4e79", "mlp_res2c": "#c55a11", "mlp_res2": "#2e7d32",
       "cnn_plain": "#4f81bd", "cnn_res": "#6aa84f", "cnn_resc": "#e69138", "cnn_res1": "#a61c00",
       "cnn_plaind": "#7030a0", "cnn_preln": "#00838f", "cnn_plainln": "#795548",
       "vit_pre": "#d81b60", "vit_premu": "#8e24aa", "vit_post": "#fb8c00"}
LAB = {"mlp_plain": "MLP plain", "mlp_res2c": r"MLP res. $s=1/\sqrt{8}$", "mlp_res2": r"MLP res. $s=1/\sqrt{D}$",
       "cnn_plain": "CNN plain", "cnn_res": r"CNN res. $s=1/\sqrt{D}$", "cnn_resc": r"CNN res. $s=1/\sqrt{8}$",
       "cnn_res1": r"CNN res. $s=1$", "cnn_plaind": "CNN plain (Dirac)", "cnn_preln": "CNN pre-norm res.",
       "cnn_plainln": "CNN post-norm plain", "vit_pre": "ViT pre-LN", "vit_premu": r"ViT pre-LN, $s=1/\sqrt{D}$",
       "vit_post": "ViT post-LN"}
MK = {k: {"mlp": "o", "cnn": "s", "vit": "^"}[k[:3]] for k in COL}
MLP_ARCH = ("mlp_plain", "mlp_res2c", "mlp_res2")
CNN_ARCH = ("cnn_plain", "cnn_res", "cnn_resc", "cnn_res1", "cnn_plaind", "cnn_preln", "cnn_plainln")
VIT_ARCH = ("vit_pre", "vit_premu", "vit_post")
ARCH = MLP_ARCH + CNN_ARCH + VIT_ARCH
S2 = {"mlp_plain": 1.0, "mlp_res2c": 1 / 8, "mlp_res2": None}


def below(ax, ncol=2, y=-0.22, fs=6.2):
    ax.legend(frameon=False, fontsize=fs, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              columnspacing=0.8, handlelength=1.6)


def log2_axis(ax, ticks, axis="x"):
    if axis == "x":
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    else:
        ax.set_yscale("log", base=2)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("  wrote", name + ".pdf")


def mean(x):
    return float(np.mean(x))


# ====================================================================== MLP: noise floors and depth laws
def mlp_curves(dataset, width=32):
    R = collections.defaultdict(dict)
    for r in records("mlp/noise_floor.txt"):
        if r["dataset"] == dataset and int(r["width"]) == width:
            R[("mlp_" + r["arch"], r["mode"], int(r["depth"]))][int(r["seed"])] = pairs(r["curve"])
    return R


def depth_law(R, arch, boot=200):
    """Noise floors per depth (PTQ = infer, QAT = train), exponents with t-intervals, ratio and difference
    of the exponents with seed-bootstrap intervals, and the tolerance gain per depth."""
    ds = sorted({d for a, m, d in R if a == arch and m == "infer"})
    both = [set(R[(arch, m, d)]) for m in ("infer", "train") for d in ds if (arch, m, d) in R]
    seeds = sorted(set.intersection(*both))

    def fit(mode, sd):
        e = [eta_c([R[(arch, mode, d)][s] for s in sd]) for d in ds]
        a, c = loglog(ds, e)
        return -a, c, e

    ai, ci, ei = fit("infer", seeds)
    at, ct, et = fit("train", seeds)
    b = []
    for _ in range(boot):
        sd = list(RNG.choice(seeds, len(seeds)))
        try:
            x, y = fit("train", sd)[0], fit("infer", sd)[0]
            b.append((x / y, x - y))
        except Exception:
            pass
    b = np.array(b)
    ok = abs(ai) > 0.1
    return dict(depths=ds, seeds=len(seeds), eta_infer=ei, eta_train=et, a_ptq=ai, a_ptq_ci=ci, a_qat=at,
                a_qat_ci=ct, ratio=at / ai if ok else float("nan"),
                ratio_ci=list(np.percentile(b[:, 0], [2.5, 97.5])) if ok else None,
                delta=at - ai, delta_ci=list(np.percentile(b[:, 1], [2.5, 97.5])),
                gain=list(np.array(et) / np.array(ei)))


print("MLP depth laws")
RF, RM = mlp_curves("fashion"), mlp_curves("mnist")
LAW = {("fashion", a): depth_law(RF, a) for a in MLP_ARCH}
LAW.update({("mnist", a): depth_law(RM, a) for a in ("mlp_plain", "mlp_res2c")})
for k, v in LAW.items():
    print("  %-22s a_ptq %.3f a_qat %.3f ratio %.3f delta %.3f" % (k, v["a_ptq"], v["a_qat"], v["ratio"], v["delta"]))
NUM["mlp_laws"] = {"%s_%s" % k: v for k, v in LAW.items()}


# ====================================================================== MLP: amplification, width, quantization
def mlp_amplification(width=32):
    """Per architecture: seed-level rows with depth, G (logit, KL), linearity and median margin."""
    R = collections.defaultdict(list)
    for r in records("mlp/amplification.txt"):
        if int(r["width"]) != width:
            continue
        R["mlp_" + r["arch"]].append(dict(
            D=int(r["depth"]), seed=int(r["seed"]), fp=float(r["fp"]), M=float(r["M"]),
            G_logit=mean([float(r[f"G_logit@{e}"]) for e in ("0.001", "0.005", "0.01")]),
            G=mean([float(r[f"G_kl@{e}"]) for e in ("0.005", "0.01")]),
            lin=float(r["G_kl@0.001"]) / float(r["G_kl@0.01"])))
    return R


print("MLP amplification")
AMP = mlp_amplification()
EXP, GROW = {}, {}
for a in MLP_ARCH:
    L = LAW[("fashion", a)]
    ec = dict(zip(L["depths"], L["eta_infer"]))
    rows = [r for r in AMP[a] if r["D"] in ec]
    rho, rc = loglog([r["D"] for r in rows], [r["G"] for r in rows])
    ds = sorted(ec)
    lam = [mean([r["G"] for r in rows if r["D"] == d]) * ec[d] for d in ds]
    rl, rlc = loglog([r["D"] for r in rows], [r["G_logit"] for r in rows])
    mu, _ = loglog([r["D"] for r in rows], [r["M"] for r in rows])
    EXP[a] = dict(rho=rho, rho_ci=rc, alpha=L["a_ptq"], alpha_ci=L["a_ptq_ci"], lam=(ds, lam),
                  lin=mean([r["lin"] for r in rows]), rho_logit=rl, rho_logit_ci=rlc, mu=mu)
    d = np.array([r["D"] for r in AMP[a]])
    g2 = np.array([r["G"] ** 2 for r in AMP[a]])
    sl, slc, g0 = linfit(d, g2)
    GROW[a] = dict(slope=sl, ci=slc, G0sq=g0, s2=S2[a],
                   G2_by_D={int(x): mean(g2[d == x]) for x in sorted(set(d))},
                   G2_sd={int(x): float(np.std(g2[d == x])) for x in sorted(set(d))})
    print("  %-10s rho %.3f+-%.3f alpha %.3f+-%.3f  G2 slope %.2f+-%.2f" % (a, rho, rc, L["a_ptq"], L["a_ptq_ci"], sl, slc))

print("MLP width study")
WIDTH = {}
for width in (32, 128, 512):
    A = mlp_amplification(width)
    for a in ("mlp_plain", "mlp_res2c"):
        fp = {x: mean([r["fp"] for r in A[a] if r["D"] == x]) for x in {r["D"] for r in A[a]}}
        failed = sorted(x for x, v in fp.items() if v < 0.7)
        if failed:
            print("  %-10s width %3d: depths %s excluded (training failed, accuracy < 0.7)" % (a, width, failed))
        rows = [r for r in A[a] if r["D"] not in failed]
        d = np.array([r["D"] for r in rows])
        sl, slc, _ = linfit(d, np.array([r["G_logit"] ** 2 for r in rows]))
        if a == "mlp_plain":
            # the amplification exponent is compared with the noise-floor exponent over the same depth range
            top = max(LAW[("fashion", a)]["depths"])
            d_rho, g_rho = zip(*[(r["D"], r["G_logit"]) for r in rows if r["D"] <= top])
        else:
            d_rho, g_rho = d, [r["G_logit"] for r in rows]
        rl, rlc = loglog(d_rho, g_rho)
        entry = dict(kappa=sl / S2[a], kappa_ci=slc / S2[a], rho=rl, rho_ci=rlc, alpha=None, alpha_ci=None,
                     depths=sorted(set(d.tolist())), excluded=failed)
        if width == 32:
            entry.update(alpha=LAW[("fashion", a)]["a_ptq"], alpha_ci=LAW[("fashion", a)]["a_ptq_ci"])
        elif a == "mlp_plain":
            Rw = mlp_curves("fashion", width)
            ds = sorted({k[2] for k in Rw if k[0] == a and k[1] == "infer"})
            e = [eta_c(list(Rw[(a, "infer", x)].values())) for x in ds]
            al, alc = loglog(ds, e)
            entry.update(alpha=-al, alpha_ci=alc)
        WIDTH[(a, width)] = entry
        print("  %-10s width %3d kappa %.2f+-%.2f rho_logit %.2f+-%.2f alpha %s" % (
            a, width, entry["kappa"], entry["kappa_ci"], rl, rlc, entry["alpha"]))
NUM["width"] = {"%s_%d" % k: v for k, v in WIDTH.items()}

print("MLP quantization")
Q = group([r for r in records("mlp/quantize.txt") if int(r["width"]) == 32], "arch", "depth")
BITS = {}
eta_mlp = collections.defaultdict(list)
for (arch, depth), rows in Q.items():
    for r in rows:
        if arch == "plain":
            for b, (e,) in cells(r["eta_q"]).items():
                eta_mlp[b].append(e)
GAMMA = {"mlp": gamma({b: mean(v) for b, v in eta_mlp.items()})}
DYN_CAL = {}
for a in MLP_ARCH:
    ds = sorted(int(d) for (x, d) in Q if "mlp_" + x == a)
    out = {"depths": ds}
    for kind in ("cal", "dyn"):
        bc, lo, hi = [], [], []
        for d in ds:
            rows = Q[(a[4:], str(d))]
            fp = [float(r["fp"]) for r in rows]
            acc = [{b: v[0] for b, v in cells(r[kind]).items()} for r in rows]
            f = lambda idx: bit_floor(mean([fp[i] for i in idx]), {b: mean([acc[i][b] for i in idx]) for b in acc[0]})
            bc.append(f(range(len(rows))))
            bs = [f(RNG.integers(0, len(rows), len(rows))) for _ in range(300)]
            lo.append(float(np.nanpercentile(bs, 2.5)))
            hi.append(float(np.nanpercentile(bs, 97.5)))
        out[kind] = dict(bc=bc, lo=lo, hi=hi, slope=float(np.polyfit(np.log2(ds), bc, 1)[0]))
    BITS[a] = out
    print("  %-10s calibrated slope %.3f  dynamic slope %.3f" % (a, out["cal"]["slope"], out["dyn"]["slope"]))
print("  gamma MLP %.3f" % GAMMA["mlp"])
NUM["mlp_bits"] = BITS
NUM["eta_q_mlp"] = {int(b): mean(v) for b, v in eta_mlp.items()}


# ====================================================================== MLP: layer map, margins, counteraction
print("MLP layer map")
QL = collections.defaultdict(list)
for r in records("mlp/layer_map.txt"):
    QL[(r["mode"], int(r["depth"]))].append(np.array([float(x) for x in r["q"].split(",")]))
LP = LAW[("fashion", "mlp_plain")]
MECH = {}
for D in sorted({d for _, d in QL}):
    P, T = np.mean(QL[("ptq", D)], 0), np.mean(QL[("qat", D)], 0)
    g = LP["gain"][LP["depths"].index(D)]
    Gf = float(np.sqrt(P.sum() / T.sum()))
    third = max(1, D // 3)
    MECH[D] = dict(gain=g, G_factor=Gf, Lambda_factor=g / Gf, G2_ratio=float(T.sum() / P.sum()),
                   ratio_first=float(T[1:-third].sum() / P[1:-third].sum()) if D > 3 else float("nan"),
                   ratio_last=float(T[-third:].sum() / P[-third:].sum()), qp=P.tolist(), qq=T.tolist())
    print("  D=%2d gain %.2f G factor %.2f Lambda factor %.2f" % (D, g, Gf, g / Gf))
NUM["mechanism"] = MECH

print("MLP margins")


def margin_table(mode):
    R = collections.defaultdict(list)
    for r in records(f"mlp/margins_{mode}.txt"):
        c = np.array([[float(x) for x in t.split(":")] for t in r["tags"]])
        R[(int(r["depth"]), float(r["ls"]), float(r["lnoise"]))].append((float(r["med_nmargin"]), float(r["frac_nm<0.25"]), c))
    return {k: dict(margin=mean([x[0] for x in v]), low=mean([x[1] for x in v]), eta=eta_c([x[2] for x in v], 0.2))
            for k, v in R.items()}


MG = {"PTQ": margin_table("infer"), "QAT": margin_table("train")}
MARG = {}
for mode, T in MG.items():
    for D in (8, 16):
        base = T[(D, 0.0, 0.0)]
        for k, v in T.items():
            if k[0] == D:
                MARG[f"{mode}_{k[0]}_{k[1]}_{k[2]}"] = dict(margin_ratio=v["margin"] / base["margin"],
                                                           low_ratio=base["low"] / v["low"], eta_ratio=v["eta"] / base["eta"],
                                                           eta_times_D=v["eta"] * D)
qat_eD = [v["eta_times_D"] for k, v in MARG.items() if k.startswith("QAT")]
NUM["margins"] = dict(conditions=MARG, qat_eta_times_D=(mean(qat_eD), float(np.std(qat_eD))))
print("  QAT eta*D %.2f +- %.2f" % NUM["margins"]["qat_eta_times_D"])

print("MLP counteraction")
CM = group(records("mlp/counteraction.txt"), "tags", "depth", "train")
COUNTER = {"mlp_cancel_all": [float(r["cancel"]) for rows in CM.values() for r in rows]}
for arch in ("plain", "res2", "res2c"):
    for mode in ("fp", "qat"):
        g = [mean([float(r["gain"]) for r in rows]) for k, rows in CM.items() if k[0] == (arch,) and k[2] == mode]
        COUNTER[f"mlp_{arch}_{mode}_gain_range"] = (min(g), max(g))
c = np.array(COUNTER["mlp_cancel_all"])
COUNTER["mlp_cancel_mean_sd"] = (float(c.mean()), float(c.std()))


# ====================================================================== CNNs and ViTs
def vision_rows(family, epochs):
    R = collections.defaultdict(list)
    for r in records(f"{family}/measure.txt"):
        if int(r["epochs"]) != epochs:
            continue
        row = dict(D=int(r["depth"]), seed=int(r["seed"]), fp=float(r["fp"]), M=float(r["M"]), ec=float(r["eta_c"]),
                   G=mean([float(r[f"G_kl@{e}"]) for e in ("0.005", "0.01")]),
                   G_logit=mean([float(r[f"G_logit@{e}"]) for e in ("0.001", "0.005", "0.01")]),
                   lin=float(r["G_kl@0.001"]) / float(r["G_kl@0.01"]), lsq=cells(r["lsq"]))
        if "minmax" in r:
            row["minmax"] = cells(r["minmax"])
            row["kurt"] = float(r["kurt"])
        R[r["tags"][0]].append(row)
    return R


def by_depth(rows, f):
    ds = sorted({r["D"] for r in rows})
    return ds, [mean([f(r) for r in rows if r["D"] == d]) for d in ds]


def bit_slope(rows, quantizer="lsq"):
    ds = sorted({r["D"] for r in rows})
    bc = []
    for d in ds:
        rr = [r for r in rows if r["D"] == d]
        bc.append(bit_floor(mean([r["fp"] for r in rr]), {b: mean([r[quantizer][b][0] for r in rr]) for b in rr[0][quantizer]}))
    m = np.isfinite(bc)
    slope = float(np.polyfit(np.log2(np.array(ds)[m]), np.array(bc)[m], 1)[0]) if m.sum() > 1 else float("nan")
    return ds, bc, slope


def harm_ratio(rows, quantizer="lsq"):
    """eta_q(b_c) / eta_c per depth: quantization error at the bit floor relative to the Gaussian noise floor."""
    ds, bc, _ = bit_slope(rows, quantizer)
    out = {}
    for d, b in zip(ds, bc):
        rr = [r for r in rows if r["D"] == d]
        eq = eta_at_bits({k: mean([r[quantizer][k][1] for r in rr]) for k in rr[0][quantizer]}, b)
        out[d] = eq / mean([r["ec"] for r in rr])
    return out


print("CNN and ViT exponents")
VR = vision_rows("cnn", 15)
VR.update(vision_rows("vit", 30))
CNN30 = vision_rows("cnn", 30)
for a in CNN_ARCH + VIT_ARCH:
    rows = [r for r in VR[a] if np.isfinite(r["ec"])]
    D = [r["D"] for r in rows]
    rho, rc = loglog(D, [r["G"] for r in rows])
    al, ac = loglog(D, [r["ec"] for r in rows])
    mu, _ = loglog(D, [r["M"] for r in rows])
    rl, rlc = loglog(D, [r["G_logit"] for r in rows])
    ds, lam = by_depth(rows, lambda r: r["ec"] * r["G"])
    EXP[a] = dict(rho=rho, rho_ci=rc, alpha=-al, alpha_ci=ac, lam=(ds, lam), lin=mean([r["lin"] for r in rows]),
                  rho_logit=rl, rho_logit_ci=rlc, mu=mu)
    print("  %-12s alpha %+.3f+-%.3f rho %+.3f+-%.3f Lambda %s" % (a, -al, ac, rho, rc, np.round(lam, 2).tolist()))


def pooled_eta(rows_by_arch, archs, quantizer):
    E = collections.defaultdict(list)
    for a in archs:
        for r in rows_by_arch[a]:
            for b, v in r[quantizer].items():
                E[b].append(v[1])
    return {b: mean(v) for b, v in E.items()}


CNN_Q = [a for a in CNN_ARCH if a != "cnn_plaind"]
ETA_B = {"MLP, LSQ-style": NUM["eta_q_mlp"], "CNN, LSQ-style": pooled_eta(VR, CNN_Q, "lsq"),
         "CNN, min-max": pooled_eta(VR, CNN_Q, "minmax"), "ViT, LSQ-style": pooled_eta(VR, VIT_ARCH, "lsq")}
GAMMA.update(cnn=gamma(ETA_B["CNN, LSQ-style"]), cnn_minmax=gamma(ETA_B["CNN, min-max"]), vit=gamma(ETA_B["ViT, LSQ-style"]))
print("  gamma", {k: round(v, 3) for k, v in GAMMA.items()})

SLOPE, HARM = {}, {}
for a in MLP_ARCH:
    SLOPE[a] = BITS[a]["cal"]["slope"]
for a in CNN_ARCH + VIT_ARCH:
    SLOPE[a] = bit_slope(VR[a])[2] if a != "cnn_plaind" else float("nan")
    if a != "cnn_plaind":
        HARM[a] = harm_ratio(VR[a])
res1_minmax = bit_slope(VR["cnn_res1"], "minmax")[2]
res1_30 = dict(bit_slope=bit_slope(CNN30["cnn_res1"])[2],
               alpha=-loglog([r["D"] for r in CNN30["cnn_res1"]], [r["ec"] for r in CNN30["cnn_res1"]])[0],
               fp={d: v for d, v in zip(*by_depth(CNN30["cnn_res1"], lambda r: r["fp"]))})
allh = [v for a, h in HARM.items() for v in h.values() if np.isfinite(v)]
print("  harm ratio range %.2f-%.2f; res1 min-max slope %.3f; res1 30 epochs %s" % (min(allh), max(allh), res1_minmax, res1_30))
NUM.update(gamma=GAMMA, bit_slopes=SLOPE, harm=HARM, harm_range=(min(allh), max(allh)), res1_minmax_slope=res1_minmax,
           res1_30=res1_30, eta_b=ETA_B)

for a in MLP_ARCH:
    L = LAW[("fashion", a)]
    h = {}
    for d, ec in zip(L["depths"], L["eta_infer"]):
        rows = Q.get((a[4:], str(d)))
        if not rows:
            continue
        eta = {b: mean([cells(r["eta_q"])[b][0] for r in rows]) for b in cells(rows[0]["eta_q"])}
        k = BITS[a]["depths"].index(d)
        h[d] = eta_at_bits(eta, BITS[a]["cal"]["bc"][k]) / ec
    HARM[a] = h
allh = [v for a, h in HARM.items() for v in h.values() if np.isfinite(v)]
NUM["harm"], NUM["harm_range"] = HARM, (min(allh), max(allh))
print("  harm ratio range incl. MLPs %.2f-%.2f" % NUM["harm_range"])

print("margin-corrected predictor")
MPRED = {a: dict(err_G=abs(EXP[a]["alpha"] - EXP[a]["rho"]), err_MG=abs(EXP[a]["alpha"] - (EXP[a]["rho"] - EXP[a]["mu"])))
         for a in ARCH}
NUM["margin_predictor"] = MPRED
print("  M/G better than 1/G in %d of %d architectures" % (sum(v["err_MG"] < v["err_G"] for v in MPRED.values()), len(MPRED)))

print("CNN counteraction")
CC = collections.defaultdict(list)
for r in records("cnn/counteraction.txt"):
    if r["tags"][0] != "cnn_plaind":
        CC[(r["tags"][0], int(r["depth"]))].append(float(r["cancel"]))
allc = np.array([c for v in CC.values() for c in v])
COUNTER.update(cnn_cancel_median=float(np.median(allc)), cnn_cancel_iqr=list(np.percentile(allc, [25, 75])),
               cnn_models=len(allc))
NUM["counteraction"] = COUNTER
print("  MLP cancel %.3f +- %.3f; CNN median %.3f IQR %s" % (*COUNTER["mlp_cancel_mean_sd"], COUNTER["cnn_cancel_median"],
                                                             np.round(COUNTER["cnn_cancel_iqr"], 3).tolist()))

# ====================================================================== QAT in CNNs and ViTs
def qat_law(family, arch, epochs, p_infer, p_train):
    R = collections.defaultdict(dict)
    for r in records(f"{family}/qat_noise.txt"):
        if r["tags"][0] == arch and int(r["epochs"]) == epochs:
            R[(r["tags"][1], int(r["depth"]))][int(r["seed"])] = np.array([[float(x) for x in t.split(":")] for t in r["tags"][2:]])
    ds = sorted({d for m, d in R})
    seeds = sorted(set.intersection(*[set(v) for v in R.values()]))

    def fit(sd):
        ei = [eta_c([R[("infer", d)][s] for s in sd], p_infer) for d in ds]
        et = [eta_c([R[("train", d)][s] for s in sd], p_train) for d in ds]
        return ei, et

    ei, et = fit(seeds)
    out = dict(depths=ds, seeds=len(seeds), eta_infer=ei, eta_train=et, gain=list(np.array(et) / np.array(ei)))
    ai = -np.polyfit(np.log(ds), np.log(ei), 1)[0]
    at = -np.polyfit(np.log(ds), np.log(et), 1)[0]
    out.update(a_ptq=float(ai), a_qat=float(at), delta=float(at - ai))
    if len(ds) > 2:
        b = []
        for _ in range(300):
            sd = list(RNG.choice(seeds, len(seeds)))
            try:
                i, t = fit(sd)
                b.append(-np.polyfit(np.log(ds), np.log(t), 1)[0] + np.polyfit(np.log(ds), np.log(i), 1)[0])
            except Exception:
                pass
        out["delta_ci"] = list(np.percentile(b, [2.5, 97.5]))
        out["a_ptq_ci"], out["a_qat_ci"] = loglog(ds, ei)[1], loglog(ds, et)[1]
    return out


print("QAT in CNNs and ViTs")
CQ = {"15 epochs": qat_law("cnn", "cnn_res1", 15, 0.4, 0.6), "30 epochs": qat_law("cnn", "cnn_res1", 30, 0.4, 0.6)}
VQ = qat_law("vit", "vit_pre", 30, 0.4, 0.8)
for k, v in list(CQ.items()) + [("ViT pre-LN", VQ)]:
    print("  %-12s gain %s delta %.3f %s" % (k, np.round(v["gain"], 2).tolist(), v["delta"], np.round(v.get("delta_ci", []), 3)))
NUM["qat_cnn"], NUM["qat_vit"] = CQ, VQ


# ====================================================================== pretrained language models
print("Language models")
LLM = {"pt": {}, "pc": {}}
for r in records("llm/measure.txt"):
    k = [float(r[f"G_kl@{e}"]) for e in ("0.002", "0.005", "0.01")]
    LLM[r["mode"]][r["tags"][0][4:]] = dict(D=int(r["depth"]), H=int(r["width"]), fp=float(r["fp"]), ec=float(r["eta_c"]),
                                            G=mean(k), lin=k[0] / k[2], bc=float(r["b_c"]),
                                            G_raw=float(r["G_raw@0.005"]), G_cen=float(r["G_cen@0.005"]))
GPT = sorted([m for m in LLM["pt"] if m.startswith("gpt2")], key=lambda m: LLM["pt"][m]["D"])
gD = [LLM["pt"][m]["D"] for m in GPT]
ga, gac = loglog(gD, [LLM["pt"][m]["ec"] for m in GPT])
gr, grc = loglog(gD, [LLM["pt"][m]["G"] for m in GPT])
GPT2 = dict(alpha=-ga, alpha_ci=gac, rho=gr, rho_ci=grc,
            Lambda={m: LLM["pt"][m]["ec"] * LLM["pt"][m]["G"] for m in GPT},
            Lambda_pc={m: LLM["pc"][m]["ec"] * LLM["pc"][m]["G"] for m in GPT if m in LLM["pc"]})
for mode in LLM:
    for m, v in LLM[mode].items():
        v["Lambda"] = v["ec"] * v["G"]
        v["linear"] = abs(v["lin"] - 1) < 0.15
print("  GPT-2 alpha %.3f+-%.3f rho %.3f+-%.3f Lambda %s" % (-ga, gac, gr, grc, np.round(list(GPT2["Lambda"].values()), 2).tolist()))
NUM["llm"], NUM["gpt2"] = LLM, GPT2

trained = [x for a in ARCH for x in EXP[a]["lam"][1]]
ok = [x for x in trained if x > 1.0]
NUM["lambda_all"] = dict(mean=mean(trained), cv=float(np.std(trained) / np.mean(trained)), min=min(trained), max=max(trained),
                         min_excl_outliers=min(ok))
NUM["exponents"] = EXP
NUM["growth"] = GROW
lins = {a: EXP[a]["lin"] for a in ARCH}
NUM["linearity_range"] = (min(lins.values()), max(lins.values()))
print("  Lambda over trained architectures: mean %.3f CV %.3f range %.2f-%.2f" % (
    NUM["lambda_all"]["mean"], NUM["lambda_all"]["cv"], NUM["lambda_all"]["min"], NUM["lambda_all"]["max"]))


# ====================================================================== figures
print("figures")
# ---- amplification
fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.45))
for a in MLP_ARCH:
    g = GROW[a]
    ds = list(g["G2_by_D"])
    lab = LAB[a] + (r" (slope/$s^2$ = %.1f)" % (g["slope"] / g["s2"]) if g["s2"] else "")
    ax[0].errorbar(ds, list(g["G2_by_D"].values()), yerr=list(g["G2_sd"].values()), fmt=MK[a], color=COL[a], capsize=2, label=lab)
    xx = np.linspace(0, max(ds) * 1.03, 40)
    ax[0].plot(xx, g["G0sq"] + g["slope"] * xx, "-", color=COL[a], lw=1)
ax[0].set_xlabel("depth $D$")
ax[0].set_ylabel(r"$G^2$")
ax[0].set_title(r"(a) MLPs: $G^2$ grows with $D$")
below(ax[0], ncol=1)
for i, archs, ticks, title in ((1, CNN_ARCH, [4, 8, 16, 32, 64], "(b) CNNs (CIFAR-10)"), (2, VIT_ARCH, [2, 4, 8, 16, 32], "(c) ViTs (CIFAR-10)")):
    for a in archs:
        ds, g = by_depth(VR[a], lambda r: r["G"])
        ax[i].plot(ds, g, MK[a] + "-", color=COL[a], label=LAB[a])
    log2_axis(ax[i], ticks)
    ax[i].set_yscale("log", base=2)
    ax[i].yaxis.set_major_formatter(ScalarFormatter())
    ax[i].set_xlabel("depth $D$")
    ax[i].set_title(title)
    below(ax[i], ncol=1 if i == 2 else 2, fs=5.9)
ax[1].set_ylabel("$G$")
save(fig, "fig_amplification")

# ---- predictor
fig, ax = plt.subplots(figsize=(3.5, 3.2))
ax.plot([-0.65, 0.85], [-0.65, 0.85], "k--", lw=0.8, zorder=0)
ax.text(0.5, 0.42, r"$\alpha=\rho$", rotation=40, fontsize=8)
for a in ARCH:
    e = EXP[a]
    ax.errorbar(e["rho"], e["alpha"], xerr=e["rho_ci"], yerr=e["alpha_ci"], fmt=MK[a], color=COL[a], capsize=2, ms=4.6,
                label=LAB[a], elinewidth=0.8)
ax.errorbar(gr, -ga, xerr=grc, yerr=gac, fmt="*", color="k", ms=9, capsize=2, label="GPT-2 (12–48 layers)")
ax.set_xlabel(r"amplification exponent $\rho$ (full precision)")
ax.set_ylabel(r"collapse exponent $\alpha_{\mathrm{PTQ}}$")
ax.set_xlim(-0.65, 0.85)
ax.set_ylim(-0.65, 0.85)
ax.set_aspect("equal")
ax.legend(frameon=False, fontsize=6.0, loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.3)
save(fig, "fig_predictor")

# ---- residual scaling and the collapse constant
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
for a in ("mlp_res2", "mlp_res2c", "mlp_plain"):
    L = LAW[("fashion", a)]
    ax[0].plot(L["depths"], L["eta_infer"], MK[a] + "-", color=COL[a])
    ax[0].plot(L["depths"], L["eta_train"], MK[a] + "--", color=COL[a], mfc="white")
hd = [Line2D([], [], color=COL[a], marker="o", ls="", label=LAB[a].replace("MLP ", "")) for a in MLP_ARCH]
hd += [Line2D([], [], color="k", ls="-", label="PTQ"), Line2D([], [], color="k", ls="--", marker="o", mfc="white", label="QAT")]
ax[0].legend(handles=hd, frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=0.8)
log2_axis(ax[0], [4, 8, 16, 32, 64])
log2_axis(ax[0], [0.1, 0.2, 0.4, 0.8], "y")
ax[0].set_xlabel("depth $D$")
ax[0].set_ylabel(r"noise floor $\eta_c$")
ax[0].set_title("(a) MLPs (Fashion-MNIST)")
ax[1].axhspan(1.0, 1.8, color="0.92", zorder=0)
for a in ARCH:
    ds, lam = EXP[a]["lam"]
    ax[1].plot(ds, lam, MK[a] + "-", color=COL[a], ms=3.2, lw=0.9, label=LAB[a])
ax[1].plot(gD, list(GPT2["Lambda"].values()), "k*-", ms=7, lw=1, label="GPT-2")
log2_axis(ax[1], [2, 4, 8, 16, 32, 64])
ax[1].set_xlabel("depth $D$")
ax[1].set_ylabel(r"$\Lambda = \eta_c\,G$")
ax[1].set_ylim(0, 2.2)
ax[1].set_title(r"(b) $\eta_c G$ across families")
below(ax[1], ncol=3, fs=5.4, y=-0.2)
save(fig, "fig_residual")

# ---- language models
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.55))
x = np.array(gD)
ec = np.array([LLM["pt"][m]["ec"] for m in GPT])
G = np.array([LLM["pt"][m]["G"] for m in GPT])
ax[0].plot(x, ec, "o-", color="#c00000", label=r"noise floor $\eta_c$ (measured)")
ax[0].plot(x, np.exp(np.mean(np.log(ec * G))) / G, "s--", color="#1f4e79", label=r"$\bar\Lambda / G$ (predicted from amplification)")
log2_axis(ax[0], [int(v) for v in x])
log2_axis(ax[0], [0.125, 0.18, 0.25, 0.35], "y")
ax[0].set_xlabel("layers (GPT-2 small $\\to$ XL)")
ax[0].set_ylabel(r"$\eta_c$")
ax[0].set_title(r"(a) GPT-2: $\alpha=%.2f$, $\rho=%.2f$" % (-ga, gr))
below(ax[0], ncol=1)
names = sorted(LLM["pt"], key=lambda m: (m[:4], LLM["pt"][m]["D"], LLM["pt"][m]["H"]))
w = 0.38
for j, (mode, c, lab) in enumerate((("pt", "#1f4e79", "per-tensor"), ("pc", "#9dc3e6", "per-channel"))):
    for i, m in enumerate(names):
        v = LLM[mode][m]
        ax[1].bar(i + (j - 0.5) * w, v["Lambda"], w, color=c if v["linear"] else "white", edgecolor=c,
                  hatch=None if v["linear"] else "////", label=lab if i == 0 else None)
        if v["Lambda"] > 5:
            ax[1].text(i + (j - 0.5) * w, 4.75, "%.1f" % v["Lambda"], ha="center", fontsize=5.5, rotation=90)
ax[1].bar([0], [0], color="white", edgecolor="gray", hatch="////", label="outside linear regime")
ax[1].axhspan(1.0, 1.8, color="0.92", zorder=0)
ax[1].set_xticks(np.arange(len(names)))
ax[1].set_xticklabels([m.replace("pythia-", "P-").replace("gpt2-", "G-").replace("gpt2", "G-small") for m in names],
                      rotation=45, ha="right", fontsize=6.5)
ax[1].set_ylabel(r"$\Lambda = \eta_c\,G$")
ax[1].set_title("(b) all nine pretrained LLMs")
ax[1].set_ylim(0, 5)
ax[1].legend(frameon=False, fontsize=6.0, loc="upper left")
save(fig, "fig_llm")


# ---- noise to bits
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.55))
gkey = {"MLP, LSQ-style": "mlp", "CNN, LSQ-style": "cnn", "CNN, min-max": "cnn_minmax", "ViT, LSQ-style": "vit"}
for (k, v), c, m in zip(ETA_B.items(), ("#1f4e79", "#6aa84f", "#a61c00", "#d81b60"), ("o", "s", "D", "^")):
    b = sorted(v)
    ax[0].plot(b, [v[x] for x in b], m + "-", color=c, label=r"%s ($\gamma=%.2f$)" % (k.replace("-", "–"), GAMMA[gkey[k]]))
bb = np.array([1, 8])
ax[0].plot(bb, ETA_B["CNN, LSQ-style"][4] * 2.0 ** (-0.5 * (bb - 4)), "k:", lw=1, label=r"theory $\gamma=\frac{1}{2}$")
ax[0].plot(bb, ETA_B["CNN, min-max"][4] * 2.0 ** (-1.0 * (bb - 4)), "k--", lw=0.9, label=r"theory $\gamma=1$")
ax[0].set_yscale("log", base=2)
ax[0].set_xlabel("bit-width $b$")
ax[0].set_ylabel(r"relative quantization error $\eta_q(b)$")
ax[0].set_title("(a) bits-to-noise rate")
below(ax[0], ncol=2)
for a in MLP_ARCH:
    B = BITS[a]["cal"]
    xl = np.log2(BITS[a]["depths"])
    y = np.array(B["bc"])
    pred = EXP[a]["alpha"] / GAMMA["mlp"]
    ax[1].errorbar(xl, y, yerr=[y - np.array(B["lo"]), np.array(B["hi"]) - y], fmt=MK[a], color=COL[a], capsize=2, label=LAB[a])
    ax[1].plot(xl, np.mean(y - pred * xl) + pred * xl, "--", color=COL[a], lw=1)
ax[1].plot([], [], "k--", lw=1, label=r"prediction $\alpha_{\mathrm{PTQ}}/\gamma$")
ax[1].set_xlabel(r"$\log_2 D$")
ax[1].set_ylabel(r"bit floor $b_c$ (calibrated PTQ)")
ax[1].set_title("(b) MLP bit floors (10 seeds)")
below(ax[1], ncol=2)
save(fig, "fig_bits")

# ---- QAT paradox
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
for ds_, ls in (("fashion", "-"), ("mnist", "--")):
    L = LAW[(ds_, "mlp_plain")]
    nm = "F" if ds_ == "fashion" else "M"
    open_ = "white" if ds_ == "mnist" else None
    ax[0].plot(L["depths"], L["eta_infer"], "o" + ls, color="#1f4e79", mfc=open_, label=r"PTQ, %s ($\alpha=%.2f$)" % (nm, L["a_ptq"]))
    ax[0].plot(L["depths"], L["eta_train"], "^" + ls, color="#c00000", mfc=open_, label=r"QAT, %s ($\alpha=%.2f$)" % (nm, L["a_qat"]))
log2_axis(ax[0], [4, 8, 16, 24])
log2_axis(ax[0], [0.08, 0.15, 0.3, 0.6], "y")
ax[0].set_xlabel("depth $D$")
ax[0].set_ylabel(r"noise floor $\eta_c$")
ax[0].set_title("(a) plain MLP: QAT higher but steeper")
below(ax[0], ncol=2, fs=6.0)
for key, lab, c, ls in ((("fashion", "mlp_plain"), "MLP plain, F", "#1f4e79", "-"), (("mnist", "mlp_plain"), "MLP plain, M", "#1f4e79", "--"),
                        (("fashion", "mlp_res2c"), r"MLP $s=1/\sqrt{8}$, F", "#c55a11", "-"),
                        (("mnist", "mlp_res2c"), r"MLP $s=1/\sqrt{8}$, M", "#c55a11", "--"),
                        (("fashion", "mlp_res2"), r"MLP $s=1/\sqrt{D}$, F", "#2e7d32", "-")):
    L = LAW[key]
    ax[1].plot(L["depths"], L["gain"], "o" + ls, color=c, label=lab, mfc="white" if ls == "--" else None, ms=3.6)
for k, ls in (("15 epochs", "-"), ("30 epochs", ":")):
    ax[1].plot(CQ[k]["depths"], CQ[k]["gain"], "s" + ls, color="#a61c00", label=r"CNN $s=1$, %s" % k.replace("epochs", "ep."),
               mfc="white" if ls == ":" else None, ms=3.6)
ax[1].plot(VQ["depths"], VQ["gain"], "^-", color="#d81b60", label="ViT pre-LN", ms=3.8)
ax[1].axhline(1, color="gray", lw=0.7)
log2_axis(ax[1], [2, 4, 8, 16, 32, 64])
ax[1].set_xlabel("depth $D$")
ax[1].set_ylabel(r"tolerance gain $g=\eta_c^{\mathrm{QAT}}/\eta_c^{\mathrm{PTQ}}$")
ax[1].set_ylim(0.8, 2.45)
ax[1].set_title("(b) QAT benefit decays with depth")
below(ax[1], ncol=2, fs=5.8)
save(fig, "fig_qat")

# ---- what does not set the floor
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
for mode, mk in (("PTQ", "o"), ("QAT", "^")):
    for depth, c in ((8, "#1f4e79"), (16, "#c55a11")):
        pts = sorted((v["margin_ratio"], v["eta_ratio"]) for k, v in MARG.items() if k.startswith(f"{mode}_{depth}_"))
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], mk, color=c, mfc="white" if mode == "QAT" else c,
                   label="%s, depth %d" % (mode, depth))
xx = np.linspace(0.9, 5.2, 20)
ax[0].plot(xx, xx, "k:", lw=1, label=r"$\eta_c\propto$ margin")
ax[0].axhline(1, color="gray", lw=0.7)
ax[0].set_xlabel("median normalised margin (relative to baseline)")
ax[0].set_ylabel(r"$\eta_c$ (relative to baseline)")
ax[0].set_ylim(0.6, 2.0)
ax[0].set_title("(a) margins barely move the floor")
below(ax[0], ncol=3, fs=6.0)
for a in ("cnn_res", "cnn_resc", "cnn_res1", "cnn_plaind"):
    ds, k = by_depth(VR[a], lambda r: r["kurt"])
    ax[1].plot(ds, k, MK[a] + "-", color=COL[a], label=LAB[a])
log2_axis(ax[1], [4, 8, 16, 32, 64])
ax[1].set_yscale("log")
ax[1].set_xlabel("depth $D$")
ax[1].set_ylabel("excess kurtosis of activations")
ax[1].set_title("(b) activation tails grow with depth (CNNs)")
below(ax[1], ncol=2)
save(fig, "fig_negative")
NUM["kurtosis"] = {a: dict(zip(*by_depth(VR[a], lambda r: r["kurt"]))) for a in CNN_ARCH}

# ---- mechanism of the QAT paradox
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.55))
for D, c in ((8, "#9dc3e6"), (16, "#2e75b6"), (24, "#1f4e79")):
    x = (np.arange(D) + 0.5) / D
    ax[0].plot(x, MECH[D]["qp"], "-", color=c, lw=1.1, label="PTQ, $D=%d$" % D)
    ax[0].plot(x, MECH[D]["qq"], "--", color=c, lw=1.1, label="QAT, $D=%d$" % D)
ax[0].set_xlabel("relative layer position $l/D$")
ax[0].set_ylabel(r"layer sensitivity $q_l$")
ax[0].set_title("(a) QAT lowers every layer's sensitivity")
below(ax[0], ncol=3, fs=6.0)
Ds = sorted(MECH)
ax[1].plot(Ds, [MECH[D]["gain"] for D in Ds], "o-", color="k", label=r"tolerance gain $g$")
ax[1].plot(Ds, [MECH[D]["G_factor"] for D in Ds], "s--", color="#2e75b6", label=r"amplification factor $G_{\mathrm{PTQ}}/G_{\mathrm{QAT}}$")
ax[1].plot(Ds, [MECH[D]["Lambda_factor"] for D in Ds], "^:", color="#c00000",
           label=r"collapse factor $\Lambda_{\mathrm{QAT}}/\Lambda_{\mathrm{PTQ}}$")
ax[1].axhline(1, color="gray", lw=0.7)
log2_axis(ax[1], Ds)
ax[1].set_xlabel("depth $D$")
ax[1].set_title(r"(b) $g = (G_{\mathrm{PTQ}}/G_{\mathrm{QAT}})\,(\Lambda_{\mathrm{QAT}}/\Lambda_{\mathrm{PTQ}})$")
below(ax[1], ncol=1, fs=6.3)
save(fig, "fig_mechanism")


# ====================================================================== tables
def ci(v, c):
    return r"$%+.2f$ {\scriptsize[%+.2f, %+.2f]}" % (v, v - c, v + c)


def clean(text):
    """Remove the sign from zeros printed as -0.00 or +0.00."""
    return text.replace("-0.00", "0.00").replace("+0.00", "0.00")


def write_table(name, header, rows, spec):
    rows = [clean(r) for r in rows]
    lines = [r"\begin{tabular}{%s}" % spec, r"\toprule", header + r" \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(lines) + "\n")
    print("  wrote", name + ".tex")


print("tables")
rows = []
for a in ARCH:
    e, fam = EXP[a], a[:3]
    g = GAMMA[fam]
    name = LAB[a].split(" ", 1)[1] if fam != "vit" else LAB[a][4:]
    bs = "--" if not np.isfinite(SLOPE[a]) else "$%+.2f$" % SLOPE[a]
    rows.append(r"%s & %s & %s & %s & %.2f--%.2f & %s & $%+.2f$ \\" % (
        {"mlp": "MLP", "cnn": "CNN", "vit": "ViT"}[fam], name, ci(e["rho"], e["rho_ci"]), ci(e["alpha"], e["alpha_ci"]),
        min(e["lam"][1]), max(e["lam"][1]), bs, e["alpha"] / g))
lam = list(GPT2["Lambda"].values())
rows += [r"\midrule", r"LLM & GPT-2, 12--48 layers & %s & %s & %.2f--%.2f & -- & -- \\" % (
    ci(gr, grc), ci(-ga, gac), min(lam), max(lam))]
write_table("tab_exponents", r"Family & Architecture & $\rho$ (amplification) & $\alpha_{\mathrm{PTQ}}$ (collapse) & "
            r"$\Lambda=\eta_c G$ & bit slope & $\alpha_{\mathrm{PTQ}}/\gamma$", rows, "llccccc")

rows = []
for key in (("fashion", "mlp_plain"), ("mnist", "mlp_plain"), ("fashion", "mlp_res2c"), ("mnist", "mlp_res2c"), ("fashion", "mlp_res2")):
    L = LAW[key]
    rat = "n/a" if L["ratio_ci"] is None else r"$%.2f$ {\scriptsize[%.2f, %.2f]}" % (L["ratio"], *L["ratio_ci"])
    rows.append(r"%s & %s & %d & $%.2f\pm%.2f$ & $%.2f\pm%.2f$ & %s & $%+.2f$ {\scriptsize[%+.2f, %+.2f]} \\" % (
        "Fashion-MNIST" if key[0] == "fashion" else "MNIST", LAB[key[1]], L["seeds"], L["a_ptq"], L["a_ptq_ci"],
        L["a_qat"], L["a_qat_ci"], rat, L["delta"], *L["delta_ci"]))
for k, v in CQ.items():
    rows.append(r"CIFAR-10 & CNN res. $s=1$ (%s)$^\dagger$ & %d & $%.2f$ & $%.2f$ & n/a & $%+.2f$ \\" % (
        k, v["seeds"], v["a_ptq"], v["a_qat"], v["delta"]))
rows.append(r"CIFAR-10 & ViT pre-LN & %d & $%.2f\pm%.2f$ & $%.2f\pm%.2f$ & n/a & $%+.2f$ {\scriptsize[%+.2f, %+.2f]} \\" % (
    VQ["seeds"], VQ["a_ptq"], VQ["a_ptq_ci"], VQ["a_qat"], VQ["a_qat_ci"], VQ["delta"], *VQ["delta_ci"]))
write_table("tab_qat", r"Dataset & Architecture & seeds & $\alpha_{\mathrm{PTQ}}$ & $\alpha_{\mathrm{QAT}}$ & "
            r"$\alpha_{\mathrm{QAT}}/\alpha_{\mathrm{PTQ}}$ & $\Delta\alpha$", rows, "llccccc")

rows = []
for m in names:
    v = LLM["pt"][m]
    nm = m.replace("gpt2", "GPT-2").replace("pythia", "Pythia").replace("-medium", " medium").replace("-large", " large").replace("-xl", " XL")
    lam_txt = "%.2f" % v["Lambda"] if v["linear"] else r"\textit{(%.2f)}" % v["Lambda"]
    rows.append(r"%s & %d & %d & %.3f & %.3f & %.2f & %.2f & %s \\" % (
        nm if m != "gpt2" else "GPT-2 small", v["D"], v["H"], v["fp"], v["ec"], v["G"], v["lin"], lam_txt))
write_table("tab_llm", r"Model & layers & width & accuracy & $\eta_c$ & $G$ & linearity & $\Lambda=\eta_c G$", rows, "lrrccccc")

rows = []
for a, nm in (("mlp_plain", "plain"), ("mlp_res2c", r"residual, $s=1/\sqrt8$")):
    for width in (32, 128, 512):
        w = WIDTH[(a, width)]
        al = "--" if w["alpha"] is None else r"$%.2f\pm%.2f$" % (w["alpha"], w["alpha_ci"])
        rows.append(r"%s & %d & $%.2f$ {\scriptsize[%.2f, %.2f]} & $%.2f\pm%.2f$ & %s \\" % (
            nm, width, w["kappa"], w["kappa"] - w["kappa_ci"], w["kappa"] + w["kappa_ci"], w["rho"], w["rho_ci"], al))
write_table("tab_width", r"Architecture & width & $\kappa$ & $\rho$ & $\alpha_{\mathrm{PTQ}}$", rows, "lrccc")


def to_json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=to_json)
print("  wrote numbers.json")
