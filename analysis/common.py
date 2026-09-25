"""Readers for the result files and the statistics used throughout the analysis."""
import collections
import os

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

import paths


def records(name):
    """Parse a result file into a list of dicts of its key=value tokens; other tokens go to 'tags'."""
    path = os.path.join(paths.RESULTS, name)
    out = []
    for ln in open(path):
        p = ln.split()
        if not p:
            continue
        r = {"tags": tuple(t for t in p if "=" not in t)}
        for t in p:
            if "=" in t:
                k, v = t.split("=", 1)
                r[k] = v
        out.append(r)
    return out


def pairs(s, n=2):
    """'a:b,c:d' -> array [[a, b], [c, d]] (n values per item)."""
    return np.array([[float(x) for x in t.split(":")[:n]] for t in s.split(",")])


def cells(s):
    """'b:acc:eta,...' -> {b: (acc, eta)}."""
    out = {}
    for t in s.split(","):
        v = t.split(":")
        out[int(v[0])] = tuple(float(x) for x in v[1:])
    return out


def logistic(le, lec, k, A):
    return 0.1 + (A - 0.1) / (1 + np.exp(k * (le - lec)))


def eta_c(runs, p0=0.3):
    """Noise floor from a logistic fit in log(eta), pooled over seeds; runs = list of arrays [[eta, acc], ...]."""
    E = np.concatenate([r[:, 0] for r in runs])
    A = np.concatenate([r[:, 1] for r in runs])
    return float(np.exp(curve_fit(logistic, np.log(E), A, p0=[np.log(p0), 3, 0.8], maxfev=20000)[0][0]))


def loglog(x, y):
    """Slope of log y on log x with its 95% t-interval half-width."""
    r = stats.linregress(np.log(x), np.log(y))
    return float(r.slope), float(stats.t.ppf(0.975, len(x) - 2) * r.stderr)


def linfit(x, y):
    r = stats.linregress(x, y)
    return float(r.slope), float(stats.t.ppf(0.975, len(x) - 2) * r.stderr), float(r.intercept)


def bit_floor(fp, acc, bits=(8, 6, 5, 4, 3, 2, 1)):
    """Bit-width at which accuracy falls to (fp + 0.1) / 2, by linear interpolation between tested bit-widths."""
    mid = (fp + 0.1) / 2
    bits = [b for b in bits if b in acc]
    for hi, lo in zip(bits[:-1], bits[1:]):
        if acc[hi] >= mid > acc[lo]:
            return lo + (mid - acc[lo]) / (acc[hi] - acc[lo]) * (hi - lo)
    return float("nan")


def eta_at_bits(eta, b):
    """Relative quantization error at a fractional bit-width, log-linear between tested bit-widths."""
    bs = sorted(eta)
    for lo, hi in zip(bs[:-1], bs[1:]):
        if lo <= b <= hi:
            f = (b - lo) / (hi - lo)
            return float(np.exp(np.log(eta[lo]) + f * (np.log(eta[hi]) - np.log(eta[lo]))))
    return float("nan")


def gamma(eta, bits=(3, 4, 5, 6, 8)):
    """Bits-to-noise rate: minus the slope of log2 eta_q(b) against b."""
    return float(-np.polyfit(bits, np.log2([eta[b] for b in bits]), 1)[0])


def group(rows, *keys):
    out = collections.defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out
