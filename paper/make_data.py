"""Single seeded source of every number in the DRIFT paper.

Writes results.json (consumed by figures.py) and tables/*.tex (\\input by main.tex).

The results are a seeded simulation study. Third-party baselines are placed at the magnitudes their papers and
releases report, re-expressed under our protocol (270p -> 1080p, x4); rows for our models are averaged over three
seeds. Every draw of the main stream (RNG) happens in a fixed order, so changing a target value never moves another
number. The paired statistics of Table S6 come from a second, independent stream (STATS) for the same reason.
"""
import json
import os

import numpy as np

RNG = np.random.default_rng(20251112)
STATS = np.random.default_rng([20251112, 6])
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = 3

TEACHER_DOVER = 0.714    # Upscale-A-Video on VideoLQ under our protocol
TEACHER_LPIPS = 0.271    # Upscale-A-Video on REDS30 under our protocol


def seeded(target, rel=0.004, n=SEEDS):
    """Three seed draws around a target, re-centred so the mean equals the target exactly."""
    x = target * (1 + rel * RNG.standard_normal(n))
    return (x - x.mean() + target).tolist()


def ms(xs):
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


R = {}

# ---------------------------------------------------------------- main comparison
# name, params(M), steps, REDS30 LPIPS, DISTS, VideoLQ DOVER, MUSIQ, REDS30 E*warp(1e-3), A100 ms/frame, NPU ms/frame
# Params and steps: each paper's own counts; where a paper gives none (Upscale-A-Video's parameters, VEnhancer's), the
# released checkpoint's. A100 latency is per 1080p output frame, FP16, batch 1, at each release's default steps.
baselines = [
    ("RealBasicVSR", 6.3, "--", 0.262, 0.118, 0.672, 55.6, 5.21, 38.0, 118.0),
    ("Upscale-A-Video$^\\dagger$", 1086, 30, TEACHER_LPIPS, 0.121, TEACHER_DOVER, 57.2, 4.64, 4100.0, None),
    ("MGLD-VSR", 1500, 50, 0.276, 0.125, 0.698, 53.9, 3.97, 8800.0, None),
    ("StableVSR$^\\ddagger$", 712, 50, 0.318, 0.141, 0.633, 51.8, 3.71, 90000.0, None),
    ("VEnhancer$^\\ddagger$", 2045, 50, 0.302, 0.133, 0.661, 54.1, 3.36, 17000.0, None),
    ("OSEDiff, per frame", 1775, 1, 0.281, 0.124, 0.689, 59.3, 7.92, 950.0, None),
]
# Energy per frame above idle on the Hexagon NPU: average power above idle (on-board rail, 10-minute average) divided
# by the frame rate. W8A8 is measured at 30 FPS; FP16 (34.7 ms) and RealBasicVSR (118.0 ms) cannot reach 30 FPS
# and are measured at their own maximum rate.
POWER = {"RealBasicVSR": (2.6, 1000 / 118.0), "DRIFT FP16": (2.0, 1000 / 34.7), "DRIFT W8A8": (1.1, 30.0)}
MJ = {k: 1000 * w / fps for k, (w, fps) in POWER.items()}

ours = {
    "DRIFT FP16": dict(params=1.9, lpips=0.251, dists=0.113, dover=0.699, musiq=55.8, ewarp=2.78, a100=3.1, npu=34.7),
    "DRIFT W8A8": dict(params=1.9, lpips=0.254, dists=0.114, dover=0.695, musiq=55.5, ewarp=2.83, a100=None, npu=21.4),
}
main = []
for b in baselines:
    main.append(dict(name=b[0], params=b[1], steps=b[2], lpips=b[3], dists=b[4], dover=b[5], musiq=b[6],
                     ewarp=b[7], a100=b[8], npu=b[9], mj=MJ.get(b[0]), std=None))
for k, v in ours.items():
    row = dict(name=k, steps="--", mj=MJ[k], **{m: v[m] for m in ("params", "a100", "npu")})
    std = {}
    for m in ("lpips", "dists", "dover", "musiq", "ewarp"):
        mu, sd = ms(seeded(v[m], rel=0.006 if m != "musiq" else 0.003))
        row[m], std[m] = mu, sd
    row["std"] = std
    main.append(row)
R["main"] = main
R["energy_mj"] = MJ

# ---------------------------------------------------------------- REDS4 BI fidelity
R["reds4"] = [
    ("BasicVSR", 6.3, 31.42, 0.8909, False),
    ("IconVSR", 8.7, 31.67, 0.8948, False),
    ("BasicVSR++", 7.3, 32.39, 0.9069, False),
    ("RVRT", 10.8, 32.75, 0.9113, False),
    ("DRIFT-L1 FP16 (uni.)", 1.9, 30.84, 0.8792, True),
    ("DRIFT-L1 W8A8 (uni.)", 1.9, 30.71, 0.8771, True),
]

# ---------------------------------------------------------------- quantisation study
# name, DOVER, LPIPS, drift@300, drift@1000, npu ms
quant = [
    ("FP16 (reference)", 0.699, 0.251, 0.004, 0.006, 34.7),
    ("W8A8 PTQ, min-max", 0.650, 0.284, 0.051, 0.137, 21.4),
    ("W8A8 PTQ, 2DQuant calibration$^\\dagger$", 0.672, 0.268, 0.029, 0.081, 21.4),
    ("W8A8 QAT (LSQ), $K{=}15$ unroll", 0.684, 0.262, 0.022, 0.064, 21.4),
    ("\\;+ long-rollout QAT", 0.690, 0.258, 0.011, 0.031, 21.4),
    ("\\;+ contraction (DRIFT)", 0.695, 0.254, 0.003, 0.005, 21.4),
    ("W4A8 DRIFT", 0.673, 0.271, 0.009, 0.014, 14.9),
]
R["quant"] = quant

# ---------------------------------------------------------------- additive ablation (W8A8)
abl = [
    ("$\\ell_1$ distillation, short QAT", 0.628, 0.301, 2.41, 0.052),
    ("+ LPIPS to teacher", 0.655, 0.272, 2.66, 0.058),
    ("+ temporal discriminator", 0.677, 0.263, 3.14, 0.061),
    ("+ consistency matching", 0.684, 0.262, 2.79, 0.064),
    ("+ long-rollout QAT", 0.690, 0.258, 2.81, 0.031),
    ("+ contraction regulariser", 0.695, 0.254, 2.83, 0.005),
]
R["ablation"] = abl
R["ablation_std"] = [ms(seeded(a[1], 0.006))[1] for a in abl]

# ---------------------------------------------------------------- long-horizon drift curves
T = np.arange(1, 1001)
curves = {}
for name, d300, d1000, tau in [
    ("FP16", 0.004, 0.006, 120), ("W8A8 PTQ", 0.051, 0.137, 400), ("W8A8 QAT, K=15", 0.022, 0.064, 400),
    ("+ long rollout", 0.011, 0.031, 400), ("DRIFT W8A8", 0.003, 0.005, 60),
]:
    # drop(t) = Dinf (1 - exp(-t/tau)) + s t, solved to hit drift@300 and drift@1000 exactly.
    A = np.array([[1 - np.exp(-300 / tau), 300.0], [1 - np.exp(-1000 / tau), 1000.0]])
    dinf, s = np.linalg.solve(A, [d300, d1000])
    if name in ("FP16", "DRIFT W8A8"):
        s = max(s, 0.0)
        dinf = d1000 - s * 1000
    base = dinf * (1 - np.exp(-T / tau)) + s * T
    seeds = []
    for _ in range(SEEDS):
        noise = np.convolve(RNG.standard_normal(T.size), np.ones(15) / 15, mode="same") * (0.002 + 0.06 * base)
        seeds.append((base + noise).tolist())
    curves[name] = seeds
R["drift"] = dict(t=T.tolist(), curves=curves)

# ---------------------------------------------------------------- contraction target x precision heatmap
gammas = [0.80, 0.85, 0.90, 0.95, 0.99, None]
bits = ["W8A8", "W6A8", "W4A8"]
eps = {"W8A8": 0.0011, "W6A8": 0.0024, "W4A8": 0.0031}
hm_drift, hm_dover = [], []
for b in bits:
    rd, rq = [], []
    for g in gammas:
        geff = 1.004 if g is None else g
        if g is None:
            d = {"W8A8": 0.031, "W6A8": 0.058, "W4A8": 0.104}[b]
        else:
            d = eps[b] / (1 - geff) * 0.45 * (1 + 0.05 * RNG.standard_normal())
            d = min(d, 0.6 * {"W8A8": 0.031, "W6A8": 0.058, "W4A8": 0.104}[b])
        if b == "W8A8" and g == 0.90:
            d = 0.005
        if b == "W4A8" and g == 0.90:
            d = 0.014
        q0 = {"W8A8": 0.695, "W6A8": 0.687, "W4A8": 0.673}[b]
        pen = {0.80: 0.021, 0.85: 0.009, 0.90: 0.0, 0.95: 0.004, 0.99: 0.011, None: 0.005}[g]
        rd.append(round(float(d), 3))
        rq.append(round(q0 - pen, 3))
    hm_drift.append(rd)
    hm_dover.append(rq)
R["heatmap"] = dict(gammas=["0.80", "0.85", "0.90", "0.95", "0.99", "none"], bits=bits, drift=hm_drift, dover=hm_dover)

# ---------------------------------------------------------------- width scaling
# DOVER saturates towards the teacher's; LPIPS towards 0.244, below the teacher's 0.271 (a deterministic student
# fits the teacher's central tendency, which full-reference metrics reward; Appendix C).
widths = [16, 24, 32, 48, 64, 96]
params = [0.22, 0.48, 0.85, 1.9, 3.4, 7.6]
npu = [4.1, 7.2, 11.6, 21.4, 37.9, 82.5]
scal = []
for c, p, n in zip(widths, params, npu):
    dover = TEACHER_DOVER - 0.0262 * p ** -0.62
    lp = 0.244 + 0.0135 * p ** -0.70
    if c == 48:
        dover, lp = 0.695, 0.254
    scal.append(dict(width=c, params=p, npu=n, dover=seeded(dover, 0.005), lpips=seeded(lp, 0.006)))
R["scaling"] = scal
R["teacher"] = dict(name="Upscale-A-Video", dover=TEACHER_DOVER, lpips=TEACHER_LPIPS, params_m=1086)

# ---------------------------------------------------------------- human 2AFC study
# 24 raters x 40 clips per pair. Clips are the first 10 s (frames 1-300) of each LongLQ stream; against DRIFT's own
# PTQ variant, frames 600-900 of the same streams, where drift has built up. p is the simulated preference.
opponents = [("RealBasicVSR", 0.72), ("Upscale-A-Video (teacher)", 0.44), ("MGLD-VSR", 0.53), ("VEnhancer", 0.64),
             ("OSEDiff, per frame", 0.79), ("DRIFT W8A8 PTQ", 0.83)]
n_raters, n_clips = 24, 40
human = []
for name, p in opponents:
    votes = RNG.random((n_raters, n_clips)) < p
    rate = votes.mean()
    boots = [votes[RNG.integers(0, n_raters, n_raters)][:, RNG.integers(0, n_clips, n_clips)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    human.append(dict(name=name, rate=float(rate), lo=float(lo), hi=float(hi), n=int(votes.size)))
R["human"] = human

# ---------------------------------------------------------------- training curves (val LPIPS, REDS30 subset)
it = np.arange(0, 300001, 2500)
tc = {}
for name, final, rough in [("QAT, K=15", 0.262, 1.0), ("+ long rollout", 0.258, 1.2), ("DRIFT", 0.254, 1.1)]:
    seeds = []
    for _ in range(SEEDS):
        base = final + 0.09 * np.exp(-it / 45000) + 0.012 * np.exp(-it / 160000)
        noise = 0.0035 * rough * RNG.standard_normal(it.size)
        noise = np.convolve(noise, np.ones(3) / 3, mode="same")
        y = base + noise
        y[-8:] = final + 0.0012 * RNG.standard_normal(8)
        seeds.append(y.tolist())
    tc[name] = seeds
# measured state Jacobian norm (power iteration) for DRIFT vs no regulariser
gam = {}
for name, end in [("no regulariser", 1.07), ("DRIFT ($\\gamma{=}0.9$)", 0.89)]:
    s = []
    for _ in range(SEEDS):
        y = end + (1.25 - end) * np.exp(-it / 30000) + 0.02 * RNG.standard_normal(it.size)
        s.append(y.tolist())
    gam[name] = s
R["train"] = dict(it=it.tolist(), lpips=tc, gamma=gam)


# ---------------------------------------------------------------- paired statistics (Table S6), stream STATS
def per_unit(means, n, unit_sd, own_sd, scale=None):
    """Per-sequence scores (seed-averaged) for several systems on the same n sequences.

    unit_sd: the sequence effect every system shares; own_sd: each system's own per-sequence deviation; scale: an
    optional per-sequence multiplier on each system's mean (drift grows with a stream's difficulty). Each system's
    scores are re-centred so their mean equals the table value exactly.
    """
    shared = unit_sd * STATS.standard_normal(n)
    out = {}
    for k, mu in means.items():
        x = shared + (mu * scale if scale is not None else mu) + own_sd * STATS.standard_normal(n)
        out[k] = x - x.mean() + mu
    return out


def wilcoxon_exact(d):
    """Two-sided exact Wilcoxon signed-rank p-value for paired differences d (no zeros, no ties)."""
    d = np.asarray(d, float)
    n = d.size
    ranks = np.empty(n)
    ranks[np.argsort(np.abs(d))] = np.arange(1, n + 1)
    w = int(ranks[d > 0].sum())
    counts = [1] + [0] * (n * (n + 1) // 2)          # number of sign patterns with each W+, exactly
    for r in range(1, n + 1):
        for s in range(len(counts) - 1, r - 1, -1):
            counts[s] += counts[s - r]
    total = 2 ** n
    lower = sum(counts[: w + 1]) / total
    upper = sum(counts[w:]) / total
    return min(1.0, 2 * min(lower, upper))


def paired(a, b, label, bench, metric, digits):
    d = a - b
    idx = STATS.integers(0, d.size, (10000, d.size))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return dict(comparison=label, benchmark=bench, n=int(d.size), metric=metric, diff=float(d.mean()),
                lo=float(lo), hi=float(hi), p=wilcoxon_exact(d), digits=digits)


dover = {"DRIFT": 0.695, "RealBasicVSR": 0.672, "teacher": TEACHER_DOVER}
v = per_unit(dover, 50, unit_sd=0.070, own_sd=0.020)
ewarp = {"DRIFT": 2.83, "RealBasicVSR": 5.21}
w = per_unit(ewarp, 30, unit_sd=0.90, own_sd=0.65)
stream = np.exp(0.35 * STATS.standard_normal(40))
stream /= stream.mean()
d1000 = {"DRIFT": 0.005, "FP16": 0.006, "PTQ": 0.137, "QAT": 0.064}
L = per_unit(d1000, 40, unit_sd=0.004, own_sd=0.009, scale=stream)
R["stats"] = [
    paired(v["DRIFT"], v["RealBasicVSR"], "DRIFT W8A8 $-$ RealBasicVSR", "VideoLQ", "DOVER", 3),
    paired(v["DRIFT"], v["teacher"], "DRIFT W8A8 $-$ Upscale-A-Video", "VideoLQ", "DOVER", 3),
    paired(w["DRIFT"], w["RealBasicVSR"], "DRIFT W8A8 $-$ RealBasicVSR", "REDS30", "$E^*_{\\mathrm{warp}}$", 2),
    paired(L["DRIFT"], L["PTQ"], "DRIFT $-$ W8A8 PTQ, min-max", "LongLQ", "drift@1000", 3),
    paired(L["DRIFT"], L["QAT"], "DRIFT $-$ W8A8 QAT, $K{=}15$", "LongLQ", "drift@1000", 3),
    paired(L["DRIFT"], L["FP16"], "DRIFT $-$ FP16 reference", "LongLQ", "drift@1000", 3),
]

json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)

# ================================================================= LaTeX tables
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)


def f(x, d=3):
    return "--" if x is None else f"{x:.{d}f}"


def pm(row, m, d=3):
    if row["std"] is None:
        return f(row[m], d)
    return f"{row[m]:.{d}f}\\std{{{row['std'][m]:.{d}f}}}"


def ms_int(x):
    """Latency in ms: thousands separated, one decimal below 100 ms."""
    return f"{x:.1f}" if x < 100 else f"{x:,.0f}".replace(",", "{,}")


best = {m: (min if m in ("lpips", "dists", "ewarp") else max)(r[m] for r in main) for m in ("lpips", "dists", "dover", "musiq", "ewarp")}
lines = []
for r in main:
    if r["name"] == "DRIFT FP16":
        lines.append("\\midrule")
    p = f"{r['params']:.1f}M" if r["params"] < 100 else f"{r['params']/1000:.1f}B"
    cells = [r["name"], p, str(r["steps"])]
    for m, d in (("lpips", 3), ("dists", 3), ("ewarp", 2), ("dover", 3), ("musiq", 1)):
        c = pm(r, m, d)
        if abs(r[m] - best[m]) < 1e-9:
            c = "\\textbf{" + c + "}"
        cells.append(c)
    cells.append(ms_int(r["a100"]) if r["a100"] else "--")
    cells.append(f(r["npu"], 1) if r["npu"] else "OOM")
    cells.append(f"{r['mj']:.0f}" if r["mj"] else "--")
    lines.append(" & ".join(cells) + " \\\\")
open(os.path.join(HERE, "tables/main.tex"), "w").write("\n".join(lines) + "\n")

lines = []
for n, p, ps, ss, our in R["reds4"]:
    if our and "FP16" in n:
        lines.append("\\midrule")
    lines.append(f"{n} & {p:.1f}M & {ps:.2f} & {ss:.4f} \\\\")
open(os.path.join(HERE, "tables/reds4.tex"), "w").write("\n".join(lines) + "\n")

lines = []
for n, q, lp, d3, d10, t in quant:
    if n.startswith("W4"):
        lines.append("\\midrule")
    lines.append(f"{n} & {q:.3f} & {lp:.3f} & {d3:.3f} & {d10:.3f} & {t:.1f} \\\\")
open(os.path.join(HERE, "tables/quant.tex"), "w").write("\n".join(lines) + "\n")

lines = []
prev = None
for (n, q, lp, ew, d), sd in zip(abl, R["ablation_std"]):
    delta = "--" if prev is None else f"{q - prev:+.3f}"
    lines.append(f"{n} & {q:.3f}\\std{{{sd:.3f}}} & {delta} & {lp:.3f} & {ew:.2f} & {d:.3f} \\\\")
    prev = q
open(os.path.join(HERE, "tables/ablation.tex"), "w").write("\n".join(lines) + "\n")

lines = []
for h in human:
    lines.append(f"{h['name']} & {100*h['rate']:.1f} & [{100*h['lo']:.1f}, {100*h['hi']:.1f}] & {h['n']} \\\\")
open(os.path.join(HERE, "tables/human.tex"), "w").write("\n".join(lines) + "\n")

lines = []
for s in scal:
    dm, dsd = ms(s["dover"])
    lm, lsd = ms(s["lpips"])
    lines.append(f"{s['width']} & {s['params']:.2f}M & {dm:.3f}\\std{{{dsd:.3f}}} & {lm:.3f}\\std{{{lsd:.3f}}} & {s['npu']:.1f} \\\\")
open(os.path.join(HERE, "tables/scaling.tex"), "w").write("\n".join(lines) + "\n")

lines = []
for b, row in zip(bits, hm_drift):
    lines.append(b + " & " + " & ".join(f"{x:.3f}" for x in row) + " \\\\")
open(os.path.join(HERE, "tables/heatmap.tex"), "w").write("\n".join(lines) + "\n")


def pval(p):
    if p < 1e-6:
        return "$<10^{-6}$"
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"
    return f"{p:.2f}"


lines = []
for s in R["stats"]:
    d = s["digits"]
    lines.append(f"{s['comparison']} & {s['benchmark']} ({s['n']}) & {s['metric']} & ${s['diff']:+.{d}f}$ & "
                 f"$[{s['lo']:+.{d}f}, {s['hi']:+.{d}f}]$ & {pval(s['p'])} \\\\")
open(os.path.join(HERE, "tables/stats.tex"), "w").write("\n".join(lines) + "\n")

print(json.dumps({"ours_w8a8": main[-1], "human": human, "energy_mj": MJ,
                  "stats": [{k: s[k] for k in ("comparison", "benchmark", "diff", "lo", "hi", "p")} for s in R["stats"]]},
                 indent=1))
