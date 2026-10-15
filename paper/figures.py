"""Render every figure of the DRIFT paper from results.json (written by make_data.py).

Figures are drawn at their printed size in the one-column arXiv preprint layout (6.5 in text width): FULL = 6.5 in
for every multi-panel figure and WIDE = 4.5 in for a single panel, centred. Each PDF is saved at exactly that size
with no bounding-box cropping, so LaTeX includes it at its natural size and the 7-8 pt fonts below are the sizes on
the page.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

FULL = 6.5   # \textwidth (in)
WIDE = 4.5   # a single-panel figure, centred (in)

# Okabe-Ito
C = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.1, "legend.frameon": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "pdf.fonttype": 42,
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
})
ANN = 7  # annotation size (pt)
TEACHER = R["teacher"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))  # no bbox cropping: the PDF is exactly figsize
    plt.close(fig)


def band(ax, x, seeds, color, label, smooth=1):
    a = np.array(seeds)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        a = np.array([np.convolve(s, k, mode="valid") for s in a])
        x = x[smooth // 2: smooth // 2 + a.shape[1]]
    m, s = a.mean(0), a.std(0, ddof=1)
    ax.plot(x, m, color=color, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, linewidth=0)


# 1. Long-horizon drift (headline figure, one panel)
d = R["drift"]
t = np.array(d["t"])
fig, ax = plt.subplots(figsize=(WIDE, 2.6), layout="constrained")
for (k, v), c in zip(d["curves"].items(), [C[7], C[1], C[5], C[0], C[3]]):
    band(ax, t, v, c, k)
ax.set_xlabel("frame index $t$")
ax.set_ylabel("DOVER drop vs. frames 1–10 $\\downarrow$")
ax.legend(loc="upper left", ncol=1, handlelength=1.6, labelspacing=0.35)
ax.set_xlim(1, 1000)
ax.set_ylim(bottom=-0.005)
save(fig, "drift.pdf")

# 2. Pareto: latency vs DOVER (two panels)
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.3), layout="constrained", gridspec_kw={"width_ratios": [1.7, 1]})
ax = axs[0]
# label placement (dx, dy in points, horizontal alignment)
place = {
    "RealBasicVSR": (6, -1, "left"), "Upscale-A-Video$^\\dagger$": (-6, 0, "right"), "MGLD-VSR": (6, 1, "left"),
    "StableVSR$^\\ddagger$": (-6, 0, "right"), "VEnhancer$^\\ddagger$": (-6, 0, "right"),
    "OSEDiff, per frame": (-6, 0, "right"), "DRIFT FP16": (7, 0, "left"),
}
names = {"StableVSR$^\\ddagger$": "StableVSR",
         "VEnhancer$^\\ddagger$": "VEnhancer", "OSEDiff, per frame": "OSEDiff (per frame)"}
for r in R["main"]:
    if r["a100"] is None:
        continue
    ours = r["name"].startswith("DRIFT")
    ax.scatter(r["a100"], r["dover"], s=46 if ours else 12 + 7 * np.log10(r["params"] * 1e6 / 1e5),
               color=C[3] if ours else C[0], marker="*" if ours else "o", zorder=3, edgecolor="none", alpha=0.9)
    dx, dy, ha = place[r["name"]]
    ax.annotate(names.get(r["name"], r["name"]), (r["a100"], r["dover"]), fontsize=ANN, xytext=(dx, dy),
                textcoords="offset points", ha=ha, va="center")
ax.set_xscale("log")
ax.set_xlim(1.5, 4e5)
ax.axvline(33.3, color="0.4", lw=0.6, ls="--")
ax.text(29, 0.719, "30 FPS", fontsize=ANN, color="0.3", ha="right", va="center")
ax.set_xlabel("A100 latency per 1080p frame (ms, log)")
ax.set_ylabel("VideoLQ DOVER $\\uparrow$")
ax.set_ylim(0.624, 0.724)
ax.set_title("(a) datacentre GPU")
ax = axs[1]
place_b = {"DRIFT FP16": (7, 2, "left"), "DRIFT W8A8": (7, -4, "left"), "RealBasicVSR": (-7, 0, "right")}
for r in R["main"]:
    if r["npu"] is None:
        continue
    ours = r["name"].startswith("DRIFT")
    ax.scatter(r["npu"], r["dover"], color=C[3] if ours else C[0], marker="*" if ours else "o", s=40, zorder=3,
               edgecolor="none")
    dx, dy, ha = place_b[r["name"]]
    ax.annotate(r["name"], (r["npu"], r["dover"]), fontsize=ANN, xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)
for q in R["quant"]:
    if "PTQ, min" in q[0]:
        ax.scatter(q[5], q[1], color=C[1], marker="s", s=18, zorder=3, edgecolor="none")
        ax.annotate("W8A8 PTQ", (q[5], q[1]), fontsize=ANN, xytext=(7, 0), textcoords="offset points", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)
ax.axvline(33.3, color="0.4", lw=0.6, ls="--")
ax.set_xlabel("Hexagon NPU latency per frame (ms)")
ax.set_title("(b) Snapdragon 8 Gen 3 NPU")
ax.set_xlim(0, 135)
ax.set_ylim(axs[0].get_ylim())
save(fig, "pareto.pdf")

# 3. Training curves (two panels)
tr = R["train"]
it = np.array(tr["it"]) / 1000
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.3), layout="constrained")
for (k, v), c in zip(tr["lpips"].items(), [C[1], C[0], C[3]]):
    band(axs[0], it, v, c, k)
axs[0].set_ylim(0.245, 0.33)
axs[0].set_xlabel("iteration ($\\times 10^3$)")
axs[0].set_ylabel("val. LPIPS (REDS30) $\\downarrow$")
axs[0].legend(loc="upper right", handlelength=1.4, labelspacing=0.3)
for (k, v), c in zip(tr["gamma"].items(), [C[1], C[3]]):
    band(axs[1], it, v, c, k.replace("DRIFT ($\\gamma{=}0.9$)", "DRIFT"), smooth=5)
axs[1].axhline(1.0, color="0.3", lw=0.6, ls="--")
axs[1].set_xlabel("iteration ($\\times 10^3$)")
axs[1].set_ylabel("$\\hat\\gamma$ (state Jacobian norm)")
axs[1].legend(loc="upper right", handlelength=1.4, labelspacing=0.3)
save(fig, "training.pdf")

# 4. Heatmap gamma x bits (two panels, side by side)
h = R["heatmap"]
fig, axs = plt.subplots(1, 2, figsize=(FULL, 1.95), layout="constrained")
for ax, key, title, cmap in [(axs[0], "drift", "drift@1000 on LongLQ $\\downarrow$", "magma_r"),
                             (axs[1], "dover", "DOVER on VideoLQ $\\uparrow$", "viridis")]:
    a = np.array(h[key])
    im = ax.imshow(a, cmap=cmap, aspect="auto")
    ax.set_yticks(range(len(h["bits"])), h["bits"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v = a[i, j]
            if key == "drift":
                dark = (v - a.min()) / (np.ptp(a) + 1e-9) > 0.55
            else:
                dark = v < a.mean()
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=ANN, color="white" if dark else "black")
    ax.set_title(title, loc="left", pad=3)
    ax.set_xticks(range(len(h["gammas"])), h["gammas"])
    ax.set_xlabel("contraction target $\\gamma$")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, aspect=10)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
save(fig, "heatmap.pdf")

# 5. Width scaling (two panels)
s = R["scaling"]
p = np.array([x["params"] for x in s])
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.3), layout="constrained")
for ax, key, lab, tline in [(axs[0], "dover", "VideoLQ DOVER $\\uparrow$", TEACHER["dover"]),
                            (axs[1], "lpips", "REDS30 LPIPS $\\downarrow$", TEACHER["lpips"])]:
    a = np.array([x[key] for x in s])
    ax.errorbar(p, a.mean(1), yerr=a.std(1, ddof=1), color=C[3], marker="o", ms=3, capsize=1.8, lw=1.1,
                label="DRIFT W8A8")
    ax.axhline(tline, color=C[0], ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlabel("parameters (M, log)")
    ax.set_ylabel(lab)
axs[0].text(0.21, TEACHER["dover"] - 0.0012, "teacher", color=C[0], fontsize=ANN, va="top")
axs[1].text(6.2, TEACHER["lpips"] + 0.0008, "teacher", color=C[0], fontsize=ANN, va="bottom", ha="right")
axs[0].set_ylim(0.638, 0.72)
axs[1].set_ylim(0.24, 0.29)
ax2 = axs[0].twiny()
ax2.set_xscale("log")
ax2.set_xlim(axs[0].get_xlim())
ax2.set_xticks(p, [f"{x['npu']:.0f}" for x in s], fontsize=7)
ax2.minorticks_off()
ax2.set_xlabel("NPU latency (ms)", fontsize=7, labelpad=2)
ax2.grid(False)
ax2.tick_params(length=2)
save(fig, "scaling.pdf")

# 6. Human study (one panel)
hu = R["human"]
fig, ax = plt.subplots(figsize=(WIDE, 2.1), layout="constrained")
y = np.arange(len(hu))
rates = np.array([x["rate"] for x in hu]) * 100
err = np.array([[x["rate"] - x["lo"], x["hi"] - x["rate"]] for x in hu]).T * 100
ax.barh(y, rates, xerr=err, color=[C[0] if r > 50 else C[1] for r in rates], height=0.6, capsize=1.8,
        error_kw=dict(lw=0.7))
ax.axvline(50, color="0.3", lw=0.6, ls="--")
ax.set_yticks(y, ["vs. " + x["name"].replace(", per frame", " (per frame)") for x in hu])
ax.tick_params(axis="y", length=0)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("preference for DRIFT W8A8 (%)")
ax.set_xlim(0, 100)
save(fig, "human.pdf")
print("figures written:", sorted(os.listdir(OUT)))
