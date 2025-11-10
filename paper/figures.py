"""Render every figure of the DRIFT paper from results.json (written by make_data.py)."""
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

# Okabe-Ito
C = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.1, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def band(ax, x, seeds, color, label, smooth=1):
    a = np.array(seeds)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        a = np.array([np.convolve(s, k, mode="valid") for s in a])
        x = x[smooth // 2: smooth // 2 + a.shape[1]]
    m, s = a.mean(0), a.std(0, ddof=1)
    ax.plot(x, m, color=color, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, linewidth=0)


# 1. Pareto: latency vs DOVER
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.1), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
for i, r in enumerate(R["main"]):
    if r["a100"] is None:
        continue
    ours = r["name"].startswith("DRIFT")
    ax.scatter(r["a100"], r["dover"], s=12 + 6 * np.log10(r["params"] * 1e6 / 1e5), color=C[3] if ours else C[0],
               marker="*" if ours else "o", zorder=3, edgecolor="none", alpha=0.9)
    ax.annotate(r["name"].replace(" (teacher)", "$^\\dagger$"), (r["a100"], r["dover"]), fontsize=5.5,
                xytext=(3, 2), textcoords="offset points")
ax.set_xscale("log")
ax.axvline(33.3, color="0.4", lw=0.6, ls="--")
ax.text(36, 0.757, "30 FPS", fontsize=6, color="0.3")
ax.set_xlabel("A100 latency per 1080p frame (ms, log)")
ax.set_ylabel("VideoLQ DOVER $\\uparrow$")
ax.set_title("(a) datacentre GPU")
ax = axs[1]
for r in R["main"]:
    if r["npu"] is None:
        continue
    ours = r["name"].startswith("DRIFT")
    ax.scatter(r["npu"], r["dover"], color=C[3] if ours else C[0], marker="*" if ours else "o", s=30, zorder=3)
    off = {"DRIFT FP16": (4, 2), "DRIFT W8A8": (4, -8), "RealBasicVSR": (-30, 5)}[r["name"]]
    ax.annotate(r["name"], (r["npu"], r["dover"]), fontsize=5.5, xytext=off, textcoords="offset points")
for q in R["quant"]:
    if "PTQ, min" in q[0]:
        ax.scatter(q[5], q[1], color=C[1], marker="s", s=14, zorder=3)
        ax.annotate("W8A8 PTQ", (q[5], q[1]), fontsize=5.5, xytext=(3, -7), textcoords="offset points")
ax.axvline(33.3, color="0.4", lw=0.6, ls="--")
ax.set_xlabel("Hexagon NPU latency (ms)")
ax.set_title("(b) Snapdragon 8 Gen 3")
ax.set_xlim(0, 135)
fig.savefig(os.path.join(OUT, "pareto.pdf"))
plt.close(fig)

# 2. Training curves
tr = R["train"]
it = np.array(tr["it"]) / 1000
fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.9))
for (k, v), c in zip(tr["lpips"].items(), [C[1], C[0], C[3]]):
    band(axs[0], it, v, c, k)
axs[0].set_ylim(0.245, 0.33)
axs[0].set_xlabel("iteration ($\\times 10^3$)")
axs[0].set_ylabel("val. LPIPS (REDS30) $\\downarrow$")
axs[0].legend(loc="upper right")
for (k, v), c in zip(tr["gamma"].items(), [C[1], C[3]]):
    band(axs[1], it, v, c, k, smooth=5)
axs[1].axhline(1.0, color="0.3", lw=0.6, ls="--")
axs[1].set_xlabel("iteration ($\\times 10^3$)")
axs[1].set_ylabel("$\\hat\\gamma$ (state Jacobian norm)")
axs[1].legend(loc="upper right")
fig.savefig(os.path.join(OUT, "training.pdf"))
plt.close(fig)

# 3. Long-horizon drift
d = R["drift"]
t = np.array(d["t"])
fig, ax = plt.subplots(figsize=(5.5, 2.0))
for (k, v), c in zip(d["curves"].items(), [C[7], C[1], C[5], C[0], C[4]]):
    band(ax, t, v, c, k)
ax.set_xlabel("frame index $t$")
ax.set_ylabel("DOVER drop vs. $t{=}10$ $\\downarrow$")
ax.legend(loc="upper left", ncol=2)
ax.set_xlim(1, 1000)
fig.savefig(os.path.join(OUT, "drift.pdf"))
plt.close(fig)

# 4. Heatmap gamma x bits
h = R["heatmap"]
fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.7))
for ax, key, title, cmap in [(axs[0], "drift", "drift@1000 $\\downarrow$", "magma_r"), (axs[1], "dover", "DOVER $\\uparrow$", "viridis")]:
    a = np.array(h[key])
    im = ax.imshow(a, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(h["gammas"])), [("$\\gamma$=" + g if g != "none" else "none") for g in h["gammas"]], fontsize=5.5, rotation=30)
    ax.set_yticks(range(len(h["bits"])), h["bits"])
    ax.grid(False)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v = a[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5.5,
                    color="white" if (v - a.min()) / (np.ptp(a) + 1e-9) > (0.55 if key == "drift" else 2) or (key == "dover" and v < a.mean()) else "black")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=5.5)
axs[1].set_yticks([])
fig.subplots_adjust(wspace=0.12)
fig.savefig(os.path.join(OUT, "heatmap.pdf"))
plt.close(fig)

# 5. Width scaling
s = R["scaling"]
p = np.array([x["params"] for x in s])
fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.9))
for ax, key, lab, tline in [(axs[0], "dover", "VideoLQ DOVER $\\uparrow$", 0.748), (axs[1], "lpips", "REDS30 LPIPS $\\downarrow$", 0.244)]:
    a = np.array([x[key] for x in s])
    ax.errorbar(p, a.mean(1), yerr=a.std(1, ddof=1), color=C[3], marker="o", ms=3, capsize=2, lw=1, label="DRIFT W8A8")
    ax.axhline(tline, color=C[0], ls="--", lw=0.8, label="teacher (DOVE)")
    ax.set_xscale("log")
    ax.set_xlabel("student parameters (M, log)")
    ax.set_ylabel(lab)
    ax.legend(loc="lower right" if key == "dover" else "upper right")
ax2 = axs[0].twiny()
ax2.set_xscale("log")
ax2.set_xlim(axs[0].get_xlim())
ax2.set_xticks(p, [f"{x['npu']:.0f}" for x in s], fontsize=5.5)
ax2.minorticks_off()
ax2.set_xlabel("NPU ms", fontsize=6)
ax2.grid(False)
fig.savefig(os.path.join(OUT, "scaling.pdf"))
plt.close(fig)

# 6. Human study
hu = R["human"]
fig, ax = plt.subplots(figsize=(3.3, 1.9))
y = np.arange(len(hu))
rates = np.array([x["rate"] for x in hu]) * 100
err = np.array([[x["rate"] - x["lo"], x["hi"] - x["rate"]] for x in hu]).T * 100
ax.barh(y, rates, xerr=err, color=[C[0] if r > 50 else C[1] for r in rates], height=0.6, capsize=2,
        error_kw=dict(lw=0.7))
ax.axvline(50, color="0.3", lw=0.6, ls="--")
ax.set_yticks(y, ["vs. " + x["name"] for x in hu])
ax.invert_yaxis()
ax.set_xlabel("preference for DRIFT W8A8 (%)")
ax.set_xlim(0, 100)
fig.savefig(os.path.join(OUT, "human.pdf"))
plt.close(fig)
print("figures written:", sorted(os.listdir(OUT)))
