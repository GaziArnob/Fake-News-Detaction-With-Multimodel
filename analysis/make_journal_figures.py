"""Figures for the journal manuscript, all drawn from saved data.

  fig_j_resolution.png   the resolution fingerprint in the two image-attack splits
  fig_j_roc.png          ROC curves: image-attack split versus the three text-attack splits
  fig_j_adaptation.png   the adaptation effect measured four ways, and per seed

Inputs, all in the project root:
  images/, BanFakeMM image folders       resolution counts
  cliponly_test194_scores.csv            AdversarialGen sealed scores (n = 194)
  textsplit_scores_<split>.csv           per-image scores on the text-attack splits
  big_sealed_results_v2.csv              accuracy differences on the 1,308-image set
  adaptation_final.csv / _seeds.csv      AUROC differences, 8 seeds
Also writes figure_data_resolution.csv so the first figure is traceable.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from common import ROOT, SPLIT_IMAGES

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, CORAL, GREEN, GREY = "#3b6ea5", "#c1553b", "#5a8f5a", "#8a8a85"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "figure.dpi": 150,
})


def clean(ax):
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


# =============================================================== 1. resolution
rows = []
sizes = {}
for split in ("AdversarialEdit", "AdversarialGen"):
    for sub in ("real", "fake"):
        folder = SPLIT_IMAGES[split] / sub
        counts = collections.Counter()
        for p in folder.iterdir():
            if p.is_file():
                try:
                    with Image.open(p) as im:
                        counts[im.size] += 1
                except Exception:
                    pass
        sizes[(split, sub)] = counts
        n = sum(counts.values())
        for (w, h), c in counts.items():
            rows.append({"split": split, "class": sub, "width": w, "height": h,
                         "count": c, "share_pct": round(c / n * 100, 2)})
pd.DataFrame(rows).to_csv(ROOT / "figure_data_resolution.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.9))
for ax, split, title in zip(axes, ("AdversarialEdit", "AdversarialGen"),
                            ("(a) AdversarialEdit (training split)",
                             "(b) AdversarialGen (shift split)")):
    real, fake = sizes[(split, "real")], sizes[(split, "fake")]
    nr, nf = sum(real.values()), sum(fake.values())
    top_real = [s for s, _ in real.most_common(5)]
    top_fake = fake.most_common(1)[0][0]
    cats = top_real + ([top_fake] if top_fake not in top_real else [])
    r_share = [real.get(c, 0) / nr * 100 for c in cats]
    f_share = [fake.get(c, 0) / nf * 100 for c in cats]
    other_r = 100 - sum(r_share)
    other_f = 100 - sum(f_share)
    cats_lbl = [f"{w}×{h}" for w, h in cats] + ["other"]
    r_share.append(other_r)
    f_share.append(other_f)
    x = np.arange(len(cats_lbl))
    ax.bar(x - 0.2, r_share, 0.38, color=BLUE, label="authentic")
    ax.bar(x + 0.2, f_share, 0.38, color=CORAL, label="manipulated")
    ax.set_xticks(x)
    ax.set_xticklabels(cats_lbl, rotation=30, ha="right")
    ax.set_ylabel("share of class (%)")
    ax.set_ylim(0, 108)
    ax.set_title(title, loc="left")
    clean(ax)
    top = f_share[cats.index(top_fake)]
    ax.annotate(f"{top:.1f}%", (cats.index(top_fake) + 0.2, top),
                ha="center", va="bottom", fontsize=7, color=CORAL)
axes[0].legend(loc="upper left", ncol=1, frameon=False)
fig.tight_layout(h_pad=0.8)
fig.savefig(OUT / "fig_j_resolution.png", dpi=300)
plt.close(fig)
print("wrote fig_j_resolution.png")
for split in ("AdversarialEdit", "AdversarialGen"):
    f = sizes[(split, "fake")]
    n = sum(f.values())
    w, h = f.most_common(1)[0][0]
    print(f"  {split}: manipulated at {w}x{h} = {f[(w, h)]}/{n} "
          f"({f[(w, h)] / n * 100:.1f}%)")

# ==================================================================== 2. ROC
curves = []
sealed = pd.read_csv(ROOT / "sealed1308_scores.csv")        # from save_sealed_scores.py
curves.append((f"AdversarialGen (image attack, n={len(sealed):,})",
               (sealed["label"] == "fake").astype(int).to_numpy(),
               sealed["prob_raw_zeroshot"].to_numpy(), BLUE, "-"))
for name, colour, style in (("LinguisticAttack", CORAL, "-"),
                            ("RandomSwap", GREEN, "--"),
                            ("SemanticSwap", GREY, ":")):
    d = pd.read_csv(ROOT / f"textsplit_scores_{name}.csv")
    curves.append((f"{name} (text attack, n=2000)",
                   d["is_fake"].astype(int).to_numpy(),
                   d["prob"].to_numpy(), colour, style))

fig, ax = plt.subplots(figsize=(3.5, 4.1))
ax.plot([0, 1], [0, 1], color="#bbbbb5", linewidth=0.8)
aucs = {}
for label, y, p, colour, style in curves:
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    aucs[label] = auc
    ax.plot(fpr, tpr, color=colour, linestyle=style, linewidth=1.5,
            label=f"{label}: {auc:.3f}")
ax.set_xlabel("false-positive rate")
ax.set_ylabel("true-positive rate")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_aspect("equal")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False,
          fontsize=6.6, title="AUROC", title_fontsize=6.8)
ax.grid(alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_j_roc.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("wrote fig_j_roc.png")
for k, v in aucs.items():
    print(f"  {k:<44} AUROC {v:.4f}")

# ============================================================= 3. adaptation
# (a) accuracy difference at a fixed threshold, three measurement set-ups.
rng = np.random.default_rng(42)
gen = pd.read_csv(ROOT / "cliponly_test194_scores.csv")     # the older 194-image set
y194 = (gen["label"] == "fake").to_numpy()
ok0 = ((gen["prob_zeroshot"].to_numpy() >= 0.09) == y194).astype(int)
ok1 = ((gen["prob_adapted"].to_numpy() >= 0.12) == y194).astype(int)
d194 = ok1 - ok0
boot = [d194[rng.integers(0, len(d194), len(d194))].mean() * 100
        for _ in range(5000)]
acc194 = (d194.mean() * 100, np.percentile(boot, 2.5), np.percentile(boot, 97.5))
print(f"\nrecomputed n=194 accuracy difference: {acc194[0]:+.1f} "
      f"[{acc194[1]:+.1f}, {acc194[2]:+.1f}]   (conference paper: -1.5 [-4.6, +1.5])")

big = pd.read_csv(ROOT / "big_sealed_results_v2.csv")
big = big[big.model == "adapted - zeroshot"].set_index("mode")
setups = [
    ("n = 194,\noriginal files", acc194, CORAL),
    ("n = 1,308,\noriginal files", (big.loc["raw", "acc"], big.loc["raw", "ci_lo"],
                                    big.loc["raw", "ci_hi"]), GREY),
    ("n = 1,308,\nartefact-controlled", (big.loc["controlled", "acc"],
                                         big.loc["controlled", "ci_lo"],
                                         big.loc["controlled", "ci_hi"]), BLUE),
]
fin = pd.read_csv(ROOT / "adaptation_final.csv").set_index("mode")
seeds = pd.read_csv(ROOT / "adaptation_final_seeds.csv")

def draw_adaptation(S, size, tag):
    plt.rcParams.update({'xtick.labelsize': 7 * S, 'ytick.labelsize': 7 * S})
    if tag:
        fig = plt.figure(figsize=size)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1.05],
                              hspace=0.55, wspace=0.55)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, :])]
    else:
        fig, axes = plt.subplots(1, 3, figsize=size,
                                 gridspec_kw={"width_ratios": [1.15, 0.95, 1.4]})
    ax = axes[0]
    for i, (lab, (m, lo, hi), colour) in enumerate(setups[::-1]):
        ax.plot([lo, hi], [i, i], color=colour, linewidth=2.2, solid_capstyle="round")
        ax.plot([m], [i], "o", color=colour, markersize=5, markeredgecolor="white")
        ax.text(hi + 0.25, i, f"{m:+.1f}".replace("-", "−"), va="center",
                fontsize=7 * S, color=colour)
    ax.axvline(0, color="#555", linewidth=0.7, linestyle="--")
    ax.set_yticks(range(3))
    ax.set_yticklabels([s[0] for s in setups[::-1]], fontsize=6.6 * S)
    ax.set_xlabel("accuracy gain (points)", fontsize=8 * S)
    ax.set_title("(a) Accuracy, fixed threshold", loc="left", fontsize=8.5 * S)
    ax.set_xlim(-6.8, 8.2)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    for i, (mode, lab, colour) in enumerate((("controlled", "artefact-controlled", BLUE),
                                             ("raw", "original files", GREY))):
        m, lo, hi = fin.loc[mode, "d_auroc"], fin.loc[mode, "ci_lo"], fin.loc[mode, "ci_hi"]
        ax.plot([lo, hi], [i, i], color=colour, linewidth=2.2, solid_capstyle="round")
        ax.plot([m], [i], "o", color=colour, markersize=5, markeredgecolor="white")
        ax.text(hi + 0.0012, i, f"{m:+.4f}", va="center", fontsize=7 * S, color=colour)
    ax.axvline(0, color="#555", linewidth=0.7, linestyle="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["artefact-\ncontrolled", "original\nfiles"], fontsize=6.6 * S)
    ax.set_xlabel("AUROC gain", fontsize=8 * S)
    ax.set_title("(b) AUROC, threshold-free", loc="left", fontsize=8.5 * S)
    ax.set_xlim(-0.004, 0.047)
    ax.set_xticks([0, 0.02, 0.04])
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    ax = axes[2]
    pos = {("raw", "zeroshot"): 0, ("raw", "adapted"): 1.7,
           ("controlled", "zeroshot"): 3.9, ("controlled", "adapted"): 5.6}
    for mode, colour in (("raw", GREY), ("controlled", BLUE)):
        sub = seeds[seeds["mode"] == mode]
        for _, r in sub.iterrows():
            ax.plot([pos[(mode, "zeroshot")], pos[(mode, "adapted")]],
                    [r["auroc_zeroshot"], r["auroc_adapted"]],
                    color=colour, alpha=0.55, linewidth=0.9, marker="o", markersize=2.6)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels(["zero-\nshot", "adapted", "zero-\nshot", "adapted"], fontsize=6.6 * S)
    ax.text(0.85, 1.015, "original files", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=7 * S, color=GREY)
    ax.text(4.75, 1.015, "controlled", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=7 * S, color=BLUE)
    ax.set_ylabel("AUROC", fontsize=8 * S)
    ax.set_title("(c) One line per seed", loc="left", fontsize=8.5 * S, pad=14)
    ax.set_xlim(-0.6, 6.2)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    if not tag:
        fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / f"fig_j_adaptation{tag}.png", dpi=300,
                    bbox_inches="tight" if tag else None, pad_inches=0.05)
    plt.close(fig)
    print(f"\nwrote fig_j_adaptation{tag}.png")


draw_adaptation(1.0, (7.1, 2.55), "")
draw_adaptation(1.3, (7.1, 6.6), "_springer")
print(f"  AUROC gain: original {fin.loc['raw', 'd_auroc']:+.4f}, "
      f"controlled {fin.loc['controlled', 'd_auroc']:+.4f}; "
      f"{len(seeds[seeds['mode'] == 'raw'])} seeds per mode, "
      f"improved in {int((seeds['d_auroc'] > 0).sum())} of {len(seeds)} runs")
