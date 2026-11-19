"""Is the gap between [-4.6, +1.5] and [-5.2, +1.5] just bootstrap noise?"""
import sys

import numpy as np
import pandas as pd

from common import ROOT

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
g = pd.read_csv(ROOT / "cliponly_test194_scores.csv")
y = (g["label"] == "fake").to_numpy()
d = (((g["prob_adapted"].to_numpy() >= 0.12) == y).astype(int)
     - ((g["prob_zeroshot"].to_numpy() >= 0.09) == y).astype(int))
print(f"n = {len(d)}   mean difference {d.mean() * 100:+.2f} points   "
      f"discordant: {int((d > 0).sum())} better / {int((d < 0).sum())} worse\n")

print(f"{'resamples':>10} {'seed':>6}   95% CI")
for n_boot in (5000, 200000):
    for seed in (42, 1, 2):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(d), (n_boot, len(d)))
        means = d[idx].mean(axis=1) * 100
        print(f"{n_boot:>10} {seed:>6}   [{np.percentile(means, 2.5):+.2f}, "
              f"{np.percentile(means, 97.5):+.2f}]")
print("\n(exact bootstrap distribution is discrete: step =", f"{100 / len(d):.3f} points)")
