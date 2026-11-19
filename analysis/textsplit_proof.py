"""Two checks that turn "accuracy is 50 percent" into proof that the task is
ill-posed on the text-attack splits.

1. AUROC per split. Threshold-independent: if the score distributions of the
   manipulated and authentic sets coincide, AUROC is 0.5 and no operating
   point can help.
2. Byte-identical pairs. Where a split's manipulated image is the same file as
   one of its authentic images, the model must assign exactly the same
   probability. If it does, the 50 percent is arithmetic, not a deficiency.

Produces the AUROC column of Table II and the pair evidence in Sec. IV-D.
Output: textsplit_proof.csv
"""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ROOT

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SPLITS = ("LinguisticAttack", "RandomSwap", "SemanticSwap")

print(f"{'split':<18}{'n':>6}{'AUROC':>9}{'mean p(fake)':>15}"
      f"{'mean p(real)':>15}")
auroc_rows = []
for name in SPLITS:
    df = pd.read_csv(ROOT / f"textsplit_scores_{name}.csv")
    y = df["is_fake"].astype(bool).to_numpy()
    p = df["prob"].to_numpy()
    auc = roc_auc_score(y.astype(int), p)
    print(f"{name:<18}{len(df):>6}{auc:>9.4f}"
          f"{p[y].mean():>15.4f}{p[~y].mean():>15.4f}")
    auroc_rows.append({"split": name, "n": len(df), "auroc": round(auc, 4),
                       "mean_prob_fakeclass": round(float(p[y].mean()), 4),
                       "mean_prob_realclass": round(float(p[~y].mean()), 4)})

print("\nBYTE-IDENTICAL PAIRS WITHIN EACH SPLIT")
print(f"{'split':<18}{'pairs':>7}{'max |delta p|':>16}  verdict")
pair_rows = []
for name in SPLITS:
    df = pd.read_csv(ROOT / f"textsplit_scores_{name}.csv")
    by_hash = defaultdict(list)
    for path, fake, prob in df.itertuples(index=False):
        try:
            h = hashlib.md5(Path(path).read_bytes()).hexdigest()
        except OSError:
            continue
        by_hash[h].append((bool(fake), float(prob)))

    deltas = []
    for group in by_hash.values():
        if {f for f, _ in group} == {True, False}:
            probs = [p for _, p in group]
            deltas.append(max(probs) - min(probs))
    worst = max(deltas) if deltas else None
    if deltas:
        verdict = ("identical scores" if worst < 1e-9
                   else f"DIFFER by {worst:.2e}")
        print(f"{name:<18}{len(deltas):>7}{worst:>16.2e}  {verdict}")
    else:
        print(f"{name:<18}{0:>7}{'-':>16}  no within-split duplicate pairs")
    pair_rows.append({"split": name, "cross_label_pairs": len(deltas),
                      "max_abs_delta_prob": worst})

out = pd.merge(pd.DataFrame(auroc_rows), pd.DataFrame(pair_rows), on="split")
out.to_csv(ROOT / "textsplit_proof.csv", index=False)
print(f"\nwrote {ROOT / 'textsplit_proof.csv'}")
