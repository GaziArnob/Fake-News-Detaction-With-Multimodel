"""Empirical confirmation that the image-only model is blind to the
text-attack splits, as the ceiling analysis predicts.

Fits the zero-shot CLIP-only model on AdversarialEdit and scores
LinguisticAttack, RandomSwap and SemanticSwap. Prediction under test:
fake-recall near zero and accuracy at chance, because every manipulated image
in those splits is byte-identical to an authentic one.

Produces the measured columns of Table II.
Outputs: textsplit_results.csv, textsplit_scores_<split>.csv
"""
import sys

import numpy as np
import pandas as pd

from common import ROOT, SPLIT_IMAGES, base_raw_features, embed, metrics

sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_ablation as ab

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SPLITS = {k: v for k, v in SPLIT_IMAGES.items()
          if k in ("LinguisticAttack", "RandomSwap", "SemanticSwap")}

frame, cols = base_raw_features()
labels = frame["label"].eq("fake").astype(int).to_numpy()
print(f"training rows {len(frame)}, clip columns {len(cols)}", flush=True)

wanted = [p for base in SPLITS.values() for sub in ("real", "fake")
          for p in (base / sub).iterdir() if p.is_file()]
cache = embed(wanted, "textsplits")

model = ab.build_pipeline(False, [], True, cols)
model.fit(frame[cols].reset_index(drop=True), labels)
print("\nzero-shot CLIP-only model fitted on the AdversarialEdit split\n")

rows = []
print(f"{'split':<18}{'tau':>6}{'acc':>8}{'fake-rec':>10}{'real-rec':>10}"
      f"   confusion")
for name, base in SPLITS.items():
    paths, is_fake = [], []
    for sub in ("real", "fake"):
        for p in (base / sub).iterdir():
            if p.is_file() and str(p) in cache:
                paths.append(str(p))
                is_fake.append(sub == "fake")
    is_fake = np.array(is_fake)
    X = pd.DataFrame([cache[p] for p in paths], columns=cols)
    prob = model.predict_proba(X)[:, 1]
    for tau in (0.50, 0.09):
        m = metrics(prob, is_fake, tau)
        print(f"{name:<18}{tau:>6.2f}{m['acc']:>8.1f}{m['rec_f']:>10.1f}"
              f"{m['rec_r']:>10.1f}   tp={m['tp']} fp={m['fp']} "
              f"fn={m['fn']} tn={m['tn']}")
        rows.append({"split": name, "tau": tau, "n": len(is_fake),
                     "acc": m["acc"], "rec_f": m["rec_f"], "rec_r": m["rec_r"],
                     "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                     "tn": m["tn"]})
    pd.DataFrame({"image_path": paths, "is_fake": is_fake,
                  "prob": prob}).to_csv(
        ROOT / f"textsplit_scores_{name}.csv", index=False)

pd.DataFrame(rows).to_csv(ROOT / "textsplit_results.csv", index=False)
print("\nwrote textsplit_results.csv and the per-split score files")
