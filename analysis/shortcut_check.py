"""Is the metadata shortcut present in every BanFakeMM split, or only where
the pipeline generates an image?

A classifier is given only file metadata -- width, height, area, aspect ratio,
file size, format and colour mode, with no pixel content -- and asked to
separate manipulated from authentic. The same feature set as the Table I row,
so the five splits are directly comparable.

This is the within-benchmark control for the artefact finding: if the shortcut
appears only in the splits whose manipulated images are synthesised, it is a
property of image-generating pipelines rather than of this corpus as a whole.

Output: shortcut_by_split.csv
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from common import ROOT, SPLIT_IMAGES

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GENERATES_IMAGES = {"AdversarialEdit", "AdversarialGen"}
rows = []

for name, base in SPLIT_IMAGES.items():
    records = []
    for sub in ("real", "fake"):
        folder = base / sub
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if not p.is_file():
                continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    fmt, mode = im.format or "", im.mode
            except Exception:
                continue
            records.append({"w": w, "h": h, "area": w * h,
                            "aspect": w / max(h, 1),
                            "size_bytes": p.stat().st_size,
                            "is_png": int(fmt == "PNG"),
                            "is_rgb": int(mode == "RGB"),
                            "fake": int(sub == "fake")})
    df = pd.DataFrame(records)
    if df.empty:
        continue

    label = name + (" (train)" if name == "AdversarialEdit" else "")
    kind = ("image-manipulating" if name in GENERATES_IMAGES
            else "text-manipulating")
    print(f"\n{'=' * 72}\n{label}   ({len(df)} images, {kind})")
    for lab, who in ((0, "authentic"), (1, "manipulated")):
        d = df[df.fake == lab]
        top = collections.Counter(zip(d.w, d.h)).most_common(2)
        share = top[0][1] / len(d) * 100 if len(d) else 0
        print(f"  {who:<12} top sizes: {top}")
        print(f"  {'':<12} most common covers {share:.1f}% of the class")

    X, y = df.drop(columns=["fake"]), df["fake"].to_numpy()
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    flag = "  <-- SHORTCUT" if acc.mean() > 0.70 else ""
    print(f"  METADATA-ONLY: acc {acc.mean() * 100:.2f}% "
          f"(+-{acc.std() * 100:.2f}), AUROC {auc.mean():.4f}{flag}")
    rows.append({"split": label, "kind": kind, "n": len(df),
                 "acc": round(float(acc.mean()) * 100, 2),
                 "std": round(float(acc.std()) * 100, 2),
                 "auroc": round(float(auc.mean()), 4)})

out = pd.DataFrame(rows)
out.to_csv(ROOT / "shortcut_by_split.csv", index=False)
print(f"\n{'=' * 72}\nSUMMARY")
print(out.to_string(index=False))
print(f"\nwrote {ROOT / 'shortcut_by_split.csv'}")
