"""Does the metadata artefact appear in a SECOND Bangla multimodal benchmark?

Section IV-C shows that in BanFakeMM's two image-manipulating splits, file
metadata alone separates manipulated from authentic images perfectly. The
obvious reviewer question is whether that is a property of one corpus or of
constructed benchmarks of this kind.

This applies the identical diagnostic to MultiBanFakeDetect
(doi:10.17632/k5pbz9795f.1, CC BY 4.0), the dataset of the closest prior
system: a classifier given only width, height, area, aspect ratio, file size,
format and colour mode -- no pixel content -- under 5-fold cross-validation.

It also repeats the image-reuse check of Section IV-D: are any manipulated
images byte-identical to authentic ones, as they are in BanFakeMM's three
text-attack splits?

Output: second_benchmark.csv
"""
from __future__ import annotations

import collections
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from common import ROOT

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = Path(r"E:\Project\Fake News Detection Data\_multibanfakedetect")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
FAKE_HINTS = ("fake", "false", "misinformation", "rumor", "rumour",
              "clickbait")
REAL_HINTS = ("real", "true", "authentic", "non-fake", "nonfake", "genuine")


def label_from_path(p: Path):
    """Infer the label from the directory names on the path."""
    parts = [s.lower() for s in p.relative_to(BASE).parts]
    if any(h in s for s in parts for h in REAL_HINTS):
        return "real"
    if any(h in s for s in parts for h in FAKE_HINTS):
        return "fake"
    return None


def survey():
    files = [p for p in BASE.rglob("*") if p.is_file()
             and p.suffix.lower() in IMAGE_EXT]
    print(f"image files found: {len(files)}")
    if not files:
        print("\nDirectory layout, to adjust the label rule:")
        for d in sorted({p.parent for p in BASE.rglob("*") if p.is_file()}):
            n = sum(1 for f in d.iterdir() if f.is_file())
            print(f"  {d.relative_to(BASE)}: {n} files")
        raise SystemExit(1)
    labels = [label_from_path(p) for p in files]
    unknown = sum(l is None for l in labels)
    print(f"labelled: {len(files) - unknown}, unlabelled: {unknown}")
    if unknown:
        print("  unlabelled examples:")
        for p, l in zip(files, labels):
            if l is None:
                print(f"    {p.relative_to(BASE)}")
                break
        print("  directories seen:")
        for d in sorted({str(p.parent.relative_to(BASE)) for p in files})[:12]:
            print(f"    {d}")
    keep = [(p, l) for p, l in zip(files, labels) if l]
    print(f"usable: {len(keep)}  "
          f"({sum(l == 'fake' for _, l in keep)} manipulated)")
    return keep


def metadata_table(items):
    rows = []
    for p, lab in items:
        try:
            with Image.open(p) as im:
                w, h = im.size
                fmt, mode = im.format or "", im.mode
            rows.append({"w": w, "h": h, "area": w * h,
                         "aspect": w / max(h, 1),
                         "size_bytes": p.stat().st_size,
                         "is_png": int(fmt == "PNG"),
                         "is_rgb": int(mode == "RGB"),
                         "fake": int(lab == "fake")})
        except Exception:
            continue
    return pd.DataFrame(rows)


items = survey()
df = metadata_table(items)
print(f"\nmetadata table: {len(df)} rows, "
      f"{int(df.fake.sum())} manipulated\n")

for lab, name in ((0, "authentic"), (1, "manipulated")):
    d = df[df.fake == lab]
    top = collections.Counter(zip(d.w, d.h)).most_common(3)
    share = top[0][1] / len(d) * 100 if len(d) else 0
    print(f"  {name:<12} top sizes: {top}")
    print(f"  {'':<12} most common covers {share:.1f}% of the class")

X = df.drop(columns=["fake"])
y = df["fake"].to_numpy()
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
print(f"\nMETADATA-ONLY classifier (no pixels): "
      f"acc {acc.mean() * 100:.2f}% (+-{acc.std() * 100:.2f}), "
      f"AUROC {auc.mean():.4f}")
print(f"  BanFakeMM reference: 100.00% / AUROC 1.0000")

print("\nIMAGE REUSE: are manipulated images byte-identical to authentic ones?")
by_label = {"real": set(), "fake": set()}
for p, lab in items:
    try:
        by_label[lab].add(hashlib.md5(p.read_bytes()).hexdigest())
    except OSError:
        continue
shared = by_label["real"] & by_label["fake"]
print(f"  distinct authentic hashes   : {len(by_label['real'])}")
print(f"  distinct manipulated hashes : {len(by_label['fake'])}")
print(f"  shared between the classes  : {len(shared)}")
print(f"  BanFakeMM reference: 0 in the image-attack splits, "
      f"1000/1000 in the text-attack splits")

pd.DataFrame([{
    "benchmark": "MultiBanFakeDetect",
    "n_images": len(df),
    "n_manipulated": int(df.fake.sum()),
    "metadata_only_acc": round(float(acc.mean()) * 100, 2),
    "metadata_only_acc_std": round(float(acc.std()) * 100, 2),
    "metadata_only_auroc": round(float(auc.mean()), 4),
    "shared_hashes_across_labels": len(shared),
}]).to_csv(ROOT / "second_benchmark.csv", index=False)
print(f"\nwrote {ROOT / 'second_benchmark.csv'}")
