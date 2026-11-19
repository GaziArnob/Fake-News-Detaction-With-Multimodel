"""Second-benchmark follow-up: the metadata-only result with duplicates grouped, plus
the per-folder (Train/Validation/Test) picture. Output: second_benchmark_grouped.csv"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = Path(r"E:\Project\Fake News Detection Data\_multibanfakedetect\extracted")
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".jfif"}

rows = []
for p in sorted(BASE.rglob("*")):
    if not (p.is_file() and p.suffix.lower() in EXT):
        continue
    n = p.name.lower()
    lab = 1 if "fake" in n else 0 if "real" in n else None
    if lab is None:
        continue
    with Image.open(p) as im:
        w, h = im.size
        rows.append({"w": w, "h": h, "area": w * h, "aspect": w / max(h, 1),
                     "size_bytes": p.stat().st_size, "is_png": int(im.format == "PNG"),
                     "is_rgb": int(im.mode == "RGB"), "fake": lab,
                     "split": p.parent.name,
                     "md5": hashlib.md5(p.read_bytes()).hexdigest()})
df = pd.DataFrame(rows)
print(f"{len(df)} images, {int(df.fake.sum())} fake; label read from file name")
print(df.groupby("split").fake.agg(["size", "mean"]).round(3))

feat = ["w", "h", "area", "aspect", "size_bytes", "is_png", "is_rgb"]
X, y = df[feat], df.fake.to_numpy()
# duplicates: identical bytes get one group; drop images whose hash occurs in both classes
both = df.groupby("md5").fake.nunique()
conflict = set(both[both > 1].index)
clean = df[~df.md5.isin(conflict)].drop_duplicates("md5")
print(f"\nhashes in both classes: {len(conflict)}; after dropping them and de-duplicating: {len(clean)} images")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
res = []
for name, d in (("all images, ungrouped KFold", df), ("de-duplicated, conflicts removed", clean)):
    g = d.md5.to_numpy()
    cv = GroupKFold(n_splits=5)
    a = cross_val_score(clf, d[feat], d.fake, cv=cv, groups=g, scoring="accuracy")
    u = cross_val_score(clf, d[feat], d.fake, cv=cv, groups=g, scoring="roc_auc")
    print(f"  {name:<36} acc {a.mean()*100:.2f} (+-{a.std()*100:.2f})  AUROC {u.mean():.4f}")
    res.append({"setting": name, "n": len(d), "acc": round(a.mean() * 100, 2),
                "acc_std": round(a.std() * 100, 2), "auroc": round(u.mean(), 4)})
# official split: train on Train, test on Test
tr, te = df[df.split == "Train"], df[df.split == "Test"]
clf.fit(tr[feat], tr.fake)
pr = clf.predict_proba(te[feat])[:, 1]
from sklearn.metrics import roc_auc_score
acc = ((pr >= .5) == te.fake).mean() * 100
print(f"\nTrain->Test (official folders): acc {acc:.2f}  AUROC {roc_auc_score(te.fake, pr):.4f}  (n_test={len(te)})")
res.append({"setting": "Train->Test official", "n": len(te), "acc": round(acc, 2),
            "acc_std": "", "auroc": round(roc_auc_score(te.fake, pr), 4)})
pd.DataFrame(res).to_csv(Path(__file__).resolve().parents[1] / "second_benchmark_grouped.csv", index=False)
