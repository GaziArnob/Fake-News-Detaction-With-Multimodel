"""Per attack split: what fraction of the FAKE class is an unmodified
authentic image, and therefore undetectable from pixels alone?

Reference pool is the benchmark's full authentic corpus (master_enriched real,
7,040 images). A manipulated row whose image reproduces an authentic image did
not have its image manipulated: the attack lives in the text. Image-only
fake-recall is capped at 1 minus that fraction, by construction.

Produces Table II (ceiling column) of the journal manuscript.
Output: split_ceilings.csv
"""
import csv
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

from common import AUTH_POOL, CACHE, ROOT, SPLIT_IMAGES, hashes, min_hamming

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print(f"authentic reference pool: {AUTH_POOL}")
auth_files = sorted(p for p in AUTH_POOL.iterdir() if p.is_file())
auth_paths, auth_ph, _ = hashes(auth_files, "authpool")
print(f"  {len(auth_paths)} authentic images")
auth_bytes = {hashlib.md5(Path(p).read_bytes()).hexdigest() for p in auth_paths}

rows = []
for name, base in SPLIT_IMAGES.items():
    fake_dir = base / "fake"
    if not fake_dir.exists():
        print(f"\n{name}: no fake/ folder, skipped")
        continue
    label = name + (" (train)" if name == "AdversarialEdit" else "")
    print(f"\n{'=' * 70}\n{label}")
    files = sorted(p for p in fake_dir.iterdir() if p.is_file())
    paths, ph, _ = hashes(files, f"fake_{name}")
    n = len(paths)

    exact = sum(1 for p in paths
                if hashlib.md5(Path(p).read_bytes()).hexdigest() in auth_bytes)
    d = min_hamming(ph, auth_ph)
    near2, near6 = int((d <= 2).sum()), int((d <= 6).sum())

    print(f"  {n} manipulated images")
    print(f"  exact byte match to an authentic image : {exact:>4} "
          f"({exact / n * 100:.1f}%)")
    print(f"  pHash <= 2                             : {near2:>4} "
          f"({near2 / n * 100:.1f}%)")
    print(f"  pHash <= 6                             : {near6:>4} "
          f"({near6 / n * 100:.1f}%)")
    print(f"  => image-only fake-recall CEILING      : "
          f"{(1 - near2 / n) * 100:.1f}%")
    rows.append({"split": label, "n_fake": n, "exact": exact,
                 "phash_le2": near2, "phash_le6": near6,
                 "ceiling_pct": round((1 - near2 / n) * 100, 1)})

out = ROOT / "split_ceilings.csv"
with io.open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out}")
