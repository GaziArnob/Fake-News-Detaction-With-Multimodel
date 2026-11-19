"""How do the two image sets map to article_id?

AdversarialGen CSVs carry image_path like 'real/1090.jpg' plus article_id, so
that side is a direct filename lookup. The TRAINING images live in the
project's images/ folder under bare numeric names and need checking: if the
stem is the article id, mapping is trivial; if not, it has to go through a
content hash against master_enriched.
"""
import csv
import hashlib
import io
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = Path(r"E:\Project\Fake News Detection Data")
PROJ = Path(r"E:\Project\Fake News Detection")

# --- AdversarialGen side ----------------------------------------------------
gen_rows = []
for part in ("train", "val", "test"):
    gen_rows += list(csv.DictReader(
        io.open(DATA / "AdversarialGen" / "AdversarialGen" / f"{part}.csv",
                encoding="utf-8")))
gen_by_name = {Path(r["image_path"]).name: r["article_id"] for r in gen_rows}
print(f"AdversarialGen rows {len(gen_rows)}, unique filenames {len(gen_by_name)}")
sample = list(gen_by_name.items())[:3]
print(f"  e.g. {sample}")

# does every actual file have a row?
gen_imgs = [p for sub in ("real", "fake")
            for p in (DATA / "AdversarialGen" / "AdversarialGen" / "images"
                      / sub).iterdir() if p.is_file()]
hit = sum(1 for p in gen_imgs if p.name in gen_by_name)
print(f"  files {len(gen_imgs)}, with an article_id: {hit}")

# --- master_enriched, for the training side ---------------------------------
master = list(csv.DictReader(
    io.open(DATA / "master_enriched" / "master_enriched" / "master_enriched.csv",
            encoding="utf-8")))
m_by_name = {Path(r["image_path"]).name: r["article_id"] for r in master}
print(f"\nmaster_enriched rows {len(master)}, unique filenames {len(m_by_name)}")
print(f"  e.g. {list(m_by_name.items())[:3]}")

# --- do project training filenames map by stem? ------------------------------
print("\nTRAINING images (project images/):")
for sub in ("real", "fake"):
    files = [p for p in (PROJ / "images" / sub).iterdir() if p.is_file()][:400]
    direct = sum(1 for p in files if p.name in m_by_name)
    prefixed = sum(1 for p in files
                   if f"{sub}_{p.stem}.jpg" in m_by_name
                   or f"{sub}_{p.stem}.png" in m_by_name)
    print(f"  {sub:<5} sampled {len(files)}: name match {direct}, "
          f"'{sub}_<stem>' match {prefixed}")

# --- content-hash spot check -------------------------------------------------
print("\ncontent-hash spot check (10 training images vs master_enriched):")
m_dir = DATA / "master_enriched" / "master_enriched" / "images"
m_hash = {}
for sub in ("real", "fake"):
    for p in (m_dir / sub).iterdir():
        if p.is_file():
            m_hash.setdefault(hashlib.md5(p.read_bytes()).hexdigest(),
                              []).append(p.name)
print(f"  hashed {len(m_hash)} distinct master images")
for sub in ("real", "fake"):
    files = [p for p in (PROJ / "images" / sub).iterdir() if p.is_file()][:5]
    for p in files:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        names = m_hash.get(h, [])
        aid = [m_by_name.get(n) for n in names]
        print(f"  {sub}/{p.name:<14} -> master {names or 'NO MATCH'}  "
              f"article_id {aid}")
