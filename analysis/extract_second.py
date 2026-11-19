"""Extract the MultiBanFakeDetect archive and report its layout.

The archive is third-party data, so paths are checked before extraction and
the contents are surveyed afterwards -- the label rule in second_benchmark.py
depends on how the directories are named, and that has to be seen rather than
assumed.
"""
import collections
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = Path(r"E:\Project\Fake News Detection Data\_multibanfakedetect")
SRC = BASE / "MultiBanFakeDetect.zip"

if not SRC.exists():
    raise SystemExit(f"archive not found: {SRC}")
print(f"archive: {SRC.stat().st_size / 1024**3:.2f} GB")

with zipfile.ZipFile(SRC) as z:
    names = z.namelist()
    print(f"entries: {len(names)}")

    unsafe = [n for n in names
              if n.startswith("/") or Path(n).is_absolute()
              or ".." in n.replace("\\", "/").split("/")]
    if unsafe:
        print(f"REFUSING to extract -- {len(unsafe)} unsafe paths:")
        for n in unsafe[:5]:
            print(f"  {n}")
        raise SystemExit(1)
    print("path check: clean")

    print("\ntop-level entries:")
    for t in sorted({n.split("/")[0] for n in names})[:12]:
        print(f"  {t}")

    ext = collections.Counter(Path(n).suffix.lower() for n in names if not n.endswith("/"))
    print(f"\nfile types: {dict(ext.most_common(8))}")

    z.extractall(BASE)

print("\nextracted. Directories holding files:")
dirs = collections.Counter()
for p in BASE.rglob("*"):
    if p.is_file() and p.suffix.lower() != ".zip":
        dirs[str(p.parent.relative_to(BASE))] += 1
for d, n in sorted(dirs.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {n:>6}  {d}")

csvs = [p for p in BASE.rglob("*.csv")] + [p for p in BASE.rglob("*.xlsx")]
print(f"\ntabular files: {len(csvs)}")
for p in csvs[:8]:
    print(f"  {p.relative_to(BASE)}")
    if p.suffix.lower() == ".csv":
        try:
            import pandas as pd
            d = pd.read_csv(p, nrows=3)
            print(f"      columns: {list(d.columns)[:12]}")
        except Exception as exc:
            print(f"      (unreadable: {exc})")
