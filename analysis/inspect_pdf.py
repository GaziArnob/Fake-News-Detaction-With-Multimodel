"""Page count and sanity check on a compiled PDF."""
import re
import sys
from pathlib import Path

import pypdf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PDF = Path(sys.argv[1] if len(sys.argv) > 1
           else r"E:\Project\Fake News Detection\paper\journal_banfakemm.pdf")

r = pypdf.PdfReader(str(PDF))
print(f"{PDF.name}: {len(r.pages)} pages, {PDF.stat().st_size / 1024:.0f} KB")
print(f"metadata: {dict(r.metadata or {})}")

text = "\n".join((p.extract_text() or "") for p in r.pages)
print(f"\nextracted characters: {len(text)}")

# Did anything fail to resolve?
for pat, what in ((r"\?\?", "unresolved reference (??)"),
                  (r"\[\?\]", "unresolved citation ([?])")):
    n = len(re.findall(pat, text))
    print(f"  {what:<32} {n}")

print("\nsection headings found:")
for m in re.finditer(r"\n([IVX]+\.\s+[A-Z][A-Za-z \-]+)\n", text):
    print(f"  {m.group(1).strip()}")

print("\ntables and figures referenced in the text:")
for kind in ("TABLE", "Fig."):
    hits = sorted(set(re.findall(kind + r"\s*([IVX0-9]+)", text)))
    print(f"  {kind:<6} {hits}")
