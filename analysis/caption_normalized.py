"""BLIP captions for the RE-ENCODED images (images_normalized/), same model and
generation settings as news_guard.features (blip-image-captioning-base,
max_new_tokens=40), so the caption pathway can be compared with CLIP-only under the
artefact control. Resumable: appends to the output CSV and skips finished rows.

Output: captions_normalized.csv  (image_path = the ORIGINAL image path, as in
extracted_features.csv, plus caption)
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\Project\Fake News Detection")
OUT = ROOT / "captions_normalized.csv"
MODEL = "Salesforce/blip-image-captioning-base"

man = pd.read_csv(ROOT / "images_normalized_manifest.csv")
man = man[man["error"].isna()].reset_index(drop=True)
done = set(pd.read_csv(OUT)["image_path"]) if OUT.exists() else set()
todo = man[~man["source"].isin(done)]
print(f"{len(man)} images, {len(done)} done, {len(todo)} to do", flush=True)

proc = BlipProcessor.from_pretrained(MODEL)
model = BlipForConditionalGeneration.from_pretrained(MODEL).eval()
if not OUT.exists():
    OUT.write_text("image_path,caption\n", encoding="utf-8")

t0 = time.time()
buf = []
for i, r in enumerate(todo.itertuples(), 1):
    img = Image.open(ROOT / r.normalized).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=40)
    cap = proc.decode(ids[0], skip_special_tokens=True).strip()
    buf.append({"image_path": r.source, "caption": cap})
    if len(buf) >= 50 or i == len(todo):
        pd.DataFrame(buf).to_csv(OUT, mode="a", header=False, index=False,
                                 encoding="utf-8")
        buf = []
        el = time.time() - t0
        print(f"{i}/{len(todo)}  {el / 60:.1f} min  eta {(el / i) * (len(todo) - i) / 60:.0f} min",
              flush=True)
print("done", flush=True)
