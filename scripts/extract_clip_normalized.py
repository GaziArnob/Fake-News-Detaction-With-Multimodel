"""Extract CLIP visual embeddings from the confound-stripped normalized images.

Same CLIP model/config as news_guard.features, but run over
images_normalized/ (uniform 384x384 RGB, no EXIF, single JPEG quality) so
the resulting embeddings cannot carry the resolution/format/EXIF
fingerprint that check_dataset_artifacts.py found separates real/fake with
100% accuracy on metadata alone.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from news_guard.features import CLIP_MODEL_ID

MANIFEST_CSV = Path("images_normalized_manifest.csv")
OUTPUT_CSV = Path("clip_visual_features_normalized.csv")
CHECKPOINT_EVERY = 100


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = pd.read_csv(MANIFEST_CSV)
    manifest = manifest[manifest["error"].fillna("").eq("")].reset_index(drop=True)
    print(f"Images to process: {len(manifest)}")

    print("Loading CLIP ...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    model.eval()

    rows = []
    start = time.time()
    for count, record in enumerate(manifest.to_dict("records"), start=1):
        path = record["normalized"]
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model.get_image_features(**inputs)
                output = getattr(output, "pooler_output", output)
                vector = torch.nn.functional.normalize(output, dim=1).cpu().numpy()[0]
            row = {"image_path": record["source"], "label": record["label"], "clip_error": ""}
            row.update({f"clip_{i:03d}": float(v) for i, v in enumerate(vector)})
        except Exception as exc:  # noqa: BLE001
            row = {"image_path": record["source"], "label": record["label"], "clip_error": f"{type(exc).__name__}: {exc}"}
            row.update({f"clip_{i:03d}": 0.0 for i in range(512)})
        rows.append(row)

        if count % CHECKPOINT_EVERY == 0 or count == len(manifest):
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
            elapsed = time.time() - start
            rate = count / elapsed
            remaining = (len(manifest) - count) / rate if rate > 0 else float("inf")
            print(f"Processed {count}/{len(manifest)} ({elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min remaining)")

    print("Done.")


if __name__ == "__main__":
    main()
