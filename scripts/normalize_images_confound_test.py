"""Strip the resolution/format/EXIF confound found by check_dataset_artifacts.py.

Real and fake images turned out to have completely different resolution,
format, color mode, and EXIF distributions (metadata alone gave 100%
accuracy with zero image content). This script re-saves every image
through one identical pipeline -- fixed square resolution, RGB, JPEG at a
fixed quality, no EXIF -- so any later model can no longer use that
metadata fingerprint. Content is preserved; only the confound is removed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

SOURCE_ROOT = Path("images")
OUTPUT_ROOT = Path("images_normalized")
TARGET_SIZE = (384, 384)
JPEG_QUALITY = 90
MANIFEST_CSV = Path("images_normalized_manifest.csv")


def normalize_one(source_path: Path, dest_path: Path) -> dict:
    try:
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Save fresh (no exif kwarg passed) so no EXIF block is carried over.
            img.save(dest_path, format="JPEG", quality=JPEG_QUALITY)
        return {"source": str(source_path), "normalized": str(dest_path), "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {"source": str(source_path), "normalized": "", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    features_df = pd.read_csv("extracted_features.csv")
    rows = []
    for _, record in features_df.iterrows():
        source_path = Path(record["image_path"])
        label = record["label"]
        dest_path = OUTPUT_ROOT / label / source_path.with_suffix(".jpg").name
        result = normalize_one(source_path, dest_path)
        result["label"] = label
        rows.append(result)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_CSV, index=False)
    ok = manifest["error"].eq("").sum()
    print(f"Normalized {ok}/{len(manifest)} images -> {OUTPUT_ROOT}")
    if ok != len(manifest):
        print("Failures:")
        print(manifest[manifest["error"].ne("")][["source", "error"]].to_string(index=False))
    print(f"Manifest saved -> {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
