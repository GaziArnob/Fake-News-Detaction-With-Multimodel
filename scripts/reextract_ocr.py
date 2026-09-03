"""Re-run OCR (bn+en, preprocessed, confidence-filtered) over extracted_features.csv.

The original extraction used an English-only EasyOCR reader on this
Bangladeshi image set, which forced Bengali glyphs into Latin/digit
lookalikes. This script re-runs only the OCR + similarity step (BLIP
captions are reused as-is, since captioning was not the broken part) and
writes the result back into extracted_features.csv, checkpointing so it can
be safely interrupted and resumed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import easyocr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from news_guard.config import get_settings
from news_guard.features import SBERT_MODEL_ID, extract_ocr_text, OCR_LANGUAGES

FEATURES_CSV = Path("extracted_features.csv")
CHECKPOINT_EVERY = 20
MARKER_COLUMN = "ocr_v2"


def sbert_similarity(embedder: SentenceTransformer, ocr_text: str, caption: str) -> float:
    if not ocr_text or not caption:
        return float("nan")
    embeddings = embedder.encode(
        [ocr_text, caption],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return float(np.dot(embeddings[0], embeddings[1]))


def save_checkpoint(dataframe: pd.DataFrame) -> None:
    temp_csv = FEATURES_CSV.with_name("extracted_features.tmp.csv")
    dataframe.to_csv(temp_csv, index=False)
    while True:
        try:
            os.replace(temp_csv, FEATURES_CSV)
            return
        except PermissionError:
            print("Close extracted_features.csv in Excel. Retrying in 5 seconds...")
            time.sleep(5)


def main() -> None:
    settings = get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = pd.read_csv(FEATURES_CSV)
    if MARKER_COLUMN not in df.columns:
        df[MARKER_COLUMN] = False
    df[MARKER_COLUMN] = df[MARKER_COLUMN].fillna(False).astype(bool)
    df["ocr_text"] = df["ocr_text"].fillna("").astype(str)
    df["error"] = df["error"].fillna("").astype(str)
    df["ocr_length"] = df["ocr_length"].astype(float)
    df["similarity_score"] = df["similarity_score"].astype(float)

    pending = df.loc[~df[MARKER_COLUMN]]
    print(f"Already re-extracted: {(df[MARKER_COLUMN]).sum():,}")
    print(f"Remaining: {len(pending):,}")
    if pending.empty:
        print("Nothing to do.")
        return

    print("Loading EasyOCR reader (bn+en) ...")
    reader = easyocr.Reader(
        OCR_LANGUAGES,
        model_storage_directory=str(settings.project_root / "models" / "easyocr"),
        verbose=False,
    )
    print("Loading SBERT embedder ...")
    embedder = SentenceTransformer(SBERT_MODEL_ID, device=device)

    processed_since_checkpoint = 0
    start = time.time()
    total = len(pending)
    for count, (index, row) in enumerate(pending.iterrows(), start=1):
        image_path = row["image_path"]
        caption = str(row.get("caption", "") or "")
        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
            ocr_text = extract_ocr_text(image, reader)
            similarity_score = sbert_similarity(embedder, ocr_text, caption)
            error = ""
        except Exception as exc:  # noqa: BLE001 - mirrors notebook's per-image guard
            ocr_text = ""
            similarity_score = float("nan")
            error = f"{type(exc).__name__}: {exc}"

        df.at[index, "ocr_text"] = ocr_text
        df.at[index, "ocr_length"] = float(len(ocr_text))
        df.at[index, "similarity_score"] = similarity_score
        df.at[index, "error"] = error
        df.at[index, MARKER_COLUMN] = True
        processed_since_checkpoint += 1

        if processed_since_checkpoint >= CHECKPOINT_EVERY or count == total:
            save_checkpoint(df)
            processed_since_checkpoint = 0
            elapsed = time.time() - start
            rate = count / elapsed if elapsed > 0 else 0.0
            remaining = (total - count) / rate if rate > 0 else float("inf")
            print(
                f"Processed {count:,}/{total:,} "
                f"({elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min remaining)"
            )

    print("Done.")


if __name__ == "__main__":
    main()
