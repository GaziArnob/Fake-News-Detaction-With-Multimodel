"""Shortcut-learning / dataset-artifact check.

The ablation study showed CLIP visual features alone reach ~95.6% accuracy,
matching the full pipeline. That is a red flag: it could mean the model is
reading genuine visual content, or it could mean real/fake images were
sourced differently and the model is exploiting a trivial metadata
signature (resolution, compression, screenshot chrome) instead.

This script extracts ONLY non-content metadata (file size, dimensions,
format, color mode, EXIF presence/software tag) for every image and trains
the same repeated-k-fold XGBoost protocol on that alone. If this "sees
nothing" baseline scores well above chance, the dataset has a leak the
paper must address before trusting the 95%+ headline number.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

FEATURES_CSV = Path("dataset_artifact_features.csv")
FOLD_CSV = Path("dataset_artifact_fold_metrics.csv")
N_SPLITS = 5
N_REPEATS = 2
SEED = 42

NUMERIC_COLUMNS = ["file_size_bytes", "width", "height", "aspect_ratio", "megapixels"]
CATEGORICAL_COLUMNS = ["format", "mode", "has_exif", "exif_software"]


def extract_metadata(image_path: str) -> dict:
    path = Path(image_path)
    try:
        file_size = path.stat().st_size
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format or "unknown"
            mode = img.mode
            exif = img.getexif()
            has_exif = bool(exif) and len(exif) > 0
            software = str(exif.get(305, "")) if has_exif else ""  # 305 = Software tag
        return {
            "image_path": image_path,
            "file_size_bytes": file_size,
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height else 0.0,
            "megapixels": (width * height) / 1_000_000,
            "format": fmt,
            "mode": mode,
            "has_exif": has_exif,
            "exif_software": software if software else "none",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "image_path": image_path, "file_size_bytes": 0, "width": 0, "height": 0,
            "aspect_ratio": 0.0, "megapixels": 0.0, "format": "error", "mode": "error",
            "has_exif": False, "exif_software": "error", "error": f"{type(exc).__name__}: {exc}",
        }


def build_pipeline() -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    classifier = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", max_depth=4,
        n_estimators=300, learning_rate=0.08, random_state=SEED, n_jobs=4, tree_method="hist",
    )
    return Pipeline([("features", features), ("xgb", classifier)])


def main() -> None:
    features_df = pd.read_csv("extracted_features.csv")
    print(f"Extracting metadata for {len(features_df)} images ...")

    rows = [extract_metadata(p) for p in features_df["image_path"]]
    meta_df = pd.DataFrame(rows).merge(features_df[["image_path", "label"]], on="image_path")
    meta_df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved -> {FEATURES_CSV}")

    print("\n--- Real vs Fake distribution (numeric metadata) ---")
    real = meta_df[meta_df["label"] == "real"]
    fake = meta_df[meta_df["label"] == "fake"]
    for col in NUMERIC_COLUMNS:
        ks_stat, p_value = stats.ks_2samp(real[col], fake[col])
        flag = " <-- SIGNIFICANT DIFFERENCE" if p_value < 0.01 else ""
        print(
            f"{col:18s} real_mean={real[col].mean():12.2f}  fake_mean={fake[col].mean():12.2f}  "
            f"KS p-value={p_value:.4g}{flag}"
        )

    print("\n--- Real vs Fake distribution (categorical metadata) ---")
    for col in CATEGORICAL_COLUMNS:
        print(f"\n{col}:")
        print(pd.crosstab(meta_df[col], meta_df["label"], normalize="columns").round(3))

    print("\n--- Training XGBoost on metadata-ONLY features (no image content) ---")
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    labels = meta_df["label"].to_numpy()
    results = []
    start = time.time()
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(meta_df, labels), start=1):
        train_frame = meta_df.iloc[train_idx]
        test_frame = meta_df.iloc[test_idx]
        y_train = train_frame["label"].eq("fake").astype(int)
        y_test = test_frame["label"].eq("fake").astype(int)
        model = build_pipeline()
        model.fit(train_frame, y_train)
        predictions = model.predict(test_frame)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        results.append({"fold": fold_id, "accuracy": acc, "f1_fake": f1})
        print(f"Fold {fold_id:2d}/{N_SPLITS * N_REPEATS}: accuracy={acc:.4f} f1={f1:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(FOLD_CSV, index=False)
    print(f"\nTotal time: {time.time() - start:.1f}s")
    print(f"\nMetadata-only accuracy: {results_df['accuracy'].mean():.4f} +/- {results_df['accuracy'].std():.4f}")
    print(f"(chance level = 0.50; full pipeline = 0.956)")
    print(f"\nSaved per-fold metrics -> {FOLD_CSV}")


if __name__ == "__main__":
    main()
