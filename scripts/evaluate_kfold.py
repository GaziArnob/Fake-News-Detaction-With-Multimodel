"""Repeated stratified k-fold evaluation of the tuned multimodal XGBoost model.

A single 80/20 split (as used for the historical 94.75% figure) has real
variance on a 400-image test set: two extra mistakes swing accuracy by
0.5pp. This script fits and evaluates the exact same pipeline
(news_guard.training.build_model_pipeline) across many stratified folds and
reports mean +/- std, so the reported accuracy reflects the model rather
than one lucky/unlucky split.

CISF fusion is fit on each fold's train rows only and transformed onto that
fold's test rows (matching real inference-time behaviour), so there is no
leakage between folds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

from news_guard.config import get_settings
from news_guard.features import SBERT_MODEL_ID, SemanticFusionReference
from news_guard.training import BASE_NUMERIC_COLUMNS, build_model_pipeline, load_base_training_data

FOLD_METRICS_CSV = Path("cross_validation_multimodal_fold_metrics.csv")
SUMMARY_CSV = Path("cross_validation_multimodal_summary.csv")


def run_fold(
    fold_id: int,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    clip_columns: list[str],
    embedder: SentenceTransformer,
) -> dict:
    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    train_semantic = embedder.encode(
        train_frame["semantic_text"].tolist(), normalize_embeddings=True, convert_to_numpy=True
    )
    test_semantic = embedder.encode(
        test_frame["semantic_text"].tolist(), normalize_embeddings=True, convert_to_numpy=True
    )
    fusion, train_components, train_mean, train_max = SemanticFusionReference.fit(train_semantic)
    test_components, test_mean, test_max = fusion.transform(test_semantic)

    train_frame["cisf_neighbour_mean_similarity"] = train_mean
    train_frame["cisf_neighbour_max_similarity"] = train_max
    test_frame["cisf_neighbour_mean_similarity"] = test_mean
    test_frame["cisf_neighbour_max_similarity"] = test_max
    for index, column in enumerate(fusion.component_columns):
        train_frame[column] = train_components[:, index]
        test_frame[column] = test_components[:, index]

    numeric_columns = BASE_NUMERIC_COLUMNS + fusion.component_columns
    for column in numeric_columns + clip_columns:
        train_frame[column] = pd.to_numeric(train_frame[column], errors="coerce").fillna(0.0)
        test_frame[column] = pd.to_numeric(test_frame[column], errors="coerce").fillna(0.0)

    model = build_model_pipeline(numeric_columns, clip_columns)
    y_train = train_frame["label"].eq("fake").astype(int)
    y_test = test_frame["label"].eq("fake").astype(int)

    model.fit(train_frame, y_train)
    predictions = model.predict(test_frame)

    return {
        "fold": fold_id,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_fake": f1_score(y_test, predictions),
        "precision_fake": precision_score(y_test, predictions, zero_division=0),
        "recall_fake": recall_score(y_test, predictions, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=5, help="Folds per repeat (default: 5)")
    parser.add_argument("--repeats", type=int, default=2, help="Number of repeats (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    frame, clip_columns = load_base_training_data(settings)
    print(f"Total usable rows: {len(frame)}")
    print(f"Plan: {args.splits}-fold x {args.repeats} repeats = {args.splits * args.repeats} fold-fits")

    embedder = SentenceTransformer(SBERT_MODEL_ID)
    splitter = RepeatedStratifiedKFold(
        n_splits=args.splits, n_repeats=args.repeats, random_state=args.seed
    )

    results = []
    start = time.time()
    labels = frame["label"].to_numpy()
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(frame, labels), start=1):
        fold_start = time.time()
        metrics = run_fold(
            fold_id,
            frame.iloc[train_idx].copy(),
            frame.iloc[test_idx].copy(),
            clip_columns,
            embedder,
        )
        results.append(metrics)
        print(
            f"Fold {fold_id:2d}/{args.splits * args.repeats}: "
            f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1_fake']:.4f} "
            f"({time.time() - fold_start:.1f}s)"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(FOLD_METRICS_CSV, index=False)

    summary_rows = []
    for metric in ["accuracy", "f1_fake", "precision_fake", "recall_fake"]:
        summary_rows.append({
            "metric": metric,
            "mean": results_df[metric].mean(),
            "std": results_df[metric].std(),
            "min": results_df[metric].min(),
            "max": results_df[metric].max(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print()
    print(f"Total time: {(time.time() - start) / 60:.1f} min")
    print()
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"Saved per-fold metrics -> {FOLD_METRICS_CSV}")
    print(f"Saved summary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
