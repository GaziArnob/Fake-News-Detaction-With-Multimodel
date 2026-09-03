"""Repeated k-fold CV on CLIP features extracted from confound-stripped images.

Compares directly against the original clip_only ablation result (95.57% +/-
0.92%, computed on images with wildly different resolution/format/EXIF
between classes). If accuracy here collapses toward chance, the original
number was driven by the metadata confound, not real visual content.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

N_SPLITS = 5
N_REPEATS = 2
SEED = 42
FOLD_CSV = Path("clip_normalized_fold_metrics.csv")


def build_pipeline(clip_columns: list[str]) -> Pipeline:
    features = ColumnTransformer(
        transformers=[("clip_pca", PCA(n_components=64, random_state=SEED), clip_columns)]
    )
    classifier = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        colsample_bytree=0.6834834444556088, gamma=0.14286681792194078,
        learning_rate=0.07423304665511389, max_depth=3, min_child_weight=2,
        n_estimators=443, reg_alpha=1.007198483880919e-05, reg_lambda=4.9111315080437645,
        subsample=0.8661185283697008, random_state=SEED, n_jobs=4, tree_method="hist",
    )
    return Pipeline([("features", features), ("xgb", classifier)])


def main() -> None:
    df = pd.read_csv("clip_visual_features_normalized.csv")
    df = df[df["clip_error"].fillna("").eq("")].reset_index(drop=True)
    clip_columns = [c for c in df.columns if c.startswith("clip_") and c[5:].isdigit()]
    assert len(clip_columns) == 512
    print(f"Rows: {len(df)}, CLIP columns: {len(clip_columns)}")

    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    labels = df["label"].to_numpy()
    results = []
    start = time.time()
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(df, labels), start=1):
        train_frame = df.iloc[train_idx]
        test_frame = df.iloc[test_idx]
        y_train = train_frame["label"].eq("fake").astype(int)
        y_test = test_frame["label"].eq("fake").astype(int)
        model = build_pipeline(clip_columns)
        model.fit(train_frame, y_train)
        predictions = model.predict(test_frame)
        metrics = {
            "fold": fold_id,
            "accuracy": accuracy_score(y_test, predictions),
            "f1_fake": f1_score(y_test, predictions),
            "precision_fake": precision_score(y_test, predictions, zero_division=0),
            "recall_fake": recall_score(y_test, predictions, zero_division=0),
        }
        results.append(metrics)
        print(f"Fold {fold_id:2d}/{N_SPLITS*N_REPEATS}: accuracy={metrics['accuracy']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(FOLD_CSV, index=False)
    print(f"\nTotal time: {time.time()-start:.1f}s")
    print(f"\nNormalized CLIP-only accuracy: {results_df['accuracy'].mean():.4f} +/- {results_df['accuracy'].std():.4f}")
    print("(original clip_only on un-normalized images was 0.9557 +/- 0.0092)")
    print(f"Saved -> {FOLD_CSV}")


if __name__ == "__main__":
    main()
