"""Full pipeline (TF-IDF text + CISF + CLIP) re-tested with confound-stripped
CLIP features, to get the paper's confound-controlled headline number.

Text/OCR/caption/CISF are unaffected by image resolution, so they are
loaded as-is from extracted_features.csv; only the CLIP columns come from
clip_visual_features_normalized.csv (384x384 RGB, no EXIF, fixed JPEG
quality for both classes).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from news_guard.features import SBERT_MODEL_ID, SemanticFusionReference, prepare_text_columns
from news_guard.training import BASE_NUMERIC_COLUMNS, TEXT_MAX_FEATURES, TEXT_MIN_DOCUMENT_FREQUENCY

N_SPLITS = 5
N_REPEATS = 2
SEED = 42
FOLD_CSV = Path("full_normalized_fold_metrics.csv")


def load_data() -> tuple[pd.DataFrame, list[str]]:
    text_df = pd.read_csv("extracted_features.csv")
    clip_df = pd.read_csv("clip_visual_features_normalized.csv")
    clip_columns = [c for c in clip_df.columns if c.startswith("clip_") and c[5:].isdigit()]
    assert len(clip_columns) == 512

    base = text_df.loc[text_df["error"].fillna("").eq("") & text_df["label"].isin(["real", "fake"])].copy()
    merged = base.merge(clip_df[["image_path", "clip_error"] + clip_columns], on="image_path", how="inner")
    merged = merged.loc[merged["clip_error"].fillna("").eq("")].copy().reset_index(drop=True)
    return prepare_text_columns(merged), clip_columns


def build_pipeline(numeric_columns: list[str], clip_columns: list[str]) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(
                tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2),
                min_df=TEXT_MIN_DOCUMENT_FREQUENCY, max_features=TEXT_MAX_FEATURES,
                sublinear_tf=True, dtype=np.float32,
            ), "processed_text"),
            ("tabular", "passthrough", numeric_columns),
            ("clip_pca", PCA(n_components=64, random_state=SEED), clip_columns),
        ],
        sparse_threshold=0.30,
    )
    classifier = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        colsample_bytree=0.6834834444556088, gamma=0.14286681792194078,
        learning_rate=0.07423304665511389, max_depth=3, min_child_weight=2,
        n_estimators=443, reg_alpha=1.007198483880919e-05, reg_lambda=4.9111315080437645,
        subsample=0.8661185283697008, random_state=SEED, n_jobs=4, tree_method="hist",
    )
    return Pipeline([("features", features), ("xgb", classifier)])


def run_fold(fold_id, train_frame, test_frame, clip_columns, embedder) -> dict:
    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    train_semantic = embedder.encode(train_frame["semantic_text"].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    test_semantic = embedder.encode(test_frame["semantic_text"].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    fusion, train_components, train_mean, train_max = SemanticFusionReference.fit(train_semantic)
    test_components, test_mean, test_max = fusion.transform(test_semantic)

    train_frame["cisf_neighbour_mean_similarity"] = train_mean
    train_frame["cisf_neighbour_max_similarity"] = train_max
    test_frame["cisf_neighbour_mean_similarity"] = test_mean
    test_frame["cisf_neighbour_max_similarity"] = test_max
    for i, col in enumerate(fusion.component_columns):
        train_frame[col] = train_components[:, i]
        test_frame[col] = test_components[:, i]

    numeric_columns = BASE_NUMERIC_COLUMNS + fusion.component_columns
    for col in numeric_columns + clip_columns:
        train_frame[col] = pd.to_numeric(train_frame[col], errors="coerce").fillna(0.0)
        test_frame[col] = pd.to_numeric(test_frame[col], errors="coerce").fillna(0.0)

    model = build_pipeline(numeric_columns, clip_columns)
    y_train = train_frame["label"].eq("fake").astype(int)
    y_test = test_frame["label"].eq("fake").astype(int)
    model.fit(train_frame, y_train)
    predictions = model.predict(test_frame)

    return {
        "fold": fold_id,
        "accuracy": accuracy_score(y_test, predictions),
        "f1_fake": f1_score(y_test, predictions),
        "precision_fake": precision_score(y_test, predictions, zero_division=0),
        "recall_fake": recall_score(y_test, predictions, zero_division=0),
    }


def main() -> None:
    frame, clip_columns = load_data()
    print(f"Rows: {len(frame)}")

    embedder = SentenceTransformer(SBERT_MODEL_ID)
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    labels = frame["label"].to_numpy()

    results = []
    start = time.time()
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(frame, labels), start=1):
        fold_start = time.time()
        metrics = run_fold(fold_id, frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy(), clip_columns, embedder)
        results.append(metrics)
        print(f"Fold {fold_id:2d}/{N_SPLITS*N_REPEATS}: accuracy={metrics['accuracy']:.4f} ({time.time()-fold_start:.1f}s)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(FOLD_CSV, index=False)
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
    print(f"\nFull pipeline (normalized CLIP) accuracy: {results_df['accuracy'].mean():.4f} +/- {results_df['accuracy'].std():.4f}")
    print("(original full pipeline on un-normalized images was 0.9560 +/- 0.0070)")
    print(f"Saved -> {FOLD_CSV}")


if __name__ == "__main__":
    main()
