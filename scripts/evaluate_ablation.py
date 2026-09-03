"""Ablation study: repeated stratified k-fold CV for each feature configuration.

Isolates the contribution of each modality/technique in the multimodal
pipeline (OCR text, BLIP caption, CISF semantic fusion, CLIP visual
features) by fitting the same XGBoost classifier on different feature
subsets, each evaluated with the same repeated k-fold protocol used for the
full-pipeline number (see evaluate_kfold.py) so the comparison is
apples-to-apples and not a single lucky/unlucky split.

Each configuration's numeric feature set is restricted to what that
configuration's inputs can actually derive (e.g. "caption_only" excludes
ocr_length/similarity_score, which need OCR text) so no modality leaks
signal into an ablation meant to exclude it.
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

from news_guard.config import get_settings
from news_guard.features import SBERT_MODEL_ID, SemanticFusionReference, tokenize_and_stem
from news_guard.training import TEXT_MAX_FEATURES, TEXT_MIN_DOCUMENT_FREQUENCY, load_base_training_data

N_SPLITS = 5
N_REPEATS = 2
SEED = 42
TEXT_COLUMN = "ablation_text"
FOLD_CSV = Path("ablation_fold_metrics.csv")
SUMMARY_CSV = Path("ablation_summary.csv")

# name -> (text_source, use_cisf, use_clip)
#   text_source: "combined" (caption+ocr, matches production), "ocr", "caption", or None
CONFIGS = {
    "ocr_only": ("ocr", False, False),
    "caption_only": ("caption", False, False),
    "text_only": ("combined", False, False),
    "text_plus_cisf": ("combined", True, False),
    "clip_only": (None, False, True),
    "text_plus_clip": ("combined", False, True),
    "full_text_cisf_clip": ("combined", True, True),
}


def make_processed_text(frame: pd.DataFrame, source: str | None) -> pd.DataFrame:
    frame = frame.copy()
    if source == "combined":
        frame["caption"] = frame["caption"].fillna("").astype(str)
        frame["ocr_text"] = frame["ocr_text"].fillna("").astype(str)
        raw = "caption: " + frame["caption"].str.strip() + " ocr: " + frame["ocr_text"].str.strip()
    elif source in ("ocr", "caption"):
        column = "ocr_text" if source == "ocr" else "caption"
        raw = frame[column].fillna("").astype(str).str.strip()
    else:
        raw = pd.Series([""] * len(frame), index=frame.index)
    frame[TEXT_COLUMN] = raw.map(lambda value: " ".join(tokenize_and_stem(value)))
    frame["token_count"] = frame[TEXT_COLUMN].str.split().str.len().fillna(0).astype(float)
    frame["unique_token_ratio"] = frame[TEXT_COLUMN].map(
        lambda value: len(set(value.split())) / max(len(value.split()), 1)
    )
    return frame


def numeric_columns_for(source: str | None) -> list[str]:
    columns = []
    if source in ("combined", "ocr"):
        columns += ["ocr_length", "has_ocr_text"]
    if source in ("combined", "caption"):
        columns += ["caption_length"]
    if source == "combined":
        columns += ["similarity_score"]
    if source is not None:
        columns += ["token_count", "unique_token_ratio"]
    return columns


def build_pipeline(use_text: bool, numeric_columns: list[str], use_clip: bool, clip_columns: list[str]) -> Pipeline:
    transformers = []
    if use_text:
        transformers.append((
            "tfidf",
            TfidfVectorizer(
                tokenizer=str.split,
                token_pattern=None,
                lowercase=False,
                ngram_range=(1, 2),
                min_df=TEXT_MIN_DOCUMENT_FREQUENCY,
                max_features=TEXT_MAX_FEATURES,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            TEXT_COLUMN,
        ))
    if numeric_columns:
        transformers.append(("tabular", "passthrough", numeric_columns))
    if use_clip:
        transformers.append(("clip_pca", PCA(n_components=64, random_state=SEED), clip_columns))

    features = ColumnTransformer(transformers=transformers, sparse_threshold=0.30)
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        colsample_bytree=0.6834834444556088,
        gamma=0.14286681792194078,
        learning_rate=0.07423304665511389,
        max_depth=3,
        min_child_weight=2,
        n_estimators=443,
        reg_alpha=1.007198483880919e-05,
        reg_lambda=4.9111315080437645,
        subsample=0.8661185283697008,
        random_state=SEED,
        n_jobs=4,
        tree_method="hist",
        importance_type="gain",
    )
    return Pipeline([("features", features), ("xgb", classifier)])


def run_fold(
    config_name: str,
    text_source: str | None,
    use_cisf: bool,
    use_clip: bool,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    clip_columns: list[str],
    embedder: SentenceTransformer | None,
) -> dict:
    train_frame = make_processed_text(train_frame.reset_index(drop=True), text_source)
    test_frame = make_processed_text(test_frame.reset_index(drop=True), text_source)
    numeric_columns = numeric_columns_for(text_source)

    if use_cisf:
        train_semantic_text = (
            "caption: " + train_frame["caption"].fillna("").astype(str).str.strip()
            + " [SEP] ocr: " + train_frame["ocr_text"].fillna("").astype(str).str.strip()
        )
        test_semantic_text = (
            "caption: " + test_frame["caption"].fillna("").astype(str).str.strip()
            + " [SEP] ocr: " + test_frame["ocr_text"].fillna("").astype(str).str.strip()
        )
        train_semantic = embedder.encode(train_semantic_text.tolist(), normalize_embeddings=True, convert_to_numpy=True)
        test_semantic = embedder.encode(test_semantic_text.tolist(), normalize_embeddings=True, convert_to_numpy=True)
        fusion, train_components, train_mean, train_max = SemanticFusionReference.fit(train_semantic)
        test_components, test_mean, test_max = fusion.transform(test_semantic)
        train_frame["cisf_neighbour_mean_similarity"] = train_mean
        train_frame["cisf_neighbour_max_similarity"] = train_max
        test_frame["cisf_neighbour_mean_similarity"] = test_mean
        test_frame["cisf_neighbour_max_similarity"] = test_max
        for index, column in enumerate(fusion.component_columns):
            train_frame[column] = train_components[:, index]
            test_frame[column] = test_components[:, index]
        numeric_columns = numeric_columns + ["cisf_neighbour_mean_similarity", "cisf_neighbour_max_similarity"] + fusion.component_columns

    for column in numeric_columns + (clip_columns if use_clip else []):
        train_frame[column] = pd.to_numeric(train_frame[column], errors="coerce").fillna(0.0)
        test_frame[column] = pd.to_numeric(test_frame[column], errors="coerce").fillna(0.0)

    model = build_pipeline(text_source is not None, numeric_columns, use_clip, clip_columns)
    y_train = train_frame["label"].eq("fake").astype(int)
    y_test = test_frame["label"].eq("fake").astype(int)
    model.fit(train_frame, y_train)
    predictions = model.predict(test_frame)

    return {
        "config": config_name,
        "accuracy": accuracy_score(y_test, predictions),
        "f1_fake": f1_score(y_test, predictions),
        "precision_fake": precision_score(y_test, predictions, zero_division=0),
        "recall_fake": recall_score(y_test, predictions, zero_division=0),
    }


def main() -> None:
    settings = get_settings()
    frame, clip_columns = load_base_training_data(settings)
    print(f"Total usable rows: {len(frame)}")
    print(f"Configs: {list(CONFIGS)}")

    needs_embedder = any(use_cisf for _, use_cisf, _ in CONFIGS.values())
    embedder = SentenceTransformer(SBERT_MODEL_ID) if needs_embedder else None

    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    labels = frame["label"].to_numpy()
    # Precompute the same folds once so every config sees identical splits.
    folds = list(splitter.split(frame, labels))

    all_results = []
    overall_start = time.time()
    for config_name, (text_source, use_cisf, use_clip) in CONFIGS.items():
        config_start = time.time()
        for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
            fold_start = time.time()
            metrics = run_fold(
                config_name, text_source, use_cisf, use_clip,
                frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy(),
                clip_columns, embedder,
            )
            metrics["fold"] = fold_id
            all_results.append(metrics)
            print(
                f"[{config_name:20s}] fold {fold_id:2d}/{len(folds)}: "
                f"accuracy={metrics['accuracy']:.4f} ({time.time() - fold_start:.1f}s)"
            )
        print(f"-> {config_name} done in {(time.time() - config_start) / 60:.1f} min\n")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(FOLD_CSV, index=False)

    summary = (
        results_df.groupby("config")[["accuracy", "f1_fake", "precision_fake", "recall_fake"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reindex(list(CONFIGS)).reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"\nTotal time: {(time.time() - overall_start) / 60:.1f} min\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved per-fold metrics -> {FOLD_CSV}")
    print(f"Saved summary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
