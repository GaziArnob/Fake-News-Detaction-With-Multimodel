"""Train and save a deployable version of the tuned multimodal XGBoost model."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import Settings
from .features import (
    SBERT_MODEL_ID,
    TEXT_MAX_FEATURES,
    TEXT_MIN_DOCUMENT_FREQUENCY,
    SemanticFusionReference,
    prepare_text_columns,
)


BASE_NUMERIC_COLUMNS = [
    "similarity_score",
    "ocr_length",
    "caption_length",
    "has_ocr_text",
    "token_count",
    "unique_token_ratio",
    "cisf_neighbour_mean_similarity",
    "cisf_neighbour_max_similarity",
]


def load_base_training_data(settings: Settings) -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_csv(settings.project_root / "extracted_features.csv")
    clip = pd.read_csv(settings.project_root / "clip_visual_features.csv")
    clip_columns = [
        column for column in clip.columns
        if column.startswith("clip_") and column[5:].isdigit()
    ]
    if len(clip_columns) != 512:
        raise ValueError("clip_visual_features.csv must contain 512 CLIP columns.")
    base = features.loc[
        features["error"].fillna("").eq("") & features["label"].isin(["real", "fake"])
    ].copy()
    base = base.merge(
        clip[["image_path", "clip_error"] + clip_columns], on="image_path", how="inner"
    )
    base = base.loc[base["clip_error"].fillna("").eq("")].copy().reset_index(drop=True)
    if base["label"].nunique() != 2:
        raise ValueError("Both real and fake labeled images are required.")
    return prepare_text_columns(base), clip_columns


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result


def build_training_frame(
    raw_frame: pd.DataFrame,
    clip_columns: list[str],
    embedder: SentenceTransformer,
) -> tuple[pd.DataFrame, SemanticFusionReference, list[str]]:
    frame = prepare_text_columns(raw_frame)
    semantic_embeddings = embedder.encode(
        frame["semantic_text"].tolist(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    fusion, components, mean_similarity, max_similarity = SemanticFusionReference.fit(semantic_embeddings)
    frame["cisf_neighbour_mean_similarity"] = mean_similarity
    frame["cisf_neighbour_max_similarity"] = max_similarity
    for index, column in enumerate(fusion.component_columns):
        frame[column] = components[:, index]
    numeric_columns = BASE_NUMERIC_COLUMNS + fusion.component_columns
    frame = _coerce_numeric(frame, numeric_columns + clip_columns)
    return frame, fusion, numeric_columns


def build_model_pipeline(numeric_columns: list[str], clip_columns: list[str]) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            (
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
                "processed_text",
            ),
            ("tabular", "passthrough", numeric_columns),
            ("clip_pca", PCA(n_components=64, random_state=42), clip_columns),
        ],
        sparse_threshold=0.30,
    )
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
        random_state=42,
        n_jobs=4,
        tree_method="hist",
        importance_type="gain",
    )
    return Pipeline([("features", features), ("xgb", classifier)])


def train_and_save(
    settings: Settings,
    additional_features: pd.DataFrame | None = None,
) -> dict:
    """Fit on all approved labelled records and save an inference artifact."""
    settings.create_runtime_directories()
    frame, clip_columns = load_base_training_data(settings)
    raw_columns = [
        "image_path",
        "label",
        "ocr_text",
        "caption",
        "similarity_score",
        "ocr_length",
        "caption_length",
    ] + clip_columns
    frame = frame[raw_columns].copy()
    if additional_features is not None and not additional_features.empty:
        expected = set(raw_columns)
        missing = expected.difference(additional_features.columns)
        if missing:
            raise ValueError(f"Verified feature rows are missing columns: {sorted(missing)[:5]}")
        frame = pd.concat([frame, additional_features[raw_columns]], ignore_index=True)
    embedder = SentenceTransformer(SBERT_MODEL_ID)
    frame, fusion, numeric_columns = build_training_frame(frame, clip_columns, embedder)
    model = build_model_pipeline(numeric_columns, clip_columns)
    model.fit(frame, frame["label"].eq("fake").astype(int))
    artifact = {
        "model": model,
        "fusion": fusion,
        "clip_columns": clip_columns,
        "numeric_columns": numeric_columns,
        "label_mapping": {0: "real", 1: "fake"},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(frame)),
        "class_counts": frame["label"].value_counts().to_dict(),
    }
    joblib.dump(artifact, settings.model_path)
    return artifact
