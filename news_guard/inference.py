"""Run the saved multimodal classifier against a newly uploaded image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Settings
from .features import ImageFeatureExtractor


@dataclass
class LocalPrediction:
    label: str
    fake_probability: float
    confidence: float
    claim: str
    ocr_text: str
    caption: str
    similarity_score: float
    raw_features: dict

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "fake_probability": self.fake_probability,
            "confidence": self.confidence,
            "claim": self.claim,
            "ocr_text": self.ocr_text,
            "caption": self.caption,
            "similarity_score": self.similarity_score,
        }


class ProductionClassifier:
    def __init__(self, settings: Settings) -> None:
        if not settings.model_path.exists():
            raise FileNotFoundError(
                "Production model is missing. Run: python scripts/train_production_model.py"
            )
        self.settings = settings
        self.artifact = joblib.load(settings.model_path)
        self.extractor = ImageFeatureExtractor(settings)

    def predict(self, image_path: str | Path) -> LocalPrediction:
        extracted = self.extractor.extract(image_path)
        components, mean_similarity, max_similarity = self.artifact["fusion"].transform(
            np.asarray([extracted["semantic_embedding"]])
        )
        extracted["cisf_neighbour_mean_similarity"] = float(mean_similarity[0])
        extracted["cisf_neighbour_max_similarity"] = float(max_similarity[0])
        for index, column in enumerate(self.artifact["fusion"].component_columns):
            extracted[column] = float(components[0, index])
        for index, column in enumerate(self.artifact["clip_columns"]):
            extracted[column] = float(extracted["clip_embedding"][index])
        row = pd.DataFrame([extracted])
        for column in self.artifact["numeric_columns"]:
            row[column] = pd.to_numeric(row.get(column, 0.0), errors="coerce").fillna(0.0)
        probabilities = self.artifact["model"].predict_proba(row)[0]
        fake_probability = float(probabilities[1])
        label = "fake" if fake_probability >= 0.5 else "real"
        claim = extracted["caption"].strip()
        return LocalPrediction(
            label=label,
            fake_probability=fake_probability,
            confidence=max(fake_probability, 1.0 - fake_probability),
            claim=claim,
            ocr_text=extracted["ocr_text"],
            caption=extracted["caption"],
            similarity_score=float(extracted["similarity_score"]),
            raw_features=extracted,
        )
