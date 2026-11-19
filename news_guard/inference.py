"""Run the saved multimodal classifier against a newly uploaded image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Settings
from .features import ImageFeatureExtractor

# Selected on the 150-image AdversarialGen VALIDATION partition
# (adversarialgen_val150_manifest.csv), which is disjoint from both the
# adaptation-training set and the sealed 200-image test partition. The rule
# was fixed before the sweep: argmax fake-class F1 over a 0.01 grid, ties
# broken by the median of the tied run -> 0.21 for this model
# (multimodal_model.joblib). Sweep: adversarialgen_val150_sweep.csv.
#
# Scored once on the sealed test partition at this threshold: 92.0% accuracy,
# 94% fake-recall, 90% real-recall, against 86.0% / 73% / 99% at the default
# 0.50 cut-off (adversarialgen_leakfree_results.csv).
#
# NOTE: an earlier version of this constant was 0.30, chosen by sweeping the
# test partition itself. That value is superseded and should not be restored;
# the numbers it produced (90.0% / 91.5%) were selection artifacts.
FAKE_DECISION_THRESHOLD = 0.21


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
        label = "fake" if fake_probability >= FAKE_DECISION_THRESHOLD else "real"
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
