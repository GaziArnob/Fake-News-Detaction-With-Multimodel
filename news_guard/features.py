"""Feature generation shared by production training and the upload app."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import easyocr
import numpy as np
import pandas as pd
import torch
from nltk.stem import SnowballStemmer
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.neighbors import NearestNeighbors
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

from .config import Settings


SEED = 42
TEXT_MAX_FEATURES = 5_000
TEXT_MIN_DOCUMENT_FREQUENCY = 3
# Multilingual so CISF semantic fusion and similarity_score work on Bengali
# OCR/caption text too, not just English (see OCR_LANGUAGES below).
SBERT_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
STEMMER = SnowballStemmer("english")
STOP_WORDS = ENGLISH_STOP_WORDS.difference({"no", "not", "nor", "never"})

# Bengali script block (U+0980-U+09FF). No stemmer is used for Bengali here
# (NLTK/Snowball has no Bengali support); a compact stopword list keeps the
# highest-frequency function words out of the TF-IDF vocabulary.
BENGALI_STOP_WORDS = frozenset({
    "এবং", "বা", "এই", "ওই", "সে", "তিনি", "তারা", "আমি", "আমরা", "তুমি",
    "তোমরা", "আপনি", "আপনারা", "এটা", "ওটা", "এটি", "ওটি", "যে", "যা",
    "যিনি", "যারা", "কি", "কী", "কেন", "কীভাবে", "কোথায়", "কখন", "না",
    "নেই", "নয়", "ছিল", "ছিলেন", "হয়", "হয়েছে", "হবে", "করে", "করেছে",
    "করেছেন", "করবে", "থেকে", "দিয়ে", "জন্য", "সঙ্গে", "সাথে", "মধ্যে",
    "পর", "আগে", "পরে", "একটি", "একটা", "দুটি", "অনেক", "সব", "সকল",
    "প্রতি", "তবে", "কিন্তু", "অথবা", "ও", "তো", "যদি", "তাহলে", "তাই",
    "এমন", "এত", "এতটা", "কোন", "কোনো", "হচ্ছে", "হয়ে", "রয়েছে", "আছে",
    "এর", "তার", "তাদের", "আমার", "আমাদের", "এখন", "তখন", "যখন",
})
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z']+|[ঀ-৿]+")

# This dataset is Bangladeshi news imagery (BANGLADESH BANK, TAKA, BUET, ...):
# an English-only reader forces Bengali glyphs into nonsense Latin/digit
# lookalikes, so both scripts must be loaded together.
OCR_LANGUAGES = ["bn", "en"]
OCR_CONFIDENCE_THRESHOLD = 0.35
OCR_MIN_SIDE_PIXELS = 800


def preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """Grayscale + autocontrast + upscale small images before OCR detection."""
    gray = ImageOps.autocontrast(image.convert("L"))
    width, height = gray.size
    shortest_side = min(width, height)
    if 0 < shortest_side < OCR_MIN_SIDE_PIXELS:
        scale = OCR_MIN_SIDE_PIXELS / shortest_side
        gray = gray.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    return np.asarray(gray)


def extract_ocr_text(
    image: Image.Image,
    reader: "easyocr.Reader",
    confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
) -> str:
    """Run EasyOCR with confidence filtering; low-confidence noise is dropped."""
    processed = preprocess_for_ocr(image)
    results = reader.readtext(processed, detail=1, paragraph=False)
    kept = [
        text.strip()
        for _, text, confidence in results
        if confidence >= confidence_threshold and text.strip()
    ]
    return " ".join(kept)


def tokenize_and_stem(text: str) -> list[str]:
    """Token order is preserved across scripts so English/Bengali bigrams
    (e.g. code-mixed captions) still line up correctly in the TF-IDF ngrams.
    English tokens are lowercased+stemmed; Bengali has no stemmer available
    so its tokens pass through as-is after stopword filtering."""
    tokens = TOKEN_PATTERN.findall(str(text))
    processed = []
    for token in tokens:
        if token[0].isascii():
            token = token.lower()
            if token not in STOP_WORDS and len(token) > 1:
                processed.append(STEMMER.stem(token))
        else:
            if token not in BENGALI_STOP_WORDS and len(token) > 1:
                processed.append(token)
    return processed


def prepare_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["ocr_text"] = prepared["ocr_text"].fillna("").astype(str)
    prepared["caption"] = prepared["caption"].fillna("").astype(str)
    prepared["combined_text"] = (
        "caption: " + prepared["caption"].str.strip()
        + " ocr: " + prepared["ocr_text"].str.strip()
    )
    prepared["semantic_text"] = (
        "caption: " + prepared["caption"].str.strip()
        + " [SEP] ocr: " + prepared["ocr_text"].str.strip()
    )
    prepared["processed_text"] = prepared["combined_text"].map(
        lambda value: " ".join(tokenize_and_stem(value))
    )
    prepared["token_count"] = prepared["processed_text"].str.split().str.len().fillna(0).astype(float)
    prepared["unique_token_ratio"] = prepared["processed_text"].map(
        lambda value: len(set(value.split())) / max(len(value.split()), 1)
    )
    prepared["has_ocr_text"] = prepared["ocr_text"].str.strip().ne("").astype(float)
    return prepared


def _normalize(rows: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.clip(norm, 1e-12, None)


@dataclass
class SemanticFusionReference:
    reference_embeddings: np.ndarray
    pca: PCA
    n_neighbors: int = 5
    alpha: float = 0.65

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        n_components: int = 32,
        n_neighbors: int = 5,
        alpha: float = 0.65,
    ) -> tuple["SemanticFusionReference", np.ndarray, np.ndarray, np.ndarray]:
        normalized = _normalize(np.asarray(embeddings, dtype=np.float32))
        fused, mean_similarity, max_similarity = cls._fuse(
            queries=normalized,
            candidates=normalized,
            n_neighbors=n_neighbors,
            alpha=alpha,
            exclude_self=True,
        )
        pca = PCA(
            n_components=min(n_components, fused.shape[0] - 1, fused.shape[1]),
            random_state=SEED,
        )
        return (
            cls(normalized, pca, n_neighbors=n_neighbors, alpha=alpha),
            pca.fit_transform(fused),
            mean_similarity,
            max_similarity,
        )

    @staticmethod
    def _fuse(
        queries: np.ndarray,
        candidates: np.ndarray,
        n_neighbors: int,
        alpha: float,
        exclude_self: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = min(n_neighbors + int(exclude_self), len(candidates))
        if count < 1:
            raise ValueError("At least one reference embedding is required for CISF.")
        search = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute")
        search.fit(candidates)
        distances, positions = search.kneighbors(queries)
        fused_rows, mean_rows, max_rows = [], [], []
        for index, (query, row_distances, row_positions) in enumerate(
            zip(queries, distances, positions)
        ):
            similarity = 1.0 - row_distances
            if exclude_self:
                keep = row_positions != index
                row_positions, similarity = row_positions[keep], similarity[keep]
            row_positions, similarity = row_positions[:n_neighbors], similarity[:n_neighbors]
            if not len(row_positions):
                raise ValueError("CISF could not find a non-self semantic neighbour.")
            weights = np.clip(similarity, 0.0, None)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)
            weights /= weights.sum()
            neighbour = (candidates[row_positions] * weights[:, None]).sum(axis=0)
            fused = alpha * query + (1.0 - alpha) * neighbour
            fused_rows.append(fused / max(float(np.linalg.norm(fused)), 1e-12))
            mean_rows.append(float(similarity.mean()))
            max_rows.append(float(similarity.max()))
        return np.vstack(fused_rows), np.asarray(mean_rows), np.asarray(max_rows)

    @property
    def component_columns(self) -> list[str]:
        return [f"cisf_fused_pca_{index:02d}" for index in range(self.pca.n_components_)]

    def transform(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = _normalize(np.asarray(embeddings, dtype=np.float32))
        fused, mean_similarity, max_similarity = self._fuse(
            queries=normalized,
            candidates=self.reference_embeddings,
            n_neighbors=self.n_neighbors,
            alpha=self.alpha,
            exclude_self=False,
        )
        return self.pca.transform(fused), mean_similarity, max_similarity


class ImageFeatureExtractor:
    """Lazy multimodal feature extraction for a newly uploaded image."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._reader = None
        self._blip_processor = None
        self._blip_model = None
        self._clip_processor = None
        self._clip_model = None
        self._embedder = None

    @property
    def reader(self):
        if self._reader is None:
            self._reader = easyocr.Reader(
                OCR_LANGUAGES,
                model_storage_directory=str(self.settings.project_root / "models" / "easyocr"),
            )
        return self._reader

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(SBERT_MODEL_ID, device=str(self.device))
        return self._embedder

    @property
    def blip(self):
        if self._blip_processor is None or self._blip_model is None:
            self._blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID).to(self.device)
            self._blip_model.eval()
        return self._blip_processor, self._blip_model

    @property
    def clip(self):
        if self._clip_processor is None or self._clip_model is None:
            self._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device)
            self._clip_model.eval()
        return self._clip_processor, self._clip_model

    def extract(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        ocr_text = extract_ocr_text(image, self.reader)
        blip_processor, blip_model = self.blip
        blip_inputs = blip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated = blip_model.generate(**blip_inputs, max_new_tokens=40)
        caption = blip_processor.decode(generated[0], skip_special_tokens=True).strip()
        semantic_text = f"caption: {caption} [SEP] ocr: {ocr_text}"
        embeddings = self.embedder.encode(
            [ocr_text or " ", caption or " ", semantic_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        clip_processor, clip_model = self.clip
        clip_inputs = clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            clip_output = clip_model.get_image_features(**clip_inputs)
            clip_vector = getattr(clip_output, "pooler_output", clip_output)
            clip_vector = torch.nn.functional.normalize(clip_vector, dim=1).cpu().numpy()[0]
        base = prepare_text_columns(pd.DataFrame([{
            "ocr_text": ocr_text,
            "caption": caption,
        }])).iloc[0].to_dict()
        base.update({
            "image_path": str(image_path.resolve()),
            "similarity_score": float(np.dot(embeddings[0], embeddings[1])),
            "ocr_length": float(len(ocr_text)),
            "caption_length": float(len(caption)),
            "semantic_embedding": embeddings[2],
            "clip_embedding": clip_vector.astype(np.float32),
        })
        return base
