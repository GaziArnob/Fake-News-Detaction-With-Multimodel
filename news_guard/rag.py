"""Local ChromaDB retrieval plus opt-in Serper live search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import requests
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from .config import Settings
from .features import CLIP_MODEL_ID, SBERT_MODEL_ID


@dataclass(frozen=True)
class EvidenceItem:
    title: str
    text: str
    url: str
    source_type: str
    score: float | None = None

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "source_type": self.source_type,
            "score": self.score,
        }


class LocalEvidenceStore:
    """Persistent local evidence collection. Documents must be independently vetted."""

    collection_name = "trusted_news_evidence"

    def __init__(self, settings: Settings, embedding_model: str = SBERT_MODEL_ID) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(embedding_model)

    def count(self) -> int:
        return self.collection.count()

    def index_documents(self, records: Iterable[dict]) -> int:
        prepared = []
        for index, record in enumerate(records):
            document = str(record.get("text", "")).strip()
            if not document:
                continue
            source_id = str(record.get("id") or f"evidence-{index}")
            prepared.append({
                "id": source_id,
                "document": document,
                "metadata": {
                    "title": str(record.get("title", "Untitled evidence"))[:500],
                    "url": str(record.get("url", ""))[:2_000],
                    "publisher": str(record.get("publisher", ""))[:500],
                    "source_type": "local_trusted",
                },
            })

        if not prepared:
            return 0

        embeddings = self.embedder.encode(
            [item["document"] for item in prepared],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
        self.collection.upsert(
            ids=[item["id"] for item in prepared],
            documents=[item["document"] for item in prepared],
            metadatas=[item["metadata"] for item in prepared],
            embeddings=embeddings,
        )
        return len(prepared)

    def query(self, claim: str, limit: int = 5, min_score: float = 0.30) -> list[EvidenceItem]:
        """min_score filters out low-similarity matches that are topically
        unrelated to the claim (e.g. score ~0.09-0.12 for an unrelated
        story) -- passing those to the verifier LLMs as "evidence" invites
        a confusing or wrong verdict instead of a clean "insufficient"."""
        if not claim.strip() or self.count() == 0:
            return []
        embedding = self.embedder.encode(
            [claim], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        result = self.collection.query(
            query_embeddings=embedding,
            n_results=min(limit, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadata = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            EvidenceItem(
                title=item_metadata.get("title", "Untitled evidence"),
                text=document,
                url=item_metadata.get("url", ""),
                source_type="local_trusted",
                score=round(1.0 - float(distance), 4),
            )
            for document, item_metadata, distance in zip(documents, metadata, distances)
            if (1.0 - float(distance)) >= min_score
        ]


class ImageEvidenceStore:
    """CLIP-embedding image-to-image evidence retrieval.

    LocalEvidenceStore's text search depends entirely on BLIP's caption --
    if the caption mis-describes a text-dense or cluttered screenshot, the
    claim it searches on is wrong and retrieval fails outright, even when
    the very same image was already fact-checked and sitting in this
    project's own evidence corpus. This store instead compares the
    *uploaded image's own pixels* against known fact-checked images via
    CLIP, independent of caption quality. It loads CLIP itself for this
    purpose only -- it does not touch or share state with the classifier's
    own BLIP/CLIP/SBERT feature-extraction pipeline in features.py.
    """

    collection_name = "trusted_news_evidence_images"

    def __init__(self, settings: Settings, clip_model_id: str = CLIP_MODEL_ID) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_processor = None
        self._clip_model = None
        self._clip_model_id = clip_model_id

    @property
    def clip(self):
        if self._clip_processor is None or self._clip_model is None:
            self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_id)
            self._clip_model = CLIPModel.from_pretrained(self._clip_model_id).to(self.device)
            self._clip_model.eval()
        return self._clip_processor, self._clip_model

    def count(self) -> int:
        return self.collection.count()

    def _embed(self, image_path: str | Path) -> list[float]:
        processor, model = self.clip
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            features = getattr(features, "pooler_output", features)
            features = torch.nn.functional.normalize(features, dim=1)
        return features.cpu().numpy()[0].tolist()

    def index_images(self, records: Iterable[dict]) -> int:
        """records: dicts with id, image_path, title, url, publisher."""
        prepared = []
        for record in records:
            image_path = record.get("image_path")
            if not image_path or not Path(image_path).exists():
                continue
            try:
                embedding = self._embed(image_path)
            except Exception:
                continue
            prepared.append({
                "id": str(record["id"]),
                "embedding": embedding,
                "metadata": {
                    "title": str(record.get("title", "Untitled evidence"))[:500],
                    "url": str(record.get("url", ""))[:2_000],
                    "publisher": str(record.get("publisher", ""))[:500],
                },
            })
        if not prepared:
            return 0
        self.collection.upsert(
            ids=[item["id"] for item in prepared],
            documents=[item["metadata"]["title"] for item in prepared],
            metadatas=[item["metadata"] for item in prepared],
            embeddings=[item["embedding"] for item in prepared],
        )
        return len(prepared)

    def query_by_image(self, image_path: str | Path, limit: int = 5, min_score: float = 0.75) -> list[EvidenceItem]:
        """min_score is much higher than the text store's (0.30): CLIP
        cosine similarity between genuinely unrelated photos still lands
        around 0.55-0.60, so only a near-duplicate / re-shared image should
        be treated as a match here."""
        if self.count() == 0:
            return []
        embedding = self._embed(image_path)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(limit, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        metadata = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            EvidenceItem(
                title=item_metadata.get("title", "Untitled evidence"),
                text=f"Visually matches a previously fact-checked image: {item_metadata.get('title', '')}",
                url=item_metadata.get("url", ""),
                source_type="local_trusted_image_match",
                score=round(1.0 - float(distance), 4),
            )
            for item_metadata, distance in zip(metadata, distances)
            if (1.0 - float(distance)) >= min_score
        ]


def index_jsonl_file(store: LocalEvidenceStore, source_path: Path) -> int:
    if not source_path.exists():
        raise FileNotFoundError(f"Evidence file not found: {source_path}")
    with source_path.open(encoding="utf-8") as file_handle:
        records = [json.loads(line) for line in file_handle if line.strip()]
    return store.index_documents(records)


def search_serper(claim: str, api_key: str | None, limit: int = 5) -> list[EvidenceItem]:
    """Retrieve web results; results are evidence leads, not truth labels."""
    if not api_key or not claim.strip():
        return []
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": claim, "num": limit},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        EvidenceItem(
            title=str(item.get("title", "Untitled result")),
            text=str(item.get("snippet", "")),
            url=str(item.get("link", "")),
            source_type="serper_live",
            score=None,
        )
        for item in payload.get("organic", [])[:limit]
    ]


def build_evidence_context(items: Iterable[EvidenceItem], character_limit: int = 8_000) -> str:
    """Per-item text is truncated (not skipped/dropped) so one long article
    can never crowd out every other piece of evidence -- previously the
    first item alone exceeding character_limit made the whole context empty,
    even with several short, on-topic items right behind it."""
    blocks: list[str] = []
    used = 0
    for number, item in enumerate(items, start=1):
        if used >= character_limit:
            break
        remaining = character_limit - used
        header = f"[{number}] {item.title}\nURL: {item.url}\n"
        text = item.text
        if len(header) + len(text) > remaining:
            text = text[: max(remaining - len(header), 0)].rstrip() + "..."
        block = (header + text).strip()
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks) or "No retrieved evidence is available."
