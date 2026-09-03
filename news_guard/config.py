"""Project paths and opt-in API configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)
# The notebook has already downloaded its models. Avoid network checks during
# local app use; set HF_HUB_OFFLINE=0 before starting Python only if you intend
# to download or update a Hugging Face model deliberately.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    model_path: Path = PROJECT_ROOT / "models" / "multimodal_model.joblib"
    chroma_path: Path = PROJECT_ROOT / "rag" / "chroma_store"
    trusted_documents_path: Path = PROJECT_ROOT / "rag" / "trusted_documents.jsonl"
    feedback_path: Path = PROJECT_ROOT / "data" / "verified_feedback.csv"
    verified_images_path: Path = PROJECT_ROOT / "data" / "verified_images"
    upload_path: Path = PROJECT_ROOT / "data" / "uploads"
    independent_manifest_path: Path = PROJECT_ROOT / "independent_test_manifest.csv"
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    # Llama 3.3 70B was retired from Groq's free/developer tier in August 2026.
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    serper_api_key: str | None = os.getenv("SERPER_API_KEY")

    def create_runtime_directories(self) -> None:
        for directory in (
            self.model_path.parent,
            self.chroma_path,
            self.feedback_path.parent,
            self.verified_images_path,
            self.upload_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.create_runtime_directories()
    return settings
