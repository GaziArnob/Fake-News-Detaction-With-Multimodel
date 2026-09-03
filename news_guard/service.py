"""One orchestration layer for local classification, RAG, verification, and feedback."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import Settings, get_settings
from .social import SocialProfile, assess_social_capital

if TYPE_CHECKING:
    from .inference import ProductionClassifier
    from .rag import LocalEvidenceStore


class NewsVerificationService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._classifier = None
        self._evidence_store = None

    @property
    def classifier(self) -> ProductionClassifier:
        if self._classifier is None:
            from .inference import ProductionClassifier

            self._classifier = ProductionClassifier(self.settings)
        return self._classifier

    @property
    def evidence_store(self) -> LocalEvidenceStore:
        if self._evidence_store is None:
            from .rag import LocalEvidenceStore

            self._evidence_store = LocalEvidenceStore(self.settings)
        return self._evidence_store

    def service_status(self) -> dict:
        from .updates import update_readiness

        return {
            "production_model": self.settings.model_path.exists(),
            "local_rag_documents": self.evidence_store.count(),
            "gemini_configured": bool(self.settings.gemini_api_key),
            "groq_configured": bool(self.settings.groq_api_key),
            "serper_configured": bool(self.settings.serper_api_key),
            "update_queue": update_readiness(self.settings),
        }

    def analyze(self, image_path: str | Path, social_profile: SocialProfile | None = None) -> dict:
        from .rag import search_serper
        from .verification import run_dual_verification

        local_prediction = self.classifier.predict(image_path)
        claim = local_prediction.claim
        local_evidence = self.evidence_store.query(claim) if claim else []
        try:
            live_evidence = search_serper(claim, self.settings.serper_api_key) if claim else []
            live_search_status = "completed" if self.settings.serper_api_key else "not_configured"
        except Exception as exc:
            live_evidence = []
            live_search_status = f"error: {exc}"
        evidence = local_evidence + live_evidence
        verification = run_dual_verification(claim, evidence, self.settings) if claim else {
            "status": "not_run",
            "consensus": None,
            "gemini": {"status": "not_run", "rationale": "No claim text was extracted."},
            "groq": {"status": "not_run", "rationale": "No claim text was extracted."},
        }
        social = assess_social_capital(social_profile or SocialProfile())
        return {
            "local_prediction": local_prediction.as_dict(),
            "evidence": [item.as_dict() for item in evidence],
            "local_evidence_count": len(local_evidence),
            "live_evidence_count": len(live_evidence),
            "live_search_status": live_search_status,
            "dual_verification": verification,
            "social_capital": social,
            "disclaimer": (
                "The classifier estimates a dataset-trained risk signal. Retrieved sources and AI reviewers "
                "are evidence aids, not a substitute for human fact-checking."
            ),
        }

    def save_human_verdict(
        self,
        image_path: str | Path,
        label: str,
        reviewer: str,
        source_url: str = "",
        notes: str = "",
    ) -> dict:
        from .updates import record_verified_example

        return record_verified_example(
            self.settings, image_path, label, reviewer, source_url, notes
        )
