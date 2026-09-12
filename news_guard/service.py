"""One orchestration layer for local classification, RAG, verification, and feedback."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import Settings, get_settings
from .social import SocialProfile, assess_social_capital

if TYPE_CHECKING:
    from .inference import ProductionClassifier
    from .rag import ImageEvidenceStore, LocalEvidenceStore

# Unrelated photos score ~0.55-0.60 on CLIP cosine similarity; a genuine
# same-image re-share (even resized + re-compressed) still scores ~0.90+.
# Above this, treat it as the same image rather than routing through
# claim-based LLM verification, which checks "does evidence support this
# claim" and fails whenever the claim (BLIP's caption) is wrong even though
# the matched evidence is exactly on point.
IMAGE_MATCH_HIGH_CONFIDENCE = 0.90


class NewsVerificationService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._classifier = None
        self._evidence_store = None
        self._image_evidence_store = None

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

    @property
    def image_evidence_store(self) -> ImageEvidenceStore:
        if self._image_evidence_store is None:
            from .rag import ImageEvidenceStore

            self._image_evidence_store = ImageEvidenceStore(self.settings)
        return self._image_evidence_store

    def service_status(self) -> dict:
        from .updates import update_readiness

        return {
            "production_model": self.settings.model_path.exists(),
            "local_rag_documents": self.evidence_store.count(),
            "local_rag_images": self.image_evidence_store.count(),
            "gemini_configured": bool(self.settings.gemini_api_key),
            "groq_configured": bool(self.settings.groq_api_key),
            "serper_configured": bool(self.settings.serper_api_key),
            "update_queue": update_readiness(self.settings),
        }

    def analyze(
        self,
        image_path: str | Path,
        social_profile: SocialProfile | None = None,
        user_claim: str | None = None,
    ) -> dict:
        from .rag import search_serper
        from .verification import run_dual_verification

        local_prediction = self.classifier.predict(image_path)
        # BLIP describes what is visually IN the image ("a man on a red
        # background"), never what is being CLAIMED about it. A real viral
        # image is almost always shared alongside some claim text (a post
        # caption, a forwarded message); when the caller supplies it, use it
        # as the claim instead of the caption -- it is never worse, and it
        # is the only way this layer can verify what is actually asserted
        # rather than just what the picture visually shows.
        user_claim = (user_claim or "").strip()
        claim = user_claim or local_prediction.claim
        claim_source = "user_provided" if user_claim else "blip_caption"
        local_evidence = self.evidence_store.query(claim) if claim else []
        # Caption-based retrieval fails outright when BLIP mis-describes a
        # text-dense or cluttered screenshot; this compares the uploaded
        # image's own pixels against known fact-checked images, independent
        # of caption quality, and its matches are merged in alongside the
        # text-based results.
        image_evidence = self.image_evidence_store.query_by_image(image_path)
        local_evidence = local_evidence + image_evidence

        direct_match = next(
            (item for item in image_evidence if (item.score or 0) >= IMAGE_MATCH_HIGH_CONFIDENCE),
            None,
        )

        try:
            live_evidence = search_serper(claim, self.settings.serper_api_key) if claim else []
            live_search_status = "completed" if self.settings.serper_api_key else "not_configured"
        except Exception as exc:
            live_evidence = []
            live_search_status = f"error: {exc}"
        evidence = local_evidence + live_evidence

        if direct_match is not None:
            # Skip the claim-based LLM call entirely: it asks "does this
            # evidence support THIS claim text", which fails whenever the
            # claim (BLIP's caption) is wrong -- irrelevant here, since the
            # image itself, not its caption, is what was matched.
            verification = {
                "status": "direct_image_match",
                "consensus": None,
                "matched_article": {
                    "title": direct_match.title,
                    "url": direct_match.url,
                    "similarity": direct_match.score,
                },
                "gemini": {"status": "skipped", "rationale": "Bypassed: high-confidence direct image match found."},
                "groq": {"status": "skipped", "rationale": "Bypassed: high-confidence direct image match found."},
            }
        else:
            verification = run_dual_verification(claim, evidence, self.settings) if claim else {
                "status": "not_run",
                "consensus": None,
                "gemini": {"status": "not_run", "rationale": "No claim text was extracted."},
                "groq": {"status": "not_run", "rationale": "No claim text was extracted."},
            }
        social = assess_social_capital(social_profile or SocialProfile())
        return {
            "local_prediction": local_prediction.as_dict(),
            "verification_claim": claim,
            "claim_source": claim_source,
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
