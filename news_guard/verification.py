"""Gemini + Groq evidence-aware, dual-model verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .rag import EvidenceItem, build_evidence_context


@dataclass(frozen=True)
class ModelVerdict:
    provider: str
    status: str
    verdict: str | None
    confidence: float | None
    rationale: str
    cited_urls: list[str]
    raw_response: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "status": self.status,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "cited_urls": self.cited_urls,
        }


def _prompt(claim: str, evidence: list[EvidenceItem]) -> str:
    context = build_evidence_context(evidence)
    return f"""You are one independent fact-checking reviewer.
Assess the claim using only the supplied evidence. Do not treat search ranking,
social engagement, or a classifier probability as proof. If the evidence is not
enough, choose insufficient.

Claim:
{claim}

Evidence:
{context}

Return valid JSON only with this exact shape:
{{"verdict":"supported|contradicted|insufficient","confidence":0-100,
"rationale":"short evidence-grounded explanation","cited_urls":["https://..."]}}
"""


def _parse(provider: str, response_text: str) -> ModelVerdict:
    match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if not match:
        return ModelVerdict(provider, "unparseable", None, None, response_text.strip(), [], response_text)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ModelVerdict(provider, "unparseable", None, None, response_text.strip(), [], response_text)

    verdict = str(payload.get("verdict", "")).lower().strip()
    if verdict not in {"supported", "contradicted", "insufficient"}:
        verdict = "insufficient"
    try:
        confidence = max(0.0, min(float(payload.get("confidence", 0)), 100.0))
    except (TypeError, ValueError):
        confidence = 0.0
    urls = payload.get("cited_urls", [])
    if not isinstance(urls, list):
        urls = []
    return ModelVerdict(
        provider=provider,
        status="completed",
        verdict=verdict,
        confidence=confidence,
        rationale=str(payload.get("rationale", "No rationale supplied.")).strip(),
        cited_urls=[str(url) for url in urls if str(url).startswith("http")],
        raw_response=response_text,
    )


def verify_with_gemini(claim: str, evidence: list[EvidenceItem], settings: Settings) -> ModelVerdict:
    if not settings.gemini_api_key:
        return ModelVerdict("gemini", "not_configured", None, None, "GEMINI_API_KEY is not configured.", [])
    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.interactions.create(
            model=settings.gemini_model,
            input=_prompt(claim, evidence),
        )
        return _parse("gemini", str(response.output_text))
    except Exception as exc:  # External services should never stop local analysis.
        return ModelVerdict("gemini", "error", None, None, str(exc), [])


def verify_with_groq(claim: str, evidence: list[EvidenceItem], settings: Settings) -> ModelVerdict:
    if not settings.groq_api_key:
        return ModelVerdict("groq", "not_configured", None, None, "GROQ_API_KEY is not configured.", [])
    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": "Return only the JSON requested by the user."},
                {"role": "user", "content": _prompt(claim, evidence)},
            ],
            temperature=0,
            max_completion_tokens=500,
        )
        return _parse("groq", response.choices[0].message.content or "")
    except Exception as exc:
        return ModelVerdict("groq", "error", None, None, str(exc), [])


def run_dual_verification(claim: str, evidence: list[EvidenceItem], settings: Settings) -> dict:
    gemini = verify_with_gemini(claim, evidence, settings)
    groq = verify_with_groq(claim, evidence, settings)
    completed = [item for item in (gemini, groq) if item.status == "completed"]
    verdicts = {item.verdict for item in completed}
    if len(completed) == 2 and len(verdicts) == 1:
        consensus = completed[0].verdict
        status = "agreement"
    elif not completed:
        consensus = None
        status = "not_run"
    else:
        consensus = None
        status = "needs_human_review"
    return {
        "status": status,
        "consensus": consensus,
        "gemini": gemini.as_dict(),
        "groq": groq.as_dict(),
    }
