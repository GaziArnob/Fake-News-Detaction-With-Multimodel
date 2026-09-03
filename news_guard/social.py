"""Transparent, optional social-capital signals.

These signals are never treated as a fact-check by themselves. They are shown
separately because the present training data has no account/network labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log1p


@dataclass(frozen=True)
class SocialProfile:
    account_age_days: int | None = None
    followers: int | None = None
    following: int | None = None
    verified_account: bool | None = None
    historical_accuracy: float | None = None
    source_reputation: float | None = None
    reshare_count: int | None = None


def assess_social_capital(profile: SocialProfile) -> dict:
    """Return an explainable reliability signal with explicit data coverage."""
    indicators: list[tuple[str, float, float]] = []
    notes: list[str] = []

    if profile.account_age_days is not None:
        value = min(max(profile.account_age_days, 0) / 365.0, 1.0)
        indicators.append(("account age", value, 0.15))
        if profile.account_age_days < 30:
            notes.append("Account is newer than 30 days.")

    if profile.followers is not None and profile.following is not None:
        ratio = (max(profile.followers, 0) + 1) / (max(profile.following, 0) + 1)
        value = min(log1p(ratio) / log1p(10), 1.0)
        indicators.append(("audience ratio", value, 0.10))

    if profile.verified_account is not None:
        indicators.append(("platform verification", 1.0 if profile.verified_account else 0.45, 0.10))
        if not profile.verified_account:
            notes.append("The account is not platform-verified; this is not proof of falsehood.")

    for name, value, weight in (
        ("historical accuracy", profile.historical_accuracy, 0.40),
        ("source reputation", profile.source_reputation, 0.25),
    ):
        if value is not None:
            clipped = min(max(float(value), 0.0), 1.0)
            indicators.append((name, clipped, weight))
            if clipped < 0.5:
                notes.append(f"{name.title()} is below 0.50.")

    if profile.reshare_count is not None and profile.reshare_count > 10_000:
        notes.append("High resharing can increase reach but does not verify the claim.")

    if not indicators:
        return {
            "status": "not_assessed",
            "reliability_score": None,
            "coverage": 0,
            "notes": ["No social/source data was supplied."],
            "profile": asdict(profile),
        }

    total_weight = sum(weight for _, _, weight in indicators)
    score = sum(value * weight for _, value, weight in indicators) / total_weight
    return {
        "status": "assessed",
        "reliability_score": round(score * 100, 1),
        "coverage": len(indicators),
        "signals_used": [name for name, _, _ in indicators],
        "notes": notes or ["No social-risk flags were triggered by the supplied fields."],
        "profile": asdict(profile),
    }
