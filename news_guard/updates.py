"""Human-approved feedback queue and controlled dynamic model updates."""

from __future__ import annotations

import csv
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Settings
from .features import ImageFeatureExtractor


FEEDBACK_COLUMNS = [
    "record_id",
    "created_at",
    "image_path",
    "label",
    "reviewer",
    "source_url",
    "notes",
    "feature_status",
]


def record_verified_example(
    settings: Settings,
    source_image: str | Path,
    label: str,
    reviewer: str,
    source_url: str = "",
    notes: str = "",
) -> dict:
    """Copy one human-verified image into the update queue; never auto-label it."""
    if label not in {"real", "fake"}:
        raise ValueError("label must be 'real' or 'fake'.")
    if not reviewer.strip():
        raise ValueError("A reviewer name or ID is required for an approved update.")
    settings.create_runtime_directories()
    record_id = uuid.uuid4().hex
    source = Path(source_image)
    destination = settings.verified_images_path / label / f"{record_id}{source.suffix.lower()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    record = {
        "record_id": record_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_path": str(destination.resolve()),
        "label": label,
        "reviewer": reviewer.strip(),
        "source_url": source_url.strip(),
        "notes": notes.strip(),
        "feature_status": "queued",
    }
    write_header = not settings.feedback_path.exists()
    with settings.feedback_path.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=FEEDBACK_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return record


def prepare_feedback_features(settings: Settings) -> pd.DataFrame:
    """Extract heavy image features once for queued, human-approved examples."""
    if not settings.feedback_path.exists():
        return pd.DataFrame()
    feedback = pd.read_csv(settings.feedback_path)
    output_path = settings.feedback_path.with_name("verified_features.csv")
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    completed_ids = set(existing.get("record_id", pd.Series(dtype=str)).astype(str))
    pending = feedback.loc[~feedback["record_id"].astype(str).isin(completed_ids)].copy()
    if pending.empty:
        return existing
    extractor = ImageFeatureExtractor(settings)
    rows = []
    for record in pending.to_dict("records"):
        extracted = extractor.extract(record["image_path"])
        row = {key: value for key, value in extracted.items() if key not in {"semantic_embedding", "clip_embedding"}}
        row.update({
            "record_id": record["record_id"],
            "label": record["label"],
            "approved_image_path": record["image_path"],
        })
        for index, value in enumerate(extracted["clip_embedding"]):
            row[f"clip_{index:03d}"] = float(value)
        rows.append(row)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(output_path, index=False)
    feedback.loc[feedback["record_id"].astype(str).isin({row["record_id"] for row in rows}), "feature_status"] = "ready"
    feedback.to_csv(settings.feedback_path, index=False)
    return combined


def update_readiness(settings: Settings, minimum_new_examples: int = 50) -> dict:
    if not settings.feedback_path.exists():
        return {"ready": False, "approved_rows": 0, "reason": "No human-verified examples queued."}
    feedback = pd.read_csv(settings.feedback_path)
    counts = feedback["label"].value_counts().to_dict()
    total = int(len(feedback))
    ready = total >= minimum_new_examples and len(counts) == 2
    return {
        "ready": ready,
        "approved_rows": total,
        "class_counts": counts,
        "minimum_new_examples": minimum_new_examples,
        "reason": "Ready to retrain." if ready else "Collect at least 50 verified images across both labels.",
    }
