"""Extract approved feedback features and retrain only when the queue is ready."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news_guard.config import get_settings
from news_guard.training import train_and_save
from news_guard.updates import prepare_feedback_features, update_readiness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-new-examples", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    readiness = update_readiness(settings, args.minimum_new_examples)
    if not readiness["ready"] and not args.force:
        raise SystemExit(f"Not retraining: {readiness['reason']} Current state: {readiness}")
    features = prepare_feedback_features(settings)
    if features.empty:
        raise SystemExit("No approved feedback images are available for retraining.")
    artifact = train_and_save(settings, additional_features=features)
    print(f"Model updated with {artifact['training_rows']} total labeled rows.")
