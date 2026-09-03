"""Create models/multimodal_model.joblib for the upload application."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news_guard.config import get_settings
from news_guard.training import train_and_save


if __name__ == "__main__":
    artifact = train_and_save(get_settings())
    print(f"Saved production model: {get_settings().model_path}")
    print(f"Training rows: {artifact['training_rows']}")
    print(f"Class counts: {artifact['class_counts']}")
