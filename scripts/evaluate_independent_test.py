"""Evaluate only never-before-seen, externally sourced and manually labelled images."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from news_guard.config import get_settings
from news_guard.inference import ProductionClassifier


REQUIRED_COLUMNS = {"image_path", "label"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=get_settings().independent_manifest_path
    )
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest is missing: {sorted(missing)}")
    if not set(manifest["label"]).issubset({"real", "fake"}):
        raise SystemExit("Labels must be exactly real or fake.")
    classifier = ProductionClassifier(get_settings())
    results = []
    for record in manifest.to_dict("records"):
        prediction = classifier.predict(record["image_path"])
        results.append({
            **record,
            "predicted_label": prediction.label,
            "fake_probability": prediction.fake_probability,
            "correct_prediction": prediction.label == record["label"],
        })
    result_frame = pd.DataFrame(results)
    output_path = args.manifest.with_name("independent_test_results.csv")
    result_frame.to_csv(output_path, index=False)
    print(f"Accuracy: {accuracy_score(result_frame['label'], result_frame['predicted_label']):.4f}")
    print(f"Fake F1: {f1_score(result_frame['label'], result_frame['predicted_label'], pos_label='fake'):.4f}")
    print("Confusion matrix (real, fake):")
    print(confusion_matrix(result_frame["label"], result_frame["predicted_label"], labels=["real", "fake"]))
    print(classification_report(result_frame["label"], result_frame["predicted_label"], digits=4))
    print(f"Saved per-image results: {output_path}")
