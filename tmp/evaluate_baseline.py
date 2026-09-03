"""Run the notebook's baseline Logistic Regression evaluation."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


features_df = pd.read_csv(Path("extracted_features.csv"))
labels = ["real", "fake"]
model_features = ["similarity_score", "ocr_length", "caption_length"]
seed = 42

training_df = features_df.dropna(subset=model_features + ["label"]).copy()
training_df = training_df.loc[training_df["error"].fillna("").eq("")].copy()

X_train, X_test, y_train, y_test = train_test_split(
    training_df[model_features],
    training_df["label"],
    test_size=0.20,
    stratify=training_df["label"],
    random_state=seed,
)

classifier = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1_000, random_state=seed)),
])
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print(f"Rows used for training: {len(training_df)}")
print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"precision (fake): {precision_score(y_test, predictions, pos_label='fake', zero_division=0):.4f}")
print(f"recall (fake): {recall_score(y_test, predictions, pos_label='fake', zero_division=0):.4f}")
print(f"F1 (fake): {f1_score(y_test, predictions, pos_label='fake', zero_division=0):.4f}")
print("confusion matrix [real, fake]:")
print(confusion_matrix(y_test, predictions, labels=labels))
