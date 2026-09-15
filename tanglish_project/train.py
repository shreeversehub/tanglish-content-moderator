"""
Trains a TF-IDF + Logistic Regression classifier to detect offensive /
trolling content in Tanglish (Tamil-English code-mixed) text.

Why this approach for a hackathon MVP:
  - Trains in seconds on a laptop CPU, no GPU/internet needed at runtime
  - Character n-grams handle Tanglish spelling variation well
    (e.g. "semma", "sema", "simma" all share overlapping character chunks)
  - Easy to explain/defend in a demo: "which words/character patterns
    pushed this toward offensive" via the model's learned coefficients

Run:  python3 train.py
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from data_utils import load_split

print("Loading data...")
train_df = load_split("train")
dev_df = load_split("dev")
test_df = load_split("test")

print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")

# Character n-grams (3-5 chars) are more robust than word n-grams for
# Tanglish, since the same word gets spelled many different ways
# (e.g. "semma" / "sema" / "simma" all mean "great") and there's no
# fixed dictionary of "correct" spelling to rely on.
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # dataset is ~75/25 imbalanced
        C=5.0,
    )),
])

print("Training...")
pipeline.fit(train_df["text"], train_df["binary_label"])

print("\n=== Dev set performance ===")
dev_pred = pipeline.predict(dev_df["text"])
print(classification_report(
    dev_df["binary_label"], dev_pred,
    target_names=["Not Offensive", "Offensive"],
))

print("\n=== Test set performance ===")
test_pred = pipeline.predict(test_df["text"])
print(classification_report(
    test_df["binary_label"], test_pred,
    target_names=["Not Offensive", "Offensive"],
))
print(f"Test F1 (offensive class): {f1_score(test_df['binary_label'], test_pred):.4f}")

joblib.dump(pipeline, "tanglish_model.joblib")
print("\nSaved model to tanglish_model.joblib")
