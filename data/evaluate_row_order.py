
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "CIC-IDS2017"
    / "MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

FEATURES = [
    "Flow Duration",
    "Total Length of Bwd Packets",
    "Total Length of Fwd Packets",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Destination Port",
]

LABEL_COLUMN = "Label"

print("Reading the original CSV...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip()

# Normalize labels and numeric features.
df[LABEL_COLUMN] = (
    df[LABEL_COLUMN].astype(str).str.strip().str.upper()
)

df["target"] = np.where(
    df[LABEL_COLUMN] == "BENIGN", "BENIGN", "ATTACK"
)

for column in FEATURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=FEATURES + ["target"], inplace=True)
df = df[(df[FEATURES] >= 0).all(axis=1)].reset_index(drop=True)

# Keep original file order; do not shuffle.
split_index = int(len(df) * 0.75)
train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print("\nDataset size:", len(df))
print("Training rows:", len(train))
print("Testing rows:", len(test))

print("\nTraining label counts:")
print(train["target"].value_counts().to_string())

print("\nTesting label counts:")
print(test["target"].value_counts().to_string())

# A meaningful evaluation requires both classes in both partitions.
if train["target"].nunique() < 2 or test["target"].nunique() < 2:
    print(
        "\nWARNING: One partition contains only one class. "
        "Accuracy would be misleading; do not interpret it "
        "as a valid two-class evaluation."
    )
else:
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining diagnostic model...")
    model.fit(train[FEATURES], train["target"])

    predictions = model.predict(test[FEATURES])

    print("\n=== ROW-ORDER HOLDOUT RESULTS ===")
    print(f"Accuracy: {accuracy_score(test['target'], predictions):.6f}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(
        test["target"],
        predictions,
        labels=["BENIGN", "ATTACK"],
    ))

    print("\nClassification report:")
    print(classification_report(
        test["target"],
        predictions,
        labels=["BENIGN", "ATTACK"],
        zero_division=0,
    ))

    print("\nFeature importance:")
    for name, importance in sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name}: {importance:.4f}")

print("\nOriginal file order was preserved.")
print("This is a diagnostic split, not a timestamp-based test.")
