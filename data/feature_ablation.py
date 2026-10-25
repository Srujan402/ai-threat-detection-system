
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "cic_ids2017_prepared.csv"

ALL_FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]

EXPERIMENTS = {
    "All five features": ALL_FEATURES,
    "Without destination port": [
        "flow_duration", "bytes_in", "bytes_out", "packets"
    ],
    "Only duration and packet count": [
        "flow_duration", "packets"
    ],
    "Only byte counts": ["bytes_in", "bytes_out"],
    "Only destination port": ["dst_port"],
}

df = pd.read_csv(DATA_PATH)
df.columns = [str(c).strip().lower() for c in df.columns]
df = df[ALL_FEATURES + ["label"]].dropna().copy()

df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["benign", "attack"])].copy()

X = df[ALL_FEATURES]
y = df["label"]

# Use identical row indices for every experiment.
train_idx, test_idx = train_test_split(
    list(range(len(df))),
    test_size=0.25,
    random_state=42,
    stratify=y,
)

results = []

for name, features in EXPERIMENTS.items():
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X.iloc[train_idx][features], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx][features])
    actual = y.iloc[test_idx]

    accuracy = accuracy_score(actual, predictions)
    matrix = confusion_matrix(
        actual, predictions, labels=["benign", "attack"]
    )

    results.append({
        "experiment": name,
        "features": ", ".join(features),
        "accuracy": accuracy,
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
    })

    print(f"\n{'=' * 60}")
    print(name)
    print(f"Features: {', '.join(features)}")
    print(f"Accuracy: {accuracy:.6f}")
    print("Confusion matrix (rows=true, columns=predicted)")
    print("Label order: benign, attack")
    print(matrix)
    print(classification_report(
        actual,
        predictions,
        labels=["benign", "attack"],
        zero_division=0,
    ))

print("\n=== SUMMARY ===")
print(pd.DataFrame(results).to_string(index=False))
print("\nExperiment complete. No model or dataset files were modified.")
