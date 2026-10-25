
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "cic_ids2017_prepared.csv"

FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]

df = pd.read_csv(DATA_PATH)
df.columns = [str(c).strip().lower() for c in df.columns]
df = df[FEATURES + ["label"]].dropna().copy()

df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["benign", "attack"])].copy()

X = df[FEATURES]
y = df["label"]

print("\n=== LABEL RATE BY DESTINATION PORT ===")
port_stats = (
    df.groupby("dst_port")["label"]
    .agg(
        rows="size",
        attack_rate=lambda s: (s == "attack").mean(),
    )
    .sort_values("rows", ascending=False)
)

print("Most frequent ports:")
print(port_stats.head(20).to_string())

print("\nPorts with highest attack rates (minimum 100 rows):")
print(
    port_stats[port_stats["rows"] >= 100]
    .sort_values("attack_rate", ascending=False)
    .head(20)
    .to_string()
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print(importance.to_string())

print("\n=== PERMUTATION IMPORTANCE ON HELD-OUT TEST DATA ===")
permutation = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=3,
    random_state=42,
    scoring="accuracy",
    n_jobs=-1,
)

permutation_scores = pd.DataFrame({
    "feature": FEATURES,
    "mean_accuracy_drop": permutation.importances_mean,
    "std": permutation.importances_std,
}).sort_values("mean_accuracy_drop", ascending=False)

print(permutation_scores.to_string(index=False))

print("\nInspection complete. No model or dataset files were modified.")
