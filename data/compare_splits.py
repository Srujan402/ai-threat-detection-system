
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit

DATA_PATH = Path(__file__).parent / "cic_ids2017_prepared.csv"

FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]

LABEL = "label"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
df.columns = [str(c).strip().lower() for c in df.columns]

df = df[FEATURES + [LABEL]].copy()
df[LABEL] = df[LABEL].astype(str).str.strip().str.lower()

df = df[df[LABEL].isin(["benign", "attack"])].copy()
df = df.dropna(subset=FEATURES + [LABEL])

X = df[FEATURES]
y = df[LABEL]

# Identical five-feature vectors receive the same group ID.
groups = pd.factorize(
    pd.MultiIndex.from_frame(X)
)[0]

def evaluate_split(name, train_idx, test_idx):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = len(train_groups & test_groups)

    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)
    print(f"Training rows: {len(train_idx):,}")
    print(f"Test rows: {len(test_idx):,}")
    print(f"Training class counts:\n{y_train.value_counts()}")
    print(f"Test class counts:\n{y_test.value_counts()}")
    print(f"Feature groups appearing in both sets: {overlap:,}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.6f}")
    print("\nConfusion matrix (rows=true, columns=predicted)")
    print("Label order: benign, attack")
    print(confusion_matrix(
        y_test, predictions, labels=["benign", "attack"]
    ))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["benign", "attack"],
        zero_division=0,
    ))

# Test A: reproduce the original random row split.
indices = list(range(len(df)))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)
evaluate_split("A: Random row split", train_idx, test_idx)

# Test B: keep identical feature vectors in one partition.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE,
)
train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)
evaluate_split("B: Grouped feature-vector split", train_idx, test_idx)

print("\nComparison complete. No files were modified.")

