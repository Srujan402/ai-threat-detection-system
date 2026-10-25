
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "cic_ids2017_prepared.csv"

FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]
LABEL = "label"

df = pd.read_csv(DATA_PATH)

required = FEATURES + [LABEL]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("\n=== DATASET OVERVIEW ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Missing values in required columns:\n{df[required].isna().sum()}")
print(f"\nLabel counts:\n{df[LABEL].value_counts(dropna=False)}")

print("\n=== EXACT DUPLICATE ROWS ===")
exact_mask = df.duplicated(keep=False)
print(f"Rows belonging to duplicate groups: {exact_mask.sum():,}")
print(f"Duplicate rows after first occurrence: {df.duplicated().sum():,}")

print("\n=== DUPLICATE FEATURE VECTORS ===")
feature_mask = df.duplicated(subset=FEATURES, keep=False)
print(f"Rows sharing a feature vector: {feature_mask.sum():,}")
print(
    "Repeated feature vectors after first occurrence: "
    f"{df.duplicated(subset=FEATURES).sum():,}"
)

print("\n=== FEATURE VECTORS WITH CONFLICTING LABELS ===")
label_counts_per_vector = (
    df.groupby(FEATURES, dropna=False)[LABEL]
    .nunique()
)
conflicting_vectors = label_counts_per_vector[
    label_counts_per_vector > 1
]
print(f"Conflicting feature vectors: {len(conflicting_vectors):,}")

print("\n=== DUPLICATE FEATURES ACROSS LABELS ===")
if not conflicting_vectors.empty:
    examples = (
        df.merge(
            conflicting_vectors.rename("label_count"),
            left_on=FEATURES,
            right_index=True,
        )
        .sort_values(FEATURES)
        .head(10)
    )
    print(examples[FEATURES + [LABEL]].to_string(index=False))
else:
    print("No conflicting feature vectors found.")

print("\n=== FEATURE CARDINALITY ===")
for col in FEATURES:
    print(f"{col}: {df[col].nunique(dropna=False):,} unique values")

print("\n=== NUMERIC FEATURE SUMMARY ===")
print(df[FEATURES].describe().to_string())

print("\nAudit complete. No model or dataset files were modified.")
