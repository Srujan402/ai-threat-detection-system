
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "CIC-IDS2017"
    / "MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

OUTPUT_FILE = BASE_DIR / "cic_ids2017_prepared.csv"

MAX_PER_CLASS = 50000
FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]

benign_parts = []
attack_parts = []
benign_count = 0
attack_count = 0

print("Reading CIC-IDS2017 in chunks...")

for chunk in pd.read_csv(INPUT_FILE, chunksize=50000, low_memory=False):
    chunk.columns = [str(c).strip() for c in chunk.columns]

    required = [
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Label",
    ]

    missing = [c for c in required if c not in chunk.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    labels = chunk["Label"].astype(str).str.strip().str.upper()

    prepared = pd.DataFrame({
        "flow_duration": pd.to_numeric(
            chunk["Flow Duration"], errors="coerce"
        ),
        "bytes_in": pd.to_numeric(
            chunk["Total Length of Bwd Packets"], errors="coerce"
        ),
        "bytes_out": pd.to_numeric(
            chunk["Total Length of Fwd Packets"], errors="coerce"
        ),
        "packets": (
            pd.to_numeric(chunk["Total Fwd Packets"], errors="coerce")
            + pd.to_numeric(
                chunk["Total Backward Packets"], errors="coerce"
            )
        ),
        "dst_port": pd.to_numeric(
            chunk["Destination Port"], errors="coerce"
        ),
        "label": np.where(labels == "BENIGN", "BENIGN", "ATTACK"),
    })

    prepared.replace([np.inf, -np.inf], np.nan, inplace=True)
    prepared.dropna(subset=FEATURES, inplace=True)

    # Keep only valid, non-negative network measurements.
    prepared = prepared[(prepared[FEATURES] >= 0).all(axis=1)]

    benign = prepared[prepared["label"] == "BENIGN"]
    attack = prepared[prepared["label"] == "ATTACK"]

    if benign_count < MAX_PER_CLASS and not benign.empty:
        take = benign.head(MAX_PER_CLASS - benign_count)
        benign_parts.append(take)
        benign_count += len(take)

    if attack_count < MAX_PER_CLASS and not attack.empty:
        take = attack.head(MAX_PER_CLASS - attack_count)
        attack_parts.append(take)
        attack_count += len(take)

    print(f"Collected BENIGN={benign_count}, ATTACK={attack_count}")

    if benign_count >= MAX_PER_CLASS and attack_count >= MAX_PER_CLASS:
        break

if benign_count < 2 or attack_count < 2:
    raise ValueError(
        "Not enough examples of both classes. Check the source CSV labels."
    )

result = pd.concat(benign_parts + attack_parts, ignore_index=True)
result = result.sample(frac=1, random_state=42).reset_index(drop=True)
result.to_csv(OUTPUT_FILE, index=False)

print("\nPreparation complete.")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Rows: {len(result)}")
print("\nLabel counts:")
print(result["label"].value_counts().to_string())
print("\nColumns:")
print(result.columns.tolist())
