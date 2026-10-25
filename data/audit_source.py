
from pathlib import Path
import pandas as pd

SOURCE = (
    Path(__file__).resolve().parent
    / "CIC-IDS2017"
    / "MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

CHUNK_SIZE = 50_000

if not SOURCE.is_file():
    raise FileNotFoundError(f"Source CSV not found: {SOURCE}")

total_rows = 0
label_counts = {}
chunk_number = 0

print(f"Auditing: {SOURCE}")
print("Reading the complete source CSV in chunks...")

for chunk in pd.read_csv(SOURCE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_number += 1
    total_rows += len(chunk)

    chunk.columns = [str(c).strip() for c in chunk.columns]

    if "Label" not in chunk.columns:
        raise ValueError("Source CSV does not contain a Label column.")

    labels = chunk["Label"].astype(str).str.strip().str.upper()
    counts = labels.value_counts(dropna=False)

    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + int(count)

    print(
        f"Chunk {chunk_number}: {len(chunk):,} rows; "
        f"cumulative rows: {total_rows:,}"
    )

print("\n=== COMPLETE SOURCE SUMMARY ===")
print(f"Total rows: {total_rows:,}")
print(f"Chunks read: {chunk_number}")
print("\nOriginal label counts:")
for label, count in sorted(label_counts.items()):
    print(f"{label}: {count:,}")

print("\nAudit complete. No files were modified.")
