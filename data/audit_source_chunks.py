
from pathlib import Path
import pandas as pd
import numpy as np

SOURCE = (
    Path(__file__).resolve().parent
    / "CIC-IDS2017"
    / "MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

CHUNK_SIZE = 50_000

for chunk_number, chunk in enumerate(
    pd.read_csv(SOURCE, chunksize=CHUNK_SIZE, low_memory=False),
    start=1,
):
    chunk.columns = [str(c).strip() for c in chunk.columns]
    labels = chunk["Label"].astype(str).str.strip().str.upper()

    print(f"\n{'=' * 65}")
    print(f"CHUNK {chunk_number}: {len(chunk):,} rows")
    print("=" * 65)
    print("Original labels:")
    print(labels.value_counts().to_string())

    numeric_columns = {
        "flow_duration": "Flow Duration",
        "bytes_in": "Total Length of Bwd Packets",
        "bytes_out": "Total Length of Fwd Packets",
        "dst_port": "Destination Port",
    }

    for output_name, source_name in numeric_columns.items():
        values = pd.to_numeric(chunk[source_name], errors="coerce")
        print(
            f"{output_name}: "
            f"median={values.median():,.2f}, "
            f"p95={values.quantile(0.95):,.2f}"
        )

    packets = (
        pd.to_numeric(chunk["Total Fwd Packets"], errors="coerce")
        + pd.to_numeric(chunk["Total Backward Packets"], errors="coerce")
    )
    print(
        f"packets: median={packets.median():,.2f}, "
        f"p95={packets.quantile(0.95):,.2f}"
    )

print("\nChunk audit complete. No files were modified.")
