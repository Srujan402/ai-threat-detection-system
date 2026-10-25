
from pathlib import Path
import pandas as pd

DATA_DIR = (
    Path(__file__).resolve().parent
    / "CIC-IDS2017"
    / "MachineLearningCVE"
)

for file_path in sorted(DATA_DIR.glob("*.csv")):
    print(f"\n{'=' * 70}")
    print(file_path.name)

    try:
        # Inspect the original header and find the label column.
        header = pd.read_csv(file_path, nrows=0, low_memory=False)
        columns = [str(c).strip() for c in header.columns]

        label_index = next(
            (i for i, name in enumerate(columns)
             if name.lower() == "label"),
            None,
        )

        if label_index is None:
            print("ERROR: Label column not found.")
            print("Last 10 columns:", columns[-10:])
            continue

        original_label_column = header.columns[label_index]
        counts = {}
        total_rows = 0

        for chunk in pd.read_csv(
            file_path,
            usecols=[original_label_column],
            chunksize=50000,
            low_memory=False,
        ):
            labels = (
                chunk[original_label_column]
                .astype(str)
                .str.strip()
            )

            total_rows += len(labels)

            for label, count in labels.value_counts().items():
                counts[label] = counts.get(label, 0) + int(count)

        print(f"Total rows: {total_rows:,}")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count:,}")

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
