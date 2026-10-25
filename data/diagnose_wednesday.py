
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CIC-IDS2017" / "MachineLearningCVE"
MODEL_PATH = (
    ROOT / "config" / "artifacts"
    / "network_model_multifile_experiment.joblib"
)

TEST_FILE = "Wednesday-workingHours.pcap_ISCX.csv"

FEATURE_MAP = {
    "Flow Duration": "flow_duration",
    "Total Length of Bwd Packets": "bytes_in",
    "Total Length of Fwd Packets": "bytes_out",
    "Total Fwd Packets": "fwd_packets",
    "Total Backward Packets": "bwd_packets",
    "Destination Port": "dst_port",
}

FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]


def main():
    print("Loading experimental model...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["estimator"]

    path = DATA_DIR / TEST_FILE
    header = pd.read_csv(path, nrows=0, low_memory=False)
    actual_columns = {
        str(c).strip(): c for c in header.columns
    }

    required = list(FEATURE_MAP) + ["Label"]
    missing = [c for c in required if c not in actual_columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = pd.read_csv(
        path,
        usecols=[actual_columns[c] for c in required],
        low_memory=False,
    )
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=FEATURE_MAP)

    for col in [
        "flow_duration", "bytes_in", "bytes_out",
        "fwd_packets", "bwd_packets", "dst_port",
    ]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(
                ",", "", regex=False
            ),
            errors="coerce",
        )

    df["packets"] = df["fwd_packets"] + df["bwd_packets"]
    df["attack_type"] = df["Label"].astype(str).str.strip()

    df = df.replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna(subset=FEATURES + ["attack_type"])

    # Use all valid rows to measure category-level performance.
    df["actual_attack"] = (
        df["attack_type"].str.upper() != "BENIGN"
    ).astype(int)

    predictions = model.predict(df[FEATURES])
    df["predicted_attack"] = predictions.astype(int)

    print(f"\nValid rows evaluated: {len(df):,}")
    print("\nResults by original label:")

    rows = []
    for label, group in df.groupby("attack_type"):
        y_true = group["actual_attack"]
        y_pred = group["predicted_attack"]

        rows.append({
            "attack_type": label,
            "records": len(group),
            "detected": int(((y_true == 1) & (y_pred == 1)).sum()),
            "missed": int(((y_true == 1) & (y_pred == 0)).sum()),
            "false_alarms": int(((y_true == 0) & (y_pred == 1)).sum()),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    result = pd.DataFrame(rows).sort_values(
        ["records", "attack_type"],
        ascending=[False, True],
    )

    print(result.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
        },
    ))


if __name__ == "__main__":
    main()
