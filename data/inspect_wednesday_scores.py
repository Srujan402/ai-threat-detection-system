
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CIC-IDS2017" / "MachineLearningCVE"
MODEL_PATH = (
    ROOT / "config" / "artifacts"
    / "network_model_multifile_experiment.joblib"
)

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
    bundle = joblib.load(MODEL_PATH)
    model = bundle["estimator"]

    path = DATA_DIR / "Wednesday-workingHours.pcap_ISCX.csv"
    header = pd.read_csv(path, nrows=0, low_memory=False)
    actual_columns = {str(c).strip(): c for c in header.columns}

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

    df["actual_attack"] = (
        df["attack_type"].str.upper() != "BENIGN"
    ).astype(int)

    # Probability assigned to the ATTACK class.
    scores = model.predict_proba(df[FEATURES])[:, 1]
    df["attack_score"] = scores

    print("\nAttack-score distribution by actual label:")
    print(
        df.groupby("attack_type")["attack_score"]
        .agg(["count", "min", "median", "mean", "max"])
        .round(4)
        .to_string()
    )

    print("\nThreshold experiment on the same held-out file:")
    print("Threshold | Attack recall | False positives | Missed attacks")

    y_true = df["actual_attack"].to_numpy()

    for threshold in [0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01]:
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        recall = tp / (tp + fn) if (tp + fn) else 0.0

        print(
            f"{threshold:9.2f} | {recall:13.4f} | "
            f"{fp:15,} | {fn:14,}"
        )


if __name__ == "__main__":
    main()
