
from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CIC-IDS2017" / "MachineLearningCVE"
PREPARED = ROOT / "data" / "cic_ids2017_prepared.csv"

FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]

# Maximum records sampled from each class in each evaluation file.
MAX_PER_CLASS = 10000

RAW_FEATURES = [
    "Flow Duration",
    "Total Length of Bwd Packets",
    "Total Length of Fwd Packets",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Destination Port",
    "Label",
]

FILES = [
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]


def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", "", regex=False),
        errors="coerce",
    )


def prepare_raw_file(path):
    # CIC-IDS2017 column names can contain leading/trailing spaces.
    header = pd.read_csv(path, nrows=0, low_memory=False)
    actual_names = list(header.columns)
    stripped_to_actual = {
        str(name).strip(): name for name in actual_names
    }

    missing = [c for c in RAW_FEATURES if c not in stripped_to_actual]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    usecols = [stripped_to_actual[c] for c in RAW_FEATURES]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns={
        "Flow Duration": "flow_duration",
        "Total Length of Bwd Packets": "bytes_in",
        "Total Length of Fwd Packets": "bytes_out",
        "Destination Port": "dst_port",
        "Total Fwd Packets": "fwd_packets",
        "Total Backward Packets": "bwd_packets",
    })

    for col in [
        "flow_duration", "bytes_in", "bytes_out",
        "dst_port", "fwd_packets", "bwd_packets",
    ]:
        df[col] = clean_numeric(df[col])

    df["packets"] = df["fwd_packets"] + df["bwd_packets"]

    labels = df["Label"].astype(str).str.strip().str.upper()
    df["target"] = (labels != "BENIGN").astype(int)

    df = df[FEATURES + ["target"]].replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()

    # Limit evaluation size while retaining both classes where available.
    parts = []
    for _, group in df.groupby("target"):
        parts.append(
            group.sample(
                n=min(len(group), MAX_PER_CLASS),
                random_state=42,
            )
        )

    if not parts:
        return df

    return pd.concat(parts).sample(frac=1, random_state=42)


def main():
    print("Loading prepared training dataset...")
    train_df = pd.read_csv(PREPARED, low_memory=False)
    train_df.columns = [str(c).strip() for c in train_df.columns]

    missing = [c for c in FEATURES + ["label"] if c not in train_df.columns]
    if missing:
        raise ValueError(f"Prepared CSV missing columns: {missing}")

    train_df = train_df[FEATURES + ["label"]].copy()
    train_df[FEATURES] = train_df[FEATURES].apply(clean_numeric)
    train_df = train_df.replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()

    X_train = train_df[FEATURES]
    y_train = (
        train_df["label"].astype(str).str.strip().str.upper()
        .map({"BENIGN": 0, "ATTACK": 1})
    )

    valid = y_train.notna()
    X_train = X_train.loc[valid]
    y_train = y_train.loc[valid].astype(int)

    print(f"Training rows: {len(X_train):,}")
    print("Training a temporary Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    results = {}

    for filename in FILES:
        path = DATA_DIR / filename
        print(f"\n{'=' * 72}\n{filename}")

        if not path.exists():
            print("SKIPPED: file not found")
            continue

        try:
            test_df = prepare_raw_file(path)

            if test_df.empty:
                print("SKIPPED: no valid rows")
                continue

            X_test = test_df[FEATURES]
            y_test = test_df["target"].astype(int)
            predictions = model.predict(X_test)

            cm = confusion_matrix(
                y_test, predictions, labels=[0, 1]
            )
            report = classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["BENIGN", "ATTACK"],
                output_dict=True,
                zero_division=0,
            )

            print(f"Evaluated rows: {len(test_df):,}")
            print(f"True class counts: {y_test.value_counts().to_dict()}")
            print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
            print("Confusion matrix [[TN, FP], [FN, TP]]:")
            print(cm)
            print(
                "BENIGN recall: "
                f"{report['BENIGN']['recall']:.4f}"
            )
            print(
                "ATTACK precision: "
                f"{report['ATTACK']['precision']:.4f}"
            )
            print(
                "ATTACK recall: "
                f"{report['ATTACK']['recall']:.4f}"
            )
            print(
                "ATTACK F1: "
                f"{report['ATTACK']['f1-score']:.4f}"
            )

            results[filename] = {
                "rows": int(len(test_df)),
                "class_counts": {
                    str(k): int(v)
                    for k, v in y_test.value_counts().to_dict().items()
                },
                "accuracy": float(accuracy_score(y_test, predictions)),
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
            }

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")

    output = ROOT / "data" / "cross_file_evaluation.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output}")
    print("The existing model artifacts were not modified.")


if __name__ == "__main__":
    main()
