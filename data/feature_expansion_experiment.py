
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from train_multifile_experiment import (
    DATA_DIR,
    OUT_DIR,
    TRAIN_FILES,
)

VALIDATION_FILE = "Tuesday-WorkingHours.pcap_ISCX.csv"
TEST_FILE = "Wednesday-workingHours.pcap_ISCX.csv"

BASE_FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]

EXTRA_MAP = {
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "SYN Flag Count": "syn_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "Flow Packets/s": "flow_packets_s",
    "Flow Bytes/s": "flow_bytes_s",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Packet Length Mean": "packet_length_mean",
    "Packet Length Std": "packet_length_std",
}

EXTRA_FEATURES = list(EXTRA_MAP.values())
ALL_FEATURES = BASE_FEATURES + EXTRA_FEATURES
MAX_PER_CLASS = 30000
TARGET_FPR = 0.01


def load_data(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    header = pd.read_csv(path, nrows=0, low_memory=False)
    original = {str(c).strip(): c for c in header.columns}

    required = [
        "Flow Duration",
        "Total Length of Bwd Packets",
        "Total Length of Fwd Packets",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Destination Port",
        "Label",
        *EXTRA_MAP.keys(),
    ]

    missing = [c for c in required if c not in original]
    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

    df = pd.read_csv(
        path,
        usecols=[original[c] for c in required],
        low_memory=False,
    )
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns={
        "Flow Duration": "flow_duration",
        "Total Length of Bwd Packets": "bytes_in",
        "Total Length of Fwd Packets": "bytes_out",
        "Total Fwd Packets": "fwd_packets",
        "Total Backward Packets": "bwd_packets",
        "Destination Port": "dst_port",
        **EXTRA_MAP,
    })

    labels = df["Label"].astype(str).str.strip()
    df["attack_type"] = labels
    df["target"] = (labels.str.upper() != "BENIGN").astype(int)

    numeric_features = [
        "flow_duration", "bytes_in", "bytes_out",
        "fwd_packets", "bwd_packets", "dst_port",
        *EXTRA_FEATURES,
    ]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["packets"] = df["fwd_packets"] + df["bwd_packets"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["target"])

    # Sample benign and attack rows separately.
    parts = []
    for _, group in df.groupby("target"):
        parts.append(
            group.sample(
                n=min(len(group), MAX_PER_CLASS),
                random_state=42,
            )
        )

    return pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=42
    )


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "attack_recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "attack_precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "attack_f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
    }


def choose_threshold(y_true, scores):
    # Search observed validation scores, never Wednesday scores.
    candidates = np.unique(np.append(scores, 1.000001))
    best = None

    for threshold in candidates:
        result = evaluate(y_true, scores, threshold)

        if result["fpr"] <= TARGET_FPR:
            if best is None or (
                result["attack_recall"],
                result["attack_precision"],
                result["threshold"],
            ) > (
                best["attack_recall"],
                best["attack_precision"],
                best["threshold"],
            ):
                best = result

    return best


def main():
    print("Loading training captures...")
    training_parts = []

    for filename in TRAIN_FILES:
        part = load_data(filename)
        print(f"{filename}: {len(part):,} rows")
        training_parts.append(part)

    train_df = pd.concat(training_parts, ignore_index=True)

    print("\nLoading Tuesday validation data...")
    val_df = load_data(VALIDATION_FILE)

    print("Loading Wednesday held-out test data...")
    test_df = load_data(TEST_FILE)

    results = {}

    experiments = {
        "baseline_5_features": BASE_FEATURES,
        "expanded_15_features": ALL_FEATURES,
    }

    for name, features in experiments.items():
        print(f"\n{'=' * 65}\n{name}\n{'=' * 65}")

        model = make_model()
        model.fit(train_df[features], train_df["target"])

        val_scores = model.predict_proba(
            val_df[features]
        )[:, 1]

        threshold_result = choose_threshold(
            val_df["target"].to_numpy(), val_scores
        )

        if threshold_result is None:
            print("No threshold met the validation FPR target.")
            continue

        threshold = threshold_result["threshold"]

        # Freeze threshold before evaluating Wednesday.
        test_scores = model.predict_proba(
            test_df[features]
        )[:, 1]

        test_result = evaluate(
            test_df["target"].to_numpy(),
            test_scores,
            threshold,
        )

        print("\nTuesday validation:")
        print(json.dumps(threshold_result, indent=2))

        print("\nWednesday held-out test:")
        print(json.dumps(test_result, indent=2))

        results[name] = {
            "features": features,
            "training_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "validation_metrics": threshold_result,
            "test_metrics": test_result,
        }

        # Experimental artifact only; never overwrite production.
        model_path = OUT_DIR / f"{name}_experiment.joblib"
        joblib.dump({
            "model_type": name,
            "features": features,
            "estimator": model,
            "threshold": threshold,
            "training_files": TRAIN_FILES,
            "validation_file": VALIDATION_FILE,
            "test_file": TEST_FILE,
        }, model_path)

        print(f"Experimental model saved: {model_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "feature_expansion_results.json"
    report_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved: {report_path}")
    print("Production model was not modified.")


if __name__ == "__main__":
    main()
