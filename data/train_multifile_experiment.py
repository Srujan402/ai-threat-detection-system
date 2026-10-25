
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
OUT_DIR = ROOT / "config" / "artifacts"

# Keep the Wednesday capture completely out of training.
TEST_FILE = "Wednesday-workingHours.pcap_ISCX.csv"


TRAIN_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]


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

MAX_PER_CLASS_PER_FILE = 30000


def load_capture(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    header = pd.read_csv(path, nrows=0, low_memory=False)
    original_columns = {
        str(column).strip(): column for column in header.columns
    }

    required = list(FEATURE_MAP) + ["Label"]
    missing = [name for name in required if name not in original_columns]
    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

    df = pd.read_csv(
        path,
        usecols=[original_columns[name] for name in required],
        low_memory=False,
    )
    df.columns = [str(column).strip() for column in df.columns]
    df = df.rename(columns=FEATURE_MAP)

    for column in [
        "flow_duration", "bytes_in", "bytes_out",
        "fwd_packets", "bwd_packets", "dst_port",
    ]:
        df[column] = pd.to_numeric(
            df[column].astype(str).str.strip().str.replace(
                ",", "", regex=False
            ),
            errors="coerce",
        )

    original_labels = df["Label"].astype(str).str.strip()
    df["attack_type"] = original_labels
    df["target"] = (
        original_labels.str.upper() != "BENIGN"
    ).astype(int)

    df["packets"] = df["fwd_packets"] + df["bwd_packets"]
    df = df[FEATURES + ["target", "attack_type"]]
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=FEATURES + ["target"])

    # Sample each class separately so benign-heavy files remain usable.
    sampled = []
    for _, group in df.groupby("target"):
        sampled.append(
            group.sample(
                n=min(len(group), MAX_PER_CLASS_PER_FILE),
                random_state=42,
            )
        )

    if not sampled:
        return df.iloc[0:0]

    return pd.concat(sampled).sample(frac=1, random_state=42)


def main():
    print("Loading training captures...")
    train_parts = []

    for filename in TRAIN_FILES:
        part = load_capture(filename)
        print(f"{filename}: {len(part):,} sampled rows")
        train_parts.append(part)

    train_df = pd.concat(train_parts, ignore_index=True)

    X_train = train_df[FEATURES]
    y_train = train_df["target"].astype(int)

    print("\nTraining class counts:")
    print(y_train.value_counts().rename(index={0: "BENIGN", 1: "ATTACK"}))

    print(f"\nLoading held-out capture: {TEST_FILE}")
    test_df = load_capture(TEST_FILE)

    X_test = test_df[FEATURES]
    y_test = test_df["target"].astype(int)

    print("\nHeld-out class counts:")
    print(y_test.value_counts().rename(index={0: "BENIGN", 1: "ATTACK"}))

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining experimental model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )

    print("\nHeld-out results")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "network_model_multifile_experiment.joblib"
    report_path = OUT_DIR / "multifile_experiment_evaluation.json"

    # Separate filenames protect the currently deployed model.
    import joblib

    joblib.dump(
        {
            "model_type": "random_forest_multifile_experiment",
            "features": FEATURES,
            "estimator": model,
            "training_files": TRAIN_FILES,
            "held_out_file": TEST_FILE,
        },
        model_path,
    )

    results = {
        "training_files": TRAIN_FILES,
        "held_out_file": TEST_FILE,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "held_out_attack_labels": {
            str(k): int(v)
            for k, v in test_df.loc[
                test_df["target"] == 1, "attack_type"
            ].value_counts().items()
        },
    }

    report_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nExperimental model saved: {model_path}")
    print(f"Evaluation saved: {report_path}")
    print("Existing network_model.joblib was not modified.")


if __name__ == "__main__":
    main()
