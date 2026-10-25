
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# Import existing project configuration and helpers
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "data") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "data"))

from train_multifile_experiment import (
    FEATURES,
    TRAIN_FILES,
    load_capture,
    DATA_DIR,
    OUT_DIR,
)

ARTIFACT_DIR = PROJECT_ROOT / "config" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "network_model_threshold_experiment.joblib"
REPORT_PATH = ARTIFACT_DIR / "threshold_experiment_results.json"

VALIDATION_FILE = (
    "Tuesday-WorkingHours.pcap_ISCX.csv"
)

TEST_FILE = (
    "Wednesday-workingHours.pcap_ISCX.csv"
)

MAX_ROWS_PER_CLASS = 30000
RANDOM_STATE = 42
TARGET_FPR = 0.01


# --------------------------------------------------
# Evaluation helpers
# --------------------------------------------------

def evaluate(y_true, scores, threshold):
    """Evaluate predictions using the specified score threshold."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    attack_recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fpr),
        "attack_recall": float(attack_recall),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "attack_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "attack_f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }


def print_result(result):
    print(
        f"Threshold={result['threshold']:.2f} | "
        f"FPR={result['false_positive_rate']:.2%} | "
        f"Attack recall={result['attack_recall']:.2%} | "
        f"Precision={result['attack_precision']:.2%} | "
        f"F1={result['attack_f1']:.4f} | "
        f"TP={result['tp']:,} | "
        f"FP={result['fp']:,} | "
        f"FN={result['fn']:,} | "
        f"TN={result['tn']:,}"
    )


# --------------------------------------------------
# Prepare capture data
# --------------------------------------------------



def prepare_capture(filename, max_rows_per_class=None):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = load_capture(path)

    # Normalize column names.
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    # The existing loader may already return prepared data.
    prepared_features = [
        "flow_duration",
        "bytes_in",
        "bytes_out",
        "packets",
        "dst_port",
    ]

    missing_features = [
        column for column in FEATURES
        if column.lower() not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in {filename}: {missing_features}. "
            f"Columns found: {df.columns.tolist()}"
        )

    # Accept either prepared or original CIC-IDS2017 labels.
    if "target" in df.columns:
        y = pd.to_numeric(df["target"], errors="coerce")
        labels = (
            df["attack_type"].astype(str).str.strip()
            if "attack_type" in df.columns
            else y.map({0: "BENIGN", 1: "ATTACK"}).astype(str)
        )

    elif "label" in df.columns:
        labels = df["label"].astype(str).str.strip()
        y = (~labels.str.upper().eq("BENIGN")).astype(int)

    else:
        raise ValueError(
            f"No target, Label, or label column in {filename}. "
            f"Columns found: {df.columns.tolist()}"
        )

    # Use the exact feature names expected by the model.
    feature_map = {
        feature: feature.lower()
        for feature in FEATURES
    }

    X = df[
        [feature_map[feature] for feature in FEATURES]
    ].copy()

    X.columns = FEATURES

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove invalid features and labels.
    y = pd.to_numeric(y, errors="coerce")
    valid = X.notna().all(axis=1) & y.isin([0, 1])

    X = X.loc[valid]
    y = y.loc[valid].astype(int)
    labels = labels.loc[valid]

    # Sample each class reproducibly.
    if max_rows_per_class is not None:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = []

        for class_value in (0, 1):
            indices = y.index[y == class_value].to_numpy()

            if len(indices) > max_rows_per_class:
                indices = rng.choice(
                    indices,
                    size=max_rows_per_class,
                    replace=False,
                )

            selected.extend(indices.tolist())

        X = X.loc[selected]
        y = y.loc[selected]
        labels = labels.loc[selected]

    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        labels.reset_index(drop=True),
    )



# --------------------------------------------------
# Load training sources and train temporary model
# --------------------------------------------------

def main():
    print("=" * 65)
    print("AI THREAT DETECTION - THRESHOLD VALIDATION EXPERIMENT")
    print("=" * 65)

    X_train_parts = []
    y_train_parts = []

    print("\nLoading training captures...")

    for filename in TRAIN_FILES:
        X_part, y_part, _ = prepare_capture(
            filename,
            max_rows_per_class=MAX_ROWS_PER_CLASS,
        )

        X_train_parts.append(X_part)
        y_train_parts.append(y_part)

        print(
            f"Training source {filename}: "
            f"{len(X_part):,} rows | "
            f"BENIGN={(y_part == 0).sum():,} | "
            f"ATTACK={(y_part == 1).sum():,}"
        )

    X_train = pd.concat(
        X_train_parts,
        ignore_index=True,
    )
    y_train = pd.concat(
        y_train_parts,
        ignore_index=True,
    )

    print("\nCombined training data:")
    print(f"Rows: {len(X_train):,}")
    print(f"BENIGN: {(y_train == 0).sum():,}")
    print(f"ATTACK: {(y_train == 1).sum():,}")

    print("\nTraining temporary Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Temporary model saved to: {MODEL_PATH}")

    # --------------------------------------------------
    # Tuesday validation - not used for training
    # --------------------------------------------------

    print("\nLoading Tuesday validation capture...")

    X_val, y_val, original_labels = prepare_capture(
        VALIDATION_FILE,
        max_rows_per_class=MAX_ROWS_PER_CLASS,
    )

    validation_scores = model.predict_proba(X_val)[:, 1]

    print(f"Validation rows: {len(X_val):,}")
    print(f"BENIGN: {(y_val == 0).sum():,}")
    print(f"ATTACK: {(y_val == 1).sum():,}")

    # --------------------------------------------------
    # Diagnose scores by original attack family
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("TUESDAY SCORE DIAGNOSTICS BY ORIGINAL LABEL")
    print("=" * 65)

    diagnostic_df = pd.DataFrame({
        "original_label": original_labels,
        "binary_label": y_val,
        "score": validation_scores,
    })

    for label, group in diagnostic_df.groupby("original_label"):
        print(f"\nLabel: {label}")
        print(f"Rows: {len(group):,}")
        print(
            f"Mean score: {group['score'].mean():.4f}"
        )
        print(
            f"Median score: {group['score'].median():.4f}"
        )

        for threshold in [0.01, 0.05, 0.10, 0.30, 0.50, 0.90, 0.99]:
            detection_rate = (
                group["score"] >= threshold
            ).mean()

            print(
                f"Score >= {threshold:.2f}: "
                f"{detection_rate:.2%}"
            )

    # --------------------------------------------------
    # Select threshold using Tuesday only
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("THRESHOLD SELECTION ON TUESDAY")
    print("=" * 65)

    # Include observed scores so that the search is not
    # limited to arbitrary increments such as 0.01.
    candidates_to_test = np.unique(
        np.concatenate([
            validation_scores,
            np.array([0.0, 1.0]),
        ])
    )

    candidates = []

    for threshold in candidates_to_test:
        result = evaluate(
            y_val,
            validation_scores,
            threshold,
        )

        if result["false_positive_rate"] <= TARGET_FPR:
            candidates.append(result)

    if not candidates:
        print(
            "\nNo threshold met the target FPR. "
            "This should not normally happen because the maximum "
            "observed score can produce zero false positives."
        )
        return

    # Maximize attack recall while meeting the FPR target.
    # If recall ties, prefer lower FPR; then higher threshold.
    best = max(
        candidates,
        key=lambda item: (
            item["attack_recall"],
            -item["false_positive_rate"],
            item["threshold"],
        ),
    )

    print("\nSelected validation threshold:")
    print_result(best)

    print("\nSelected threshold is frozen before Wednesday testing.")

    # --------------------------------------------------
    # Evaluate once on held-out Wednesday capture
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL HELD-OUT TEST: WEDNESDAY")
    print("=" * 65)

    X_test, y_test, test_labels = prepare_capture(
        TEST_FILE,
        max_rows_per_class=None,
    )

    test_scores = model.predict_proba(X_test)[:, 1]

    test_result = evaluate(
        y_test,
        test_scores,
        best["threshold"],
    )

    print(f"Test rows: {len(X_test):,}")
    print_result(test_result)

    # Report performance by original label as well.
    print("\nWednesday results by original label:")

    test_df = pd.DataFrame({
        "original_label": test_labels,
        "binary_label": y_test,
        "score": test_scores,
    })

    for label, group in test_df.groupby("original_label"):
        predicted_attack = (
            group["score"] >= best["threshold"]
        )

        if (group["binary_label"] == 1).any():
            attack_recall = predicted_attack.mean()
            print(
                f"{label}: rows={len(group):,}, "
                f"detected={int(predicted_attack.sum()):,}, "
                f"missed={int((~predicted_attack).sum()):,}, "
                f"detection_rate={attack_recall:.2%}"
            )
        else:
            print(
                f"{label}: rows={len(group):,}, "
                f"false_alarms={int(predicted_attack.sum()):,}, "
                f"false_positive_rate={predicted_attack.mean():.2%}"
            )

    # --------------------------------------------------
    # Save experiment report
    # --------------------------------------------------

    report = {
        "experiment": "multifile_threshold_validation",
        "features": list(FEATURES),
        "training_sources": list(TRAIN_FILES),
        "validation_source": VALIDATION_FILE,
        "test_source": TEST_FILE,
        "target_false_positive_rate": TARGET_FPR,
        "selected_threshold": best["threshold"],
        "validation_metrics": best,
        "test_metrics": test_result,
        "warning": (
            "This is an experimental evaluation, not a guarantee "
            "of production performance. The chosen features may "
            "not generalize to unseen attack families."
        ),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nExperiment report saved to: {REPORT_PATH}")
    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
