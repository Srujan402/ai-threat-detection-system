import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_expansion_experiment import (
    BASE_FEATURES,
    ALL_FEATURES,
    DATA_DIR,
    OUT_DIR,
    TRAIN_FILES,
    VALIDATION_FILE,
    TEST_FILE,
    TARGET_FPR,
    load_data,
    make_model,
    evaluate,
    choose_threshold,
)

RATE_FEATURES = ["flow_packets_s", "flow_bytes_s"]
NO_RATE_FEATURES = [f for f in ALL_FEATURES if f not in RATE_FEATURES]


def clean_features(df, features):
    """Replace infinite values with missing values; imputer handles them."""
    X = df[features].copy()
    return X.replace([np.inf, -np.inf], np.nan)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading training captures...")
    train_df = pd.concat(
        [load_data(filename) for filename in TRAIN_FILES],
        ignore_index=True,
    )
    print(f"Training rows: {len(train_df):,}")

    print("Loading Tuesday validation capture...")
    val_df = load_data(VALIDATION_FILE)

    print("Loading Wednesday held-out test capture...")
    test_df = load_data(TEST_FILE)

    # Compute rate-feature clipping limits using training data only.
    train_rates = clean_features(train_df, RATE_FEATURES)
    rate_caps = {
        feature: float(train_rates[feature].quantile(0.999))
        for feature in RATE_FEATURES
    }
    print("\nTraining-only 99.9th-percentile rate caps:")
    print(json.dumps(rate_caps, indent=2))

    experiments = {
        "expanded_15_sanitized": {
            "features": ALL_FEATURES,
            "cap_rates": False,
        },
        "expanded_15_capped": {
            "features": ALL_FEATURES,
            "cap_rates": True,
        },
        "expanded_13_without_rates": {
            "features": NO_RATE_FEATURES,
            "cap_rates": False,
        },
    }

    results = {}

    for name, config in experiments.items():
        features = config["features"]
        cap_rates = config["cap_rates"]

        print("\n" + "=" * 72)
        print(name)
        print("=" * 72)

        def prepare(df):
            X = clean_features(df, features)
            if cap_rates:
                for feature, upper in rate_caps.items():
                    if feature in X.columns:
                        X[feature] = X[feature].clip(upper=upper)
            return X

        X_train = prepare(train_df)
        X_val = prepare(val_df)
        X_test = prepare(test_df)

        model = make_model()
        model.fit(X_train, train_df["target"])

        val_scores = model.predict_proba(X_val)[:, 1]
        selected = choose_threshold(val_df["target"].to_numpy(), val_scores)

        if selected is None:
            print("No threshold met the validation FPR target.")
            continue

        threshold = selected["threshold"]
        test_scores = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(
            test_df["target"].to_numpy(),
            test_scores,
            threshold,
        )

        print("\nTuesday validation metrics:")
        print(json.dumps(selected, indent=2))
        print("\nWednesday held-out metrics:")
        print(json.dumps(test_metrics, indent=2))

        results[name] = {
            "features": features,
            "cap_rates": cap_rates,
            "rate_caps": rate_caps if cap_rates else None,
            "threshold": threshold,
            "validation_metrics": selected,
            "test_metrics": test_metrics,
        }

        artifact = {
            "model_type": name,
            "features": features,
            "estimator": model,
            "threshold": threshold,
            "cap_rates": cap_rates,
            "rate_caps": rate_caps if cap_rates else None,
            "training_files": TRAIN_FILES,
            "validation_file": VALIDATION_FILE,
            "test_file": TEST_FILE,
            "note": (
                "Experimental only. Infinite values are converted to NaN. "
                "When cap_rates is true, rate caps are training-derived."
            ),
        }

        artifact_path = OUT_DIR / f"{name}_experiment.joblib"
        joblib.dump(artifact, artifact_path)
        print(f"Experimental artifact saved: {artifact_path}")

    report_path = OUT_DIR / "rate_feature_experiment_results.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults report saved: {report_path}")

    print("\nSUMMARY")
    for name, result in results.items():
        val = result["validation_metrics"]
        test = result["test_metrics"]
        print(
            f"{name}: threshold={result['threshold']:.6f} | "
            f"Tuesday recall={val['attack_recall']:.4%}, "
            f"FPR={val['fpr']:.4%} | "
            f"Wednesday recall={test['attack_recall']:.4%}, "
            f"FPR={test['fpr']:.4%}"
        )


if __name__ == "__main__":
    main()
