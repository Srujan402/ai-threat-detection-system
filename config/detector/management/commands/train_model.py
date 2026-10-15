
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]


class Command(BaseCommand):
    help = "Train and evaluate a supervised network threat classifier."

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            required=True,
            help="Path to a prepared CSV containing network features and labels.",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"]).expanduser()

        if not csv_path.is_absolute():
            csv_path = Path(settings.BASE_DIR) / csv_path

        if not csv_path.is_file():
            raise CommandError(f"CSV file not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            raise CommandError(f"Could not read CSV: {exc}") from exc

        df.columns = [str(c).strip().lower() for c in df.columns]

        required = set(FEATURES + ["label"])
        missing = required - set(df.columns)

        if missing:
            raise CommandError(
                "Missing required columns: " + ", ".join(sorted(missing))
            )

        for feature in FEATURES:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")

        df = df.replace([float("inf"), float("-inf")], float("nan"))
        df = df.dropna(subset=FEATURES + ["label"]).copy()

        labels = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "benign": "benign",
                "normal": "benign",
                "0": "benign",
                "attack": "attack",
                "malicious": "attack",
                "anomaly": "attack",
                "1": "attack",
            })
        )

        valid = labels.isin(["benign", "attack"])
        df = df.loc[valid].copy()
        labels = labels.loc[valid]

        if len(df) < 30 or labels.nunique() != 2:
            raise CommandError(
                "Need at least 30 valid rows containing both benign and attack labels."
            )

        if labels.value_counts().min() < 2:
            raise CommandError(
                "Each class must contain at least two rows."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            df[FEATURES],
            labels,
            test_size=0.25,
            random_state=42,
            stratify=labels,
        )

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        report = classification_report(
            y_test,
            predictions,
            labels=["benign", "attack"],
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=["benign", "attack"],
        )

        artifact_dir = Path(settings.BASE_DIR) / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifact_dir / "network_model.joblib"
        report_path = artifact_dir / "evaluation.json"

        # Save a dictionary so inference code can identify the model type.
        joblib.dump(
            {
                "model_type": "random_forest",
                "features": FEATURES,
                "estimator": model,
            },
            model_path,
        )

        results = {
            "dataset": csv_path.name,
            "model_type": "RandomForestClassifier",
            "rows_used": int(len(df)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": FEATURES,
            "test_class_counts": {
                str(key): int(value)
                for key, value in y_test.value_counts().items()
            },
            "confusion_matrix_labels": ["benign", "attack"],
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
            "warning": (
                "Experimental results on the prepared CIC-IDS2017 subset. "
                "The subset uses five mapped features and is not a production "
                "validation. Results may not generalize to other networks."
            ),
        }

        report_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        self.stdout.write(self.style.SUCCESS(f"Model saved: {model_path}"))
        self.stdout.write(self.style.SUCCESS(f"Evaluation saved: {report_path}"))
        self.stdout.write(
            "\nEvaluation summary:\n"
            + classification_report(
                y_test,
                predictions,
                labels=["benign", "attack"],
                zero_division=0,
            )
        )
