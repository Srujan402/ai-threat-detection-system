# AI Threat Detection System

A Django-based cybersecurity portfolio project that combines machine-learning classification with rule-based detection to identify suspicious network activity and authentication events.

## Project Objectives

- Ingest network and system security events from CSV files.
- Identify suspicious network traffic using a trained Random Forest classifier.
- Detect potential port scanning, high outbound traffic, and repeated failed logins.
- Generate alerts with severity, evidence, and source information.
- Provide a dashboard and alert filters to support analyst triage.

## Technology Stack

- Python
- Django
- SQLite for local development
- pandas and joblib
- scikit-learn
- Bootstrap
- CIC-IDS2017 dataset for model experimentation

## Detection Capabilities

| Detection | Method |
|---|---|
| Suspicious network traffic | Machine-learning classifier |
| High outbound traffic | Threshold-based rule |
| Potential port scan | Distinct destination-port threshold |
| Repeated failed logins | Authentication-event aggregation |
| Alert triage | Severity, status, and detection-rule filters |

## Project Structure

```text
ai-threat-protection/
+-- config/
�   +-- artifacts/
�   �   +-- evaluation.json
�   +-- detector/
�   +-- config/
�   +-- manage.py
�   +-- requirements.txt
+-- data/
�   +-- prepare_cic.py
+-- detector/
�   +-- management/
+-- .gitignore
+-- README.md
```

The exact structure may evolve during development. Large datasets, local databases, virtual environments, and trained model binaries are excluded from version control.

## Local Setup

1. Install a compatible Python version.
2. Create and activate a virtual environment.
3. Install dependencies from the project requirements file.
4. Place the required trained model artifact at `config/artifacts/network_model.joblib`.
5. Run database migrations.
6. Start the Django development server.

Example commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\config\requirements.txt
python .\config\manage.py migrate
python .\config\manage.py runserver
```

The model artifact and dataset are not included in Git by default. The model must be generated using the project's training workflow or supplied separately before ML inference can work.

## Automated Tests

Run the detection tests:

```powershell
python .\config\manage.py test detector.tests --verbosity 2
```

The current detection test suite contains five tests covering high outbound traffic, ML alert evidence, port-scan thresholds, and repeated failed logins.

## Model Evaluation

The classifier was trained using a prepared subset of CIC-IDS2017. Reported metrics from a random train/test split require further validation before being interpreted as evidence of real-world detection performance.

Potential data leakage, duplicate flows, feature mapping, class balance, false positives, and false negatives should be investigated. Performance should also be tested on independent or appropriately grouped data.

## Limitations

- This is a learning and portfolio project, not a production security monitoring platform.
- Detection aggregation currently depends on the events supplied in an upload batch.
- The classifier's output is a signal for investigation, not proof of malicious activity.
- Dataset coverage and generalization to unseen traffic require further evaluation.
- Authentication, deployment security, monitoring, and operational hardening require additional work before production use.

## Future Improvements

- Add rolling time-window detection.
- Add alert deduplication and correlation.
- Improve analyst investigation notes and alert lifecycle handling.
- Validate model performance on independent data.
- Add reproducible training and evaluation scripts.
- Add dashboard screenshots and a safe sample dataset for demonstration.

## Disclaimer

Developed for educational and defensive cybersecurity purposes. Test only with systems and data you are authorized to use.
## Application Screenshots

### Security Dashboard
![Security Dashboard](docs/screenshots/dashboard.png)

### Security Alerts
![Security Alerts](docs/screenshots/alerts.png)

### Alert Details
![Alert Details](docs/screenshots/alert-details.png)
