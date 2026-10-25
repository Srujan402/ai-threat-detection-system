# Experimental Model Evaluation

## Dataset and methodology

The experiments use sampled CIC-IDS2017 network-flow CSV captures.

- Training captures: Friday DDoS, Friday PortScan, Friday Morning (Bot), and Thursday WebAttacks.
- Validation capture: Tuesday Working Hours.
- Held-out test capture: Wednesday Working Hours.
- Sampling: Up to 30,000 benign and 30,000 attack rows per capture.
- Thresholds are selected using validation data, subject to a target false-positive rate of at most 1%.

## Results

| Model | Tuesday Recall | Tuesday FPR | Wednesday Recall | Wednesday FPR | Wednesday Precision | Wednesday F1 |
|---|---:|---:|---:|---:|---:|---:|
| 15-feature sanitized | 14.53% | 0.990% | 72.41% | 0.533% | 99.27% | 0.8374 |
| 15-feature capped | 14.04% | 0.990% | 71.98% | 0.537% | 99.26% | 0.8344 |
| 13-feature without rates | 5.27% | 0.980% | 74.01% | 0.473% | 99.36% | 0.8483 |

## Interpretation

The 13-feature model produced the strongest Wednesday metrics among these three experiments. However, its Tuesday attack recall was only 5.27%, indicating substantial variation across captures.

The capped-rate experiment did not improve the measured metrics over the uncapped 15-feature experiment.

The sampled test data has an artificially balanced benign/attack distribution. Consequently, measured precision must not be interpreted as expected real-world SOC alert precision.

The Wednesday capture contains attack families not represented in the training captures. The Heartbleed sample contains only two rows, so its detection performance cannot be estimated reliably.

## Limitations

- Results are dataset-specific and do not establish production readiness.
- Cross-capture generalization remains weak.
- Performance on live traffic has not been established.
- Precision and alert volume may differ substantially under real-world class imbalance.
- Additional representative datasets and independent validation are needed.

## Model deployment decision

These experimental results do not authorize replacing the existing production model. The current production artifact remains unchanged at `config/artifacts/network_model.joblib`.

The experimental models remain separate artifacts until integration tests and further validation justify a change.
