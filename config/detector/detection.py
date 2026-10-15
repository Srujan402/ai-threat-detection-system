
import logging
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings

from .models import Alert


logger = logging.getLogger(__name__)

MODEL_PATH = Path(settings.BASE_DIR) / "artifacts" / "network_model.joblib"

# Detection thresholds. These are initial prototype values, not universal
# indicators of malicious activity.
OUTBOUND_BYTES_THRESHOLD = 10_000_000
PORT_SCAN_THRESHOLD = 10
FAILED_LOGIN_THRESHOLD = 5

FEATURES = [
    "flow_duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "dst_port",
]


def timestamp_value(event):
    """Return an event timestamp in JSON-serializable ISO format."""
    timestamp = getattr(event, "timestamp", None)

    if timestamp is None:
        return None

    if hasattr(timestamp, "isoformat"):
        return timestamp.isoformat()

    return str(timestamp)


def numeric_value(value):
    """Convert a value to a finite number, or return None if invalid."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    return number


def event_evidence(event):
    """Create a consistent evidence record for a security event."""
    return {
        "event_id": getattr(event, "event_id", None),
        "timestamp": timestamp_value(event),
        "source_type": getattr(event, "source_type", None),
        "source_ip": getattr(event, "src_ip", None),
        "destination_ip": getattr(event, "dst_ip", None),
        "source_port": getattr(event, "src_port", None),
        "destination_port": getattr(event, "dst_port", None),
        "protocol": getattr(event, "protocol", None),
        "host": getattr(event, "host", None),
        "username": getattr(event, "username", None),
    }


@lru_cache(maxsize=1)
def get_model():
    """Load the trained model once per Django server process."""
    if not MODEL_PATH.exists():
        logger.warning("ML model not found at %s", MODEL_PATH)
        return None

    try:
        saved = joblib.load(MODEL_PATH)

        if isinstance(saved, dict) and "estimator" in saved:
            return saved["estimator"]

        return saved

    except Exception:
        logger.exception("Could not load the threat detection model")
        return None


def create_alert(
    *,
    title,
    description,
    rule_name,
    severity,
    source_ip=None,
    event=None,
    evidence=None,
):
    """Persist an alert with structured evidence."""
    return Alert.objects.create(
        event=event,
        title=title,
        description=description,
        rule_name=rule_name,
        severity=severity,
        source_ip=source_ip,
        evidence=evidence or {},
    )


def analyze_network_batch(events):
    """
    Detect possible port scanning in one uploaded batch.

    Rule: at least PORT_SCAN_THRESHOLD distinct destination ports
    contacted by the same source IP.

    Limitation: this rule analyzes the supplied batch, not a rolling
    time window across all network events.
    """
    grouped = defaultdict(list)

    for event in events:
        if not getattr(event, "src_ip", None):
            continue

        if numeric_value(getattr(event, "dst_port", None)) is None:
            continue

        grouped[event.src_ip].append(event)

    alerts_created = 0

    for source_ip, source_events in grouped.items():
        ports = sorted({
            int(numeric_value(event.dst_port))
            for event in source_events
        })

        if len(ports) < PORT_SCAN_THRESHOLD:
            continue

        timestamps = [
            timestamp_value(event)
            for event in source_events
            if timestamp_value(event)
        ]

        destination_ips = sorted({
            event.dst_ip
            for event in source_events
            if getattr(event, "dst_ip", None)
        })

        create_alert(
            title="Possible Port Scan Detected",
            description=(
                f"Source {source_ip} contacted {len(ports)} distinct "
                f"destination ports in one uploaded batch. This may "
                f"indicate port scanning; verify the traffic and "
                f"destination systems before classifying it as malicious."
            ),
            rule_name="possible_port_scan",
            severity=Alert.HIGH,
            source_ip=source_ip,
            event=source_events[-1],
            evidence={
                "rule_name": "possible_port_scan",
                "reason": "Distinct destination port count reached threshold",
                "source_ip": source_ip,
                "distinct_destination_ports": len(ports),
                "threshold": PORT_SCAN_THRESHOLD,
                "destination_ports": ports[:100],
                "destination_ports_truncated": len(ports) > 100,
                "destination_ips": destination_ips[:100],
                "events_in_batch": len(source_events),
                "first_seen": min(timestamps) if timestamps else None,
                "last_seen": max(timestamps) if timestamps else None,
                "event": event_evidence(source_events[-1]),
            },
        )
        alerts_created += 1

    return alerts_created


def analyze_network_event(event):
    """
    Analyze a network event using deterministic rules and ML.

    ML predictions are indicators for analyst review, not proof of
    malicious activity.
    """
    alerts_created = 0

    source_ip = getattr(event, "src_ip", None)
    destination_ip = getattr(event, "dst_ip", None)

    # Rule 1: unusually high outbound traffic.
    bytes_out = numeric_value(getattr(event, "bytes_out", None))

    if bytes_out is not None and bytes_out >= OUTBOUND_BYTES_THRESHOLD:
        create_alert(
            title="Unusually High Outbound Traffic",
            description=(
                f"The event reports {bytes_out:,.0f} outbound bytes, "
                f"meeting the configured threshold of "
                f"{OUTBOUND_BYTES_THRESHOLD:,} bytes. Verify the "
                f"destination, transfer purpose, and normal traffic baseline."
            ),
            rule_name="high_outbound_traffic",
            severity=Alert.HIGH,
            source_ip=source_ip,
            event=event,
            evidence={
                "rule_name": "high_outbound_traffic",
                "reason": "Outbound bytes met or exceeded the configured threshold",
                "observed_bytes_out": bytes_out,
                "threshold_bytes": OUTBOUND_BYTES_THRESHOLD,
                "destination_ip": destination_ip,
                "destination_port": getattr(event, "dst_port", None),
                "protocol": getattr(event, "protocol", None),
                "event": event_evidence(event),
            },
        )
        alerts_created += 1

    # Rule 2: supervised ML classifier.
    model = get_model()

    if model is None:
        return alerts_created

    raw_features = {
        "flow_duration": getattr(event, "flow_duration", None),
        "bytes_in": getattr(event, "bytes_in", None),
        "bytes_out": getattr(event, "bytes_out", None),
        "packets": getattr(event, "packets", None),
        "dst_port": getattr(event, "dst_port", None),
    }

    features = {}

    for name in FEATURES:
        value = numeric_value(raw_features[name])

        # Do not silently replace missing model inputs with fabricated
        # measurements. A missing destination port is handled as zero,
        # matching the existing inference behavior.
        if name == "dst_port" and value is None:
            value = 0.0

        if value is None or value < 0:
            logger.warning(
                "Skipping ML inference for event %s: invalid feature %s",
                getattr(event, "pk", None),
                name,
            )
            return alerts_created

        features[name] = value

    feature_frame = pd.DataFrame(
        [[features[name] for name in FEATURES]],
        columns=FEATURES,
    )

    try:
        prediction = str(model.predict(feature_frame)[0]).strip().lower()

        probabilities = None

        if hasattr(model, "predict_proba"):
            try:
                probability_values = model.predict_proba(feature_frame)[0]
                classes = getattr(model, "classes_", [])

                probabilities = {
                    str(label): round(float(probability), 6)
                    for label, probability in zip(
                        classes, probability_values
                    )
                }
            except (ValueError, TypeError, AttributeError, IndexError):
                logger.warning(
                    "Could not obtain ML probabilities for event %s",
                    getattr(event, "pk", None),
                    exc_info=True,
                )

        if prediction == "attack":
            evidence = {
                "rule_name": "ml_network_attack",
                "reason": "The trained classifier predicted the attack class",
                "model_prediction": prediction,
                "model_probabilities": probabilities,
                "model_type": "RandomForestClassifier",
                "features": features,
                "event": event_evidence(event),
            }

            create_alert(
                title="ML Network Threat Detection",
                description=(
                    "The trained classifier marked this network flow "
                    "as an attack. Review the recorded features and "
                    "supporting network telemetry before deciding "
                    "whether the event is malicious."
                ),
                rule_name="ml_network_attack",
                severity=Alert.MEDIUM,
                source_ip=source_ip,
                event=event,
                evidence=evidence,
            )
            alerts_created += 1

    except (ValueError, TypeError, AttributeError, KeyError, IndexError):
        logger.exception(
            "ML inference failed for event %s",
            getattr(event, "pk", None),
        )

    return alerts_created


def analyze_system_events(events):
    """
    Detect repeated failed logins in one uploaded system-log batch.

    Rule: at least FAILED_LOGIN_THRESHOLD failed login events for the
    same source IP, username, and host.

    Limitation: events across separate uploads are not aggregated.
    """
    grouped = defaultdict(list)

    for event in events:
        event_type = str(getattr(event, "event_type", "") or "").lower()
        event_id = str(getattr(event, "event_id", "") or "").strip()
        message = str(getattr(event, "message", "") or "").lower()

        is_failed_login = (
            event_id == "4625"
            or "failed_login" in event_type
            or "failed login" in event_type
            or "failed login" in message
            or "authentication failure" in message
        )

        if not is_failed_login:
            continue

        key = (
            getattr(event, "src_ip", None),
            getattr(event, "username", None) or "",
            getattr(event, "host", None) or "",
        )
        grouped[key].append(event)

    alerts_created = 0

    for (source_ip, username, host), failed_events in grouped.items():
        if len(failed_events) < FAILED_LOGIN_THRESHOLD:
            continue

        timestamps = [
            timestamp_value(event)
            for event in failed_events
            if timestamp_value(event)
        ]

        event_ids = sorted({
            str(event.event_id)
            for event in failed_events
            if getattr(event, "event_id", None) is not None
        })

        create_alert(
            title="Repeated Failed Login Attempts",
            description=(
                f"Detected {len(failed_events)} failed login events "
                f"for account '{username or 'unknown'}' on "
                f"'{host or 'unknown host'}' from "
                f"{source_ip or 'an unknown source IP'}. "
                f"Review the account, source, and surrounding events."
            ),
            rule_name="repeated_failed_logins",
            severity=Alert.HIGH,
            source_ip=source_ip,
            event=failed_events[-1],
            evidence={
                "rule_name": "repeated_failed_logins",
                "reason": "Failed login count reached the configured threshold",
                "failed_attempts": len(failed_events),
                "threshold": FAILED_LOGIN_THRESHOLD,
                "source_ip": source_ip,
                "username": username,
                "host": host,
                "event_ids": event_ids,
                "first_seen": min(timestamps) if timestamps else None,
                "last_seen": max(timestamps) if timestamps else None,
                "events": [
                    event_evidence(item)
                    for item in failed_events[-20:]
                ],
                "evidence_events_truncated": len(failed_events) > 20,
            },
        )
        alerts_created += 1

    return alerts_created
