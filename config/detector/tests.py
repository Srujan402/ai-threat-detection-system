
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone

from .detection import (
    analyze_network_batch,
    analyze_network_event,
    analyze_system_events,
)
from .models import Alert, SecurityEvent


class DetectionRuleTests(TestCase):
    def make_event(self, **overrides):
        data = {
            "timestamp": timezone.now(),
            "source_type": SecurityEvent.NETWORK,
            "src_ip": "192.168.1.10",
            "dst_ip": "192.168.1.20",
            "src_port": 51515,
            "dst_port": 443,
            "protocol": "TCP",
            "host": "test-host",
            "username": "",
            "event_id": "",
            "event_type": "network",
            "message": "",
            "flow_duration": 1000,
            "bytes_in": 500,
            "bytes_out": 800,
            "packets": 10,
        }
        data.update(overrides)
        return SecurityEvent.objects.create(**data)

    def test_high_outbound_traffic_creates_alert(self):
        event = self.make_event(bytes_out=12_000_000)

        created = analyze_network_event(event)

        self.assertGreaterEqual(created, 1)
        self.assertTrue(
            Alert.objects.filter(
                event=event,
                rule_name="high_outbound_traffic",
                severity=Alert.HIGH,
            ).exists()
        )

        alert = Alert.objects.get(
            event=event,
            rule_name="high_outbound_traffic",
        )
        self.assertEqual(alert.evidence["observed_bytes_out"], 12_000_000)
        self.assertEqual(alert.evidence["threshold_bytes"], 10_000_000)
        self.assertIn("reason", alert.evidence)

    def test_port_scan_creates_alert_at_threshold(self):
        events = [
            self.make_event(
                src_ip="192.168.1.50",
                dst_port=port,
            )
            for port in range(20, 30)
        ]

        created = analyze_network_batch(events)

        self.assertEqual(created, 1)
        alert = Alert.objects.get(rule_name="possible_port_scan")
        self.assertEqual(alert.evidence["distinct_destination_ports"], 10)
        self.assertEqual(alert.evidence["threshold"], 10)
        self.assertIn("reason", alert.evidence)

    def test_port_scan_does_not_alert_below_threshold(self):
        events = [
            self.make_event(
                src_ip="192.168.1.51",
                dst_port=port,
            )
            for port in range(20, 29)
        ]

        created = analyze_network_batch(events)

        self.assertEqual(created, 0)
        self.assertFalse(
            Alert.objects.filter(rule_name="possible_port_scan").exists()
        )

    def test_repeated_failed_logins_create_alert(self):
        events = [
            self.make_event(
                source_type=SecurityEvent.SYSTEM,
                src_ip="192.168.1.60",
                username="testuser",
                host="workstation-01",
                event_id="4625",
                event_type="failed_login",
                message="Failed login",
            )
            for _ in range(5)
        ]

        created = analyze_system_events(events)

        self.assertEqual(created, 1)
        alert = Alert.objects.get(rule_name="repeated_failed_logins")
        self.assertEqual(alert.evidence["failed_attempts"], 5)
        self.assertEqual(alert.evidence["threshold"], 5)
        self.assertEqual(alert.evidence["username"], "testuser")
        self.assertIn("reason", alert.evidence)

    def test_ml_attack_prediction_creates_evidence(self):
        event = self.make_event()

        class FakeModel:
            classes_ = ["attack", "benign"]

            def predict(self, features):
                return ["attack"]

            def predict_proba(self, features):
                return [[0.95, 0.05]]

        with patch(
            "detector.detection.get_model",
            return_value=FakeModel(),
        ):
            created = analyze_network_event(event)

        self.assertEqual(created, 1)
        alert = Alert.objects.get(
            event=event,
            rule_name="ml_network_attack",
        )
        self.assertEqual(alert.evidence["model_prediction"], "attack")
        self.assertIn("features", alert.evidence)
        self.assertEqual(
            alert.evidence["model_probabilities"]["attack"],
            0.95,
        )
        self.assertIn("reason", alert.evidence)
