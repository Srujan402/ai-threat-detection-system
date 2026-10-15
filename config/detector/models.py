from django.db import models


class SecurityEvent(models.Model):
    NETWORK = "network"
    SYSTEM = "system"

    SOURCE_CHOICES = [
        (NETWORK, "Network"),
        (SYSTEM, "System Log"),
    ]

    timestamp = models.DateTimeField(db_index=True)
    source_type = models.CharField(
        max_length=20,
        choices=SOURCE_CHOICES,
        db_index=True,
    )

    src_ip = models.GenericIPAddressField(null=True, blank=True)
    dst_ip = models.GenericIPAddressField(null=True, blank=True)
    src_port = models.PositiveIntegerField(null=True, blank=True)
    dst_port = models.PositiveIntegerField(null=True, blank=True)

    protocol = models.CharField(max_length=20, blank=True)
    host = models.CharField(max_length=150, blank=True)
    username = models.CharField(max_length=150, blank=True)
    event_id = models.CharField(max_length=50, blank=True)
    event_type = models.CharField(max_length=80, blank=True)
    message = models.TextField(blank=True)

    flow_duration = models.FloatField(default=0)
    bytes_in = models.FloatField(default=0)
    bytes_out = models.FloatField(default=0)
    packets = models.FloatField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.source_type}: {self.timestamp} ({self.src_ip})"


class Alert(models.Model):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    SEVERITY_CHOICES = [
        (LOW, "Low"),
        (MEDIUM, "Medium"),
        (HIGH, "High"),
        (CRITICAL, "Critical"),
    ]

    NEW = "new"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"

    STATUS_CHOICES = [
        (NEW, "New"),
        (INVESTIGATING, "Investigating"),
        (RESOLVED, "Resolved"),
    ]

    event = models.ForeignKey(
        SecurityEvent,
        on_delete=models.CASCADE,
        related_name="alerts",
        null=True,
        blank=True,
    )

    title = models.CharField(max_length=200)
    description = models.TextField()
    rule_name = models.CharField(max_length=100)

    severity = models.CharField(
        max_length=10,
        choices=SEVERITY_CHOICES,
        default=MEDIUM,
    )

    source_ip = models.GenericIPAddressField(null=True, blank=True)
    evidence = models.JSONField(default=dict, blank=True)

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=NEW,
        db_index=True,
    )

    analyst_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.severity.upper()}] {self.title}"