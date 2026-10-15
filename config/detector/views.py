
import ipaddress
import math
import csv
import json

from django.http import HttpResponse
from django.utils import timezone
import pandas as pd
from django.contrib import messages
from django.db import transaction
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from .detection import (
    analyze_network_batch,
    analyze_network_event,
    analyze_system_events,
)
from .forms import CSVUploadForm
from .models import Alert, SecurityEvent


def parse_timestamp(value):
    parsed = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(parsed):
        return None

    return parsed.to_pydatetime()


def clean_ip(value):
    if value is None or pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return None


def safe_text(value, default=""):
    if value is None or pd.isna(value):
        return default

    return str(value).strip()


def number(value, default=0):
    if value is None or pd.isna(value):
        return default

    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return max(0, result)
    except (TypeError, ValueError):
        return default


def optional_port(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None

    try:
        port = int(float(value))
        if 0 <= port <= 65535:
            return port
    except (TypeError, ValueError, OverflowError):
        pass

    return None



def dashboard(request):
    total_events = SecurityEvent.objects.count()
    total_alerts = Alert.objects.count()
    new_alerts = Alert.objects.filter(status=Alert.NEW).count()

    open_high_critical = Alert.objects.filter(
        severity__in=[Alert.HIGH, Alert.CRITICAL],
        status__in=[Alert.NEW, Alert.INVESTIGATING],
    ).count()

    stat_cards = [
        {
            "label": "Total Events",
            "value": total_events,
            "icon": "activity",
            "color": "blue",
        },
        {
            "label": "Total Alerts",
            "value": total_alerts,
            "icon": "shield",
            "color": "purple",
        },
        {
            "label": "New Alerts",
            "value": new_alerts,
            "icon": "bell",
            "color": "orange",
        },
        {
            "label": "Open High / Critical",
            "value": open_high_critical,
            "icon": "alert",
            "color": "red",
        },
    ]

    severity_totals = {
        item["severity"]: item["total"]
        for item in Alert.objects.values("severity").annotate(
            total=Count("id")
        )
    }

    severity_chart = [
        {
            "label": label,
            "count": severity_totals.get(value, 0),
            "color": color,
        }
        for value, label, color in [
            (Alert.CRITICAL, "Critical", "#ef4444"),
            (Alert.HIGH, "High", "#f97316"),
            (Alert.MEDIUM, "Medium", "#eab308"),
            (Alert.LOW, "Low", "#22c55e"),
        ]
    ]

    recent_alerts = (
        Alert.objects.select_related("event")
        .order_by("-created_at")[:10]
    )

    return render(
        request,
        "detector/dashboard.html",
        {
            "stat_cards": stat_cards,
            "severity_chart": severity_chart,
            "recent_alerts": recent_alerts,
            "form": CSVUploadForm(),
        },
    )



@require_http_methods(["GET", "POST"])
def upload_csv(request):
    if request.method == "GET":
        return redirect("dashboard")

    form = CSVUploadForm(request.POST, request.FILES)

    if not form.is_valid():
        for field_errors in form.errors.values():
            for error in field_errors:
                messages.error(request, error)
        return redirect("dashboard")

    data_type = form.cleaned_data["data_type"]
    uploaded_file = form.cleaned_data["csv_file"]

    try:
        # Read at most 100,000 rows per upload.
        df = pd.read_csv(uploaded_file, nrows=100001)
        df.columns = [str(column).strip().lower() for column in df.columns]

        if len(df) > 100000:
            messages.error(
                request,
                "The CSV exceeds 100,000 rows. Split it into smaller files.",
            )
            return redirect("dashboard")

        if df.empty:
            messages.error(request, "The CSV file contains no data rows.")
            return redirect("dashboard")

        if df.columns.duplicated().any():
            messages.error(
                request,
                "The CSV contains duplicate column names.",
            )
            return redirect("dashboard")

        if data_type == "network":
            required = {
                "timestamp",
                "src_ip",
                "dst_ip",
                "dst_port",
                "flow_duration",
                "bytes_in",
                "bytes_out",
                "packets",
            }
        else:
            required = {"timestamp", "event_id", "event_type"}

        missing = required - set(df.columns)

        if missing:
            messages.error(
                request,
                "Missing required CSV columns: "
                + ", ".join(sorted(missing)),
            )
            return redirect("dashboard")

        created_events = []

        with transaction.atomic():
            for row in df.to_dict(orient="records"):
                timestamp = parse_timestamp(row.get("timestamp"))

                if timestamp is None:
                    continue

                event = SecurityEvent(
                    timestamp=timestamp,
                    source_type=(
                        SecurityEvent.NETWORK
                        if data_type == "network"
                        else SecurityEvent.SYSTEM
                    ),
                    src_ip=clean_ip(row.get("src_ip")),
                    dst_ip=clean_ip(row.get("dst_ip")),
                    src_port=optional_port(row.get("src_port")),
                    dst_port=optional_port(row.get("dst_port")),
                    protocol=safe_text(row.get("protocol")),
                    host=safe_text(row.get("host")),
                    username=safe_text(row.get("username")),
                    event_id=safe_text(row.get("event_id")),
                    event_type=safe_text(row.get("event_type")),
                    message=safe_text(row.get("message")),
                    flow_duration=number(row.get("flow_duration")),
                    bytes_in=number(row.get("bytes_in")),
                    bytes_out=number(row.get("bytes_out")),
                    packets=number(row.get("packets")),
                )
                event.save()
                created_events.append(event)

            if not created_events:
                raise ValueError(
                    "No valid timestamps were found. "
                    "Check the CSV timestamp column."
                )

            alerts_before = Alert.objects.count()

            if data_type == "network":
                analyze_network_batch(created_events)

                for event in created_events:
                    analyze_network_event(event)
            else:
                analyze_system_events(created_events)

            alerts_created = Alert.objects.count() - alerts_before

        messages.success(
            request,
            f"Imported {len(created_events)} events and generated "
            f"{alerts_created} alerts.",
        )

    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        messages.error(request, f"CSV processing failed: {exc}")
    except Exception:
        messages.error(
            request,
            "The upload could not be processed. Check the terminal "
            "for the technical error and verify the CSV columns.",
        )
        raise

    return redirect("dashboard")



def alerts(request):
    alert_list = Alert.objects.select_related("event").all()

    severity = request.GET.get("severity", "").strip()
    status = request.GET.get("status", "").strip()
    rule_name = request.GET.get("rule", "").strip()

    # Accept only valid severity and status values.
    valid_severities = {
        choice[0] for choice in Alert.SEVERITY_CHOICES
    }
    valid_statuses = {
        choice[0] for choice in Alert.STATUS_CHOICES
    }

    if severity in valid_severities:
        alert_list = alert_list.filter(severity=severity)
    else:
        severity = ""

    if status in valid_statuses:
        alert_list = alert_list.filter(status=status)
    else:
        status = ""

    # Get the detection rules that actually exist in the database.
    rule_choices = list(
        Alert.objects.exclude(rule_name="")
        .exclude(rule_name__isnull=True)
        .values_list("rule_name", flat=True)
        .distinct()
        .order_by("rule_name")
    )

    if rule_name in rule_choices:
        alert_list = alert_list.filter(rule_name=rule_name)
    else:
        rule_name = ""

    alert_list = alert_list.order_by("-created_at")

    return render(
        request,
        "detector/alerts.html",
        {
            "alerts": alert_list,
            "selected_severity": severity,
            "selected_status": status,
            "selected_rule": rule_name,
            "rule_choices": rule_choices,
            "severity_choices": Alert.SEVERITY_CHOICES,
            "status_choices": Alert.STATUS_CHOICES,
        },
    )


@require_http_methods(["GET", "POST"])
def alert_detail(request, alert_id):
    alert = get_object_or_404(
        Alert.objects.select_related("event"),
        pk=alert_id,
    )

    if request.method == "POST":
        new_status = request.POST.get("status", "")
        analyst_notes = request.POST.get("analyst_notes", "")

        valid_statuses = {choice[0] for choice in Alert.STATUS_CHOICES}

        if new_status not in valid_statuses:
            messages.error(request, "Please select a valid alert status.")
            return redirect("alert_detail", alert_id=alert.id)

        alert.status = new_status
        alert.analyst_notes = analyst_notes[:10000]
        alert.save(update_fields=["status", "analyst_notes"])

        messages.success(request, "Alert investigation details updated.")
        return redirect("alert_detail", alert_id=alert.id)

    return render(
        request,
        "detector/alert_detail.html",
        {"alert": alert, "status_choices": Alert.STATUS_CHOICES},
    )

def export_alerts_csv(request):
    """Export alerts using the active severity, status, and rule filters."""

    queryset = Alert.objects.select_related("event").all()

    severity = request.GET.get("severity", "").strip()
    status = request.GET.get("status", "").strip()
    rule = request.GET.get("rule", "").strip()

    # Apply only valid severity and status choices.
    valid_severities = {value for value, label in Alert.SEVERITY_CHOICES}
    valid_statuses = {value for value, label in Alert.STATUS_CHOICES}

    if severity in valid_severities:
        queryset = queryset.filter(severity=severity)

    if status in valid_statuses:
        queryset = queryset.filter(status=status)

    if rule:
        queryset = queryset.filter(rule_name=rule)

    queryset = queryset.order_by("-created_at")

    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = (
        'attachment; filename="soc_filtered_alerts.csv"'
    )
    response.write("\ufeff")  # UTF-8 BOM for Excel compatibility.

    writer = csv.writer(response)

    writer.writerow([
        "Alert ID",
        "Timestamp",
        "Title",
        "Severity",
        "Status",
        "Detection Rule",
        "Source IP",
        "Description",
        "Evidence",
    ])

    def safe_csv_value(value):
        """Reduce spreadsheet formula-injection risk for text cells."""
        text = "" if value is None else str(value)

        if text.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + text

        return text

    for alert in queryset.iterator():
        timestamp = alert.created_at

        if timestamp and timezone.is_aware(timestamp):
            timestamp = timezone.localtime(timestamp)

        writer.writerow([
            alert.pk,
            timestamp.isoformat() if timestamp else "",
            safe_csv_value(alert.title),
            safe_csv_value(alert.severity),
            safe_csv_value(alert.status),
            safe_csv_value(alert.rule_name),
            safe_csv_value(alert.source_ip),
            safe_csv_value(alert.description),
            safe_csv_value(
                json.dumps(alert.evidence or {}, ensure_ascii=False)
            ),
        ])

    return response
