
from django.contrib import admin
from .models import Alert, SecurityEvent


@admin.register(SecurityEvent)
class SecurityEventAdmin(admin.ModelAdmin):
    list_display = (
        "timestamp",
        "source_type",
        "src_ip",
        "dst_ip",
        "event_type",
        "host",
    )
    list_filter = ("source_type", "protocol", "timestamp")
    search_fields = ("src_ip", "dst_ip", "username", "host", "message")
    date_hierarchy = "timestamp"


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "severity",
        "status",
        "source_ip",
        "created_at",
    )
    list_filter = ("severity", "status", "created_at")
    search_fields = ("title", "description", "source_ip", "rule_name")
    date_hierarchy = "created_at"
