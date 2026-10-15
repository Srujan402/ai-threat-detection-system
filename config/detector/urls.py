
from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload_csv, name="upload_csv"),
    path("alerts/", views.alerts, name="alerts"),
    path("alerts/<int:alert_id>/", views.alert_detail, name="alert_detail"),
    path(
    "alerts/export/",
    views.export_alerts_csv,
    name="export_alerts_csv",
),
]
