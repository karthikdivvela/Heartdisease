"""Create the 'Heart Disease API' Grafana dashboard through Grafana's HTTP API."""
import base64
import json
import os
import urllib.request

BASE = os.getenv("GRAFANA_URL", "http://localhost:3000")
PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")
TOKEN = base64.b64encode(f"admin:{PASSWORD}".encode()).decode()


def call(method, path, body=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Basic {TOKEN}"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


ds = call("GET", "/api/datasources/name/Prometheus")
DS = {"type": "prometheus", "uid": ds["uid"]}


def panel(pid, title, expr, legend, x, y, kind="timeseries", w=12):
    return {"id": pid, "type": kind, "title": title, "datasource": DS,
            "gridPos": {"h": 8, "w": w, "x": x, "y": y},
            "targets": [{"refId": "A", "expr": expr,
                         "legendFormat": legend, "datasource": DS}]}


dashboard = {
    "uid": "heart-api", "title": "Heart Disease API", "schemaVersion": 39,
    "time": {"from": "now-15m", "to": "now"}, "refresh": "5s",
    "panels": [
        panel(1, "Requests per second",
              "sum by (endpoint, status) (rate(api_requests_total[1m]))",
              "{{endpoint}} {{status}}", 0, 0),
        panel(2, "Predictions by class", "predictions_total",
              "class {{label}}", 12, 0),
        panel(3, "p95 latency of /predict (s)",
              'histogram_quantile(0.95, sum by (le) '
              '(rate(api_request_latency_seconds_bucket{endpoint="/predict"}[1m])))',
              "p95", 0, 8),
        panel(4, "Total requests", "sum(api_requests_total)", "total",
              12, 8, kind="stat"),
    ],
}
result = call("POST", "/api/dashboards/db",
              {"dashboard": dashboard, "overwrite": True})
print("Created:", BASE + result["url"])
