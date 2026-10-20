# Heart Disease Risk - End-to-End MLOps Pipeline

MLOps (AIMLCZG523) - Assignment 01, BITS Pilani

A machine learning classifier that predicts the risk of heart disease from patient health data (UCI Heart Disease, Cleveland), served as a containerised REST API, deployed on Kubernetes (Minikube) and monitored with Prometheus and Grafana. The project covers data preparation and EDA, model training and tuning, experiment tracking with MLflow, packaging, automated tests and CI with GitHub Actions, Docker, Kubernetes and monitoring.

**Results (5-fold CV on the training split, then a held-out 20% test split):**

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression (selected) | 0.902 | 0.966 |
| Random Forest | 0.901 | 0.947 |

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/` | `download_data.py` (data download), `create_grafana_dashboard.py` (creates the Grafana dashboard) |
| `src/` | `data.py` (cleaning), `features.py` (preprocessing pipeline), `eda.py` (plots), `train.py` (training, tuning, MLflow), `predict.py` (offline inference) |
| `samples/` | example input files (`sample_patient.json`, `sample_patients.csv`) for `predict.py` |
| `app/` | `main.py` - FastAPI service (`/predict`, `/health`, `/metrics`) with request logging |
| `tests/` | pytest unit tests (8) for data, model training and the API |
| `data/` | `heart.csv` - the cleaned dataset |
| `models/` | `model.joblib` (preprocessing + classifier) and `metrics.json` |
| `reports/` | EDA and evaluation plots |
| `k8s/` | Kubernetes manifests (`deployment.yaml`, `service.yaml`, optional `ingress.yaml`) |
| `monitoring/` | Prometheus configuration and Docker Compose file for Prometheus + Grafana |
| `.github/workflows/ci.yml` | GitHub Actions pipeline |
| `Dockerfile`, `requirements.txt`, `requirements-serve.txt` | container build and pinned dependencies |
| `screenshots/` | evidence used in the report |

## Requirements

- Python 3.11 or newer (tested on 3.12 and 3.14)
- Git
- Docker (for the container, Kubernetes and monitoring steps)
- Minikube (for the Kubernetes step; it bundles `kubectl`)

## 1. Set up and run the pipeline

```bash
git clone https://github.com/karthikdivvela/Heartdisease.git
cd Heartdisease
python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/download_data.py      # optional: data/heart_raw.csv from UCI
python -m src.data                   # optional: rebuilds data/heart.csv (already in the repo)
python -m src.eda                    # EDA plots -> reports/eda/
python -m src.train                  # trains both models -> models/model.joblib + MLflow runs
mlflow ui --backend-store-uri sqlite:///mlflow.db     # open http://localhost:5000
```

`data/heart.csv` is committed, so the download step can be skipped if the UCI site is not reachable. Training prints the scores of both models and the selected one. Random seed 42 is used everywhere, so results are reproducible.

## 2. Tests and lint

```bash
pytest -v
flake8 src app tests scripts
```

All 8 tests should pass and flake8 should print nothing.

## 3. Run the API locally

## Inference (offline, without the API)

`src/predict.py` loads `models/model.joblib` and predicts for a JSON or CSV file. It prints the same fields as the API: `prediction`, `label`, `confidence` and `probability_disease`. Run it from the repository root:

```
python -m src.predict --input samples/sample_patient.json
python -m src.predict --input samples/sample_patients.csv --output results.csv
```

Expected output for the sample patient:

```
prediction      label  confidence  probability_disease
         0 no disease      0.8345               0.1655
```

```bash
uvicorn app.main:app --reload
```


Open http://localhost:8000/docs for the interactive documentation.

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | JSON with the 13 features; returns `prediction`, `label`, `confidence`, `probability_disease`. Invalid input returns HTTP 422. |
| `/health` | GET | liveness / readiness check |
| `/metrics` | GET | Prometheus metrics |

Example request:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

Response:

```json
{"prediction":0,"label":"no disease","confidence":0.8345,"probability_disease":0.1655}
```

## 4. Docker

```bash
docker build -t heart-api:latest .
docker run -d -p 8000:8000 --name heart-api heart-api:latest
docker ps                            # wait ~10 seconds until the status says (healthy)
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d "$SAMPLE"
docker logs heart-api
docker rm -f heart-api               # stop it before the monitoring stack (both use port 8000)
```

Define the request body first: `SAMPLE='{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'`

## 5. Kubernetes (Minikube)

```bash
minikube start --driver=docker --cni=bridge
minikube image load heart-api:latest
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get nodes,pods,deployments,svc       # wait until both pods are Running
```

If `kubectl` is not installed, use Minikube's bundled copy: `alias kubectl='minikube kubectl --'`.

In a **second terminal** run `minikube tunnel` and leave it running. This gives the LoadBalancer service an `EXTERNAL-IP`:

```bash
kubectl get svc heart-api
curl http://<EXTERNAL-IP>/health
curl -X POST http://<EXTERNAL-IP>/predict -H "Content-Type: application/json" -d "$SAMPLE"
kubectl logs deployment/heart-api
```

`k8s/ingress.yaml` is an optional alternative to the LoadBalancer (needs `minikube addons enable ingress`).

## 6. Monitoring (Prometheus + Grafana)

The API exposes request counts, latency and prediction counts at `/metrics`. Prometheus scrapes it every 5 seconds and Grafana draws four panels (requests per second, predictions by class, p95 latency of `/predict`, total requests).

The compose file starts the API, Prometheus and Grafana together, with Prometheus and Grafana on the host network. It was tested on Linux (a Linux lab machine and GitHub Codespaces); Docker Desktop on Windows/macOS does not support host networking.

```bash
docker rm -f heart-api                                   # free port 8000 if the Docker step is still running
docker compose -f monitoring/docker-compose.yml up -d
docker ps                                                # three containers: API, Prometheus, Grafana
curl -s localhost:9090/-/ready                           # Prometheus Server is Ready.

# send some traffic
for i in $(seq 1 60); do curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d "$SAMPLE" > /dev/null; sleep 0.5; done

# create the Grafana data source and dashboard (default login admin / admin)
curl -s -u admin:admin -X POST localhost:3000/api/datasources -H "Content-Type: application/json" \
  -d '{"name":"Prometheus","type":"prometheus","url":"http://localhost:9090","access":"proxy","isDefault":true}'
python3 scripts/create_grafana_dashboard.py
```

Then open:

- Prometheus: http://localhost:9090 (Status > Target health shows `heart-api` as UP)
- Grafana dashboard: http://localhost:3000/d/heart-api (set the time range to the last 15 minutes)

## 7. CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push to `main` and on pull requests:

1. **build-test-train:** install dependencies, lint with flake8, run pytest, retrain the model, upload the model, metrics, plots, JUnit report and MLflow database as artifacts (even if a step failed).
2. **docker** (only if the first job passes): build the image, start the container, wait for `/health`, send a sample `/predict` request and print the container logs.

Any failing step stops the pipeline. The Docker image uses the committed `models/model.joblib`.

## 8. Accessing the deployed API

| Where | Address |
|---|---|
| Docker container | http://localhost:8000/docs |
| Kubernetes LoadBalancer (while `minikube tunnel` runs) | http://`<EXTERNAL-IP>`/docs (the IP comes from `kubectl get svc heart-api`) |

The API is not exposed on a public internet address. To test it on a fresh machine, follow sections 1, 4 and 5 above.

## Notes

- This is a learning project, not a medical device.
- The dataset is small (about 300 patients), so scores should be read with care.
