# Heart Disease Risk - End-to-End MLOps Pipeline (AIMLCZG523 Assignment 01)

## Setup (tested on Python 3.14; Windows: use `.venv\Scripts\activate`)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py      # -> data/heart_raw.csv
python -m src.data                   # -> data/heart.csv (cleaned; commit this file)
python -m src.eda                    # -> reports/eda/*.png
python -m src.train                  # -> models/model.joblib, MLflow runs
mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://localhost:5000
pytest -v && flake8 src app tests scripts
uvicorn app.main:app --reload        # http://localhost:8000/docs
```

## Docker
```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

## Kubernetes (Minikube)
```bash
minikube start && eval $(minikube docker-env) && docker build -t heart-api:latest .
kubectl apply -f k8s/ && minikube tunnel      # in a second terminal
kubectl get pods,svc
```

## Monitoring
`cd monitoring && docker compose up` -> Prometheus :9090, Grafana :3000 (add Prometheus datasource `http://prometheus:9090`).
