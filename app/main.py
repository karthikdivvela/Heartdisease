"""FastAPI model-serving app with request logging and Prometheus metrics."""
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak",
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions by class", ["label"])

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    return _model


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="chest pain type")
    trestbps: float = Field(..., ge=50, le=260, description="resting blood pressure")
    chol: float = Field(..., ge=50, le=700, description="serum cholesterol mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar > 120")
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=260, description="max heart rate")
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: float | None = Field(None, ge=0, le=4)
    thal: float | None = Field(None, description="3 normal, 6 fixed, 7 reversible")

    model_config = {"json_schema_extra": {"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}}}


class Prediction(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    if request.url.path != "/metrics":
        REQUESTS.labels(request.url.path, str(response.status_code)).inc()
        LATENCY.labels(request.url.path).observe(elapsed)
        log.info("%s %s -> %s in %.1f ms", request.method, request.url.path,
                 response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    try:
        model = get_model()
    except Exception as exc:
        log.error("Model load failed: %s", exc)
        raise HTTPException(status_code=503, detail="model unavailable")
    row = pd.DataFrame([patient.model_dump()])[FEATURES]
    proba = float(model.predict_proba(row)[0, 1])
    pred = int(proba >= 0.5)
    PREDICTIONS.labels(str(pred)).inc()
    log.info("prediction=%d proba=%.3f input=%s", pred, proba, patient.model_dump())
    return Prediction(prediction=pred,
                      label="disease" if pred else "no disease",
                      confidence=round(proba if pred else 1 - proba, 4),
                      probability_disease=round(proba, 4))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
