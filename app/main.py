import time
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from app.metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY
from app.model_service import ModelService
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Medical Text Classifier", version="0.1.0")
service = ModelService("models/model.joblib")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    started = time.perf_counter()
    try:
        prediction, class_name, confidence = service.predict(request.text)
        REQUEST_COUNT.labels("/predict", "POST", "200").inc()
        return PredictionResponse(
            prediction=prediction, class_name=class_name, confidence=confidence
        )
    except Exception:
        ERROR_COUNT.labels("/predict").inc()
        REQUEST_COUNT.labels("/predict", "POST", "500").inc()
        raise
    finally:
        REQUEST_LATENCY.labels("/predict").observe(time.perf_counter() - started)

@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
