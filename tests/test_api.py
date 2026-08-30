from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_metrics() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text

def test_predict_requires_text() -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422
