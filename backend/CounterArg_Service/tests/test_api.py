# backend/CounterArg_Service/tests/test_api.py
from fastapi.testclient import TestClient
from backend.CounterArg_Service.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict():
    r = client.post("/predict", json={"claim": "The earth is flat."})
    assert r.status_code == 200
    data = r.json()
    assert "counterArgument" in data
    assert isinstance(data["counterArgument"], str)
