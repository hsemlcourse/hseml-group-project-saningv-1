from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_expected_schema():
    response = client.post("/predict", json={"message": "Free prize! Call now to claim your cash"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"label", "target", "spam_score"}
    assert payload["label"] in {"ham", "spam"}
    assert payload["target"] in {0, 1}
    assert 0 <= payload["spam_score"] <= 1
