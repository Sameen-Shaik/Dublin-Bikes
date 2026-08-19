import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_predict_endpoint_valid_request(client):
    payload = {
        "STATION_ID": 42,
        "TARGET_TIME": "2021-11-15T12:00"
    }

    response = client.post('/predict', json=payload)

    assert response.status_code == 200
    assert "predicted_available_bikes" in response.json()