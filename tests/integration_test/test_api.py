# This file contains tests for the FastAPI backend API endpoints.
import pytest
from fastapi import status
from src.app.main import app
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_health(client):
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_predict_endpoint(client):
    tweet = "I love that movie! It was great"

    response = client.post(
        "/api/predict",
        params={"tweet": tweet}
    )

    assert response.status_code == 200

    data = response.json()

    assert "id" in data
    assert "timestamp" in data
    assert data["object"] == "text-classification"
    assert "prediction" in data
    assert "metadata" in data

    # -------- Prediction --------
    prediction = data["prediction"]

    assert "label" in prediction
    assert prediction["label"] in [0, 1]

    assert "confidence" in prediction
    assert isinstance(prediction["confidence"], float)
    assert 0 <= prediction["confidence"] <= 1

    assert "toxicity" in prediction
    assert isinstance(prediction["toxicity"], str)

    # -------- Metadata --------
    metadata = data["metadata"]

    assert "latency" in metadata
    assert metadata["latency"] >= 0
    assert metadata["latency"] < 1   # seconds

    usage = metadata["usage"]
    assert usage["word_count"] == len(tweet.split())
    assert usage["total_characters"] == len(tweet)

    model = metadata["model"]
    assert model["name"] == "XGB-Classifier-Booster"
    assert "version" in model
    assert "vectorizer" in model

    assert metadata["streamable"] is False
    assert metadata["environment"] == "Production"
    assert metadata["api_version"] == "v-2.0"
 

def test_explain_endpoint(client):
    response = client.post(
        "/api/explain",
        params={"tweet": "You are stupid"}
    )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_feedback_endpoint_success(client):
    payload = {
        "request_id": "test123",
        "pred_label": 1,
        "feedback_label": 1
    }

    response = client.post("/api/feedback", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Feedback recorded successfully"