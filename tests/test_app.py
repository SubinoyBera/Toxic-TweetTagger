import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi import status
from app.app import app
from fastapi.testclient import TestClient

client =TestClient(app)

@pytest.mark.asyncio
async def test_health_check():
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    json_data = response.json()
    assert "message" in json_data
    assert "model" in json_data
    assert "version" in json_data

@pytest.mark.asyncio
async def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_predict_page():
    response = client.post("/predict", data={"tweet": "I love that movie! It was great"})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert any(
        phrase in response.text
        for phrase in [
            "highly toxic", 
            "toxic", 
            "likely to be toxic", 
            "looks safe"
        ]
    )


async def test_api_endpoint():
    response = client.post("/api", params={"tweet": "I love that movie! It was great"})
    assert response.status_code == 200
    data = response.json()
    
    assert "response" in data
    assert "metadata" in data
    assert "confidence" in data["response"]
    assert 0 <= data["response"]["confidence"] <= 1