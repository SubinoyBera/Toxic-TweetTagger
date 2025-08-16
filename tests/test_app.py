# This file contains tests for the FastAPI application endpoints.
import pytest
from fastapi import status
from app.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.mark.asyncio
async def test_health_check():
    """
    Test the health check endpoint.

    This test sends a GET request to the /health endpoint and verifies that the response is 200 OK, 
    and that the response contains the message, model name, and model version in its JSON payload.
    
    """
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    json_data = response.json()
    assert "message" in json_data
    assert "model" in json_data
    assert "version" in json_data


@pytest.mark.asyncio
async def test_home_page():
    """
    Test the home page endpoint.

    This test sends a GET request to the home endpoint and verifies that the response is 200 OK, 
    and that the response contains the HTML content type in its headers.
    
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_predict_page():
    """
    Test the predict page endpoint.

    This test sends a POST request to the /predict endpoint with a valid tweet and verifies 
    that the response is 200 OK, and that the response contains the HTML content type in its 
    headers, and that the response contains either :
    ["highly toxic", "toxic", "likely to be toxic", or "looks safe"]
    
    """
    response = client.post("/predict", data={"tweet": "I love that movie! It was great"})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Check if the response contains expected phrases
    assert any(
        phrase in response.text
        for phrase in [
            "strong", 
            "high", 
            "light", 
            "none"
        ]
    )

  
@pytest.mark.asyncio
async def test_api_endpoint():
    """
    Test the /api endpoint.

    This test sends a POST request to the /api endpoint with a sample tweet and verifies 
    that the response status is 200 OK.
    Checks if the response contains both 'response' and 'metadata' keys and validates that 
    the 'confidence' value in the 'response' is a float between 0 and 1.
    
    """
    response = client.post("/api", params={"tweet": "I love that movie! It was great"})
    assert response.status_code == 200
    data = response.json()
    
    # Validate the response structure
    assert "response" in data
    assert "metadata" in data
    assert "confidence" in data["response"]
    assert 0 <= data["response"]["confidence"] <= 1