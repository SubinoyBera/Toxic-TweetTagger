import pytest
import numpy as np
from src.app.services.inference import InferenceService

class MockVectorizer:
    def transform(self, text):
        return text

class MockBooster:
    def __init__(self, prob):
        self.prob = prob

    def inplace_predict(self, x):
        return np.array([self.prob])

class MockEventConsumer:
    def add_event(self, record):
        pass


@pytest.fixture
def service():
    booster = MockBooster(0.8)
    vectorizer = MockVectorizer()
    consumer = MockEventConsumer()

    return InferenceService(
        model_booster=booster,
        vectorizer=vectorizer,
        eval_threshold=0.5,
        prediction_event_consumer=consumer,
        model_version="1.0"
    )


def test_strong_toxicity(service):
    response = service.predict(
        request_id="123",
        input_tweet="Bad tweet",
        text="Bad tweet"
    )

    assert response["prediction"]["toxicity"] == "strong"
    assert response["prediction"]["label"] == 1


def test_high_toxicity():
    booster = MockBooster(0.6)

    service = InferenceService(
        booster,
        MockVectorizer(),
        0.5,
        MockEventConsumer(),
        "1.0"
    )

    response = service.predict("1", "tweet", "tweet")

    assert response["prediction"]["toxicity"] == "high"


def test_uncertain_toxicity():
    booster = MockBooster(0.48)

    service = InferenceService(
        booster,
        MockVectorizer(),
        0.5,
        MockEventConsumer(),
        "1.0"
    )

    response = service.predict("1", "tweet", "tweet")

    assert response["prediction"]["toxicity"] == "uncertain"
    assert response["warnings"] is not None
    assert response["warnings"]["code"] == "LOW_CONFIDENCE_MARGIN"


def test_safe_prediction():
    booster = MockBooster(0.1)

    service = InferenceService(
        booster,
        MockVectorizer(),
        0.5,
        MockEventConsumer(),
        "1.0"
    )

    response = service.predict("1", "tweet", "tweet")

    assert response["prediction"]["toxicity"] == "none"
    assert response["prediction"]["label"] == 0


def test_response_structure(service):
    response = service.predict("123", "tweet", "tweet")

    assert "prediction" in response
    assert "metadata" in response
    assert "confidence" in response["prediction"]