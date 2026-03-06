import pytest
from src.app.services.feedback import FeedbackService

class MockEventConsumer:
    def __init__(self):
        self.events = []

    def add_event(self, record):
        self.events.append(record)


@pytest.fixture
def service():
    feedback_consumer = MockEventConsumer()
    return FeedbackService(feedback_consumer)


def test_feedback_success(service):
    response = service.submit_feedback(
        request_id="abc123",
        pred_label=1,
        feedback_label=1
    )
    assert response["status"] == "success"
    assert response["message"] == "Feedback recorded successfully"


def test_feedback_event_recorded(service):
    service.submit_feedback("abc123", 1, 0)

    event = service.event_consumer_worker.events[0]

    assert event["request_id"] == "abc123"
    assert event["predicted_label"] == 1
    assert event["feedback_label"] == 0
    assert "time_stamp" in event


def test_feedback_failure():
    class BrokenConsumer:
        def add_event(self, record):
            raise RuntimeError("DB failure")

    service = FeedbackService(BrokenConsumer())

    with pytest.raises(Exception):
        service.submit_feedback("abc123", 1, 1)