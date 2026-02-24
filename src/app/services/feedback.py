import sys
from datetime import datetime, timezone
from src.core.logger import logging
from src.core.exception import AppException
from src.app.monitoring.service_metrics import FEEDBACK_REQUEST_SUCCESS, FEEDBACK_VALIDATION, FEEDBACK_REQUEST_FAILED


class FeedbackService:
    def __init__(self, feedback_event_consumer):
        self.event_consumer_worker = feedback_event_consumer

    def submit_feedback(self, request_id, pred_label, feedback_label):
        try:
            feedback_record = {
                "request_id": request_id,
                "time_stamp": datetime.now(timezone.utc).isoformat(),
                "predicted_label": pred_label,
                "feedback_label": feedback_label
            }

            self.event_consumer_worker.add_event(feedback_record)
            FEEDBACK_REQUEST_SUCCESS.inc()

            # Compare prediction vs feedback
            if pred_label == feedback_label:
                FEEDBACK_VALIDATION.labels(result="correct").inc()
            else:
                FEEDBACK_VALIDATION.labels(result="incorrect").inc()

            return {
                "status": "success",
                "request_id": request_id,
                "message": "Feedback recorded successfully",
            }

        except Exception as e:
            logging.exception(f"Failed to submit feedback for request_id: {request_id} : {e}")
            FEEDBACK_REQUEST_FAILED.inc()
            raise AppException(e, sys)