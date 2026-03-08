import sys
from datetime import datetime, timezone
from src.core.logger import logging
from src.core.exception import AppException
from src.app.monitoring.service_metrics import FEEDBACK_REQUEST_SUCCESS, USER_PREDICTION_FEEDBACK, FEEDBACK_REQUEST_FAILED


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

            if feedback_label == 1:
                USER_PREDICTION_FEEDBACK.labels(feedback="correct").inc()
            else:
                USER_PREDICTION_FEEDBACK.labels(feedback="incorrect").inc()

            return {
                "status": "success",
                "message": "Feedback recorded successfully",
            }

        except Exception as e:
            logging.exception(f"Failed to submit feedback for request_id: {request_id} : {e}")
            FEEDBACK_REQUEST_FAILED.inc()
            raise AppException(e, sys)