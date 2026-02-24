import time, sys
from datetime import datetime, timezone
from src.core.logger import logging
from src.core.exception import AppException
from src.app.monitoring.service_metrics import (PREDICTION_REQUEST_SUCCESS, PREDICTION_REQUEST_FAILED, 
                                     PREDICTION_CLASS, INFERENCE_LATENCY, PREDICTION_CONFIDENCE)

class InferenceService:
    def __init__(self, model_booster, vectorizer, eval_threshold, 
                 prediction_event_consumer, model_version):
        
        self.booster = model_booster
        self.vectorizer = vectorizer
        self.threshold = eval_threshold
        self.event_consumer_worker = prediction_event_consumer
        self.model_version = model_version

    def predict(self, request_id: str, text: str):
        """
        Model inference API endpoint.

        This endpoint accepts a POST request with a JSON payload containing a comment string.
        It returns a JSON response containing the model prediction, and metadata.
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            start_time = time.perf_counter()
        
            x = self.vectorizer.transform([text])
            prob = self.booster.inplace_predict(x)              # P(class=1)
            pred = (prob > self.threshold).astype(int)

        except Exception as e:
            logging.exception(e)
            PREDICTION_REQUEST_FAILED.inc()
            raise AppException(e, sys)

        if prob is None or len(pred) == 0:
            logging.error("No prediction made by the model")
            PREDICTION_REQUEST_FAILED.inc()
            raise RuntimeError("No prediction made by the model")

        if prob[0] > 0.70:
            toxicity = "strong"
        elif prob[0] > self.threshold + 0.05:
            toxicity = "high"
        elif prob[0] > self.threshold - 0.02:
            toxicity = "light"
        else:
            toxicity = "none"

        confidence = prob[0] if pred[0] == 1 else 1-prob[0]
        confidence_margin = round(abs(2*pred[0] - 1), 4)

        warnings = []

        if len(text.split()) <= 2:
            warnings.append({
                "type": "MIN_TOKEN_WARNING",
                "message": "Input is too short for reliable classification."
        })

        if confidence_margin < 0.10:
            message = f"""
                Prediction is close to model decision boundary. Confidence Margin: {confidence_margin}.
                Please consider manual review.
            """
            warnings.append({
                "code": "LOW_CONFIDENCE_MARGIN",
                "message": message
        })

        # Prepare the record to insert into database
        prediction_record = {
            "request_id": request_id,
            "timestamp": timestamp,
            "comment": text,
            "prediction": pred[0],
            "confidence": confidence,
        }

        # Add the record to the batch writer for asynchronous insertion into MongoDB
        self.event_consumer_worker.add_event(prediction_record)

        end_time = time.perf_counter()
        response_time = round((end_time - start_time), 4)

        PREDICTION_REQUEST_SUCCESS.inc()
        # Track latency
        INFERENCE_LATENCY.observe(response_time)
        # Track class distribution
        PREDICTION_CLASS.labels(class_label=str(pred[0])).inc()
        # Track class confidence
        PREDICTION_CONFIDENCE.labels(class_label=str(pred[0])).observe(confidence)

        response = {
            "id": request_id,
            "timestamp": timestamp,
            "object": "text-classification",
            "prediction": {
                "label": pred[0],
                "confidence": prob[0],
                "toxicity": toxicity,
            },
            "warnings": warnings if warnings else None,
            "metadata": {
                "latency": response_time,
                "usage": {
                    "word_count": len(text.split()),
                    "total_characters": len(text)
                },
                "model": {
                    "name": "XGB-Classifier-Booster",
                    "version": self.model_version,
                    "vectorizer": str(type(self.vectorizer).__name__),
                },
                "streamable": False,
                "environment": "Production",
                "api_version": "v-2.0"
            }
        }
        
        return response