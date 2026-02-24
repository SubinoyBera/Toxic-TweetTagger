from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from lime.lime_text import LimeTextExplainer
from main.workers import BufferedEventConsumerWorker
from main.schema import APIResponse, FeedbackResponse, InputData
from main.utils import Explainer, get_model_version, load_model_artifacts
from src.core.constants import DATABASE_NAME, PRODUCTION_COLLECTION_NAME, FEEDBACK_COLLECTION_NAME
from src.core.mongo_client import MongoDBClient
from src.core.logger import logging
from src.core.exception import AppException

# Count total predictions
# Total requests received at predict endpoint
REQUEST_RECEIVED = Counter(
    "predict_requests_total",
    "Total prediction requests received"
)

# Requests successfully served
REQUEST_SUCCESS = Counter(
    "predict_requests_success_total",
    "Total successful prediction responses"
)

# Requests failed
REQUEST_FAILED = Counter(
    "predict_requests_failed_total",
    "Total failed prediction requests"
)

# Prediction class distribution (hate / non-hate)
PREDICTION_CLASS = Counter(
    "prediction_class_total",
    "Count of predicted classes",
    ["class_label"]  # label dimension
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence distribution by predicted class",
    ["class_label"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Inference latency
INFERENCE_LATENCY = Histogram(
    "model_inference_seconds",
    "Model inference time in seconds"
)

# Feedback counter
FEEDBACK_COUNTER = Counter(
    "feedback_submissions_total",
    "Total feedback submissions"
)

FEEDBACK_VALIDATION = Counter(
    "feedback_validation_total",
    "Feedback validation result",
    ["result"]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This context manager sets up the FastAPI application state with a MongoDB client, a collection reference, 
    a batch writer, and a model instance. It also ensures that the batch writer is shut down and the MongoDB connection 
    is closed when the application is finished.
    """
    # application state setup
    try:
        mongo_client = MongoDBClient()
        app.state.collection = mongo_client.client[DATABASE_NAME][PRODUCTION_COLLECTION_NAME]

        app.state.vectorizer, app.state.xgb_booster, app.state.eval_threshold = load_model_artifacts()
        app.state.model_version = get_model_version()

        lime_explainer = LimeTextExplainer(class_names=["hate", "non-hate"], bow=False)
        app.state.explainer = Explainer(lime_explainer, app.state.model, app.state.vectorizer)
        app.state.prediction_event_consumer = BufferedEventConsumerWorker(mongo_client, DATABASE_NAME, PRODUCTION_COLLECTION_NAME)
        app.state.feedback_event_consumer = BufferedEventConsumerWorker(mongo_client, DATABASE_NAME, FEEDBACK_COLLECTION_NAME)
        logging.info("Infernce API app server started successfully")

    except Exception as e:
        logging.critical(f"Startup Failed: {e}", exc_info=True)
        raise AppException(e, sys)
    
    # run application
    yield
    
    # application shutdown
    app.state.prediction_event_consumer.shutdown()
    app.state.feedback_event_consumer.shutdown()
    mongo_client.close_connection()


# Initialize FastAPI application with the defined lifespan context manager
inference_api = FastAPI(lifespan=lifespan)
Instrumentator().instrument(inference_api).expose(inference_api)


@inference_api.get("/health")
def status():
    """Status endpoint for the model inference API."""
    return {"status": "ok"}


@inference_api.post("/predict", response_model=APIResponse)
def api_response(payload: InputData, request: Request):
    """
    Model inference API endpoint.

    This endpoint accepts a POST request with a JSON payload containing a comment string.
    It returns a JSON response containing the model prediction, and metadata.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    booster = request.app.state.xgb_booster
    vectorizer = request.app.state.vectorizer
    thresh = request.app.state.eval_threshold

    if payload.text is None:
        raise HTTPException(status_code=400, detail="No text provided.")

    try:
        x = vectorizer.transform([payload.text])
        prob = booster.inplace_predict(x)              # P(class=1)
        pred = (prob > thresh).astype(int)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if pred is None:
        REQUEST_FAILED.inc()
        raise HTTPException(status_code=500, detail=f"Inference failed.")

    if prob[0] > 0.70:
        toxicity = "strong"
    elif prob[0] > thresh + 0.05:
        toxicity = "high"
    elif prob[0] > thresh - 0.02:
        toxicity = "light"
    else:
        toxicity = "none"

    confidence = prob[0] if pred[0] == 1 else 1-prob[0]
    confidence_margin = round(abs(2*pred[0] - 1), 4)

    warnings = []

    if len(payload.text.split()) <= 2:
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
        "request_id": payload.request_id,
        "timestamp": timestamp,
        "comment": payload.text,
        "prediction": pred[0],
        "confidence": confidence,
    }

    # Add the record to the batch writer for asynchronous insertion into MongoDB
    request.app.state.prediction_event_consumer.add_event(prediction_record)

    end_time = time.perf_counter()
    response_time = round((end_time - start_time) * 1000, 4)

    REQUEST_SUCCESS.inc()
    # Track latency
    INFERENCE_LATENCY.observe(response_time)
    # Track class distribution
    PREDICTION_CLASS.labels(class_label=str(pred[0])).inc()
    # Track class confidence
    PREDICTION_CONFIDENCE.labels(class_label=str(pred[0])).observe(pred[0])

    response = {
        "id": payload.request_id,
        "timestamp": timestamp,
        "object": "text-classification",
        "prediction": {
            "label": pred[0],
            "confidence": prob[0],
            "toxicity": toxicity,
        },
        "warnings": warnings if warnings else None,
        "metadata": {
            "latency_ms": response_time,
            "usage": {
                "word_count": len(payload.text.split()),
                "total_characters": len(payload.text)
            },
            "model": {
                "name": "XGB-Classifier-Booster",
                "version": request.app.state.model_version,
                "vectorizer": str(type(vectorizer).__name__),
            },
            "streamable": False,
            "environment": "Production",
            "api_version": "v-2.0"
        }
    }
    
    return JSONResponse(status_code=200, content=response)


@inference_api.post("/explain")
def explain(payload: InputData, request: Request):
    try:
        explanation_html = request.app.state.explainer.explain(payload.text)
        if not explanation_html:
            raise ValueError("Returned empty explaination")
        
        return HTMLResponse(content=explanation_html)
    
    except Exception as e:
        logging.exception(f"Failed to generate explaination. RequestID: {payload.request_id} : {e}")
        raise HTTPException(status_code=503, detail="Explanation engine service is unavailable.")


@inference_api.post("/feedback")
def submit_feedback(payload: FeedbackResponse, request: Request):
    """
    Submit user feedback for a given request_id.
    """
    try:
        feedback_record = {
            "request_id": payload.request_id,
            "time_stamp": datetime.now(timezone.utc).isoformat(),
            "predicted_label": payload.predicted_label,
            "feedback_label": payload.feedback_label
        }

        request.app.state.feedback_consumer_worker.add_event(feedback_record)
        logging.info(f"Feedback submitted for request_id: {payload.request_id}")

        FEEDBACK_COUNTER.inc()

        # Compare prediction vs feedback
        if payload.predicted_label == payload.feedback_label:
            FEEDBACK_VALIDATION.labels(result="correct").inc()
        else:
            FEEDBACK_VALIDATION.labels(result="incorrect").inc()

        return {
            "status": "success",
            "request_id": payload.request_id,
            "message": "Feedback recorded successfully",
        }
    
    except Exception as e:
        logging.exception(f"Failed to submit feedback. RequestID: {payload.request_id} : {e}")
        raise HTTPException(status_code=503, detail="Feedback service is unavailable.")