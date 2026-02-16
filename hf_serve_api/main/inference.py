from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.db.db_logging import BufferedBatchWriter
from main.schema import APIResponse, FeedbackResponse, InputData
from main.utils import LimeExplainer, get_model_registry, get_model_version, load_model
from src.constant.constants import DATABASE_NAME, PRODUCTION_COLLECTION_NAME
from src.db.mongo_client import MongoDBClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This context manager sets up the FastAPI application state with a MongoDB client, a collection reference, 
    a batch writer, and a model instance. It also ensures that the batch writer is shut down and the MongoDB connection 
    is closed when the application is finished.
    """
    # application state setup
    mongo_client = MongoDBClient()
    app.state.collection = mongo_client.client[DATABASE_NAME][PRODUCTION_COLLECTION_NAME]

    app.state.batch_writer = BufferedBatchWriter(mongo_client)
    app.state.model, app.state.vectorizer = load_model()
    app.state.explainer = LimeExplainer(app.state.model)
    # run application
    yield
    # application shutdown
    app.state.batch_writer.shutdown()
    mongo_client.close_connection()


# Initialize FastAPI application with the defined lifespan context manager
inference_api = FastAPI(lifespan=lifespan)


@inference_api.get("/")
def status():
    """Status endpoint for the model inference API."""
    return JSONResponse(
        content={
            "status": 200,
            "message": "Inference API active.",
        }
    )


@inference_api.post("/get_prediction", response_model=APIResponse)
def api_response(payload: InputData, request: Request):
    """
    Model inference API endpoint.

    This endpoint accepts a POST request with a JSON payload containing a comment string.
    It returns a JSON response containing the model prediction, explanation, and metadata.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    model = request.app.state.model
    writer = request.app.state.batch_writer
    tweet = payload.comment

    try:
        explainer = LimeExplainer(model)
        explanation = explainer.explain(tweet)
        prediction = explainer.prediction
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    if prediction is None:
        raise HTTPException(status_code=500, detail="Model prediction could not be made")

    try:
        label = int(prediction["class_label"][0])
        probability_scores = prediction["class_probability_scores"][0]
        proba_class0 = float(probability_scores[0])
        proba_class1 = float(probability_scores[1])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Invalid model output format: {exc}") from exc

    if proba_class1 > 0.70:
        toxic_level = "strong"
    elif proba_class1 > 0.54:
        toxic_level = "high"
    elif proba_class1 > 0.46:
        toxic_level = "light"
    else:
        toxic_level = "none"

    confidence = round(max(proba_class0, proba_class1), 4)

    # Prepare the record to insert into database
    record = {
        "request_id": request_id,
        "timestamp": timestamp,
        "comment": tweet,
        "prediction": label,
        "confidence": confidence,
        "feedback": None,
    }
    # Add the record to the batch writer for asynchronous insertion into MongoDB
    writer.add(record)

    end_time = time.perf_counter()

    try:
        model_version = int(get_model_version())
    except (TypeError, ValueError):
        model_version = 0

    response = {
        "response": {
            "class_label": label,
            "confidence": confidence,
            "toxic_level": toxic_level,
            "pred_scores": {
                0: round(proba_class0, 4),
                1: round(proba_class1, 4),
            },
            "explanation": explanation,
        },
        "metadata": {
            "request_id": request_id,
            "timestamp": timestamp,
            "response_time_ms": round((end_time - start_time) * 1000, 4),
            "input": {
                "num_tokens": len(tweet.split()),
                "num_characters": len(tweet),
            },
            "model": type(model.model).__name__,
            "version": model_version,
            "vectorizer": type(model.vectorizer).__name__,
            "type": "Production",
            "loader_module": f"Mlflow {get_model_registry()}",
            "streamable": False,
            "api_version": "v-1.0",
            "developer": "Subinoy Bera",
        },
    }

    return response


@inference_api.post("/feedback")
def submit_feedback(payload: FeedbackResponse, request: Request):
    """
    Submit user feedback for a given request_id.
    """
    collection = request.app.state.collection

    result = collection.update_one(
        {
            "request_id": payload.request_id,
            "feedback": None,
        },
        {
            "$set": {
                "feedback": {"label": payload.label},
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Invalid request_id or feedback already submitted",
        )

    return {
        "status": "success",
        "request_id": payload.request_id,
        "message": "Feedback recorded successfully",
    }


@inference_api.get("/metrics")
def metrics():
    pass