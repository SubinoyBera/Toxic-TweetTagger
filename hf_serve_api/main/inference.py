from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from main.workers import BufferedEventConsumerWorker
from main.schema import APIResponse, FeedbackResponse, InputData
from main.utils import LimeExplainer, get_model_version, load_model_artifacts
from src.constant.constants import DATABASE_NAME, PRODUCTION_COLLECTION_NAME, FEEDBACK_COLLECTION_NAME
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
    app.state.vectorizer, app.state.xgb_booster, app.state.eval_threshold = load_model_artifacts()
    app.state.explainer = LimeExplainer(app.state.model, app.state.vectorizer)
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
        x = request.app.state.vectorizer.transform([payload.text])
        prob = request.app.state.xgb_booster.inplace_predict(x)
        pred = (prob > thresh).astype(int)
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    if pred is None:
        raise HTTPException(status_code=500, detail="Failed to generate model prediction.")
        
    prob_class1 = float(1 - prob[0])

    if prob_class1 > 0.70:
        toxicity = "strong"
    elif prob_class1 > thresh + 0.05:
        toxicity = "high"
    elif prob_class1 > thresh - 0.02:
        toxicity = "light"
    else:
        toxicity = "none"

    # Prepare the record to insert into database
    record = {
        "request_id": payload.request_id,
        "timestamp": timestamp,
        "comment": payload.text,
        "prediction": pred[0],
        "confidence": prob[0],
        "feedback": None,
    }
    # Add the record to the batch writer for asynchronous insertion into MongoDB
    request.app.state.writer.add(record)

    end_time = time.perf_counter()

    try:
        model_version = int(get_model_version())
    except (TypeError, ValueError):
        model_version = 0

    response = {
        "response": {
            "class_label": pred[0],
            "confidence": prob[0],
            "toxicity": toxicity,
        },
        "metadata": {
            "request_id": payload.request_id,
            "timestamp": timestamp,
            "response_time_ms": round((end_time - start_time) * 1000, 4),
            "input": {
                "num_tokens": len(payload.text.split()),
                "num_characters": len(payload.text),
            },
            "model": type(booster),
            "version": model_version,
            "vectorizer": type(vectorizer),
            "type": "ToxicTagger-Models",
            "loader_module": f"mlflow.pyfunc.model",
            "streamable": False,
            "api_version": "v-2.0",
            "developer": "Subinoy Bera"
        },
    }
    
    return response


@inference_api.get("/explain")
def explain(payload: InputData, request: Request):
    try:
        explanation = request.app.state.explainer.explain(payload.text)
    except:
        pass



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