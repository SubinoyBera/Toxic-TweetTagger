from fastapi import FastAPI
from fastapi.responses import JSONResponse
from main.schema import InputData, APIResponse
from datetime import datetime
from main.utils import *
import uuid, time
from src.db.mongo_client import MongoDBClient
from db_logging import BufferedBatchWriter

model = load_model()

writer = BufferedBatchWriter()

inference_api = FastAPI()

@inference_api.get("/")
def status():
    """
    Status endpoint for the model inference API.

    Returns a JSON response with a status of 200 and a message indicating
    that the API is active.

    """
    return JSONResponse(content={
        "status": 200,
        "message": "Inference API active."
        })


@inference_api.post('/get_prediction', response_model=APIResponse)
def api_response(payload: InputData):
    """
    Inference endpoint for getting prediction from the model.

    This endpoint accepts a POST request with a JSON payload containing the text to be classified.
    The response is a JSON object with the model prediction, confidence score, and other metadata.

    :param payload: InputData object containing the text to be classified.
    :return: APIResponse object containing the model prediction, confidence score, 
            and other metadata.
    """
    timestamp = datetime.now().astimezone().isoformat()
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    tweet = payload.comment
    explainer = LimeExplainer(model)
    explaination = explainer.explain(tweet)
    prediction = explainer.prediction

    if prediction is not None:
        label = int(prediction["class_label"][0])
        probability_scores = prediction["class_probability_scores"][0]
        proba_class0 = float(probability_scores[0])
        proba_class1 = float(probability_scores[1])
    else:
        raise ValueError("Model prediction could not be made.")

    if proba_class1 > 0.70:
        toxic_level = "strong"
    elif proba_class1 > 0.54:
        toxic_level = "high"
    elif proba_class1 > 0.46:
        toxic_level = "light"
    else:
        toxic_level = "none"

    record = {
        "request_id": request_id,
        "timestamp": timestamp,
        "comment": tweet,
        "prediction": label,
        "confidence": round(max(proba_class0, proba_class1), 4),
        "feedback": None
    }

    writer.add(record)

    end_time = time.perf_counter()
    
    response = {
        "prediction": {
            "class_label": label,
            "confidence": round(max(proba_class0, proba_class1), 4),
            "toxic_level": toxic_level,
            "explaination": explaination
        },
        "metadata": {
            "request_id": request_id,
            "timestamp": timestamp,
            "response_time": f"{round((end_time - start_time), 4)} sec",
            "input": {
                "num_tokens": int(len(tweet.split())),
                "num_characters": int(len([i for i in tweet])),
                "language": "en (iso 639-1code)",
            },
            "model": type(model.model).__name__,
            "model_version": get_model_version(),
            "vectorizer": type(model.vectorizer).__name__,
            "model_registry": f"Mlflow {get_model_registry()}",
            "type": "Production",
            "explainer_varient": "LimeTextExplainer",
            "streamable": False,
            "api_version": "v-1.0",
            "developer": "Subinoy Bera"
        }
    }

    return JSONResponse(status_code=200, content=response)