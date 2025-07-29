from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from .services import *
from .schema import UserInput, APIResponse
from datetime import datetime
import uuid, time

clf = Model()
model, version = clf.load_model()

# Initializing fastapi app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get('/', response_class=HTMLResponse)
async def Home(request: Request):
    return templates.TemplateResponse("index.html", {"request" : request, "result": None })


@app.get('/health')
def health_check():
    return {
        "status" : 200,
        "message": "live",
        "model" : clf.get_model_name(),
        "version" : version,
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, tweet: str = Form(...,
                                                    description="User tweet or comment to be classified")
                                                ):
    if not tweet:
        raise HTTPException(status_code=400, detail="Tweet cannot be empty.")
    
    tweet_df = preprocess(tweet)
    model_response = model.predict(tweet_df)

    probability_scores = model_response["class_probability_scores"]
    pred_score_class1 = float(probability_scores[0][1])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": True,
        "prediction_class1": round(pred_score_class1, 4),
        "user_input": tweet
    })


@app.post('/api', response_model=APIResponse)
async def api(data: UserInput, request: Request):
    timestamp = datetime.now().astimezone().isoformat()
    request_id = str(uuid.uuid4())
    
    start_time = time.time()
    tweet = data.comment
    df_tweet = preprocess(tweet)
    model_response = model.predict(df_tweet)

    label = int(model_response["class_label"][0])
    probability_scores = model_response["class_probability_scores"]
    proba_class0 = float(probability_scores[0][0])
    proba_class1 = float(probability_scores[0][1])

    end_time = time.time()

    if proba_class1 > 0.70:
        toxic_level = "strong"
    elif proba_class1 > 0.54:
        toxic_level = "high"
    elif proba_class1 > 0.46:
        toxic_level = "light"
    else:
        toxic_level = "none"


    response = {
        "response": {
            "class_label": label,
            "confidence": round(abs(proba_class0 - proba_class1), 4),
            "toxic_level": toxic_level,
            "pred_scores": {
                0: round(proba_class0, 4),
                1: round(proba_class1, 4)
            },
        },
        "metadata": {
            "request_id": request_id,
            "timestamp": timestamp,
            "response_time": f"{round((end_time - start_time), 4)} sec",
            "input": {
                "num_tokens": int(len(tweet.split())),
                "num_characters": int(len([i for i in tweet])),
                "language": "'en' (iso 639-1code)",
            },
            "model": clf.get_model_name(),
            "version": version,
            "vectorizer": clf.get_vectorizer_name(),
            "type": "production",
            "loader_module": "mlflow.pyfunc.model",
            "streamable": False,
            "api_version": "v-1.0",
            "developer": "Subinoy Bera"
        }
    }

    return JSONResponse(status_code=200, content=response)