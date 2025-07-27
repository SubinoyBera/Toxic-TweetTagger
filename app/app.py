from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from .services import *
import numpy as np
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uuid
from datetime import datetime
import time

clf = Model()
model, version = clf.load_model()


# Initializing fastapi app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get('/', response_class=HTMLResponse)
async def Home(request: Request):
    return templates.TemplateResponse("index.html", 
                                      {"request": request,
                                       "result": None})


@app.get('/health')
def health_check():
    return {
        "status" : 200,
        "model" : "XGB",
        "version" : 0.1
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    text = text
    text_df = preprocess(text)

    model_response = model.predict(text_df)

    probability_scores = model_response["class_probalility_scores"]
    confidence_class1 = float(np.round(probability_scores[0][1], 4).item())

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": True,
        "confidence": confidence_class1,
        "user_input": text
    })


@app.post('/api')
async def api(data: str, request: Request):
    start_time = time.time()
    timestamp = datetime.now().astimezone().isoformat()

    text = data
    text_df = preprocess(text)
    
    model_response = model.predict(text_df)

    label = model_response["class_label"][0]
    probability_scores = model_response["class_probalility_scores"]
    confidence = float(np.max(probability_scores)),
    proba_class0 = float(np.round(probability_scores[0][0], 4).item())
    proba_class1 = float(np.round(probability_scores[0][1], 4).item())

    if proba_class1 > 0.70:
        toxic_level = "strong"
    elif proba_class1 > 0.54:
        toxic_level = "high"
    elif proba_class1 > 0.46:
        toxic_level = "light"
    else:
        toxic_level = "none"


    response = { 
        "prediction": {
            "category" : int(label),
            "confidence" : confidence,
            "toxic_level" : toxic_level,
            "category_scores": {
                0: proba_class0,
                1: proba_class1
            },
        }
    }

    return response