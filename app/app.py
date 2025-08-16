from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
from app.utils import *
import requests

# Initilize fastapi app server
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

response: Optional[requests.Response] = None

@app.get('/', response_class=HTMLResponse)
async def Home(request: Request):
    """
    Endpoint to render the home page with a form to input a tweet or a comment 
    to be classified as toxic or non-toxic.

    The form is submitted to the /predict endpoint.
    
    """
    return templates.TemplateResponse("index.html", {"request" : request,
                                                     "result": None })


@app.get('/health')
def health_check():
    """
    Endpoint to check the health of the API server.

    Returns a 200 OK with a JSON payload {"status": 200, "message": "live"}
    if the server is alive and running.
    
    """
    url = "https://subi003-toxictagger-serveapi.hf.space/"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        model_api_status = response.json().get("message")
    else:
        model_api_status = "Inactive"
    
    return {
        "app_status" : 200,
        "message": "live",
        "model_server": "HuggingFace",
        "api_status": model_api_status
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, tweet: str = Form(...,
                                                    description="User tweet or comment to be classified")
                                                ) -> HTMLResponse :
    """
    Endpoint to predict the if the tweet or comment is toxic or not.

    This endpoint accepts a POST request with a tweet or comment to be classified.
    It preprocesses the input text, calls the model inference API for prediction,
    and returns the result to be rendered on the homepage.

    Args:
        request (Request): The request object.
        tweet (str): The user tweet or comment to be classified.

    Returns:
        HTMLResponse: The rendered HTML template with the prediction result.
    
    """
    if not tweet:
        raise HTTPException(status_code=400, detail="Tweet cannot be empty.")
    
    payload_comment = preprocess(tweet)
    
    url = "https://subi003-toxictagger-serveapi.hf.space/get_prediction"

    global response 
    response = requests.post(url, json={"comment": payload_comment})

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": True,
        "toxicity_level" : response.json().get("prediction", {}).get("toxic_level"),
        "user_input": tweet
    })


@app.get("/api_response")
async def api_response():
    """
    Endpoint to view the raw response from the model inference API.
    Returns the raw JSON response from the model inference API 
    for the prediction made.
    """
    if response is None:
        raise HTTPException(status_code=404, detail="No prediction found.")
    return response.json()


@app.get("/about.html", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request" : request,
                                                     "result": None })