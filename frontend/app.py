from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette import status
from urllib.parse import quote
from typing import Optional
from pathlib import Path
from frontend.utils import *
import requests, uuid

# Initilize fastapi app server
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
static_path = BASE_DIR / "static"

templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory=static_path), name="static")

response: Optional[requests.Response] = None

@app.get('/', response_class=HTMLResponse)
async def Home(request: Request):
    """
    Endpoint to render the home page with a form to input a tweet or a comment 
    to be classified as toxic or non-toxic.

    The form is submitted to the /predict endpoint.
    
    """
    feedback_submitted = request.query_params.get("feedback_submitted") == "true"
    feedback_error = request.query_params.get("feedback_error") == "true"

    return templates.TemplateResponse("index.html", {
        "request" : request,
        "feedback_submitted": feedback_submitted,
        "feedback_error": feedback_error,
        "result": None })


@app.get('/health')
def health_check():
    """
    Endpoint to check the health of the API server.

    Returns a 200 OK with a JSON payload {"status": 200, "message": "live"}
    if the server is alive and running.
    
    """
    url = "https://subi003-toxictweet-tagger.hf.space/api/health"
    
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


@app.post("/get_prediction", response_class=HTMLResponse)
async def get_prediction(request: Request, tweet: str = Form(...,
                                                    description="User tweet or comment to be classified")
                                                ):
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
    if not tweet or not tweet.strip():
        return RedirectResponse(
            url=f"/?error={quote('No Text found')}",
            status_code=303
        )
    
    payload_text = preprocess(tweet)
    words = payload_text.split()

    if not words:
        return RedirectResponse(
        url=f"/?error={quote('Text must contain meaningful words')}",
        status_code=303
    )

    if len(words) > 50:
        return RedirectResponse(
        url=f"/?error={quote('Comments must not contain more than 50 words')}",
        status_code=303
    )

    if not any(len(w) >= 3 for w in words):
        return RedirectResponse(
        url=f"/?error={quote('Comments must contain at least one word with 3+ characters')}",
        status_code=303
    )
    
    #url = "http://localhost:8000/api/predict"
    url = "https://subi003-toxictweet-tagger.hf.space/api/predict"

    headers = {"X-Request-ID": str(uuid.uuid4())}

    global response
    response = requests.post(
        url,
        json={"input_tweet": tweet, "text": payload_text},
        headers=headers
    )

    id = response.json().get("request_id", "N/A")
    pred_label = response.json().get("prediction", {}).get("label", None)
    toxicity = response.json().get("prediction", {}).get("toxicity", "N/A")
    confidence = response.json().get("prediction", {}).get("confidence", None)
    warnings = response.json().get("warnings", None)
    inference_time = response.json().get("metadata", {}).get("latency", None)

    if confidence is not None:
        confidence = round(confidence * 100, 2)
    else:
        confidence = 0

    if inference_time is not None:
        inference_time = round(inference_time*1000, 4)          # Convert to milliseconds

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": True,
        "id": id,
        "pred_label": pred_label,
        "toxicity": toxicity,
        "confidence": confidence,
        "warning_msg": warnings["message"] if warnings else None,
        "inference_time": inference_time,
        "user_input": tweet
    })


@app.post("/submit_feedback", response_class=HTMLResponse)
async def submit_feedback(request: Request,
                          id: str = Form(..., description="The unique request ID for the prediction"),
                          pred_label: int = Form(..., description="The predicted label for the input tweet"),
                          feedback_label: int = Form(..., description="The feedback label for the input tweet")):

    #url = "http://localhost:8000/api/submit_feedback"
    url = "https://subi003-toxictweet-tagger.hf.space/api/submit_feedback"

    try:
        response = requests.post(
            url,
            json={
                "predicted_label": pred_label,
                "feedback_label": feedback_label
            },
            headers={"X-Request-ID": id},
        )

        if response.json().get("status") != "success":
            return RedirectResponse(
                url="/?feedback_error=true",
                status_code=status.HTTP_303_SEE_OTHER
            )

    except requests.RequestException:
        # Network / timeout / connection error
        return RedirectResponse(
            url="/?feedback_error=true",
            status_code=status.HTTP_303_SEE_OTHER
        )

    return RedirectResponse(
        url="/?feedback_submitted=true",
        status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/get_explanation", response_class=HTMLResponse)
async def get_explanation(request: Request, tweet: str=Form(...)):
    if not tweet:
        raise HTTPException(status_code=400, detail="Tweet cannot be empty.")
    
    #url = "http://localhost:8000/api/explain"
    url = "https://subi003-toxictweet-tagger.hf.space/api/explain"

    response = requests.post(url, json={"input_tweet": tweet})

    html_content = response.text
    return HTMLResponse(content=html_content)


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