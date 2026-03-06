from fastapi import APIRouter, Request, Depends
from prometheus_client import generate_latest, CollectorRegistry, multiprocess, CONTENT_TYPE_LATEST
from fastapi.responses import Response, HTMLResponse

from src.app.services.inference import InferenceService
from src.app.services.explainer import ExplainerService
from src.app.services.feedback import FeedbackService
from src.app.api.dependencies import get_inference_service, get_explainer_service, get_feedback_service
from src.app.api.schemas import InferenceRequest, InferenceResponse, FeedbackRequest, ExplanationRequest

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/predict", response_model=InferenceResponse)
async def predict(request: Request, payload: InferenceRequest, service: InferenceService = Depends(get_inference_service)):
    request_id = request.state.request_id
    return service.predict(request_id, payload.input_tweet, payload.text)

@router.post("/explain")
async def explain(payload: ExplanationRequest, service: ExplainerService = Depends(get_explainer_service)):
    return HTMLResponse(service.explain(payload.input_tweet))

@router.post("/submit_feedback")
async def submit_feedback(request: Request, payload: FeedbackRequest, service: FeedbackService = Depends(get_feedback_service)):
    request_id = request.state.request_id
    return service.submit_feedback(request_id, payload.predicted_label, payload.feedback_label)

@router.get("/metrics")
def metrics():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return Response(
        generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
        headers={"Cache-Control": "no-cache"}
    )