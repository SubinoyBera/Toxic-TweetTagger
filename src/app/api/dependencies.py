from fastapi import Request

from src.app.services.inference import InferenceService
from src.app.services.explainer import ExplainerService
from src.app.services.feedback import FeedbackService


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.prediction_service


def get_explainer_service(request: Request) -> ExplainerService:
    return request.app.state.explainer_service


def get_feedback_service(request: Request) -> FeedbackService:
    return request.app.state.feedback_service