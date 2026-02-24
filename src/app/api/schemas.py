# Schema validation for the API response

from pydantic import BaseModel, Field
from typing import Annotated, Dict, Literal


class InferenceRequest(BaseModel):
    request_id: str = Field(..., description="Unique inference request identifier")
    text: Annotated[str, Field(..., description="Input text for classification")]


class PredictionResult(BaseModel):
    class_label: int
    confidence: float
    toxic_level: str
    pred_scores: Dict[int, float]
    explanation: Dict[str, float]


class MetaData(BaseModel):
    request_id: str
    timestamp: str
    response_time: str
    input: Dict[str, int]
    model: str
    version: int
    vectorizer: str
    type: str
    loader_module: str
    streamable: bool
    api_version: str
    developer: str


class InferenceResponse(BaseModel):
    response: PredictionResult
    metadata: MetaData


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=15, description="ID of the Inference request served")
    predicted_label: Literal[0, 1]
    feedback_label: Literal[0, 1]