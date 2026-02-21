# Schema validation for the API response

from pydantic import BaseModel, Field
from typing import Annotated, Dict, Literal


class InputData(BaseModel):
    request_id: str = Field(..., description="ID of incomming request")
    text: Annotated[str, Field(..., description="User tweet or comment (text)")]


class Prediction(BaseModel):
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


class APIResponse(BaseModel):
    response: Prediction
    metadata: MetaData


class FeedbackResponse(BaseModel):
    request_id: str = Field(..., min_length=15, description="Unique identifier for the inference request")
    label: Literal[0, 1]
