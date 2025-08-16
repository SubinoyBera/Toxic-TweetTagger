# Schema validation for the API response

from pydantic import BaseModel, Field
from typing import Annotated, Dict

class InputData(BaseModel):
    comment: Annotated[str, Field(..., description="User tweet or comment to be classified")]

class Prediction(BaseModel):
    class_label: int
    confidence: float
    toxic_level: str
    pred_scores: Dict[int, float]

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