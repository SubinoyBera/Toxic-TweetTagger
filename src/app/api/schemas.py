# Schema validation for the API requests and responses

from pydantic import BaseModel, Field
from typing import Annotated, Dict, Literal, Optional
from datetime import datetime


class InferenceRequest(BaseModel):
    input_tweet: Annotated[str, Field(..., description="Input tweet or comment text for classification")]
    text: Annotated[str, Field(..., description="Preprocessed text to be fed to the model for prediction")]

class ExplanationRequest(BaseModel):
    input_tweet: Annotated[str, Field(..., description="Input tweet or comment text for generating explanation")]

class PredictionResult(BaseModel):
    label: int
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    toxicity: Literal["strong", "high", "uncertain", "none"]

class ModelInfoSchema(BaseModel):
    name: str = Field(..., description="Model name")
    version: int = Field(..., description="Model version")
    vectorizer: str = Field(..., description="Vectorizer class name")

class MetadataSchema(BaseModel):
    latency: float = Field(..., ge=0, description="Response time in seconds")
    usage: Dict[str, float] 
    model: ModelInfoSchema
    streamable: bool = Field(default=False)
    environment: Literal["Standard", "Beta", "Production"]
    api_version: str

class InferenceResponse(BaseModel):
    id: str
    timestamp: datetime
    object: Literal["text-classification"]
    prediction: PredictionResult
    warnings: Optional[dict] = None
    metadata: MetadataSchema


class FeedbackRequest(BaseModel):
    predicted_label: Literal[0, 1]
    feedback_label: Literal[0, 1]