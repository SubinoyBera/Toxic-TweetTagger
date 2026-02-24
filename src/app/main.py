import sys
import yaml, json
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

import xgboost as xgb
from lime.lime_text import LimeTextExplainer
from src.core.constants import REGISTERED_MODELS_DIR
from src.app.workers import BufferedEventConsumerWorker
from src.app.services.inference import InferenceService
from src.app.services.feedback import FeedbackService
from src.app.services.explainer import ExplainerService

from src.utils import load_obj
from src.app.middleware import http_observability_middleware
from src.core.constants import DATABASE_NAME, PRODUCTION_COLLECTION_NAME, FEEDBACK_COLLECTION_NAME
from src.core.mongo_client import MongoDBClient
from src.core.logger import logging
from src.core.exception import AppException

from src.app.api.api_routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # application state setup
    try:
        mongo_client = MongoDBClient()
        # load model
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(Path(REGISTERED_MODELS_DIR, "artifacts", "booster.json"))
        # load vectorizer
        vectorizer = load_obj(Path(REGISTERED_MODELS_DIR, "artifacts"), "vectorizer.joblib")

        with open(Path(REGISTERED_MODELS_DIR, "artifacts/metrics.json"), 'r') as f:
            metrics = json.load(f)
            eval_threshold = metrics.get("threshold", 0.5)

        # get model version
        with open(Path("src/app/model/registered_model_meta"), 'r') as f:
            model_metadata = yaml.safe_load(f)
        if not model_metadata:
            raise FileNotFoundError("Failed to load file having model metadata")
        model_version = int(model_metadata.get("model_version", 0))

        # initialize workers
        prediction_event_consumer = BufferedEventConsumerWorker(mongo_client, DATABASE_NAME, PRODUCTION_COLLECTION_NAME)
        feedback_event_consumer = BufferedEventConsumerWorker(mongo_client, DATABASE_NAME, FEEDBACK_COLLECTION_NAME)
        
        # initialize services
        app.state.prediction_service = InferenceService(xgb_booster, vectorizer, eval_threshold, 
                                                        prediction_event_consumer, model_version)
        
        app.state.feedback_service = FeedbackService(feedback_event_consumer)

        lime_explainer = LimeTextExplainer(class_names=["hate", "non-hate"], bow=False)
        app.state.explainer_service = ExplainerService(lime_explainer, app.state.model, app.state.vectorizer)

        logging.info("Infernce API app server started successfully")

    except Exception as e:
        logging.critical(f"Startup Failed: {e}", exc_info=True)
        raise AppException(e, sys)
    
    # run application
    yield
    
    # application shutdown
    app.state.prediction_event_consumer.shutdown()
    app.state.feedback_event_consumer.shutdown()
    mongo_client.close_connection()

# Create FastAPI app
app = FastAPI(
    title="Hate Speech Detection API",
    version="2.0.0",
    description="Production-grade ML inference API with monitoring and feedback system.",
    lifespan=lifespan
)

# Register API routes
app.include_router(router, prefix="/api")

# Register middleware
app.middleware("http")(http_observability_middleware)