# This module is responsible for loading the production model artifacts from MLflow registry and packages in the app folder for serving.

import os, sys
import mlflow
import dagshub
from mlflow import artifacts
from src.core.logger import logging
from src.core.exception import AppException
from dotenv import load_dotenv
load_dotenv()

# get environment variables
uri = os.getenv("MLFLOW_URI")
dagshub_token = os.getenv("DAGSHUB_TOKEN")
dagshub_username = os.getenv("OWNER")

if not dagshub_token or not dagshub_username:
    raise EnvironmentError("Dagshub environment variables is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(uri)        # type: ignore

# For local use
# ==============================================================================
# repo_owner = os.getenv("OWNER")
# repo_name = os.getenv("REPO")
# 
# mlflow.set_tracking_uri(uri) 
# if repo_owner is None:
# 	raise EnvironmentError("Missing dagshub logging environment credentials.")
# dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
# ===============================================================================

def load_model():
    """
    Download the production model artifacts from MLflow model registry and to the app folder.

    Raises:
        AppException: If there is an error during the model loading process.
        EnvironmentError: If the Dagshub Token environment variable is not set.
    """
    try:
        model_name = "ToxicTagger-Models"
        stage = "Production"
        model_uri = f"models:/{model_name}/{stage}"

        logging.info(f"Downloading model artifacts from MLflow model registry")
        artifacts.download_artifacts(artifact_uri=model_uri, dst_path="src/app/model")
        logging.info("Successfully downloaded model artifacts!")

    except Exception as e:
        logging.error(f"Model loading operation failed: {e}", exc_info=True)
        raise AppException(e, sys)
    

if __name__ == "__main__":
    load_model()