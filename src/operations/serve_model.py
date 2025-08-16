# Upload the production model from registry to huggingface spaces for serving

import os, sys
import mlflow
import dagshub
from mlflow import artifacts
from pathlib import Path
from huggingface_hub import upload_folder
from src.constant.constants import HUGGINGFACE_REPO_ID
from src.core.logger import logging
from src.core.exception import AppException
from dotenv import load_dotenv
load_dotenv()

# get environment variables
uri = os.getenv("MLFLOW_URI")
dagshub_token = os.getenv("DAGSHUB_TOKEN")
dagshub_username = os.getenv("OWNER")
hf_token = os.getenv("HUGGINGFACE_TOKEN")

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

def serve_model():
    """
    This function serves the production model by downloading the model artifacts from MLflow 
    model registry and uploads it along with the inference api to Huggingface space for model serving.
    """
    try:
        model_name = "ToxicTagger-Models"
        stage = "Production"
        model_uri = f"models:/{model_name}/{stage}"

        logging.info(f"Downloading model artifacts from MLflow model registry")
        artifacts.download_artifacts(artifact_uri=model_uri, dst_path="serve_api/model")

        if not hf_token:
            raise EnvironmentError("Huggingface access token is not set")

        logging.info(f"Uploading to huggingface space for model serving")
        upload_folder(
            folder_path = Path("hf_serve_api"),
            repo_id = HUGGINGFACE_REPO_ID,
            repo_type = "space",
            token = hf_token,
            delete_patterns = ["**"]
        )
        logging.info("Successfully uploaded!")

    except Exception as e:
        logging.error(f"Model serving operation failed: {e}", exc_info=True)
        raise AppException(e, sys)