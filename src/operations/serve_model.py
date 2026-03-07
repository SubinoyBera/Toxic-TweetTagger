# Uploads the production model along with the FastAPI backend API services to Huggingface Spaces for serving

import os, sys
from pathlib import Path
from huggingface_hub import upload_folder
from src.core.constants import HUGGINGFACE_REPO_ID
from src.core.logger import logging
from src.core.exception import AppException

from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")

def serve_model():
    """
    Serves and deploys the model along with the API services to Huggingface Spaces.

    Uploads the app folder containing the FastAPI backend API services and the production model artifacts to
    Huggingface Spaces.
    """
    try:
        if not hf_token:
            raise EnvironmentError("Huggingface access token is not set")

        logging.info(f"Deploying ML application to Huggingface Spaces")
        
        upload_folder(
            folder_path = Path("src"),
            repo_id = HUGGINGFACE_REPO_ID,
            repo_type = "space",
            token = hf_token,
            delete_patterns = ["**"]
        )
        
        logging.info("Successfully uploaded!")

    except Exception as e:
        logging.error(f"Deployment failed: {e}", exc_info=True)
        raise AppException(e, sys)
    

if __name__ == "__main__":
    serve_model()