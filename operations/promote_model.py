# This script promotes a model from staging to production in MLflow.
import os, sys
import mlflow
from src.core.logger import logging
from src.core.exception import AppException
from dotenv import load_dotenv
load_dotenv()

def promote_model():
    try:
        # Set up MLflow tracking URI and authentication
        uri = os.getenv("MLFOW_URI")
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("Dagshub Token environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(uri)        # type: ignore

        client = mlflow.MlflowClient()

        model_name = "ToxicTagger-Models"
        latest_version_staging = client.get_latest_versions(name=model_name, stages=["Staging"])
        model_version_staging = latest_version_staging[0].version

        logging.info("Archiving existing production model versions")
        prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])
        for version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
        logging.info("Transitioning model version from Staging to Production")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version_staging,
            stage="Production"
        )
        logging.info(f"Model version {model_version_staging} successfully promoted to production stage")

    except Exception as e:
        logging.error(f"Failed to promote model to production stage: {e}", exc_info=True)
        raise AppException(e, sys)

if __name__ == "__main__":
    promote_model()