# This script promotes a model from staging to production in MLflow.
import os, sys
import mlflow
from src.core.logger import logging
from src.core.exception import AppException

def promote_model():
    """
    Promotes a model from staging to production in MLflow.

    This function sets up MLflow tracking URI and authentication using environment variables,
    retrieves the latest version of the model in the Staging stage, archives any existing
    Production model versions, and transitions the selected model version to the Production stage.
    Raises:
        AppException: If there is an error during the promotion process.
        EnvironmentError: If the Dagshub Token environment variable is not set.
    """
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