import sys, os
import json
import mlflow
import dagshub
from pathlib import Path
from src.core.logger import logging
from src.core.exception import AppException
from src.core.configuration import AppConfiguration
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

mlflow.set_tracking_uri(uri)         # type: ignore

# For local use
# ==================================================================================
# repo_owner = os.getenv("OWNER")
# repo_name = os.getenv("REPO")
# mlflow.set_tracking_uri(uri)
# 
# if repo_owner is None:
# 	raise EnvironmentError("Missing dagshub logging environment credentials.")
# dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True) 
# ===================================================================================


class ModelRegistration:
    def __init__(self, config = AppConfiguration()):
        """
        Initializes the DataPreprocessing object by creating a data preprocessing configuration.
        Raises:
            AppException: If error occurs during creation of data preprocessing configuration
        """
        try:
            self.registration_config = config.model_registration_config()

        except Exception as e:
            logging.error(f"Failed to create model registration configuration: {e}", exc_info=True)
            raise AppException(e, sys)


    def load_exp_info(self, exp_info_file: Path):
        """
        Loads the experiment information from the saved JSON file.
        Args:
            exp_info_file (Path): The file path to the 'experiment_info.json' file.
        Returns:
            dict: A dictionary containing the experiment information if the file is found.
        """
        try:
            if not os.path.exists(exp_info_file):
                raise FileNotFoundError("'experiment_info.json' file not found")
            with open(exp_info_file, 'r') as file:
                exp_info = json.load(file)
                return exp_info
            
        except Exception as e:
            logging.error(f"Failed to get model experiment information: {e}", exc_info=True)
            raise AppException(e, sys)
        
        
    def register(self, experiment_info: dict, register_modelname: str):
        """
        Registers a model in Mlflow using the provided experiment information and model name.
        Transitions the registered model version to the "Staging" stage.

        Args:
            experiment_info (dict): dictionary containing experiment details:
                                    'run_id' and 'model_name'.
            register_modelname (str): name under which the model to be registered in Mlflow..
        """
        try:
            model_uri = f"runs:/{experiment_info['run_id']}/{experiment_info['model']}"

            logging.info("Registering model in Mlflow")
            model_version = mlflow.register_model(model_uri, register_modelname)

            # Transition the model to "Staging" stage
            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name = register_modelname,
                version = model_version.version,
                stage = "Staging"
            )
            logging.info(f"{register_modelname} - version : {model_version.version} registered and transitioned to Staging")
        
        except Exception as e:
            logging.error(f"Error during model registration in Mlflow: {e}", exc_info=True)
            raise AppException(e, sys)


def register_model():
    """
    Main function to handle the model registration process in Mlflow.
    Initializes a RegisterModel object, loads experiment information from a JSON file,
    and registers the model in Mlflow. The registered model is transitioned to the "Staging" stage.

    Raises:
        AppException: If an error occurs during model registration.
    """
    obj = ModelRegistration()
    try:
        logging.info(f"{'='*20}Model Registration{'='*20}")
        exp_info_filepath = obj.registration_config.experiment_info_filepath

        experiment_info = obj.load_exp_info(exp_info_filepath)
        register_modelname = "ToxicTagger-Models"

        if type(experiment_info) is not dict:
            logging.error("'register_model' function expects dict type object ")
            return
        obj.register(experiment_info, register_modelname)
        logging.info(f"{'='*20}Model Registration in Mlflow Completed Successfully{'='*20} \n\n")

    except Exception as e:
        logging.error(f"Error during model registration in Mlflow: {e}", exc_info=True)
        raise AppException(e, sys)
    

if __name__ == "__main__":
    register_model()