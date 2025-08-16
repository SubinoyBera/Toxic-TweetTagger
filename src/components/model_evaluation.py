import sys, os
import json
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from .data_preprocessing import DataPreprocessing
from src.core.logger import logging
from src.core.exception import AppException
from src.core.configuration import AppConfiguration
from src.utils import create_directory, read_yaml, load_obj
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score)
import gc
import dagshub
import mlflow
from src.script.model_wrapper import CustomModel
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

mlflow.set_tracking_uri(uri)            # type: ignore

# For local use
# =================================================================================
# repo_owner = os.getenv("OWNER")
# repo_name = os.getenv("REPO")
# 
# mlflow.set_tracking_uri(uri)
# if repo_owner is None:
# 	raise EnvironmentError("Missing dagshub logging environment credentials.")
# dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True) 
# ==================================================================================


class ModelEvaluation:
    def __init__(self, config = AppConfiguration()):
        """
        Initializes the ModelEvaluation object by creating a model evaluation configuration.
        Args:
            config (AppConfiguration): The configuration object containing the application configuration.
        """
        try:
            self.evaluation_config = config.model_evaluation_config()

        except Exception as e:
            logging.error(f"Failed to create model evaluation configuration: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def save_report(self, report: dict):
        """
        Saves the model evaluation metrics to a JSON file in the "reports" directory.
        Args:
            report (dict): A dictionary containing the model evaluation metrics.
        """
        try:
            create_directory(Path("reports"))
            with open("reports/metrics.json", 'w')  as f:
                json.dump(report, f, indent=4)
        
        except Exception as e:
            logging.error(f"Failed to save model evaluation report: {e}", exc_info=True)
            raise AppException(e, sys)
        
    
    def save_experiment_info(self, model_name: str, run_id: str):
        """
        Saves the model name and the Mlflow run ID to a JSON file 
        for logging purposes.
        Args:
            model_name (str): The name of the model
            run_id (str): The Mlflow run ID
        """
        try:
            experiment_info = {
                'model' : model_name,
                'run_id' : run_id,
            }
            exp_info_path = Path("reports/experiment.json")
            with open(exp_info_path, 'w') as f:
                json.dump(experiment_info, f, indent=4)
        
        except Exception as e:
            logging.error(f"Failed to save model experiment info: {e}", exc_info=True)
            raise AppException(e, sys)
        
    def evaluate(self, model:Any, model_name:str, vectorizer:Any, vectorizer_name:str, df:pd.DataFrame) -> tuple[dict, dict]:
        """
        Evaluates the given model and vectorizer on the given dataset.
        Args:
            model (Any): The model to evaluate.
            model_name (str): The name of the model.
            vectorizer (Any): The vectorizer to use.
            vectorizer_name (str): The name of the vectorizer.
            df (pd.DataFrame): The test dataset to evaluate on.
        
        Returns:
            tuple: A tuple containing the evaluation report and the model parameters.
        """
        try:
            X_test = df.drop(columns='Label')
            # Perform vectorization
            X_test = vectorizer.transform(X_test["Content"])
            
            logging.info(f"Evaluating model: {model_name}")
            # Predict using the model
            y_pred = model.predict(X_test)
            # Get the predicted probabilities for the positive class
            proba_class1 = model.predict_proba(X_test)[:, 1]
 
            # Get the true labels
            y_test = df['Label'].values

            evaluation_report = {
                "model": model_name,
                "vectorizer": vectorizer_name,
                "accuracy": accuracy_score(y_test, y_pred),                 # type: ignore
                "precision": precision_score(y_test, y_pred),               # type: ignore
                "recall": recall_score(y_test, y_pred),                     # type: ignore
                "f1 score": f1_score(y_test, y_pred),                       # type: ignore
                "roc_auc": roc_auc_score(y_test, proba_class1)              # type: ignore
            }
            # Get model parameters
            model_params: dict = model.get_xgb_params()

            return (
                evaluation_report,
                model_params
            )
        
        except Exception as e:
            logging.error(f"Failed in model evaluation process: {e}", exc_info=True)
            raise AppException(e, sys)
    

def initiate_model_evaluation():
    """
    Initiates the model evaluation process by creating a ModelEvaluation object,
    which then evaluates the model using the evaluate method and logs the
    evaluation metrics in Mlflow.
    """
    eval_obj = ModelEvaluation()
    preprocessor = DataPreprocessing()
    logging.info(f"{'='*20}Model Evaluation{'='*20}")

    # get model name
    config_params = read_yaml(Path("params.yaml"))
    model_name = config_params.model_training.model_name
    vectorizer_name = config_params.feature_engineering.vectorizer

    test_data_path = eval_obj.evaluation_config.test_data_path
    model_path = eval_obj.evaluation_config.models_dir

    test_df = pd.read_parquet(test_data_path)
    test_df.dropna(how='any', inplace=True)
    processed_test_df = preprocessor.preprocess(test_df, filename="preprocessed_test_data.feather")

    mlflow.set_experiment("DVC Pipeline Model Experiments")
    with mlflow.start_run(run_name=model_name) as run:
        try:
            model = load_obj(location_path=model_path, obj_name=f"{model_name}.joblib")
            vectorizer = load_obj(location_path=model_path, obj_name=f"{vectorizer_name}.joblib")

            evaluation_report, model_params = eval_obj.evaluate(model=model, model_name=model_name, vectorizer=vectorizer,
                                                                vectorizer_name=vectorizer_name,df=processed_test_df)

            eval_obj.save_report(evaluation_report)
            logging.info("Logging model and different parameters in Mlflow")

            # Log model evaluation metrics in Mlflow
            # remove the model and vectorizer name from the report
            evaluation_report.pop("model", None)
            evaluation_report.pop("vectorizer", None)

            for metric_name, metric_score in evaluation_report.items():
               mlflow.log_metric(metric_name, metric_score)

            # log model parameters in Mlflow
            if model_params is not None:
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)
                mlflow.log_param("model", model_name)
                mlflow.log_param("vectorizer", vectorizer_name)
            else:
                logging.warning("No model parameters found. Skipping parameter logging")   

            # Log the Custom Model in Mlflow
            # Custom Model wraps the actual model classifer and the vectorizer into a single Python Model object
            final_model = CustomModel(model=model, vectorizer=vectorizer)
            mlflow.pyfunc.log_model(
                artifact_path = model_name,
                python_model = final_model,
                artifacts={
                    "vectorizer": os.path.join(model_path, f"{vectorizer_name}.joblib"),
                    "classifier": os.path.join(model_path, f"{model_name}.joblib")
                }
            )
            # log model evaluation metrics file to Mlfow
            mlflow.log_artifact("reports/metrics.json")
            # save model info
            eval_obj.save_experiment_info(model_name=model_name, run_id=run.info.run_id)
            logging.info("Model evaluation completed")

            #free memory
            del test_df, processed_test_df, model, vectorizer
            gc.collect()

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}", exc_info=True)
            raise AppException(e, sys)

 
if __name__ == "__main__":
    initiate_model_evaluation()