import os
import numpy as np
import json
from typing import Any
from ..components import *
from scipy.sparse import csr_matrix
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score)
import dagshub
import mlflow
import mlflow.pyfunc
from mlflow.pyfunc.model import PythonModel

from dotenv import load_dotenv
load_dotenv()

# get environment variables
uri = os.getenv("MLFOW_URI")
repo_owner = os.getenv("OWNER")
repo_name = os.getenv("REPO")

# set up connection with dagshub
mlflow.set_tracking_uri(uri)    # type: ignore

if repo_owner is None:
	raise ValueError("Missing dagshub logging environment credentials.")
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)   # type: ignore


class CustomModel(PythonModel):
    def __init__(self, model=None, vectorizer=None):
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, context, model_input: pd.DataFrame):
        texts = model_input["texts"]
        if self.vectorizer is not None and self.model is not None:
            X = self.vectorizer.transform(texts)
            X = csr_matrix(X)
            X = pd.DataFrame(X.toarray())
            class_probalility_scores = self.model.predict_proba(X)
            class_label = self.model.predict(X)

        return {
            "class_probalility_scores" : class_probalility_scores,
            "class_label" : class_label
        }


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
        
    
    def save_model_info(self, model_name: str, run_id: str):
        """
        Saves the model name and the Mlflow run ID to a JSON file 
        for logging purposes.
        Args:
            model_name (str): The name of the model
            run_id (str): The Mlflow run ID
        """
        try:
            model_info = {
                'model' : model_name,
                'run_id' : run_id,
            }

            exp_info_path = Path("reports/experiment.json")
            with open(exp_info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
        
        except Exception as e:
            logging.error(f"Failed to save model experiment info: {e}", exc_info=True)
            raise AppException(e, sys)
        
    def evaluate(self, model: Any , model_name: str, df):
        """
        Evaluates the given model on the test data and returns an evaluation report.
        Args:
            model: The model to be evaluated.
            model_name: The name of the model.
            df: The test dataframe.
            df: The test dataframe.

        Returns:
            tuple: A tuple containing the evaluation report and the model parameters.
        """
        try:
            y_test = df['Label'].values
            X_test = df.drop(columns='Label')
            logging.info(type(model))
            
            logging.info(f"Evaluating model: {model_name}")
            y_pred = model.predict(X_test)
            proba_class1 = model.predict_proba(X_test)[:, 1]

            evaluation_report = {
                "model" : model_name,
                "accuracy" : accuracy_score(y_test, y_pred),
                "precision" : precision_score(y_test, y_pred),
                "recall" : recall_score(y_test, y_pred),
                "f1 score" : f1_score(y_test, y_pred),
                "roc_auc" : roc_auc_score(y_test, proba_class1)
            }
            model_params = model.get_xgb_params()

            return (
                evaluation_report,
                model_params )
        
        except Exception as e:
            logging.error(f"Failed in model evaluation process: {e}", exc_info=True)
            raise AppException(e, sys)
    

def initiate_model_evaluation():
    """
    Main function to evaluate the model and log the evaluation metrics, parameters, and the model in Mlflow.

    This function reads the test data, loads the trained model, evaluates the model, logs the evaluation metrics and model parameters in Mlflow, and saves the model evaluation metrics to a file.

    Raises:
        AppException: If an error occurs during model evaluation.
    """
    obj = ModelEvaluation()
    # get model name
    config_params = read_yaml(Path("params.yaml"))
    model_name = config_params.model_training.model_name
    vectorizer_name = config_params.feature_engineering.vectorizer

    test_data_path = obj.evaluation_config.test_data_path
    model_path = obj.evaluation_config.models_dir

    mlflow.set_experiment("DVC Pipeline Model Experiments")
    with mlflow.start_run(run_name=model_name) as run:
        try:
            test_df = pd.read_feather(test_data_path)
            test_df.dropna(how='any', inplace=True)

            model = load_obj(location_path=model_path, obj_name=f"{model_name}.joblib")
            vectorizer = load_obj(location_path=model_path, obj_name=f"{vectorizer_name}.joblib")

            evaluation_report, model_params = obj.evaluate(model=model, model_name=model_name, df=test_df)

            obj.save_report(evaluation_report)
            logging.info("Logging model and different parameters in Mlflow")

            # log model metrics in Mlflow
            model_name = evaluation_report.pop("model", None)
            
            for metric_name, metric_score in evaluation_report.items():
               mlflow.log_metric(metric_name, metric_score)

            # log model parameters in Mlflow
            if model_params is not None:
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)
                    mlflow.log_param("models", model_name)
            else:
                logging.warning("No model parameters found. Skipping parameter logging")   

            # log model in Mlflow
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
            obj.save_model_info(model_name=model_name, run_id=run.info.run_id)
            logging.info("Model evaluation completed")

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}", exc_info=True)
            raise AppException(e, sys)

# entry point for the model evaluation process  
if __name__ == "__main__":
    initiate_model_evaluation()