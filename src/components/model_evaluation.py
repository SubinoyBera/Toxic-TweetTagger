from src.components import *
import os
import numpy as np
import json
import dagshub
import mlflow
from mlflow.xgboost import log_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dotenv import load_dotenv
load_dotenv()

# get environment variables
uri = os.getenv("MLFOW_URI")
repo_owner = os.getenv("OWNER")
repo_name = os.getenv("REPO")

if uri is not None:
	raise ValueError("MLFLOW_URI environment variable is not set.")
mlflow.set_tracking_uri(uri)    # type: ignore

if repo_owner is None:
	raise ValueError("Missing dagshub logging environment credentials.")
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)   # type: ignore


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
            with open(Path(f"reports/metrics.json")) as f:
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
            with open(Path("reports/experiment.json")) as f:
                json.dump(model_info, f, indent=4)
        
        except Exception as e:
            logging.error(f"Failed to save model experiment info: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def evaluate(self, model, model_name, df):
        """
        Evaluates the given model on the test data and returns an evaluation report.
        Args:
            model: The model to be evaluated.
            model_name: The name of the model.
            df: The test dataframe.

        Returns:
            tuple: A tuple containing the evaluation report and the model parameters.
        """
        try:
            y_test = df['Label'].values
            X_test = df.drop(columns='Label')

            pred_score = model.predict(X_test)
            y_pred = (np.array(pred_score) > 0.5).astype(int)

            evaluation_report = {
                "model" : model_name,
                "accuracy" : accuracy_score(y_test, y_pred),
                "precision" : precision_score(y_test, y_pred),
                "recall" : recall_score(y_test, y_pred),
                "f1 score" : f1_score(y_test, y_pred),
                "roc_auc" : roc_auc_score(y_test, pred_score)
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
    test_data_path = obj.evaluation_config.test_data_path
    model_path = obj.evaluation_config.models_dir
    model_name = obj.evaluation_config.trained_model_name

    mlflow.set_experiment("DVC Pipeline Model Experiments")
    with mlflow.start_run(run_name=model_name) as run:
        try:
            test_df = pd.read_csv(test_data_path)
            model = load_obj(location_path=model_path, obj_name=f"{model_name}.pkl")
            
            evaluation_report, model_params = obj.evaluate(model, model_name, test_df)
            
            obj.save_report(evaluation_report)

            logging.info("Logging model and different parameters in Mlflow")
            # log model metrics in Mlflow
            for metric_name, metric_score in evaluation_report.items():
                mlflow.log_metric(metric_name, metric_score)

            # log model parameters in Mlflow
            if model_params is not None:
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)
            else:
                logging.warning("No model parameters found. Skipping parameter logging")   
            
            # log model in Mlflow
            log_model(model, "model")

            # log model evaluation metrics file to Mlfow
            mlflow.log_artifact("reports/model_evaluation.json")

            # save model info
            obj.save_model_info(model_name=model_name, run_id=run.info.run_id)
            logging.info("Model evaluation completed")

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}", exc_info=True)
            raise AppException(e, sys)

# entry point for the model evaluation process  
if __name__ == "__main__":
    initiate_model_evaluation()