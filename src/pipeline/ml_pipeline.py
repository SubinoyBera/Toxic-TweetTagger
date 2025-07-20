from core.logger.logging import logging
from core.exception.exceptions import AppException
from components.data_ingestion import initiate_data_ingestion
from components.data_preprocessing import initiate_data_preprocessing
from components.feature_engineering import initiate_feature_engineering
from components.model_training import initiate_model_training
from components.model_evaluation import initiate_model_evaluation
from components.register_model import initiate_model_registration
import sys
from prefect import task, flow

@task
def data_ingestion():
    logging.info("\n\n STAGE:1 Data Ingestion Stage Initiated")
    initiate_data_ingestion()

@task
def data_preprocessing():
    logging.info("STAGE:2 Data Preprocessing Stage Initiated")
    initiate_data_preprocessing()

@task
def feature_engineering():
    logging.info("STAGE:3 Feature Engineering Stage Initiated")
    initiate_feature_engineering()

@task()
def model_training():
    logging.info("STAGE:4 Model Training Stage Initiated")
    initiate_model_training()

@task
def model_evaluation():
    logging.info("STAGE:5 Model Evaluation Stage Initiated")
    initiate_model_evaluation()

@task
def register_model():
    logging.info("STAGE:6 Model Registration Stage Initiated")
    initiate_model_registration()

@flow(name="ML Pipeline")
def run_pipeline():
    try:
        logging.info("ML Pipeline started")
        data_ingestion()
        data_preprocessing()
        feature_engineering()
        model_training()
        model_evaluation()
        register_model()
        logging.info("ML Pipline Completed")

    except Exception as e:
        logging.error(f"ML Pipeline Terminated: {e}", exc_info=True)
        raise AppException(e, sys)
    
# entry point for running the ML Pipeline
if __name__ == "__main__":
    run_pipeline()