import sys
from ..core.logger import logging
from ..core.exception import AppException
from ..components.data_ingestion import initiate_data_ingestion
from ..components.data_preprocessing import initiate_data_preprocessing
from ..components.feature_engineering import initiate_feature_engineering
from ..components.model_training import initiate_model_training
from ..components.model_evaluation import initiate_model_evaluation
from ..components.register_model import initiate_model_registration


def run_pipeline():
    """
    This function is the entry point for the Machine Learning pipeline.
    It starts all the stages of the pipeline one by one and logs the status of each stage.
    If any error occurs in any stage, it logs the error and raises an AppException with 
    the error message.
    If all stages are completed successfully, it logs a success message.
    """
    try:
        logging.info("STAGE:1 Data Ingestion Stage Initiated")
        #initiate_data_ingestion()

        logging.info("STAGE:2 Data Preprocessing Stage Initiated")
        #initiate_data_preprocessing()

        logging.info("STAGE:3 Feature Engineering Stage Initiated")
        #initiate_feature_engineering()

        logging.info("STAGE:4 Model Training Stage Initiated")
        #   initiate_model_training()

        logging.info("STAGE:5 Model Evaluation Stage Initiated")
        #initiate_model_evaluation()

        logging.info("STAGE:6 Model Registration Stage Initiated")
        initiate_model_registration()

        logging.info("ML Pipeline Completed")

    except Exception as e:
        logging.error(f"ML Pipeline Terminated: {e}", exc_info=True)
        raise AppException(e, sys)
    
# entry point for running the ML Pipeline
if __name__ == "__main__":
    run_pipeline()