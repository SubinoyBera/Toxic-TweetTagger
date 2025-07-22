# Project pipeline configuration 
import sys
from pathlib import Path
from ..core.logger import logging
from ..constant.constants import *
from ..utils.common import read_yaml, create_directory
from ..core.exception import AppException
from ..core.config_entity import (DataIngestionConfig, DataPreprocessingConfig, FeatureEngineeringConfig, 
                                  ModelTrainingConfig, ModelEvaluationConfig, ModelRegistrationConfig)

class AppConfiguration:
    def __init__(self, 
                config_filepath : Path = CONFIG_FILE_PATH):
        """
        Initializes the configuration object by reading configuration from config.yaml file..

        Parameters:
        config_filepath (str): Path to the configuration file.
        """
        try:
            self.config = read_yaml(config_filepath)

        except Exception as e:
            logging.error(f"Failed to load configuration: {e}", exc_info=True)
            raise AppException(e, sys)
        
    
    def data_ingestion_config(self) -> DataIngestionConfig:
        """
        Creates the configuration for Data Ingestion 
        Returns: DataIngestionConfig object
        """
        try:
            ingestion_config = self.config.data_ingestion

            ingestion_root = Path(ingestion_config.root_dir)
            raw_dir = ingestion_config.raw_data_dir
            ingested_data_dir = ingestion_config.ingested_data_dir
            download_url = ingestion_config.download_url

            create_directory(ingestion_root)

            raw_data_path = Path(ingestion_root, raw_dir)
            ingested_data_path = Path(ingestion_root, ingested_data_dir)

            ingestion_configuration = DataIngestionConfig(
                raw_data_dir = raw_data_path,
                ingested_data_dir = ingested_data_path,
                data_download_url = download_url
            )
        
            logging.info("Data Ingestion Configuration creation successfull")
            return ingestion_configuration
        
        except Exception as e:
            logging.error(f"Error while creating Data Ingestion Configuration: {e}", exc_info=True)
            raise AppException(e, sys)


    def data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        Creates the configuration for Data Preprocessing.
        Returns: DataPreprocessingConfig object
        """
        try:
            preprocessing_config = self.config.data_preprocessing
            ingestion_configuration = self.data_ingestion_config()

            preprocessing_root = Path(preprocessing_config.root_dir)
            dataset = preprocessing_config.dataset

            create_directory(preprocessing_root)

            dataset_path = Path(ingestion_configuration.ingested_data_dir, dataset)

            preprocessing_configuration = DataPreprocessingConfig(
                preprocessed_data_dir = preprocessing_root,
                ingested_dataset_path = dataset_path
            )

            return preprocessing_configuration
        
        except Exception as e:
            logging.error(f"Error while creating Data Preprocessing Configuration: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def feature_engineering_config(self) -> FeatureEngineeringConfig:
        """
        Creates the configuration for Feature Engineering.
        Returns: FeatureEngineeringConfig object
        """
        try:
            feature_eng_config = self.config.feature_engineering
            preprocessing_configuration = self.data_preprocessing_config()

            feature_eng_root_dir = Path(feature_eng_config.root_dir)
            models_root_dir = Path(feature_eng_config.models_dir)

            create_directory(feature_eng_root_dir)
            create_directory(models_root_dir)

            preprocessed_data = feature_eng_config.preprocessed_dataset
            preprocessed_data_path = Path(preprocessing_configuration.preprocessed_data_dir, preprocessed_data)

            feature_engineering_configuration = FeatureEngineeringConfig(
                models_dir = models_root_dir,
                preprocessed_data_path = preprocessed_data_path,
                train_test_data_path = feature_eng_root_dir
            )

            return feature_engineering_configuration
        
        except Exception as e:
            logging.error(f"Error while creating Feature Engineering Configuration: {e}", exc_info=True)
            raise AppException(e, sys)


    def model_training_config(self) -> ModelTrainingConfig:
        """
        Creates the configuration for Model Training.
        Returns: ModelTrainingConfig object
        """
        try:
            training_config = self.config.model_training
            feature_engineering_configuration = self.feature_engineering_config()

            models_dir_path = Path(training_config.model_dir)
            train_data_path = feature_engineering_configuration.train_test_data_path

            training_configuration = ModelTrainingConfig(
                train_data_path = train_data_path,
                models_dir = models_dir_path,
            )
        
            return training_configuration

        except Exception as e:
            logging.error(f"Error while creating Model Training Configuration: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Creates the configuration for Model Training.
        Returns: ModelTrainingConfig object
        """
        try:
            evaluation_config = self.config.model_evaluation
            feature_engineering_configuration = self.feature_engineering_config()
            training_configuration = self.model_training_config()

            models_dir_path = training_configuration.models_dir
            eval_report_filename = evaluation_config.evaluation_report
            exp_info_filename = evaluation_config.experiment_info

            reports_dir_path = evaluation_config.reports_dir
            create_directory(reports_dir_path)

            eval_report_filepath = Path(reports_dir_path, eval_report_filename)
            exp_info_filepath = Path(reports_dir_path, exp_info_filename)

            test_data_path = feature_engineering_configuration.train_test_data_path

            evaluation_configuration = ModelEvaluationConfig(
                    test_data_path = test_data_path,
                    models_dir = models_dir_path,
                    evaluation_report_filepath = eval_report_filepath,
                    experiment_info_filepath = exp_info_filepath
                )

            return evaluation_configuration

        except Exception as e:
            logging.error(f"Error while creating Model Training Configuration: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def model_registration_config(self) -> ModelRegistrationConfig:
        try:
            evaluation_configuration = self.model_evaluation_config()
            exp_info_filepath = evaluation_configuration.experiment_info_filepath

            return ModelRegistrationConfig(
                experiment_info_filepath = exp_info_filepath
            )
        
        except Exception as e:
            logging.error(f"Error while creating Model Evaluation Configuration: {e}", exc_info=True)
            raise AppException(e, sys)