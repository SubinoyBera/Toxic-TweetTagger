# Project pipeline configuration 
import sys
from pathlib import Path
from src.logger.logging import logging
from src.constants.constants import *
from src.utils.common import read_yaml, create_directory
from src.exception.app_exception import AppException
from src.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig,FeatureEngineeringConfig,
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

            ingestion_root = ingestion_config.root_dir
            raw_dir = ingestion_config.raw_data_dir
            ingested_data_dir = ingestion_config.ingested_data_dir
            download_url = ingestion_config.download_url

            create_directory(ingestion_root)

            raw_data_path = Path(ingestion_root, raw_dir)
            ingested_data_path = Path(ingestion_root, ingested_data_dir)

            self.ingestion_configuration = DataIngestionConfig(
                raw_data_dir = raw_data_path,
                ingested_data_dir = ingested_data_path,
                data_download_url = download_url
            )
        
            logging.info("Data Ingestion Configuration creation successfull")
            return self.ingestion_configuration
        
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

            preprocessing_root = Path(preprocessing_config.root_dir)
            dataset = preprocessing_config.dataset

            create_directory(preprocessing_root)

            dataset_path = Path(self.ingestion_configuration.ingested_data_dir, dataset)

            self.preprocessing_configuration = DataPreprocessingConfig(
                preprocessed_data_dir = preprocessing_root,
                ingested_dataset_path = dataset_path
            )

            return self.preprocessing_configuration
        
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

            feature_eng_root_dir = Path(feature_eng_config.root_dir)
            models_root_dir = Path(feature_eng_config.models_dir)

            create_directory(feature_eng_root_dir)
            create_directory(models_root_dir)

            preprocessed_data = feature_eng_config.preprocessed_dataset
            preprocessed_data_path = Path(self.preprocessing_configuration.preprocessed_data_dir, preprocessed_data)

            self.feature_engineering_configuration = FeatureEngineeringConfig(
                models_dir = models_root_dir,
                preprocessed_data_path = preprocessed_data_path,
                train_test_data_path = feature_eng_root_dir
            )

            return self.feature_engineering_configuration
        
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

            models_dir_path = Path(training_config.model_dir)
            model_name = training_config.model_name
            train_data_path = self.feature_engineering_configuration.train_test_data_path

            self.training_configuration = ModelTrainingConfig(
                train_data_path = train_data_path,
                models_dir = models_dir_path,
                model_name = model_name
            )
        
            return self.training_configuration

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

            models_dir_path = Path(evaluation_config.model_dir)
            model_name = self.training_configuration.model_name
            eval_report_filename = evaluation_config.evaluation_report
            exp_info_filename = evaluation_config.experiment_info

            reports_dir_path = evaluation_config.reports_dir
            create_directory(reports_dir_path)

            eval_report_filepath = Path(reports_dir_path, eval_report_filename)
            exp_info_filepath = Path(reports_dir_path, exp_info_filename)

            test_data_path = self.feature_engineering_configuration.train_test_data_path

            self.evaluation_configuration = ModelEvaluationConfig(
                    test_data_path = test_data_path,
                    models_dir = models_dir_path,
                    trained_model_name = model_name,
                    evaluation_report_filepath = eval_report_filepath,
                    experiment_info_filepath = exp_info_filepath
                )

            return self.evaluation_configuration

        except Exception as e:
            logging.error(f"Error while creating Model Training Configuration: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def model_registration_config(self) -> ModelRegistrationConfig:
        exp_info_filepath = self.evaluation_configuration.experiment_info_filepath

        return ModelRegistrationConfig(
            experiment_info_filepath = exp_info_filepath
        )