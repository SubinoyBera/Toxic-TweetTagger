# Project pipeline configuration 
import sys
from pathlib import Path
from src.logger.logging import logging
from src.constants.constants import *
from src.utils.common import read_yaml, create_directory
from src.exception.app_exception import AppException
from src.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig,
                                      FeatureEngineeringConfig, ModelTrainingConfig)

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
            train_data_path = self.feature_engineering_configuration.train_test_data_path

            training_configuration = ModelTrainingConfig(
                train_data_path = train_data_path,
                models_dir = models_dir_path
            )
        
            return training_configuration

        except Exception as e:
            logging.error(f"Error while creating Model Training Configuration: {e}", exc_info=True)
            raise AppException(e, sys)