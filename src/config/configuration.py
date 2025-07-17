import sys
from pathlib import Path
from src.logger.logging import logging
from src.constants.constants import *
from src.utils.common import read_yaml, create_directory
from src.exception.app_exception import AppException
from src.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig)

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

            ingestion_configuration = DataIngestionConfig(
                raw_data_dir = raw_data_path,
                ingested_dir = ingested_data_path,
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
            ingestion_config = self.config.ingestion_config

            preprocessing_root = preprocessing_config.root_dir
            processed_data_dir = preprocessing_config.processed_data_dir
            dataset = preprocessing_config.dataset

            create_directory(preprocessing_root)

            processed_data_dir_path = Path(preprocessing_root, processed_data_dir)
            dataset_path = Path(ingestion_config.ingested_data, dataset)

            preprocessing_configuration = DataPreprocessingConfig(
                processed_data_dir = processed_data_dir_path,
                ingested_dataset_path = dataset_path
            )

            return preprocessing_configuration
        
        except Exception as e:
            logging.error(f"Error while creating Data Preprocessing Configuration: {e}", exc_info=True)
            raise AppException(e, sys)