# Project pipeline configuration 
import sys
from pathlib import Path
from ..constant.constants import *
from ..utils import read_yaml, create_directory
from ..core.logger import logging
from ..core.exception import AppException
from ..core.config_entity import (DataIngestionConfig, DataValidationConfig, DataPreprocessingConfig, 
                                  FeatureEngineeringConfig, ModelTrainingConfig, ModelEvaluationConfig, ModelRegistrationConfig)

# 
class AppConfiguration:
    # class variable to hold the configuration
    _config = None
    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH):
        
        """
        Initializes the AppConfiguration object by loading the configuration from the given file path.
        Args:
            config_filepath (Path): The path to the configuration YAML file. Defaults to CONFIG_FILE_PATH.
        
        Raises:
            AppException: If an error occurs while loading the configuration.
        """
        if AppConfiguration._config is None:
            try:
                AppConfiguration._config = read_yaml(config_filepath)
                
            except Exception as e:
                logging.error(f"Failed to load configuration: {e}", exc_info=True)
                raise AppException(e, sys)       
            
        self.config = AppConfiguration._config

        
    def data_ingestion_config(self) -> DataIngestionConfig:
        """
        Creates the configuration for Data Ingestion 
        Returns: DataIngestionConfig object
        """
        try:
            ingestion_config = self.config.data_ingestion

            ingestion_root_dir = Path(ingestion_config.root_dir)
            ingested_data_dir = Path(ingestion_root_dir, ingestion_config.ingested_data_dir)
            ingested_dataset_name = ingestion_config.ingested_dataset
            train_data_filename = ingestion_config.train_dataset
            test_data_filename = ingestion_config.test_dataset

            create_directory(ingestion_root_dir)
            create_directory(ingested_data_dir)

            ingestion_configuration = DataIngestionConfig(
                ingested_data_path = Path(ingested_data_dir, ingested_dataset_name), 
                train_data_path = Path(ingestion_root_dir, train_data_filename),
                test_data_path = Path(ingestion_root_dir, test_data_filename)
            )
            logging.info("Data Ingestion Configuration creation successfull")
            return ingestion_configuration
        
        except Exception as e:
            logging.error(f"Error while creating Data Ingestion Configuration: {e}", exc_info=True)
            raise AppException(e, sys)


    def data_validation_config(self) -> DataValidationConfig:
        """
        Creates the configuration for Data Validation.
        Returns: DataValidationConfig object
        """
        try:
            validation_config = self.config.data_validation
            ingestion_configuration = self.data_ingestion_config()

            validation_dir = Path(validation_config.root_dir)
            reports_dir = Path(validation_config.reports_dir)
            status_filename = validation_config.status_file
            status_file_path = Path(validation_dir, status_filename)

            create_directory(validation_dir)
            create_directory(reports_dir)

            validation_configuration = DataValidationConfig(
                ingested_data_path = ingestion_configuration.ingested_data_path,
                train_data_path = ingestion_configuration.train_data_path,
                test_data_path = ingestion_configuration.test_data_path,
                validation_status_file = status_file_path
            )
            logging.info("Data Validation Configuration creation successfull")
            return validation_configuration
        
        except Exception as e:
            logging.error(f"Error while creating Data Validation Configuration: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        Creates the configuration for Data Preprocessing.
        Returns: DataPreprocessingConfig object
        """
        try:
            preprocessing_config = self.config.data_preprocessing
            ingestion_configuration = self.data_ingestion_config()

            preprocessing_dir = Path(preprocessing_config.root_dir)
            preprocessed_dataset_name = preprocessing_config.preprocessed_dataset 
            create_directory(preprocessing_dir)

            preprocessing_configuration = DataPreprocessingConfig(
                preprocessed_data_dir = preprocessing_dir,
                preprocessed_data_filename = preprocessed_dataset_name,
                data_path = ingestion_configuration.train_data_path
            )
            logging.info("Data Preprocessing Configuration creation successfull")
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
            training_dataset = feature_eng_config.training_dataset
            training_data_path = Path(feature_eng_root_dir, training_dataset)

            create_directory(feature_eng_root_dir)
            create_directory(models_root_dir)

            feature_engineering_configuration = FeatureEngineeringConfig(
                models_dir = models_root_dir,
                preprocessed_data_path = Path(preprocessing_configuration.preprocessed_data_dir, preprocessing_configuration.preprocessed_data_filename), 
                training_data_path = training_data_path
            )
            logging.info("Feature Engineering Configuration creation successfull")
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

            models_dir_path = Path(training_config.models_dir)

            training_configuration = ModelTrainingConfig(
                training_data_path = feature_engineering_configuration.training_data_path,
                models_dir = models_dir_path,
            )
            logging.info("Model Training Configuration creation successfull")
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
            training_configuration = self.model_training_config()
            ingestion_configuration = self.data_ingestion_config()

            models_dir_path = training_configuration.models_dir
            eval_report_filename = evaluation_config.evaluation_report
            exp_info_filename = evaluation_config.experiment_info

            reports_dir_path = Path(evaluation_config.reports_dir)
            create_directory(reports_dir_path)

            eval_report_filepath = Path(reports_dir_path, eval_report_filename)
            exp_info_filepath = Path(reports_dir_path, exp_info_filename)

            evaluation_configuration = ModelEvaluationConfig(
                test_data_path = ingestion_configuration.test_data_path,
                models_dir = models_dir_path,
                evaluation_report_filepath = eval_report_filepath,
                experiment_info_filepath = exp_info_filepath
            )
            logging.info("Model Evaluation Configuration creation successfull")
            return evaluation_configuration

        except Exception as e:
            logging.error(f"Error while creating Model Training Configuration: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def model_registration_config(self) -> ModelRegistrationConfig:
        try:
            evaluation_configuration = self.model_evaluation_config()
            exp_info_filepath = evaluation_configuration.experiment_info_filepath

            logging.info("Model Registration Configuration creation successfull")
            return ModelRegistrationConfig(
                experiment_info_filepath = exp_info_filepath
            )
        
        except Exception as e:
            logging.error(f"Error while creating Model Evaluation Configuration: {e}", exc_info=True)
            raise AppException(e, sys)