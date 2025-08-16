from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    ingested_data_path: Path
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class DataValidationConfig:
    ingested_data_path: Path
    train_data_path: Path
    test_data_path: Path
    validation_status_file: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    data_path: Path
    preprocessed_data_dir: Path
    preprocessed_data_filename: str

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    models_dir: Path
    preprocessed_data_path: Path
    training_data_path: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    training_data_path: Path
    models_dir: Path

@dataclass(frozen=True)
class ModelEvaluationConfig:
    test_data_path: Path
    models_dir: Path
    evaluation_report_filepath: Path
    experiment_info_filepath: Path

@dataclass(frozen=True)
class ModelRegistrationConfig:
    experiment_info_filepath: Path