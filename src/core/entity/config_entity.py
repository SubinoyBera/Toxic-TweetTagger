from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    data_download_url: str
    raw_data_dir: Path
    ingested_data_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    ingested_dataset_path: Path
    preprocessed_data_dir: Path

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    models_dir: Path
    preprocessed_data_path: Path
    train_test_data_path: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    train_data_path: Path
    models_dir: Path
    model_name: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    test_data_path: Path
    models_dir: Path
    trained_model_name: str
    evaluation_report_filepath: Path
    experiment_info_filepath: Path

@dataclass(frozen=True)
class ModelRegistrationConfig:
    experiment_info_filepath: Path