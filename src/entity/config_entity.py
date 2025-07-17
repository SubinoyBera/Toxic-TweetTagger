from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    data_download_url: str
    raw_data_dir: Path
    ingested_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    ingested_dataset_path: Path
    processed_data_dir: Path