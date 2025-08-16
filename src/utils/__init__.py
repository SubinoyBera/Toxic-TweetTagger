import os
import sys
import yaml
import joblib
from pathlib import Path
from box import ConfigBox
from ensure import ensure_annotations
from ..core.logger import logging
from ..core.exception import AppException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.
    Args:
        path_to_yaml (Path): The path to the YAML file to be read.
    Returns:
        ConfigBox: A ConfigBox object containing the parsed content of the YAML file.
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file read successfully from {path_to_yaml}")

            return ConfigBox(content)
        
    except Exception as e:
        logging.error(f"Failed to read YAML file at {path_to_yaml}: {e}", exc_info=True)
        raise AppException(e, sys) 
        
    
def create_directory(path_to_directory: Path, verbose=True):
    """
    Creates directory at the specified location path.
    Args:
        path_to_directory : Directory path location where it is to be created.
        verbose (bool, optional): If True, logs the creation of each directory. Defaults to True.
    """
    try:
        if not isinstance(path_to_directory, Path):
            raise TypeError("path_to_directory must be a Path object")
        
        if path_to_directory.exists():
            logging.info(f"Directory already exists at {path_to_directory}")
            return
        
        os.makedirs(path_to_directory, exist_ok=True)
        logging.info(f"Directory created at {path_to_directory}") if verbose else None

    except Exception as e:
        logging.error(f"Failed to create directory at {path_to_directory}: {e}", exc_info=True)
        raise AppException(e, sys)
    

def save_obj(location_path: Path, obj, obj_name: str):
    """
    Saves a given object to the given path in joblib format.
    Args:
        location_path (Path): The path to the directory where the object should be saved.
        obj (object): The object to be saved.
        obj_name (str): The name of the object to be saved (should include .joblib extension).
    """
    try:
        if not obj_name.endswith(".joblib"):
            logging.error(f"Invalid object format.")
            raise ValueError(f"Invalid file format for object: {obj_name}")
        
        with open(Path(location_path, obj_name), 'wb') as f:
            joblib.dump(obj, f)
            logging.info(f"{obj_name} saved at {location_path}")

    except Exception as e:
        logging.error(f"Failed to save object at {location_path}: {e}", exc_info=True)
        raise AppException(e, sys)


def load_obj(location_path: Path, obj_name: str):
    """
    Loads a given object from the given path.
    Args:
        location_path (Path): The directory path from where the object is to be loaded.
        obj_name (str): The name of the object to be loaded (should have .joblib extension).
    """
    try:
        if not obj_name.endswith(".joblib"):
            logging.error(f"Invalid object format.")
            raise ValueError(f"Invalid file format for object: {obj_name}")
        with open(Path(location_path, obj_name), 'rb') as f:
            obj = joblib.load(f)
            logging.info(f"{obj_name} loaded from {location_path}")
            return obj

    except Exception as e:
        logging.error(f"Failed to load object from {location_path}: {e}", exc_info=True)
        raise AppException(e, sys)