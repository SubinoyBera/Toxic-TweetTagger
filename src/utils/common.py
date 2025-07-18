import os
import sys
import yaml
import pickle
from pathlib import Path
from box import ConfigBox
from ensure import ensure_annotations
from src.logger import logging
from src.exception.app_exception import AppException
import logging

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
            content = yaml.safe_load_all(yaml_file)
            logging.info(f"YAML file read successfully from {path_to_yaml}")

            return ConfigBox(content)
        
    except Exception as e:
        logging.error(f"Failed to read YAML file at {path_to_yaml}: {e}", exc_info=True)
        raise AppException(e, sys) 
        
    
def create_directory(path_to_directory: Path, verbose=True):
    """
    Creates directories specified in the given list of paths.
    Args:
        path_to_directories (list): List of directory paths to be created.
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
    

def save_obj(location_path: Path, obj, obj_name):
    """
    Saves a given object to a given path in .pkl format using the pickle library.
    Args:
        location_path (Path): The path to the directory where the object should be saved.
        obj (object): The object to be saved.
        obj_name (str): The name of the object to be saved (should include .pkl extension).
    """
    try:
        if ".pkl" not in obj_name:
            logging.error(f"Invalid object format.")
            return
        pickle.dump(obj, open(Path(location_path, obj_name), 'wb'))

    except Exception as e:
        logging.error(f"Failed to save object at {location_path}: {e}", exc_info=True)