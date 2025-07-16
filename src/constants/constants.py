# Constants for configuration file paths
# This module defines constants for the configuration file paths used in the application.

import os
from pathlib import Path

# Get current working directory
ROOT_DIR = os.getcwd()

# Main configuration file path
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = Path(ROOT_DIR, CONFIG_FILE_NAME)
