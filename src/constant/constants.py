# Constants for configuration file paths
# This module defines constants for the configuration file paths used in the application.

import os
from pathlib import Path

# Get current working directory
ROOT_DIR = os.getcwd()

# configuration file path
CONFIG_DIR = Path(ROOT_DIR, "configs")
CONFIG_FILE = Path(CONFIG_DIR, "config.yaml")

# Params file path
PARAMS_FILE = Path(CONFIG_DIR, "params.yaml")

# Database Constants
DATABASE_NAME = "Tweets_DB"
COLLECTION_NAME = "toxic_tweet_data"

# HuggingFace Repo
HUGGINGFACE_REPO_ID = "Subi003/ToxicTagger_serveAPI"