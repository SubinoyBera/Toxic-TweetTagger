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
TRAINING_COLLECTION_NAME = "tweets_collection"
PRODUCTION_COLLECTION_NAME = "production_tweets"
FEEDBACK_COLLECTION_NAME = "prediction_feedback"

# Final registerd models dir
REGISTERED_MODELS_DIR = Path(ROOT_DIR, "src/app/model")

# HuggingFace Repo
HUGGINGFACE_REPO_ID = "Subi003/ToxicTagger_serveAPI"