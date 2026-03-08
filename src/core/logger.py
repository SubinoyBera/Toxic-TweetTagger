# This script sets up the logging configuration for the project.
# Creates a log file with a timestamp in the name and sets up logging to both a file and the console.

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

LOG_LEVEL = logging.INFO
ENV = os.getenv("ENV")

log_handlers = [logging.StreamHandler(sys.stdout)]

if ENV == "DEV":
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

    # Create a logs directory if it doesn't exist
    os.makedirs(logs_path, exist_ok = True)

    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

    log_handlers.append(logging.FileHandler(LOG_FILE_PATH))

# logging configuration
logging.basicConfig(
    format = "[ %(asctime)s ]: %(levelname)s - %(lineno)d %(module)s - %(message)s",
    level = logging.INFO,
    handlers = log_handlers
)