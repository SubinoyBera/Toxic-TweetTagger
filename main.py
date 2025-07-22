# File: main.py for starting ML pipeline execution
# This script serves as an entry point for triggering the machine learning pipeline.

import sys
import time
from src.core.logger import logging
from src.core.exception import AppException
from src.pipeline.ml_pipeline import run_pipeline

try:
    start_datetime, start = time.ctime(), time.time()
    logging.error(f"Initializing ML Pipeline ::  Start Time - {start_datetime}")
    run_pipeline()
    end_datetime, end = time.ctime(), time.time()
    logging.error(f"Pipeline executed successfully ::  End Time - {end_datetime}")
    logging.error(f"\n\n Total Pipeline Execution Time - {end - start}  seconds")
        
except Exception as e:
    logging.error(f"ML pipeline terminated: {e}", exc_info=True)
    raise AppException(e, sys)