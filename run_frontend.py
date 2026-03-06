# Script for starting the FastAPI frontend app

import uvicorn
from src.core.logger import logging


if __name__ == "__main__":
    logging.info("Starting frontend app...")
    uvicorn.run("frontend.app:app", host="0.0.0.0", port=8500, reload=True, reload_dirs=["frontend"])