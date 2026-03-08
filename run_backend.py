# Script for starting the FastAPI backend application server.

import uvicorn
from src.core.logger import logging


if __name__ == "__main__":
    logging.info("Starting app server...")
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000)