# Script for starting the FastAPI application

import uvicorn
from src.core.logger import logging

if __name__ == "__main__":
    logging.info("Starting app server...")
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, reload=True)