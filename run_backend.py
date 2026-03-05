# Script for starting the FastAPI application

import uvicorn
import gunicorn
from src.core.logger import logging

#if __name__ == "__main__":
#    gunicorn src.run_app:app -k uvicorn.workers.UvicornWorker --reload


if __name__ == "__main__":
    logging.info("Starting app server...")
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["src"])