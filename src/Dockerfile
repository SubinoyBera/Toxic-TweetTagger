FROM python:3.11.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r model/requirements.txt

RUN useradd -m appuser
USER appuser

EXPOSE 7860
ENV HOST=0.0.0.0 PORT=7860 PYTHONUNBUFFERED=1

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main.inference:inference_api", "--bind", "0.0.0.0:7860"]