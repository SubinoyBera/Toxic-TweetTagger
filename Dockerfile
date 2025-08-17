FROM python:3.11.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y

WORKDIR /app

COPY app/ /app/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]