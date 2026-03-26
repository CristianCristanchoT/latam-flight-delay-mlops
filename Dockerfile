# syntax=docker/dockerfile:1.2
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY challenge/ ./challenge/
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
