# Build context is the repository root (see docker-compose.yml)
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY app/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# only the torch-free helpers from the training package are needed at inference time
COPY src/genai/__init__.py src/genai/corruptions.py src/genai/synthetic.py /app/genai/
COPY app/backend/main.py /app/main.py
COPY app/backend/samples /app/samples
ENV MODELS_DIR=/models SAMPLES_DIR=/app/samples
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
