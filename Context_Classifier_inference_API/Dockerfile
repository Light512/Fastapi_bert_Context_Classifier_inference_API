# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy code
COPY main.py ./
COPY inference_context_classifier.py ./
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Run API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
