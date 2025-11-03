# Context Classifier API (FastAPI + Docker)

This repository provides a FastAPI-based REST API for running text classification inference using a fine-tuned transformer model (e.g., BERT, RoBERTa, etc.).  
The API loads your trained model from a mounted directory and serves predictions through HTTP endpoints.  
It also supports GPU inference inside Docker.

---

## Features

- FastAPI for high-performance inference
- Transformer-based model (from Hugging Face)
- GPU support (CUDA/ROCm) ready
- Dockerized deployment for reproducibility
- Swagger UI at `/docs` for easy testing
- Compatible with any model trained using `transformers.AutoModelForSequenceClassification`

---

## Project Structure

```
.
├── main.py                      # FastAPI app for serving predictions
├── inference_context_classifier.py  # Optional batch inference script
├── requirements.txt             # Dependencies list
├── Dockerfile                   # Docker build configuration
└── README.md                    # Project documentation
```

---

## Environment Variables

| Variable Name | Description | Default |
|----------------|-------------|----------|
| MODEL_DIR | Path to model directory containing `pytorch_model.bin`, `config.json`, and `class_names.txt` | `/opt/dlami/nvme/Folder_Final_results/model` |
| MAX_LEN | Max token length for model input | 256 |

---

## Model Folder Structure

Make sure your model directory (mounted in Docker) looks like this:

```
model/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
└── class_names.txt
```

The `class_names.txt` file should contain one class per line:
```
APPROVAL
ADDITIONAL INFORMATION REQUESTED
DENIAL
...
```

---

## Setup Instructions

### 1. Build Docker Image
```bash
docker build -t context-classifier-api .
```

### 2. Run Container (with GPU support)
```bash
docker run -d --gpus all -p 8000:8000   -e MODEL_DIR="/opt/dlami/nvme/Folder_Final_results/model"   -v /opt/dlami/nvme/Folder_Final_results/model:/opt/dlami/nvme/Folder_Final_results/model   context-classifier-api
```

If you don’t have GPU, remove the `--gpus all` flag:
```bash
docker run -d -p 8000:8000 context-classifier-api
```

---

## API Endpoints

| Method | Endpoint | Description |
|---------|-----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict the class for a given text |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{"text": "The patient requested an additional report."}'
```

### Example Response
```json
{
  "prediction": "ADDITIONAL INFORMATION REQUESTED"
}
```

---

## Test in Browser

Once running, visit:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc UI: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Clean Up Containers

```bash
docker ps -a         # Check running containers
docker stop <id>     # Stop container
docker rm <id>       # Remove container
docker rmi context-classifier-api  # Remove image (optional)
```

---

## Example Use Cases

- Internal document classification
- Automated email triage
- Medical or insurance text categorization
- Multi-domain text workflow automation

---

## Future Improvements

- Add support for batch prediction via `/predict_batch`
- Integrate MLflow for model tracking
- Add RAG / LLM agent for context-aware reasoning

---

## Author

Ashutosh Tiwari  
GitHub: [Light512](https://github.com/Light512)
