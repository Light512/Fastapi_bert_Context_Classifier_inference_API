#!/bin/bash
set -e  # Exit on any error

IMAGE_NAME="bert-inference"
CONTAINER_NAME="bert-inference-run"

# ======================================
# Step 1: Build Docker image
# ======================================
echo "Building Docker image: ${IMAGE_NAME}..."
docker build -t ${IMAGE_NAME} .

# ======================================
# Step 2: Remove any previous container
# ======================================
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Removing old container..."
    docker rm -f ${CONTAINER_NAME}
fi

# ======================================
# Step 3: Run inference container
# ======================================
echo "Starting inference job..."
docker run --gpus all \
  --name ${CONTAINER_NAME} \
  -v /opt/dlami/nvme:/opt/dlami/nvme \
  -v $(pwd):/app \
  -e DATA_DIR="infeer_Data" \
  -e MODEL_DIR="Classification_CLUSTER\Folder_Final_results\model" \
  -e BATCH_SIZE=32 \
  python:3.10 \
  bash -c "
    cd /app && \
    echo '📦 Installing dependencies...' && \
    pip install torch transformers scikit-learn pandas && \
    echo '🚀 Running inference script...' && \
    python inference_context_classifier.py
  "

# ======================================
# Step 4: Done
# ======================================
echo "Inference completed. Results saved in Final_results/model/folder_predictions.json"
