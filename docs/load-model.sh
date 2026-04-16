#!/bin/bash
# Load required MiniLM models into DJL Serving
# Usage: ./load-model.sh [BASE_URL]
BASE_URL="${1:-http://localhost:8090}"

echo "Waiting for DJL Serving at $BASE_URL..."
until curl -s -f "$BASE_URL/ping" >/dev/null 2>&1; do
  echo "  waiting..."
  sleep 2
done
echo "DJL Serving is up."

# Load all-MiniLM-L6-v2
echo "Loading all-MiniLM-L6-v2..."
curl -s -X POST "$BASE_URL/models?url=djl%3A%2F%2Fai.djl.huggingface.pytorch%2Fsentence-transformers%2Fall-MiniLM-L6-v2&model_name=all-MiniLM-L6-v2&engine=PyTorch&synchronous=true"
echo ""

# Load paraphrase-MiniLM-L3-v2
echo "Loading paraphrase-MiniLM-L3-v2..."
curl -s -X POST "$BASE_URL/models?url=djl%3A%2F%2Fai.djl.huggingface.pytorch%2Fsentence-transformers%2Fparaphrase-MiniLM-L3-v2&model_name=paraphrase-MiniLM-L3-v2&engine=PyTorch&synchronous=true"
echo ""

echo "Check status: curl -s $BASE_URL/models | jq ."
