# DJL Serving Setup Guide

This guide helps DevOps and developers get [DJL Serving](https://github.com/deepjavalibrary/djl-serving) running for the embedder module. DJL Serving provides a REST API for inference (embeddings, text generation, etc.).

## Quick Start

**Important**: DJL Serving does not auto-load models. You must load the model after the container is running.

```bash
cd /work/modules/module-embedder/docs/djl-serving   # or cd docs/djl-serving from module-embedder
docker compose up -d
./load-model.sh                    # Waits for server, then loads all-MiniLM-L6-v2
# Or manually:
# curl -X POST "http://localhost:8090/models?url=djl%3A%2F%2Fai.djl.huggingface.pytorch%2Fsentence-transformers%2Fall-MiniLM-L6-v2&model_name=all-MiniLM-L6-v2&synchronous=true"

# Working curl - single text embedding
curl -s -X POST http://localhost:8090/predictions/all-MiniLM-L6-v2 \
  -H "Content-Type: application/json" \
  -d '["Hello world"]'
```

The response is a JSON array of 384 floats (the embedding vector). The module-embedder uses `{"inputs": "text"}` format; both `["text"]` and `{"inputs": "text"}` work with sentence-transformers.

## Image Variants (CPU, GPU, ARM)

Choose the image that matches your platform:

| Platform | Image Tag | Use Case |
|----------|-----------|----------|
| **CPU (x86/AMD64)** | `deepjavalibrary/djl-serving:0.36.0-cpu` | Laptops, servers without GPU |
| **CPU full** | `deepjavalibrary/djl-serving:0.36.0-cpu-full` | More engines (ONNX, etc.) |
| **NVIDIA GPU** | `deepjavalibrary/djl-serving:0.36.0-pytorch-gpu` | CUDA GPUs (RTX, A100, etc.) |
| **ARM64 (M1/M2, Graviton)** | `deepjavalibrary/djl-serving:0.36.0-aarch64` | Apple Silicon, AWS Graviton |

**AMD GPUs**: DJL does not have native AMD/ROCm images. Use the **CPU** image; it works on AMD systems.

**ARM (aarch64)**: Use for Apple M1/M2/M3, AWS Graviton, or other ARM64 servers.

## Launching

### Option 1: Docker Compose (Recommended)

```bash
cd docs/djl-serving
docker compose up -d
```

See `docker-compose.yml` in this directory. Uses environment variables to switch images:

```bash
# CPU (default)
docker compose up -d

# NVIDIA GPU
DJL_IMAGE=deepjavalibrary/djl-serving:0.36.0-pytorch-gpu docker compose up -d

# ARM64 (e.g. Mac M1)
DJL_IMAGE=deepjavalibrary/djl-serving:0.36.0-aarch64 docker compose up -d
```

### Option 2: Docker Run

```bash
# CPU
docker run -d -p 8090:8080 --name djl-serving \
  -e MODEL_LOADING_TIMEOUT=300 \
  deepjavalibrary/djl-serving:0.36.0-cpu

# GPU (requires nvidia-docker)
docker run -d -p 8090:8080 --name djl-serving \
  --gpus all \
  -e MODEL_LOADING_TIMEOUT=300 \
  deepjavalibrary/djl-serving:0.36.0-pytorch-gpu
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_LOADING_TIMEOUT` | 240 | Seconds to wait for model load |
| `JAVA_OPTS` | (none) | JVM options, e.g. `-Xmx8g` for heap |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache (mount for persistence) |

### Loading Models

**Via management API** (model downloaded from HuggingFace on first request):

```bash
# MiniLM (384d, fast, good for embeddings)
curl -X POST "http://localhost:8090/models?url=djl%3A%2F%2Fai.djl.huggingface.pytorch%2Fsentence-transformers%2Fall-MiniLM-L6-v2&model_name=all-MiniLM-L6-v2&synchronous=true"

# BGE-M3 (1024d, higher quality) - requires custom handler in many setups
curl -X POST "http://localhost:8090/models?url=djl%3A%2F%2Fai.djl.huggingface.pytorch%2FBAAI%2Fbge-m3&model_name=bge_m3&synchronous=true"
```

**Auto-load from `/opt/ml/model`**: Mount a model directory with `serving.properties`. See [Custom Models](#custom-models).

## Example API Calls

### List models
```bash
curl -s http://localhost:8090/models | jq .
```

### Embedding (MiniLM)
```bash
# Single text
curl -s -X POST http://localhost:8090/predictions/all-MiniLM-L6-v2 \
  -H "Content-Type: application/json" \
  -d '["Hello world"]'

# Batch of texts
curl -s -X POST http://localhost:8090/predictions/all-MiniLM-L6-v2 \
  -H "Content-Type: application/json" \
  -d '["First sentence", "Second sentence", "Third sentence"]'
```

**Response**: JSON array of arrays (each inner array is a 384-d vector for MiniLM):
```json
[[0.023, -0.015, ...], [0.019, -0.012, ...], ...]
```

### Unload a model
```bash
curl -X DELETE "http://localhost:8090/models/all-MiniLM-L6-v2"
```

## Custom Models

To use a custom model (e.g. BGE-M3 with Python handler):

1. Create a directory with `serving.properties` and optional `model.py`
2. Mount it at `/opt/ml/model/<model-name>/`
3. Set `Initial Models: ALL` or load via API

Example `serving.properties`:
```properties
engine=Python
option.model_id=BAAI/bge-m3
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Container exits immediately** | Check `docker compose logs`. Often: OOM, model download failure, or wrong image for architecture |
| **Model never reaches READY** | First load downloads from HuggingFace (~400MB for MiniLM). Ensure network access and disk space |
| **Connection refused** | Wait 30-60s after start. Check port: `curl http://localhost:8090/ping` |
| **504 / timeout** | Increase `MODEL_LOADING_TIMEOUT`. Large models (BGE-M3) need 300+ seconds |
| **OOM on GPU** | Use CPU image, or unload other models. Only one big model fits on 16GB GPU |
| **Wrong architecture** | On ARM Mac: use `0.36.0-aarch64`. On AMD without NVIDIA: use `-cpu` |
| **404 on predictions** | Model name in URL must match exactly (e.g. `all-MiniLM-L6-v2`, underscores not hyphens) |

## Embedder Module Integration

The module-embedder expects DJL at `EMBEDDER_DJL_SERVING_URL` (default: `http://localhost:8090`) with model `all-MiniLM-L6-v2`. Ensure the compose port matches:

```yaml
ports:
  - "8090:8080"   # host 8090 → container 8080
```

Set `EMBEDDER_DJL_SERVING_URL=http://djl-host:8090` when embedder runs in another container.
