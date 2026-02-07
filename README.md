# MLPatron Demo

Minimal MLflow Project with PyTorch for [ML Patron](https://mlpatron.com).

## Project Structure

```
MLproject       # MLflow project definition (image + entry points)
Dockerfile      # Environment image (dependencies on top of base PyTorch image)
train.py        # Training script
```

`MLproject` references the image built from `Dockerfile`. See [Docker Image](#docker-image) for details.

## Local Development

```bash
# 1. Build the image locally (only builds, no push needed for local dev)
#    On Apple Silicon Mac, specify --platform since the base image is linux/amd64:
docker build --platform linux/amd64 -t ghcr.io/nblintao/mlpatron-demo:latest .

# 2. Start a local MLflow server
mlflow server --port 5002 --host 0.0.0.0 --allowed-hosts "*" \
  --backend-store-uri sqlite:///.mlflow/mlflow.db \
  --artifacts-destination .mlflow/mlartifacts

# 3. Run the experiment (entry_point "main" with custom parameters, see MLproject)
docker run --rm --platform linux/amd64 \
  -v "$(pwd)":/workspace \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5002 \
  ghcr.io/nblintao/mlpatron-demo:latest \
  python train.py --epochs 5 --lr 0.01 --batch_size 64 --hidden_size 128
```

View results at http://localhost:5002.

## How This Maps to ML Patron

When you submit a Run on ML Patron, the platform:

1. Reads `MLproject` from your repo (via Git API, no full clone)
2. Resolves the commit SHA and image digest
3. Creates a K8s Job that clones your code and runs the command

| Local development | You fill in on [ML Patron](https://mlpatron.com) | ML Patron |
|---|---|---|
| `docker build` + `docker push` to registry | — (from `MLproject` in your repo) | Platform pulls the image you specified in `MLproject` |
| Code is on your local machine | **Git Repository URL** + **Git Version** (branch/tag/commit, default: `main`) | Platform clones your code into `/workspace` via init container |
| Local MLflow server | — (automatic) | Platform's managed MLflow server |
| Manual `MLFLOW_TRACKING_URI` | — (automatic) | Platform injects env vars into the container |
| `docker run ... python train.py --epochs 5` | **Entry Point** (default: `main`) + **Parameters** (key-value pairs) | Platform parses `MLproject`, creates a K8s Job with the resolved command |

## Docker Image

**Requirement:** `docker_env.image` in `MLproject` must point to a publicly pullable image that has all the dependencies your training script needs (including `mlflow` for logging). The image does **not** need to contain your code — ML Patron clones it into `/workspace` at runtime.

**In this demo**, the `Dockerfile` installs `mlflow`, `matplotlib`, etc. on top of the PyTorch base image. The GitHub Actions workflow (`.github/workflows/docker.yml`) automatically builds and pushes it to `ghcr.io/nblintao/mlpatron-demo`. This is just one way to host your image — any registry works.

## Note

Do **not** call `mlflow.set_tracking_uri()` in your code. ML Patron injects `MLFLOW_TRACKING_URI` via environment variable, and MLflow reads it automatically.
