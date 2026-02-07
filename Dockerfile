FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install experiment dependencies
RUN pip install --no-cache-dir mlflow>=2.10 matplotlib

WORKDIR /workspace
