#!/usr/bin/env bash
# Local development setup script for mlpatron-demo
# Installs dependencies, starts MLflow server, and runs a test training

set -euo pipefail

MLFLOW_PORT="${MLFLOW_PORT:-5002}"
MLFLOW_DIR=".mlflow"
MLFLOW_PID_FILE="${MLFLOW_DIR}/mlflow.pid"

log() { echo "[setup] $*"; }

install_deps() {
    log "Installing Python dependencies..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    pip install "mlflow>=2.10" matplotlib --quiet
    log "Dependencies installed."
}

start_mlflow() {
    # Check if MLflow is already running on the target port
    if curl -s "http://localhost:${MLFLOW_PORT}/health" | grep -q "OK" 2>/dev/null; then
        log "MLflow server already running on port ${MLFLOW_PORT}."
        return 0
    fi

    mkdir -p "${MLFLOW_DIR}"

    log "Starting MLflow server on port ${MLFLOW_PORT}..."
    nohup mlflow server \
        --port "${MLFLOW_PORT}" \
        --host 0.0.0.0 \
        --backend-store-uri "sqlite:///${PWD}/${MLFLOW_DIR}/mlflow.db" \
        --artifacts-destination "${PWD}/${MLFLOW_DIR}/mlartifacts" \
        > "${MLFLOW_DIR}/server.log" 2>&1 &
    echo $! > "${MLFLOW_PID_FILE}"

    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -s "http://localhost:${MLFLOW_PORT}/health" | grep -q "OK" 2>/dev/null; then
            log "MLflow server is ready (PID: $(cat ${MLFLOW_PID_FILE}))."
            return 0
        fi
        sleep 1
    done

    log "ERROR: MLflow server failed to start. Check ${MLFLOW_DIR}/server.log"
    return 1
}

stop_mlflow() {
    if [ -f "${MLFLOW_PID_FILE}" ]; then
        PID=$(cat "${MLFLOW_PID_FILE}")
        if kill -0 "${PID}" 2>/dev/null; then
            log "Stopping MLflow server (PID: ${PID})..."
            kill "${PID}"
            rm -f "${MLFLOW_PID_FILE}"
            log "MLflow server stopped."
        else
            log "MLflow server (PID: ${PID}) is not running."
            rm -f "${MLFLOW_PID_FILE}"
        fi
    else
        log "No MLflow PID file found."
    fi
}

run_training() {
    local epochs="${1:-2}"
    log "Running training with ${epochs} epoch(s)..."
    MLFLOW_TRACKING_URI="http://localhost:${MLFLOW_PORT}" \
        python train.py --epochs "${epochs}" --lr 0.01 --batch_size 64 --hidden_size 128
    log "Training complete."
}

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  install     Install Python dependencies"
    echo "  start       Start MLflow tracking server"
    echo "  stop        Stop MLflow tracking server"
    echo "  train [N]   Run training (default: 2 epochs)"
    echo "  all [N]     Install deps, start MLflow, and run training"
    echo "  check       Verify the local dev setup is working"
}

check() {
    local ok=true

    log "Checking Python..."
    python --version 2>/dev/null || { log "FAIL: Python not found"; ok=false; }

    log "Checking PyTorch..."
    python -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null || { log "FAIL: PyTorch not installed"; ok=false; }

    log "Checking MLflow..."
    python -c "import mlflow; print(f'mlflow {mlflow.__version__}')" 2>/dev/null || { log "FAIL: MLflow not installed"; ok=false; }

    log "Checking matplotlib..."
    python -c "import matplotlib; print(f'matplotlib {matplotlib.__version__}')" 2>/dev/null || { log "FAIL: matplotlib not installed"; ok=false; }

    log "Checking MLflow server..."
    if curl -s "http://localhost:${MLFLOW_PORT}/health" | grep -q "OK" 2>/dev/null; then
        log "MLflow server is running on port ${MLFLOW_PORT}."
    else
        log "WARN: MLflow server is not running on port ${MLFLOW_PORT}."
        ok=false
    fi

    if $ok; then
        log "All checks passed."
    else
        log "Some checks failed. Run '$0 all' to set up everything."
        return 1
    fi
}

cd "$(dirname "$0")"

case "${1:-}" in
    install)  install_deps ;;
    start)    start_mlflow ;;
    stop)     stop_mlflow ;;
    train)    run_training "${2:-2}" ;;
    all)      install_deps; start_mlflow; run_training "${2:-2}" ;;
    check)    check ;;
    *)        usage ;;
esac
