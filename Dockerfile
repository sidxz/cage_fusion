# Base with CUDA libs for GPU inference. For CPU-only, use debian-slim or python base.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG APP_USER=appuser
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd -m ${APP_USER}
USER ${APP_USER}
WORKDIR /home/${APP_USER}

# Copy project files for dependency installation (rebuild layer only when these change)
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml uv.lock ./

# Create venv inheriting system site-packages (torch already in base image with CUDA support).
# --no-install-package torch skips re-downloading torch (~2 GB).
RUN uv venv --system-site-packages .venv && \
    uv sync --frozen --no-dev --no-install-package torch

ENV VIRTUAL_ENV=/home/appuser/.venv
ENV PATH="/home/appuser/.venv/bin:$PATH"

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} cage_fusion ./cage_fusion

# Runtime env vars
ENV CHECKPOINT_DIR=/checkpoints/nuisance-pred \
    MODEL_NAME=Cross-Prompt-Phased-bert-251020 \
    MODEL_FILE=latest_checkpoint.pt \
    BATCH_SIZE=256 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    HF_HUB_CACHE=/home/appuser/.cache/huggingface/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=10002 \
    WORKERS=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Sanity-check C++ ABI compatibility
RUN python - <<'PY'
import scipy, sklearn
print("OK: scipy/sklearn import succeeded")
PY

COPY --chown=${APP_USER}:${APP_USER} entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 10002

ENTRYPOINT ["./entrypoint.sh"]
