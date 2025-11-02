# Base with CUDA libs for GPU inference. For CPU-only, use debian-slim or python base.
#FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# Micromamba bootstrap (tiny and fast)
ARG MAMBA_USER=mamba
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y curl ca-certificates bzip2 && rm -rf /var/lib/apt/lists/*
RUN useradd -m ${MAMBA_USER}
USER ${MAMBA_USER}
WORKDIR /home/${MAMBA_USER}

# Install micromamba
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

ENV MAMBA_ROOT_PREFIX=/home/${MAMBA_USER}/mamba
ENV PATH=/home/${MAMBA_USER}/bin:$PATH
ENV MAMBA_DOCKERFILE_ACTIVATE=1
# Copy env and create it
COPY --chown=${MAMBA_USER}:${MAMBA_USER} environment.yml /home/${MAMBA_USER}/environment.yml
RUN ./bin/micromamba create -y -n cage-fusion -f environment.yml

# Copy your code
COPY --chown=${MAMBA_USER}:${MAMBA_USER} cage_fusion /home/${MAMBA_USER}/cage_fusion

# Runtime env vars
ENV CHECKPOINT_DIR=/checkpoints/nuisance-pred \
    MODEL_NAME=Cross-Prompt-Phased-bert-251020 \
    MODEL_FILE=latest_checkpoint.pt \
    BATCH_SIZE=256 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HF_HOME=/home/${MAMBA_USER}/.cache/huggingface \
    HF_HUB_CACHE=/home/${MAMBA_USER}/.cache/huggingface/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=8080 \
    WORKERS=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---- Startup script (does HF cache warm + launches uvicorn) ----
COPY --chown=${MAMBA_USER}:${MAMBA_USER} entrypoint.sh /home/${MAMBA_USER}/entrypoint.sh
RUN chmod +x /home/${MAMBA_USER}/entrypoint.sh

EXPOSE 8080

# Use the startup script (auto-activates env via micromamba run)
ENTRYPOINT ["/home/mamba/entrypoint.sh"]