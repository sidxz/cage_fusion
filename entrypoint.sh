#!/usr/bin/env bash
set -euo pipefail

# ---------- Logging ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo -e "[$(ts)] $1$2${NC}"; }
ok()  { log "${GREEN}" "$1"; }
warn(){ log "${YELLOW}" "$1"; }
err() { log "${RED}" "$1"; }

# ---------- Runtime env ----------
: "${CHECKPOINT_DIR:=/checkpoints/nuisance-pred}"
: "${MODEL_NAME:=Cross-Prompt-Phased-bert-251020}"
: "${MODEL_FILE:=latest_checkpoint.pt}"
: "${HF_HOME:=/home/mamba/.cache/huggingface}"
: "${PORT:=8080}"
: "${WORKERS:=1}"
: "${MODEL_CHECKPOINT_OVERRIDE:=}"              # e.g., "DeepChem/ChemBERTa-77M-MTR"
: "${HF_ENDPOINT:=https://huggingface.co}"
: "${HF_TOKEN:=}"

export HF_HOME HF_ENDPOINT HF_TOKEN
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ok "CAGE-FUSION API startup"
echo "  CHECKPOINT_DIR             = ${CHECKPOINT_DIR}"
echo "  MODEL_NAME                 = ${MODEL_NAME}"
echo "  MODEL_FILE                 = ${MODEL_FILE}"
echo "  HF_HOME                    = ${HF_HOME}"
echo "  HF_ENDPOINT                = ${HF_ENDPOINT}"
echo "  PORT                       = ${PORT}"
echo "  WORKERS                    = ${WORKERS}"
echo "  MODEL_CHECKPOINT_OVERRIDE  = ${MODEL_CHECKPOINT_OVERRIDE:-<none>}"

mkdir -p "${HF_HOME}"

# ---------- Verify your custom checkpoint exists ----------
if [[ ! -f "${CHECKPOINT_DIR}/${MODEL_NAME}/${MODEL_FILE}" ]]; then
  warn "Custom checkpoint not found: ${CHECKPOINT_DIR}/${MODEL_NAME}/${MODEL_FILE}"
  warn "API can still start, but predictions will fail until you mount the correct checkpoint."
fi

# ---------- Resolve HF repo id from config (or override) ----------
ok "Resolving HF backbone from config…"
HF_ID=$(micromamba run -n cage-fusion python - <<'PY'
from cage_fusion.configs import get_default_config
cfg = get_default_config()
print((cfg.get("model_checkpoint") or "").strip())
PY
)

if [[ -n "${MODEL_CHECKPOINT_OVERRIDE}" ]]; then HF_ID="${MODEL_CHECKPOINT_OVERRIDE}"; fi
if [[ -z "${HF_ID}" ]]; then err "Could not resolve 'model_checkpoint'."; exit 1; fi
ok "HF backbone: ${HF_ID}"

# ---------- Warm the cache and capture the resolved local directory ----------
ok "Warming HF cache for ${HF_ID}…"
export HF_ID
HF_RESOLVED_DIR=$(
  micromamba run -n cage-fusion env HF_ID="$HF_ID" HF_TOKEN="$HF_TOKEN" python - <<'PY'
import os
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id=os.environ["HF_ID"],
                              local_dir=None,
                              local_dir_use_symlinks=True,
                              token=os.environ.get("HF_TOKEN") or None)
print(local_dir)
PY
)
export HF_RESOLVED_DIR
ok "HF cache warm complete. Local dir: ${HF_RESOLVED_DIR}"

# ---------- OFFLINE TEST: ensure we can load from local snapshot ----------
ok "Offline self-test (transformers)…"
micromamba run -n cage-fusion env HF_RESOLVED_DIR="$HF_RESOLVED_DIR" python - <<'PY'
import os, sys, json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

p = Path(os.environ["HF_RESOLVED_DIR"])
# Basic sanity: config & tokenizer files must exist
candidates = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
have_any = any((p / f).exists() for f in candidates)
if not have_any:
    print("Snapshot dir contents:\n", "\n".join(str(x) for x in p.iterdir()))
    raise FileNotFoundError("No tokenizer/config files found in snapshot dir.")

# Try loading strictly offline
tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
mdl = AutoModel.from_pretrained(str(p), local_files_only=True)
print("OFFLINE_LOAD_OK", tok.__class__.__name__, mdl.__class__.__name__)
PY
ok "Offline self-test passed."

# ---------- Flip to offline for runtime ----------
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
ok "Transformers set to OFFLINE mode."

# ---------- Torch diag (non-fatal) ----------
micromamba run -n cage-fusion python - <<'PY' || true
import torch
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
PY

# ---------- Start FastAPI ----------
ok "Starting Uvicorn…"
exec micromamba run -n cage-fusion \
  env HF_RESOLVED_DIR="$HF_RESOLVED_DIR" TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  uvicorn cage_fusion.api.fast_router:app --host 0.0.0.0 --port "${PORT}" --workers "${WORKERS}"
