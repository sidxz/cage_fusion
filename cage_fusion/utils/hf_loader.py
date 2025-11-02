# cage_fusion/utils/hf_loader.py
import os
from typing import Tuple
from transformers import AutoTokenizer, AutoModel

def _is_offline() -> bool:
    return os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"

def _prefer_local_snapshot() -> Tuple[str, bool]:
    """
    Returns (source_path_or_id, local_files_only).
    If HF_RESOLVED_DIR exists, use it with local_files_only=True.
    Else, if offline -> raise; else use Hub id with local_files_only=False.
    """
    local_dir = os.getenv("HF_RESOLVED_DIR", "").strip()
    if local_dir and os.path.isdir(local_dir):
        return local_dir, True

    # No local snapshot
    if _is_offline():
        raise FileNotFoundError(
            "TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE is set but HF_RESOLVED_DIR "
            "does not point to a valid snapshot directory. "
            "Warm the cache first (entrypoint) or unset offline mode."
        )
    # Hub path (online)
    return "", False

def load_hf_checkpoint(hf_id_or_dir: str):
    """
    Return (tokenizer, model). If HF_RESOLVED_DIR is valid, load from it,
    otherwise load from hub using hf_id_or_dir.
    """
    src, local_only = _prefer_local_snapshot()
    src = src or hf_id_or_dir
    tok = AutoTokenizer.from_pretrained(src, local_files_only=local_only)
    mdl = AutoModel.from_pretrained(src, local_files_only=local_only)
    return tok, mdl

def load_tokenizer(hf_id_or_dir: str):
    src, local_only = _prefer_local_snapshot()
    src = src or hf_id_or_dir
    return AutoTokenizer.from_pretrained(src, local_files_only=local_only)

def load_model(hf_id_or_dir: str):
    src, local_only = _prefer_local_snapshot()
    src = src or hf_id_or_dir
    return AutoModel.from_pretrained(src, local_files_only=local_only)
