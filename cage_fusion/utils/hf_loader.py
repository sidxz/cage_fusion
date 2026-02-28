# cage_fusion/utils/hf_loader.py
import os
from transformers import AutoTokenizer, AutoModel


def _resolve_source(hf_id_or_dir: str):
    """
    Returns (source, local_files_only).
    If HF_RESOLVED_DIR is set and points to a valid directory, use it with
    local_files_only=True.  Otherwise fall back to the Hub id/path and let
    HuggingFace handle TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE natively.
    """
    local_dir = os.getenv("HF_RESOLVED_DIR", "").strip()
    if local_dir and os.path.isdir(local_dir):
        return local_dir, True
    return hf_id_or_dir, False


def load_hf_checkpoint(hf_id_or_dir: str):
    """Return (tokenizer, model) from Hub or a local snapshot directory."""
    src, local_only = _resolve_source(hf_id_or_dir)
    tok = AutoTokenizer.from_pretrained(src, local_files_only=local_only)
    mdl = AutoModel.from_pretrained(src, local_files_only=local_only)
    return tok, mdl


def load_tokenizer(hf_id_or_dir: str):
    """Return a tokenizer from Hub or a local snapshot directory."""
    src, local_only = _resolve_source(hf_id_or_dir)
    return AutoTokenizer.from_pretrained(src, local_files_only=local_only)
