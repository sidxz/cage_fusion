# cage_fusion/utils/hf_loader.py
import os
import transformers
from transformers import AutoTokenizer, AutoModel

# Suppress the verbose key-mismatch load report (UNEXPECTED/MISSING keys are
# expected when loading an encoder-only backbone from a task-specific checkpoint).
transformers.logging.set_verbosity_error()


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


def _resolve_pretrained_path(name_or_path: str) -> str:
    """Return a local directory path, downloading from Hub if needed.

    If *name_or_path* is an existing local directory it is returned unchanged.
    Otherwise it is treated as a HuggingFace Hub repo ID and downloaded via
    ``snapshot_download`` (cached in ``~/.cache/huggingface/hub``).
    """
    if os.path.isdir(name_or_path):
        return name_or_path
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=name_or_path)
    except Exception as e:
        raise FileNotFoundError(
            f"'{name_or_path}' is not a local directory and could not be "
            f"downloaded from the Hub: {e}"
        ) from e
