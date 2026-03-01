"""
benchmarks/openadmet/preprocessing.py
======================================
Per-endpoint log transforms for the OpenADMET ExpansionRx dataset.

The leaderboard evaluates predictions in log space (same scale as training).
Apply ``forward_transform`` to label columns before building the data module,
and ``inverse_transform`` to model predictions before writing submission.csv.

Transform table
---------------
LogD                         — none        (already log-scale)
KSOL                         — log10(y + 1)
HLM_CLint                    — log10(y + 1)
MLM_CLint                    — log10(y + 1)
Caco-2_Permeability_Papp_A_B — log10(y + 1)
Caco-2_Permeability_Efflux   — log10(y + 1)
MPPB                         — log10(y + 1)
MBPB                         — log10(y + 1)
MGMB                         — log10(y + 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Exact column names as they appear in the HuggingFace dataset ─────────────

LABEL_COLS: list[str] = [
    "LogD",
    "KSOL",
    "HLM_CLint",
    "MLM_CLint",
    "Caco-2_Permeability_Papp_A_B",
    "Caco-2_Permeability_Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
]

SMILES_COL = "SMILES"
NAME_COL   = "Molecule_Name"

# (transform_type, multiplier_before_log)
# "none"  → column is used as-is
# "log1p" → log10(multiplier * y + 1)   (clips to 0 first for safety)
_TRANSFORM_TABLE: dict[str, tuple[str, float]] = {
    "LogD":                         ("none",  1.0),
    "KSOL":                         ("log1p", 1.0),
    "HLM_CLint":                    ("log1p", 1.0),
    "MLM_CLint":                    ("log1p", 1.0),
    "Caco-2_Permeability_Papp_A_B": ("log1p", 1.0),
    "Caco-2_Permeability_Efflux":   ("log1p", 1.0),
    "MPPB":                         ("log1p", 1.0),
    "MBPB":                         ("log1p", 1.0),
    "MGMB":                         ("log1p", 1.0),
}


def forward_transform(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Apply per-endpoint log transforms to label columns in-place on a copy.

    NaN values are preserved unchanged.

    Args:
        df:   DataFrame containing label columns.
        cols: Subset of ``LABEL_COLS`` to transform.  Defaults to all.

    Returns:
        New DataFrame with transformed label columns.
    """
    df = df.copy()
    cols = cols or LABEL_COLS
    for col in cols:
        if col not in df.columns:
            continue
        kind, mult = _TRANSFORM_TABLE.get(col, ("none", 1.0))
        if kind == "log1p":
            vals = df[col].to_numpy(dtype=float)
            # clip to 0 before applying multiplier (measurement can't be negative)
            vals = np.where(np.isnan(vals), np.nan, np.log10(np.clip(vals, 0, None) * mult + 1))
            df[col] = vals
        # "none" → leave as-is
    return df


def inverse_transform(
    arr: np.ndarray,
    cols: list[str] | None = None,
) -> np.ndarray:
    """Invert log transforms to recover original measurement units.

    Args:
        arr:  (N, len(cols)) array of log-scale predictions.
        cols: Column names matching the order of ``arr`` columns.
              Defaults to ``LABEL_COLS``.

    Returns:
        Array in original measurement units (same shape as ``arr``).
    """
    cols = cols or LABEL_COLS
    arr  = arr.copy().astype(float)
    for i, col in enumerate(cols):
        kind, mult = _TRANSFORM_TABLE.get(col, ("none", 1.0))
        if kind == "log1p":
            # inverse: y = (10^x - 1) / mult
            arr[:, i] = (10 ** arr[:, i] - 1) / mult
    return arr
