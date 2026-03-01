"""
benchmarks/openadmet/data_loader.py
=====================================
Dataset fetching helpers for pretraining and the OpenADMET benchmark.

Functions
---------
load_openadmet()         — official train/test split from HuggingFace
load_tdc_admet()         — individual TDC ADMET regression datasets
build_pretrain_dataset() — merged wide DataFrame for Stage-1 pretraining
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("cagefusion")

# ── OpenADMET column names ────────────────────────────────────────────────────

OPENADMET_LABEL_COLS = [
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

# ── TDC task registry ─────────────────────────────────────────────────────────
# Each entry: (tdc_group, tdc_name, canonical_col_name, transform_hint)
# transform_hint is informational only — preprocessing.py handles actual transforms.

TDC_ADMET_TASKS = [
    # Lipophilicity / logD
    ("ADMET", "Lipophilicity_AstraZeneca",    "LogD_AZ",          "none"),
    # Solubility
    ("ADMET", "Solubility_AqSolDB",           "logS_AqSol",       "none"),
    # Permeability
    ("ADMET", "Caco2_Wang",                   "logPapp_Caco2",    "none"),
    ("ADMET", "PAMPA_NCATS",                  "logPapp_PAMPA",    "none"),
    # Plasma protein binding
    ("ADMET", "PPBR_AZ",                      "PPBR_pct_unbound", "log1p"),
    # Volume of distribution
    ("ADMET", "VDss_Lombardo",                "VDss_Lombardo",    "none"),
    # Half-life
    ("ADMET", "Half_Life_Obach",              "HalfLife_h",       "none"),
    # Clearance
    ("ADMET", "Clearance_Hepatocyte_AZ",      "CLint_Hepato",     "none"),
    ("ADMET", "Clearance_Microsome_AZ",       "CLint_Micro",      "none"),
    # hERG
    ("ADMET", "hERG",                         "hERG_pIC50",       "none"),
]

# MoleculeNet regression tasks (loaded via DeepChem or direct CSV)
MOLECULENET_TASKS = [
    ("ESOL",         "logS_ESOL"),
    ("FreeSolv",     "dGhyd_FreeSolv"),
    ("Lipophilicity","logD_MolNet"),
]


def load_openadmet(cache_dir: str = "/data-1/cage-fusion-admet/datasets") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official OpenADMET ExpansionRx train/test split from HuggingFace.

    Returns the ML-ready (default config, in-range measurements only) version.

    Args:
        cache_dir: Local directory to cache the downloaded dataset.

    Returns:
        ``(train_df, test_df)`` — DataFrames with columns ``SMILES``,
        ``Molecule_Name``, and the 9 ADMET endpoint columns.
        Missing measurements are represented as ``NaN``.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required.  Install with: pip install datasets"
        )

    logger.info("Loading OpenADMET ExpansionRx dataset from HuggingFace…")
    ds = load_dataset(
        "openadmet/openadmet-expansionrx-challenge-data",
        cache_dir=cache_dir,
    )
    train_df = ds["train"].to_pandas()
    test_df  = ds["test"].to_pandas()

    logger.info(
        "OpenADMET loaded: %d train / %d test molecules",
        len(train_df), len(test_df),
    )
    return train_df, test_df


def load_tdc_task(
    tdc_name: str,
    target_col: str,
    cache_dir: str = "/data-1/cage-fusion-pretrain/datasets",
) -> Optional[pd.DataFrame]:
    """Load a single TDC ADMET regression dataset.

    Args:
        tdc_name:   TDC dataset name (e.g. ``"Lipophilicity_AstraZeneca"``).
        target_col: Name to assign to the target column in the returned DataFrame.
        cache_dir:  Local directory to cache downloaded datasets.

    Returns:
        DataFrame with columns ``["SMILES", target_col]``, or ``None`` on failure.
    """
    try:
        from tdc.single_pred import ADMET
    except ImportError:
        raise ImportError(
            "The 'PyTDC' package is required.  Install with: pip install PyTDC"
        )

    try:
        data = ADMET(name=tdc_name, path=cache_dir)
        df   = data.get_data()
        # TDC returns columns: Drug, Drug_ID, Y
        df = df.rename(columns={"Drug": "SMILES", "Y": target_col})[["SMILES", target_col]]
        df = df.dropna(subset=["SMILES"])
        logger.info("TDC %-40s  %d molecules", tdc_name, len(df))
        return df
    except Exception as e:
        logger.warning("Could not load TDC task '%s': %s", tdc_name, e)
        return None


def load_moleculenet_task(
    name: str,
    target_col: str,
    cache_dir: str = "/data-1/cage-fusion-pretrain/datasets",
) -> Optional[pd.DataFrame]:
    """Load a MoleculeNet regression dataset via DeepChem.

    Args:
        name:       DeepChem MoleculeNet loader name (e.g. ``"ESOL"``).
        target_col: Name to assign to the target column.
        cache_dir:  Local cache directory.

    Returns:
        DataFrame with columns ``["SMILES", target_col]``, or ``None`` on failure.
    """
    try:
        import deepchem as dc
    except ImportError:
        raise ImportError(
            "DeepChem is required for MoleculeNet loading.  "
            "Install with: pip install deepchem"
        )

    loaders = {
        "ESOL":          dc.molnet.load_delaney,
        "FreeSolv":      dc.molnet.load_freesolv,
        "Lipophilicity": dc.molnet.load_lipo,
    }
    loader = loaders.get(name)
    if loader is None:
        logger.warning("Unknown MoleculeNet task: '%s'", name)
        return None

    try:
        tasks, datasets, _ = loader(
            featurizer="Raw", splitter=None, data_dir=cache_dir
        )
        dataset = datasets[0]
        smiles  = [d.smiles for d in dataset.X] if hasattr(dataset.X[0], "smiles") \
                  else list(dataset.ids)
        labels  = dataset.y[:, 0].tolist()
        df = pd.DataFrame({"SMILES": smiles, target_col: labels})
        df = df.dropna(subset=["SMILES"])
        logger.info("MoleculeNet %-35s  %d molecules", name, len(df))
        return df
    except Exception as e:
        logger.warning("Could not load MoleculeNet task '%s': %s", name, e)
        return None


def build_pretrain_dataset(
    tdc_cache:        str = "/data-1/cage-fusion-pretrain/datasets",
    moleculenet_cache: str = "/data-1/cage-fusion-pretrain/datasets",
    output_csv:       str = "/data-1/cage-fusion-pretrain/datasets/pretrain_merged.csv",
) -> pd.DataFrame:
    """Fetch all TDC + MoleculeNet ADMET datasets and merge into a wide DataFrame.

    Each row is one unique SMILES string.  Columns are one per endpoint; rows
    not measured for a given endpoint have ``NaN``.

    The resulting DataFrame is suitable for ``CageFusionDataModule.from_dataframes``
    with a masked MSE loss that ignores NaN targets.

    Args:
        tdc_cache:         Directory for TDC downloads.
        moleculenet_cache: Directory for MoleculeNet downloads.
        output_csv:        Path to save the merged CSV.  Saved automatically.

    Returns:
        Wide DataFrame with columns ``["SMILES", endpoint_1, ..., endpoint_N]``.
    """
    import os
    os.makedirs(tdc_cache, exist_ok=True)
    os.makedirs(moleculenet_cache, exist_ok=True)

    frames: list[pd.DataFrame] = []

    # TDC datasets
    for _, tdc_name, col_name, _ in TDC_ADMET_TASKS:
        df = load_tdc_task(tdc_name, col_name, cache_dir=tdc_cache)
        if df is not None:
            frames.append(df)

    # MoleculeNet datasets
    for mn_name, col_name in MOLECULENET_TASKS:
        df = load_moleculenet_task(mn_name, col_name, cache_dir=moleculenet_cache)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No datasets could be loaded. Check TDC/DeepChem installation.")

    # Merge on SMILES — outer join so every molecule is retained
    # Start with the first DataFrame; outer-merge each subsequent one
    merged = frames[0]
    for df in frames[1:]:
        merged = pd.merge(merged, df, on="SMILES", how="outer")

    # Deduplicate SMILES (keep first occurrence)
    merged = merged.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    logger.info(
        "Pretrain dataset: %d unique molecules, %d endpoints, %.1f%% coverage",
        len(merged),
        len(merged.columns) - 1,
        100 * merged.iloc[:, 1:].notna().mean().mean(),
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    logger.info("Saved pretrain dataset to %s", output_csv)
    return merged


def get_pretrain_label_cols(df: pd.DataFrame) -> list[str]:
    """Return all endpoint columns (everything except SMILES) from a pretrain DataFrame."""
    return [c for c in df.columns if c != "SMILES"]
