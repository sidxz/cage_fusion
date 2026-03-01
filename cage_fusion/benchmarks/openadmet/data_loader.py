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
    # Lipophilicity / logD — TDC provides already in log scale
    ("ADME", "Lipophilicity_AstraZeneca",    "LogD_AZ",          "none"),
    # Solubility — TDC provides already in log scale (logS)
    ("ADME", "Solubility_AqSolDB",           "logS_AqSol",       "none"),
    # Permeability — TDC provides already in log scale (log10 Papp)
    ("ADME", "Caco2_Wang",                   "logPapp_Caco2",    "none"),
    ("ADME", "PAMPA_NCATS",                  "logPapp_PAMPA",    "none"),
    # Plasma protein binding — raw % (0–100); log1p compresses the right tail
    ("ADME", "PPBR_AZ",                      "PPBR_pct_unbound", "log1p"),
    # Volume of distribution — raw L/kg (0.01–800+); log-normal distribution
    ("ADME", "VDss_Lombardo",                "VDss_Lombardo",    "log10"),
    # Half-life — raw hours (1–10,000+); log-normal distribution
    ("ADME", "Half_Life_Obach",              "HalfLife_h",       "log1p"),
    # Clearance — raw mL/min/kg; can be near 0; log-normal distribution
    ("ADME", "Clearance_Hepatocyte_AZ",      "CLint_Hepato",     "log1p"),
    ("ADME", "Clearance_Microsome_AZ",       "CLint_Micro",      "log1p"),
    # hERG — TDC provides as pIC50 (already log scale)
    ("Tox",  "hERG",                         "hERG_pIC50",       "none"),
    # Acute toxicity — TDC provides raw mol/kg; log10 to compress range
    ("Tox",  "LD50_Zhu",                     "AcuteTox_LD50",    "log10"),
]

# Map transform hint → vectorised numpy function applied to the raw Y column.
# All functions handle positive values; log1p is preferred when zeros are possible.
_TRANSFORMS: dict[str, object] = {
    "none":  None,
    "log10": np.log10,
    "log1p": np.log1p,
}

# ── TDC classification task registry ─────────────────────────────────────────
# Each entry: (tdc_group, tdc_name, canonical_col_name)
# All labels are binary (0/1).  No transforms needed.
# Skipped (intentionally excluded):
#   herg_central, hERG_Karim  — noisier binary versions of hERG (already covered
#                                as regression in Stage-1a)
#   ToxCast                   — very sparse & poorly curated; omit per ADMET-AI

TDC_CLASSIFICATION_TASKS = [
    # ── Absorption / Distribution ──────────────────────────────────────────────
    ("ADME", "HIA_Hou",                        "HIA"),
    ("ADME", "Pgp_Broccatelli",                "Pgp_Inhibitor"),
    ("ADME", "Bioavailability_Ma",             "Bioavailability"),
    ("ADME", "BBB_Martins",                    "BBB"),
    # ── CYP enzyme inhibition ─────────────────────────────────────────────────
    ("ADME", "CYP1A2_Veith",                   "CYP1A2_Inhibitor"),
    ("ADME", "CYP2C9_Veith",                   "CYP2C9_Inhibitor"),
    ("ADME", "CYP2C19_Veith",                  "CYP2C19_Inhibitor"),
    ("ADME", "CYP2D6_Veith",                   "CYP2D6_Inhibitor"),
    ("ADME", "CYP3A4_Veith",                   "CYP3A4_Inhibitor"),
    # ── CYP enzyme substrate ──────────────────────────────────────────────────
    ("ADME", "CYP2C9_Substrate_CarbonMangels", "CYP2C9_Substrate"),
    ("ADME", "CYP2D6_Substrate_CarbonMangels", "CYP2D6_Substrate"),
    ("ADME", "CYP3A4_Substrate_CarbonMangels", "CYP3A4_Substrate"),
    # ── Toxicity ──────────────────────────────────────────────────────────────
    ("Tox",  "AMES",                           "AMES_Mutagenicity"),
    ("Tox",  "DILI",                           "DILI"),
    ("Tox",  "Skin_Reaction",                  "Skin_Sensitizer"),
    ("Tox",  "Carcinogens_Lagunin",            "Carcinogen"),
    ("Tox",  "ClinTox",                        "ClinTox"),
    # TODO: Tox21 (12 assay endpoints) requires multi-pred loading — add later.
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

    # Normalise column names: HuggingFace uses spaces / special chars;
    # rename to underscore-based names used throughout this codebase.
    _RENAME = {
        "Molecule Name":                   "Molecule_Name",
        "HLM CLint":                       "HLM_CLint",
        "MLM CLint":                       "MLM_CLint",
        "Caco-2 Permeability Papp A>B":    "Caco-2_Permeability_Papp_A_B",
        "Caco-2 Permeability Efflux":      "Caco-2_Permeability_Efflux",
    }
    train_df = train_df.rename(columns=_RENAME)
    test_df  = test_df.rename(columns=_RENAME)

    logger.info(
        "OpenADMET loaded: %d train / %d test molecules",
        len(train_df), len(test_df),
    )
    return train_df, test_df


def load_tdc_task(
    tdc_name: str,
    target_col: str,
    tdc_group: str = "ADMET",
    cache_dir: str = "/data-1/cage-fusion-pretrain/datasets",
) -> Optional[pd.DataFrame]:
    """Load a single TDC regression dataset.

    Args:
        tdc_name:   TDC dataset name (e.g. ``"Lipophilicity_AstraZeneca"``).
        target_col: Name to assign to the target column in the returned DataFrame.
        tdc_group:  TDC single-pred group, e.g. ``"ADME"`` or ``"Tox"``.
        cache_dir:  Local directory to cache downloaded datasets.

    Returns:
        DataFrame with columns ``["SMILES", target_col]``, or ``None`` on failure.
    """
    try:
        import tdc.single_pred as tdc_sp
    except ModuleNotFoundError as e:
        if "pkg_resources" in str(e):
            raise ImportError(
                "PyTDC requires 'pkg_resources' (setuptools).  "
                "Install with: uv pip install setuptools"
            ) from e
        raise ImportError(
            "The 'PyTDC' package is required.  Install with: pip install PyTDC"
        ) from e

    loader_cls = getattr(tdc_sp, tdc_group, None)
    if loader_cls is None:
        logger.warning("Unknown TDC group '%s' for task '%s'", tdc_group, tdc_name)
        return None

    try:
        data = loader_cls(name=tdc_name, path=cache_dir)
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
    for tdc_group, tdc_name, col_name, transform in TDC_ADMET_TASKS:
        df = load_tdc_task(tdc_name, col_name, tdc_group=tdc_group, cache_dir=tdc_cache)
        if df is not None:
            fn = _TRANSFORMS.get(transform)
            if fn is not None:
                df[col_name] = fn(df[col_name].to_numpy(dtype=float))
                logger.info("  %-25s  transform=%s applied", col_name, transform)
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


def build_pretrain_classification_dataset(
    tdc_cache: str = "/data-1/cage-fusion-pretrain/datasets",
    output_csv: str = "/data-1/cage-fusion-pretrain/datasets/pretrain_classification_merged.csv",
) -> pd.DataFrame:
    """Fetch all TDC binary classification ADMET datasets and merge into a wide DataFrame.

    Each row is one unique SMILES string.  Label columns contain 0/1; rows not
    measured for a given endpoint have ``NaN`` — handled by the masked BCE loss.

    Covers 17 endpoints across absorption, distribution, CYP metabolism, and
    toxicity.  Tox21 (12 assays) is excluded pending multi-pred loader support.

    Args:
        tdc_cache:  Directory for TDC downloads.
        output_csv: Path to save the merged CSV.

    Returns:
        Wide DataFrame with columns ``["SMILES", task_1, ..., task_N]``.
    """
    import os
    os.makedirs(tdc_cache, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for tdc_group, tdc_name, col_name in TDC_CLASSIFICATION_TASKS:
        df = load_tdc_task(tdc_name, col_name, tdc_group=tdc_group, cache_dir=tdc_cache)
        if df is not None:
            # Coerce labels to {0, 1, NaN} — some TDC datasets may have floats
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            frames.append(df)

    if not frames:
        raise RuntimeError("No classification datasets could be loaded.")

    merged = frames[0]
    for df in frames[1:]:
        merged = pd.merge(merged, df, on="SMILES", how="outer")

    merged = merged.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    label_cols = [c for c in merged.columns if c != "SMILES"]
    coverage = 100 * merged[label_cols].notna().mean().mean()
    logger.info(
        "Classification pretrain dataset: %d unique molecules, %d endpoints, %.1f%% coverage",
        len(merged), len(label_cols), coverage,
    )
    for col in label_cols:
        n = merged[col].notna().sum()
        pos = merged[col].eq(1).sum()
        logger.info("  %-28s  n=%5d  pos_rate=%.1f%%", col, n, 100 * pos / max(n, 1))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    logger.info("Saved classification pretrain dataset to %s", output_csv)
    return merged
