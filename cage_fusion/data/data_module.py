"""
cage_fusion/data/data_module.py
================================
High-level data loading helper — analogous to PyTorch-Lightning's
``LightningDataModule`` or HuggingFace's dataset wrappers.

:class:`CageFusionDataModule` wraps the three-step pipeline

    featurize → ``CageFusionStreamingDataset`` → ``DataLoader``

into a single, easy-to-use class.

Quick start
-----------
**From a CSV file** (auto-splits into train/val)::

    from cage_fusion.data import CageFusionDataModule

    dm = CageFusionDataModule.from_csv(
        "my_compounds.csv",
        label_cols=["active"],
        model_checkpoint="DeepChem/ChemBERTa-77M-MTR",
    )
    print(dm.label_names)   # ["active"]
    # dm.train_loader, dm.val_loader are ready

**From existing DataFrames**::

    dm = CageFusionDataModule.from_dataframes(
        train_df=train_df,
        val_df=val_df,
        label_cols=["PAINS_A", "PAINS_B", "Aggregator"],
        model_checkpoint="DeepChem/ChemBERTa-77M-MTR",
    )

**From a MoleculeNet benchmark dataset** (requires ``deepchem``)::

    dm = CageFusionDataModule.from_moleculenet(
        "bace_classification",
        model_checkpoint="DeepChem/ChemBERTa-77M-MTR",
    )
    print(dm.label_names)   # ["Class"]
"""

from __future__ import annotations

import logging
import os
import tempfile
from functools import partial
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cage_fusion.data.dataset import CageFusionStreamingDataset
from cage_fusion.data.collator import collate_cage_fusion
from cage_fusion.featurization import featurize_and_save_streaming
from cage_fusion.featurization.featurizer_utils import normalize_auxiliary_features
from cage_fusion.utils.hf_loader import load_hf_checkpoint, load_tokenizer

logger = logging.getLogger(__name__)


def _worker_init(_):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


class CageFusionDataModule:
    """
    High-level data module: featurise → dataset → DataLoader in one call.

    Attributes
    ----------
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader | None
    label_names  : List[str]   — task names, length ``num_labels``
    scaler       : sklearn scaler fitted on the training auxiliary features
    tokenizer    : the HuggingFace tokenizer used for featurisation

    Construction
    ------------
    Use one of the three class-methods instead of ``__init__``:

    - :py:meth:`from_dataframes`  — two (or three) pandas DataFrames
    - :py:meth:`from_csv`         — a single CSV file; splits automatically
    - :py:meth:`from_moleculenet` — a DeepChem MoleculeNet dataset name
    """

    def __init__(
        self,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        label_names: List[str],
        *,
        test_loader: Optional[DataLoader] = None,
        scaler=None,
        tokenizer=None,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_names = label_names
        self.scaler = scaler
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Primary factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframes(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        label_cols: List[str],
        model_checkpoint: str = "DeepChem/ChemBERTa-77M-MTR",
        *,
        tokenizer=None,
        embedding_model=None,
        test_df: Optional[pd.DataFrame] = None,
        smiles_col: str = "SMILES",
        cache_dir: str = ".cache/features",
        batch_size: int = 128,
        num_workers: int = 0,
        scaler=None,
        skip_featurize: bool = False,
    ) -> "CageFusionDataModule":
        """
        Build a data module from train / val (and optionally test) DataFrames.

        The DataFrames must contain a ``SMILES`` column (or the column named
        by *smiles_col*) and all columns in *label_cols*.

        Args:
            train_df: Training set DataFrame.
            val_df: Validation set DataFrame.
            label_cols: Column names to use as prediction targets.
            model_checkpoint: HuggingFace model ID or local path for the
                sequence encoder tokenizer + embedding model.
            tokenizer: Pre-loaded tokenizer (skips loading if provided).
            embedding_model: Pre-loaded embedding model (skips loading if
                provided).
            test_df: Optional test set DataFrame.
            smiles_col: Name of the SMILES column.  Renamed to ``"SMILES"``
                internally if different.
            cache_dir: Directory for HDF5 feature caches.
            batch_size: DataLoader batch size.
            num_workers: DataLoader worker count.
            scaler: Pre-fitted scaler.  If *None*, fit on the training set.

        Returns:
            :class:`CageFusionDataModule` with ``train_loader``,
            ``val_loader``, and optionally ``test_loader`` ready.

        Example::

            dm = CageFusionDataModule.from_dataframes(
                train_df=train_df,
                val_df=val_df,
                label_cols=["PAINS_A", "Aggregator"],
                model_checkpoint="DeepChem/ChemBERTa-77M-MTR",
            )
        """
        # Normalise SMILES column name
        for df in [d for d in [train_df, val_df, test_df] if d is not None]:
            if smiles_col != "SMILES" and smiles_col in df.columns:
                df.rename(columns={smiles_col: "SMILES"}, inplace=True)

        os.makedirs(cache_dir, exist_ok=True)
        _scaler_path = os.path.join(cache_dir, "aux_features_scaler.pkl")

        def _ensure_norm_aux(h5_path: str, fitted_scaler) -> None:
            """Write 'auxiliary_features_normalized' if not already present."""
            import h5py as _h5py
            if fitted_scaler is None:
                return
            with _h5py.File(h5_path, "r") as _f:
                if "auxiliary_features_normalized" in _f:
                    return
            aux_dim = len(fitted_scaler.mean_)
            normalize_auxiliary_features(h5_path, fitted_scaler, aux_dim)

        # Determine which splits need featurisation
        splits = [("train", train_df), ("val", val_df)]
        if test_df is not None:
            splits.append(("test", test_df))

        _cached = {
            name: os.path.join(cache_dir, f"{name}_cage_fusion.h5")
            for name, _ in splits
        }
        _all_cached = all(os.path.isfile(p) for p in _cached.values())

        if skip_featurize and _all_cached:
            # Fast path: HDF5 caches already exist — skip ChemBERTa embedding pass
            logger.info(
                "skip_featurize=True and all HDF5 caches found in '%s' — "
                "skipping featurisation.",
                cache_dir,
            )
            if tokenizer is None:
                tokenizer = load_tokenizer(model_checkpoint)
            if scaler is None:
                if os.path.isfile(_scaler_path):
                    scaler = joblib.load(_scaler_path)
                    logger.info("Loaded scaler from '%s'.", _scaler_path)
                else:
                    logger.warning(
                        "No scaler found at '%s'; auxiliary features will be unscaled.",
                        _scaler_path,
                    )
            # Normalise aux features if the normalised dataset isn't already present
            for _name in _cached:
                _ensure_norm_aux(_cached[_name], scaler)
        else:
            if skip_featurize and not _all_cached:
                missing = [n for n, p in _cached.items() if not os.path.isfile(p)]
                logger.warning(
                    "skip_featurize=True but missing caches for splits %s — "
                    "running featurisation.",
                    missing,
                )

            # Load tokenizer + embedding model if not provided
            if tokenizer is None or embedding_model is None:
                logger.info("Loading sequence encoder from '%s'…", model_checkpoint)
                tok, emb = load_hf_checkpoint(model_checkpoint)
                tokenizer = tokenizer or tok
                embedding_model = embedding_model or emb

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embedding_model = embedding_model.to(device).eval()

            # Featurise training split (fit scaler)
            logger.info("Featurising training split (%d molecules)…", len(train_df))
            train_h5, fitted_scaler, _ = featurize_and_save_streaming(
                df=train_df,
                name="train",
                label_cols=label_cols,
                cache_dir=cache_dir,
                tokenizer=tokenizer,
                model=embedding_model,
                fit_scaler=(scaler is None),
                scaler=scaler,
            )
            scaler = fitted_scaler

            # Auto-save fitted scaler alongside the HDF5 caches for future reuse
            joblib.dump(scaler, _scaler_path)
            logger.info("Scaler cached to '%s'.", _scaler_path)

            # Write normalised aux features (raw descriptors are always stored first;
            # normalised is a second dataset keyed "auxiliary_features_normalized")
            _ensure_norm_aux(_cached["train"], scaler)

            # Featurise val split (apply fitted scaler)
            logger.info("Featurising validation split (%d molecules)…", len(val_df))
            featurize_and_save_streaming(
                df=val_df,
                name="val",
                label_cols=label_cols,
                cache_dir=cache_dir,
                tokenizer=tokenizer,
                model=embedding_model,
                fit_scaler=False,
                scaler=scaler,
            )
            _ensure_norm_aux(_cached["val"], scaler)

            if test_df is not None:
                logger.info("Featurising test split (%d molecules)…", len(test_df))
                featurize_and_save_streaming(
                    df=test_df,
                    name="test",
                    label_cols=label_cols,
                    cache_dir=cache_dir,
                    tokenizer=tokenizer,
                    model=embedding_model,
                    fit_scaler=False,
                    scaler=scaler,
                )
                _ensure_norm_aux(_cached["test"], scaler)

        train_loader = _make_loader(
            _cached["train"], tokenizer, batch_size, num_workers, shuffle=True
        )
        val_loader = _make_loader(
            _cached["val"], tokenizer, batch_size, num_workers, shuffle=False
        )

        test_loader = None
        if test_df is not None:
            test_loader = _make_loader(
                _cached["test"], tokenizer, batch_size, num_workers, shuffle=False
            )

        return cls(
            train_loader=train_loader,
            val_loader=val_loader,
            label_names=list(label_cols),
            test_loader=test_loader,
            scaler=scaler,
            tokenizer=tokenizer,
        )

    # ------------------------------------------------------------------
    # CSV convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        label_cols: List[str],
        model_checkpoint: str = "DeepChem/ChemBERTa-77M-MTR",
        *,
        smiles_col: str = "SMILES",
        val_split: float = 0.15,
        test_split: float = 0.10,
        random_state: int = 42,
        cache_dir: str = ".cache/features",
        batch_size: int = 128,
        num_workers: int = 0,
        scaler=None,
    ) -> "CageFusionDataModule":
        """
        Build a data module from a single CSV file.

        Splits the data into train / val (/ test) using stratified random
        splitting on the first label column.

        Args:
            csv_path: Path to a CSV file containing a SMILES column and all
                label columns.
            label_cols: Column names to use as prediction targets.
            model_checkpoint: HuggingFace model ID or local path.
            smiles_col: Name of the SMILES column (default ``"SMILES"``).
            val_split: Fraction of data for validation (default 0.15).
            test_split: Fraction of data for test (default 0.10).  Set to
                ``0.0`` to skip the test split.
            random_state: Random seed for reproducibility.
            cache_dir: Directory for HDF5 feature caches.
            batch_size: DataLoader batch size.
            num_workers: DataLoader worker count.

        Returns:
            :class:`CageFusionDataModule` with ready-to-use loaders.

        Example::

            dm = CageFusionDataModule.from_csv(
                "data/my_compounds.csv",
                label_cols=["active"],
                model_checkpoint="DeepChem/ChemBERTa-77M-MTR",
                val_split=0.1,
                test_split=0.1,
            )
        """
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows from '%s'", len(df), csv_path)

        # First carve out the test split (if requested)
        test_df: Optional[pd.DataFrame] = None
        if test_split > 0:
            df, test_df = train_test_split(
                df, test_size=test_split, random_state=random_state
            )

        # Split remaining into train/val
        val_size = val_split / (1.0 - test_split) if test_split > 0 else val_split
        train_df, val_df = train_test_split(
            df, test_size=val_size, random_state=random_state
        )

        return cls.from_dataframes(
            train_df=train_df.reset_index(drop=True),
            val_df=val_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True) if test_df is not None else None,
            label_cols=label_cols,
            model_checkpoint=model_checkpoint,
            smiles_col=smiles_col,
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            scaler=scaler,
        )

    # ------------------------------------------------------------------
    # Inference-only convenience
    # ------------------------------------------------------------------

    @classmethod
    def for_inference(
        cls,
        csv_path: str,
        label_cols: List[str],
        model_checkpoint: str = "DeepChem/ChemBERTa-77M-MTR",
        *,
        smiles_col: str = "SMILES",
        cache_dir: str = ".cache/features",
        batch_size: int = 128,
        num_workers: int = 0,
        scaler=None,
    ) -> "CageFusionDataModule":
        """
        Build a data module for inference only — no train/val split.

        All rows in *csv_path* are featurised and exposed via
        :attr:`test_loader`.  Pass a pre-fitted *scaler* (from the
        training ``CageFusionDataModule``) so auxiliary features are
        transformed on the same scale as during training.

        Args:
            csv_path: Path to a CSV with a SMILES column.
            label_cols: Target column names.  Pass ``[]`` for
                unlabelled inference data.
            model_checkpoint: HuggingFace model ID or local path.
            smiles_col: Name of the SMILES column.
            cache_dir: Directory for HDF5 feature cache.
            batch_size: DataLoader batch size.
            num_workers: DataLoader worker count.
            scaler: Pre-fitted scaler from training.  If *None*, a new
                scaler is fitted on this data (not recommended for
                inference — auxiliary features may be on a different
                scale).

        Example::

            dm_infer = CageFusionDataModule.for_inference(
                "data/new_compounds.csv",
                label_cols=[],
                model_checkpoint="DeepChem/ChemBERTa-77M-MTR",
                scaler=dm.scaler,
                cache_dir="data/tmp/infer_cache",
            )
            for batch in dm_infer.test_loader:
                ...
        """
        df = pd.read_csv(csv_path)
        if smiles_col != "SMILES" and smiles_col in df.columns:
            df.rename(columns={smiles_col: "SMILES"}, inplace=True)
        logger.info("Loaded %d rows from '%s' (inference mode)", len(df), csv_path)

        logger.info("Loading sequence encoder from '%s'…", model_checkpoint)
        tokenizer, embedding_model = load_hf_checkpoint(model_checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model = embedding_model.to(device).eval()

        os.makedirs(cache_dir, exist_ok=True)

        logger.info("Featurising %d molecules…", len(df))
        test_h5, fitted_scaler, _ = featurize_and_save_streaming(
            df=df,
            name="infer",
            label_cols=label_cols,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            model=embedding_model,
            fit_scaler=(scaler is None),
            scaler=scaler,
        )
        test_loader = _make_loader(test_h5, tokenizer, batch_size, num_workers, shuffle=False)
        return cls(
            train_loader=None,
            val_loader=None,
            label_names=list(label_cols),
            test_loader=test_loader,
            scaler=fitted_scaler,
            tokenizer=tokenizer,
        )

    # ------------------------------------------------------------------
    # MoleculeNet convenience (requires deepchem)
    # ------------------------------------------------------------------

    @classmethod
    def from_moleculenet(
        cls,
        dataset_name: str,
        model_checkpoint: str = "DeepChem/ChemBERTa-77M-MTR",
        *,
        splitter: str = "scaffold",
        seed: int = 42,
        data_dir: str = "data/molnet",
        cache_dir: str = ".cache/features",
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> "CageFusionDataModule":
        """
        Build a data module from a MoleculeNet benchmark dataset.

        Requires ``deepchem`` (``pip install deepchem``).

        Args:
            dataset_name: DeepChem loader name, e.g. ``"bace_classification"``,
                ``"tox21"``, ``"sider"``.  The full list is at
                https://deepchem.io/docs/api_reference/moleculenet.html
            model_checkpoint: HuggingFace model ID or local path.
            splitter: ``"scaffold"`` (default), ``"random"``, or ``"stratified"``.
            seed: Random seed.
            data_dir: Directory to download / cache MoleculeNet files.
            cache_dir: Directory for HDF5 feature caches.
            batch_size: DataLoader batch size.
            num_workers: DataLoader worker count.

        Returns:
            :class:`CageFusionDataModule`; ``label_names`` is taken from the
            DeepChem task list.

        Example::

            dm = CageFusionDataModule.from_moleculenet(
                "bace_classification",
                splitter="scaffold",
            )
            print(dm.label_names)   # ["Class"]
        """
        try:
            import deepchem as dc
            from deepchem.feat import RawFeaturizer
        except ImportError as e:
            raise ImportError(
                "DeepChem is required for 'from_moleculenet'. "
                "Install it with: pip install deepchem"
            ) from e

        logger.info("Loading MoleculeNet dataset '%s'…", dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        loader_fn = getattr(dc.molnet, f"load_{dataset_name}", None)
        if loader_fn is None:
            raise ValueError(
                f"Dataset 'load_{dataset_name}' not found in deepchem.molnet. "
                f"Check https://deepchem.io/docs/api_reference/moleculenet.html"
            )

        tasks, datasets, _ = loader_fn(
            featurizer=RawFeaturizer(),
            splitter=splitter,
            reload=True,
            data_dir=data_dir,
            seed=seed,
        )
        train_ds, val_ds, test_ds = datasets
        label_cols = list(tasks)

        def _to_df(ds) -> pd.DataFrame:
            data = {"SMILES": list(ds.ids)}
            for i, t in enumerate(label_cols):
                data[t] = ds.y[:, i]
            return pd.DataFrame(data)

        train_df = _to_df(train_ds)
        val_df = _to_df(val_ds)
        test_df = _to_df(test_ds)

        logger.info(
            "Dataset '%s': tasks=%s  train=%d  val=%d  test=%d",
            dataset_name, label_cols, len(train_df), len(val_df), len(test_df),
        )

        return cls.from_dataframes(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_cols=label_cols,
            model_checkpoint=model_checkpoint,
            cache_dir=os.path.join(cache_dir, dataset_name),
            batch_size=batch_size,
            num_workers=num_workers,
        )

    # ------------------------------------------------------------------

    def save_scaler(self, directory: str) -> None:
        """Persist the fitted scaler to *directory*/aux_features_scaler.pkl."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(directory, "aux_features_scaler.pkl"))
        logger.info("Scaler saved to '%s'.", directory)

    def __repr__(self) -> str:
        n_train = len(self.train_loader.dataset)  # type: ignore[arg-type]
        n_val = len(self.val_loader.dataset)  # type: ignore[arg-type]
        n_test = (
            len(self.test_loader.dataset) if self.test_loader else 0  # type: ignore[arg-type]
        )
        return (
            f"CageFusionDataModule("
            f"labels={self.label_names}, "
            f"train={n_train}, val={n_val}, test={n_test})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_loader(
    h5_path: str,
    tokenizer,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    ds = CageFusionStreamingDataset(
        h5_path,
        tokenizer_pad_id=tokenizer.pad_token_id,
        prefer_normalized_aux=True,
        return_ids=True,
        total_num_workers=num_workers,
        graph_cache="auto",
        single_worker_graph_cache=True,
        emb_cache_store_dtype=np.float32,
        return_emb_dtype=torch.float32,
    )
    collate_fn = partial(collate_cage_fusion, pad_token_id=tokenizer.pad_token_id)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init,
        pin_memory=torch.cuda.is_available(),
    )
