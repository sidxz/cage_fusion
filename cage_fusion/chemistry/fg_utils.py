"""
Functional-group detection using pre-compiled SMARTS patterns.

Patterns are loaded from ``cage_fusion/dt/pains3.json`` at import time.
The module exposes:

- ``FG_NAMES``            – ordered list of FG names
- ``NUM_FUNCTIONAL_GROUPS`` – vocabulary size (len of FG_NAMES)
- ``get_functional_groups(mol)`` – returns list of FG indices for a molecule
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from rdkit import Chem

logger = logging.getLogger("cagefusion")

# ── Load SMARTS patterns ──────────────────────────────────────────────────────

_FG_PATH = Path(__file__).parent.parent / "dt" / "pains3.json"

if not _FG_PATH.is_file():
    raise FileNotFoundError(
        f"Functional-group SMARTS file not found: {_FG_PATH.resolve()}\n"
        "Make sure cage_fusion/dt/pains3.json is present in your installation."
    )

with open(_FG_PATH, "r") as _f:
    _raw = json.load(_f)

if not isinstance(_raw, dict) or not _raw:
    raise ValueError(
        f"Expected a non-empty dict in {_FG_PATH.resolve()}, got {type(_raw)}"
    )

# Pre-compile SMARTS; warn about invalid patterns but don't abort
_FG_SMARTS: dict[str, Chem.Mol] = {}
_invalid: list[tuple[str, str]] = []

for _name, _pattern in _raw.items():
    _mol = Chem.MolFromSmarts(_pattern)
    if _mol is None:
        _invalid.append((_name, _pattern))
    else:
        _FG_SMARTS[_name] = _mol

if _invalid:
    logger.warning(
        "The following SMARTS patterns are invalid and were skipped:\n%s",
        "\n".join(f"  • {n}: {p}" for n, p in _invalid),
    )

if not _FG_SMARTS:
    raise RuntimeError(
        f"No valid SMARTS patterns could be loaded from {_FG_PATH.resolve()}."
    )

FG_NAMES: List[str] = list(_FG_SMARTS.keys())
NUM_FUNCTIONAL_GROUPS: int = len(FG_NAMES)

logger.debug("Loaded %d functional groups from %s", NUM_FUNCTIONAL_GROUPS, _FG_PATH.name)


# ── Public API ────────────────────────────────────────────────────────────────

def get_functional_groups(mol: Chem.Mol) -> List[int]:
    """
    Return a list of FG indices (0-indexed into ``FG_NAMES``) for *mol*.

    Parameters
    ----------
    mol:
        RDKit molecule.  Returns an empty list when *mol* is ``None``.
    """
    if mol is None:
        return []
    return [
        i
        for i, name in enumerate(FG_NAMES)
        if _FG_SMARTS[name] is not None and mol.HasSubstructMatch(_FG_SMARTS[name])
    ]
