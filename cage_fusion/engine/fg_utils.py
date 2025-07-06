import sys
import json
from pathlib import Path
from rdkit import Chem

# --- DYNAMIC INITIALIZATION FROM FILE ---
# Use Path for robust file path handling.
fg_path = Path(__file__).parent.parent / "dt/functional_groups.json"

if not fg_path.is_file():
    # Provide a clear error if the file is missing.
    sys.exit(f"Error: Functional groups file not found at {fg_path}")

with open(fg_path, "r") as f:
    FUNCTIONAL_GROUPS = json.load(f)

# Pre-compile the SMARTS patterns and check for errors.
FG_SMARTS = {}
for name, pattern in FUNCTIONAL_GROUPS.items():
    mol = Chem.MolFromSmarts(pattern)
    if mol is None:
        # This check will catch invalid SMARTS patterns immediately.
        print(
            f"Error: Invalid SMARTS pattern for functional group '{name}': {pattern}",
            file=sys.stderr,
        )
    FG_SMARTS[name] = mol

FG_NAMES = list(FUNCTIONAL_GROUPS.keys())
NUM_FUNCTIONAL_GROUPS = len(FG_NAMES)


def get_functional_groups(mol: Chem.Mol):
    """Returns a list of functional group IDs for a single molecule."""
    fg_ids = []
    if mol is None:
        return fg_ids

    for i, name in enumerate(FG_NAMES):
        smarts_mol = FG_SMARTS[name]
        # Ensure the SMARTS pattern was valid before using it.
        if smarts_mol is not None and mol.HasSubstructMatch(smarts_mol):
            fg_ids.append(i)
    return fg_ids
