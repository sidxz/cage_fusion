import sys
import json
from pathlib import Path
from rdkit import Chem

# --- DYNAMIC INITIALIZATION FROM FILE ---
fg_path = Path(__file__).parent.parent / "dt/pains2.json"

if not fg_path.is_file():
    sys.exit(f"❌ Error: Functional groups file not found at {fg_path.resolve()}")

# --- Load JSON ---
try:
    with open(fg_path, "r") as f:
        FUNCTIONAL_GROUPS = json.load(f)
except json.JSONDecodeError as e:
    sys.exit(f"❌ Error: Failed to parse JSON file at {fg_path.resolve()}.\n{e}")

if not isinstance(FUNCTIONAL_GROUPS, dict) or not FUNCTIONAL_GROUPS:
    sys.exit(
        f"❌ Error: Functional groups data is empty or malformed in {fg_path.resolve()}"
    )

# --- Pre-compile SMARTS patterns ---
FG_SMARTS = {}
invalid_smarts = []

for name, pattern in FUNCTIONAL_GROUPS.items():
    mol = Chem.MolFromSmarts(pattern)
    if mol is None:
        invalid_smarts.append((name, pattern))
    else:
        FG_SMARTS[name] = mol

if invalid_smarts:
    print(
        "❌ The following SMARTS patterns are invalid and were skipped:",
        file=sys.stderr,
    )
    for name, pattern in invalid_smarts:
        print(f"  • {name}: {pattern}", file=sys.stderr)

if not FG_SMARTS:
    sys.exit("❌ Error: No valid SMARTS patterns loaded. Cannot proceed.")

# --- Final Constants ---
FG_NAMES = list(FG_SMARTS.keys())
NUM_FUNCTIONAL_GROUPS = len(FG_NAMES)

print(f"✅ Loaded {NUM_FUNCTIONAL_GROUPS} valid functional groups from {fg_path.name}")


# --- FG Matcher Function ---
def get_functional_groups(mol: Chem.Mol):
    """
    Returns a list of functional group indices for a given RDKit molecule.
    """
    fg_ids = []
    if mol is None:
        return fg_ids

    for i, name in enumerate(FG_NAMES):
        smarts_mol = FG_SMARTS[name]
        if smarts_mol is not None and mol.HasSubstructMatch(smarts_mol):
            fg_ids.append(i)

    return fg_ids
