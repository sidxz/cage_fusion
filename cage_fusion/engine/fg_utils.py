import sys
from rdkit import Chem

# A more robust and validated list of functional groups.
# Using explicit SMARTS patterns is safer than using shorthands.
FUNCTIONAL_GROUPS = {
    "Hydroxyl": "[#6][OX2H]",
    "Carbonyl": "[#6][CX3](=O)[#6]",
    "Carboxyl": "[CX3](=O)[OX2H1]",
    "Amine_Primary": "[NH2;!$(N=O)]",
    "Amine_Secondary": "[NH1;!$(N=O)]",
    "Amine_Tertiary": "[N;!$(N=O);!$(N-N)]",
    "Ether": "[OD2]([#6])[#6]",
    "Ester": "[#6][CX3](=O)[OX2][#6]",
    "Phenyl": "c1ccccc1",
    "Nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "Thiol": "[#16X2H]",
    # Known PAINS/Nuisance motifs
    "Catechol": "c1c(O)c(O)ccc1",
    "Rhodanine": "C1C(=S)NC(=O)S1",
    "Michael_Acceptor_1": "[#6]=[#6]C(=O)",  # Enone
    "Michael_Acceptor_2": "[#6]=[#6]C#N",  # Acrylonitrile
}

# --- ROBUST INITIALIZATION ---
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
