from rdkit import Chem
import torch
from torch_geometric.data import Data


ATOM_HYB = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}

BOND_TYPES = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


# atom and bond featurisers
def atom_features(atom: Chem.rdchem.Atom) -> torch.Tensor:
    """Return a feature vector for a single atom."""
    return torch.tensor([
        atom.GetAtomicNum(),                   # 0: atomic number
        atom.GetTotalDegree(),                 # 1: number of connected atoms
        atom.GetTotalValence(),                # 2: total valence
        atom.GetFormalCharge(),                # 3: charge
        int(atom.GetIsAromatic()),             # 4: aromatic flag
        atom.GetTotalNumHs(),                  # 5: number of hydrogens
        ATOM_HYB.get(atom.GetHybridization(), 0),  # 6: hybridisation
    ], dtype=torch.float)


def bond_features(bond: Chem.rdchem.Bond) -> torch.Tensor:
    """Return a feature vector for a single bond."""
    return torch.tensor([
        BOND_TYPES.get(bond.GetBondType(), 0),   # 0: bond type
        int(bond.GetIsConjugated()),             # 1: conjugation
        int(bond.IsInRing()),                    # 2: ring membership
    ], dtype=torch.float)


# main conversion function
def smiles_to_data(smiles: str) -> Data:
    """Convert a SMILES string into a PyTorch Geometric Data graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Node features
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    # Edge indices and attributes
    edge_index_list, edge_attr_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Add both directions (undirected graph)
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list += [bf, bf]

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr_list)
    else:
        # Handle molecules without bonds (e.g. single atom)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    # Create torch_geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    return data
