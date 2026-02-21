# gflownet/src/gflownet/tasks/resolve_featurise.py

import torch
from rdkit import Chem
from torch_geometric.data import Data


ATOM_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
    'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti',
    'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
    'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
    'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
    'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Rf', 'Db', 'Sg',
    'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Fl', 'Lv', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', '*'
]

NUM_ATOM_TYPES = len(ATOM_SYMBOLS)


def categorical_type(x, permitted_list):
    return permitted_list.index(x)


def get_ring_size(atom_or_bond, max_size=12):
    if not atom_or_bond.IsInRing():
        return 0
    for i in range(max_size):
        if atom_or_bond.IsInRingSize(i):
            return i
    return 0


def get_atom_features(atom):
    atom_type_enc = categorical_type(atom.GetSymbol(), ATOM_SYMBOLS)
    n_heavy_neighbors_enc = atom.GetDegree()
    is_in_ring_enc = atom.IsInRing()
    ring_size = get_ring_size(atom)
    is_aromatic_enc = atom.GetIsAromatic()
    atomic_mass_scaled = (atom.GetMass() - 10.812) / 116.092
    vdw_radius_scaled = (Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6
    covalent_radius_scaled = (Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76
    valence = atom.GetTotalValence()

    return torch.tensor([
        atom_type_enc,
        ring_size,
        is_in_ring_enc,
        n_heavy_neighbors_enc,
        is_aromatic_enc,
        atomic_mass_scaled,
        vdw_radius_scaled,
        covalent_radius_scaled,
        valence
    ], dtype=torch.float)


def get_bond_features(bond):
    permitted_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        Chem.rdchem.BondType.UNSPECIFIED,
    ]
    bond_type_enc = categorical_type(bond.GetBondType(), permitted_bond_types)
    is_conj_enc = bond.GetIsConjugated()
    is_in_ring_enc = bond.IsInRing()
    ring_size = get_ring_size(bond)

    return torch.tensor([
        bond_type_enc,
        is_conj_enc,
        is_in_ring_enc,
        ring_size
    ], dtype=torch.float)


def mol_to_graph(mol):
    """
    RDKit Mol -> PyG Data
    MUST match ResolveGNN training exactly.
    """
    if mol is None:
        return None

    # Node features
    x = torch.stack([get_atom_features(a) for a in mol.GetAtoms()])

    # Edges
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)

        # Undirected graph → add both directions
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

