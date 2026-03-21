import torch
import os
import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs

from .resolve_proxy_model import MPNNModel, ResolveProxyWrapper
from gflownet.utils.misc import get_worker_device


def tanimoto_dist(fp1, fp2):
    """
    Compute the Tanimoto distance between two molecular fingerprints.
    Defined as 1 − Tanimoto similarity.
    """
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)


def internal_diversity(fps):
    """Mean pairwise Tanimoto distance inside a batch"""
    if len(fps) < 2:
        return 0.0
    dists = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            dists.append(tanimoto_dist(fps[i], fps[j]))
    return float(np.mean(dists))


def load_my_fragments(csv_path: str = "unique_fragments.csv"):
    """
    Load fragment SMILES with BRICS-style attachment points from a CSV file.
    Filters for valid molecules containing dummy atoms ("*") and records the
    indices of those attachment points for fragment recombination.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found!")

    df = pd.read_csv(csv_path)
    if "fragment_smiles" not in df.columns:
        raise KeyError(
            f"CSV must contain a 'fragment_smiles' column. Found columns: {list(df.columns)}"
        )

    smis = df["fragment_smiles"].dropna().astype(str).tolist()
    print(f"Found {len(smis)} SMILES in CSV")

    fragments = []
    skipped = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or "*" not in smi:
            skipped += 1
            continue
        stems = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        fragments.append((smi, stems))

    print(f"Successfully loaded {len(fragments)} valid fragments with attachment points")
    if skipped:
        print(f"Skipped {skipped} entries")
    return fragments


def load_resolve_proxy(checkpoint_path: str, device: torch.device, wrap_model=None):
    """
    Load a trained ReSolve proxy MPNN from a checkpoint and prepare it for inference.
    Reconstructs the model with training-matched hyperparameters, loads weights,
    wraps it in a ResolveProxyWrapper, and moves it to the target device.
    """
    wrap_model = wrap_model or (lambda x: x)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Extract state_dict from common checkpoint formats
    if isinstance(ckpt, dict):
        state = (
                ckpt.get("state_dict")
                or ckpt.get("model_state_dict")
                or ckpt.get("model")
                or ckpt
        )
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise ValueError(f"Could not extract a state_dict from checkpoint: {type(state)}")

    # Strip DataParallel "module." prefix if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    # These args must match training
    mpnn = MPNNModel(
        num_layers=7,
        emb_dim=128,

        # MUST match ReSolved training
        magic_number=2,
        magic_number_2=3,
        magic_number_3=1,
        magic_number_4=1,

        num_seed_points=3,
        heads=1,

        dim_atoms=8,
        dim_bond=6,
        emb_dielec=2,
        emb_refract=2,

        out_dim=1,
        num_atom_types=120,
        num_bond_types=5,
    )

    # Load weights (try strict, then fallback for debugging)
    try:
        mpnn.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("[Warn] strict=True failed, retrying strict=False")
        print(e)
        missing, unexpected = mpnn.load_state_dict(state, strict=False)
        if missing:
            print(f"[Warn] Missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[Warn] Unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    proxy = ResolveProxyWrapper(mpnn).to(device)
    proxy.device = device
    proxy = wrap_model(proxy)
    proxy.eval()
    return proxy


def remove_dummy_atoms(smiles: str) -> str | None:
    """
    Remove dummy atoms ([*], [1*], [14*], etc.) from a SMILES string by reconnecting their neighbors directly.
    Returns: Clean SMILES string or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None

    rw = Chem.RWMol(mol)
    dummy_atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]

    for idx in sorted(dummy_atoms, reverse=True):
        atom = rw.GetAtomWithIdx(idx)
        neighbors = atom.GetNeighbors()

        if len(neighbors) == 2:
            a, b = neighbors
            if not rw.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx()):
                # try to reuse bond type if possible
                bonds = atom.GetBonds()
                bond_type = bonds[0].GetBondType() if bonds else Chem.BondType.SINGLE
                rw.AddBond(a.GetIdx(), b.GetIdx(), bond_type)

        rw.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None

    return Chem.MolToSmiles(rw, canonical=True)