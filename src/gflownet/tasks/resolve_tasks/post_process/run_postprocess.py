import sqlite3
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Batch

# ---- ReSolved proxy imports ----
from ..resolve_proxy_model import MPNNModel, ResolveProxyWrapper
from ..resolve_featurise import mol_to_graph

"""
python -m gflownet.tasks.resolve_tasks.post_process.run_postprocess
"""

# Config

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "generated_objs_0.db"
TABLE_NAME = "results"

SORT_COL = "r"
SMILES_COL_DB = "smiles"
SMILES_COL = "smi"

TOP_K = 100
REMOVE_DUMMY = True
DEDUP = True

# Proxy
CHECKPOINT = "gflownet/tasks/resolve_tasks/best_model.pth"
DIELECTRIC = 78.4
REFRACTIVE = 1.333
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Outputs
OUT_PROCESSED = "gflownet/tasks/resolve_tasks/post_process/results/processed.csv"
OUT_TOP = "gflownet/tasks/resolve_tasks/post_process/results/top100.csv"
OUT_IMG = "gflownet/tasks/resolve_tasks/post_process/results/top100_molecules.png"
OUT_SCAFF_COUNTS = "gflownet/tasks/resolve_tasks/post_process/results/top100_scaffold_counts.csv"
OUT_SCAFF_STATS = "gflownet/tasks/resolve_tasks/post_process/results/top100_scaffold_reward_stats.csv"


# Step 1 — Load Database

def load_database():
    assert DB_PATH.exists(), f"{DB_PATH} not found"
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    print(f"Loaded {len(df)} rows from DB")
    return df


# Step 2 — Clean + Sort

def clean_dataframe(df):

    # Rename SMILES column if needed
    if SMILES_COL_DB in df.columns:
        df = df.rename(columns={SMILES_COL_DB: SMILES_COL})

    if REMOVE_DUMMY:
        df = df[~df[SMILES_COL].str.contains(r"\*", regex=True, na=True)]
        print(f"After removing dummy SMILES: {len(df)}")

    if SORT_COL not in df.columns:
        raise ValueError(f"{SORT_COL} not found")

    df = df.sort_values(SORT_COL, ascending=False)

    if DEDUP:
        before = len(df)
        df = df.drop_duplicates(subset=SMILES_COL, keep="first")
        print(f"Deduplicated: {before} → {len(df)}")

    return df


# Step 3 — Top-K

def select_top_k(df):
    if TOP_K is None:
        return df
    return df.head(TOP_K).reset_index(drop=True)


# Step 4 — Load Proxy

def load_proxy():
    mpnn = MPNNModel(
        num_layers=7,
        emb_dim=128,
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

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    mpnn.load_state_dict(state)

    proxy = ResolveProxyWrapper(mpnn).to(DEVICE)
    proxy.eval()
    proxy.device = DEVICE
    return proxy


# Step 5 — Score With Proxy

def score_with_proxy(df, proxy):

    scores = []

    for smi in tqdm(df[SMILES_COL], desc="Scoring"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores.append(None)
            continue

        graph = mol_to_graph(mol)
        if graph is None:
            scores.append(None)
            continue

        batch = Batch.from_data_list([graph]).to(proxy.device)

        dielec = torch.full((1,), DIELECTRIC, device=proxy.device)
        ref = torch.full((1,), REFRACTIVE, device=proxy.device)

        with torch.no_grad():
            pred = proxy(batch, dielec, ref)

        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        scores.append(float(pred[:, 0].item()))

    df["resolve_score"] = scores
    return df


# Step 6 — Scaffold Analysis

def compute_scaffolds(df):

    def get_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold)

    df["scaffold"] = df[SMILES_COL].apply(get_scaffold)
    df = df.dropna(subset=["scaffold"])

    scaffold_counts = (
        df.groupby("scaffold")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    scaffold_stats = (
        df.groupby("scaffold")
        .agg(
            n_molecules=(SORT_COL, "count"),
            mean_reward=(SORT_COL, "mean"),
            max_reward=(SORT_COL, "max"),
            min_reward=(SORT_COL, "min"),
            std_reward=(SORT_COL, "std"),
        )
        .reset_index()
        .sort_values("mean_reward", ascending=False)
    )

    scaffold_counts.to_csv(OUT_SCAFF_COUNTS, index=False)
    scaffold_stats.to_csv(OUT_SCAFF_STATS, index=False)

    print("Unique scaffolds:", df["scaffold"].nunique())

    return df


# Step 7 — Draw Grid

def draw_grid(df):

    mols = []
    legends = []

    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[SMILES_COL])
        if mol is None:
            continue

        mols.append(mol)
        legends.append(f"#{i} r={row[SORT_COL]:.2f}")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(300, 300),
        legends=legends,
    )

    img.save(OUT_IMG)
    print("Saved molecule grid.")


# Main

def main():

    df = load_database()
    df = clean_dataframe(df)

    df.to_csv(OUT_PROCESSED, index=False)
    print("Saved processed.csv")

    df_top = select_top_k(df)
    df_top.to_csv(OUT_TOP, index=False)
    print("Saved top100.csv")

    proxy = load_proxy()
    df_top = score_with_proxy(df_top, proxy)
    df_top.to_csv(OUT_TOP, index=False)

    df_top = compute_scaffolds(df_top)
    draw_grid(df_top)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
