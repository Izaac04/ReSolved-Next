# ReSolvedNext

Fragment-Based Molecular Generation with GFlowNets and Solvent-Aware Property Optimisation

---

## Overview

ReSolved-Next is a research framework integrating Generative Flow Networks (GFlowNets) with a solvent-conditioned graph neural network proxy for redox potential prediction.

The system generates novel molecules using fragment-based construction and samples them in proportion to a multi-objective reward combining:

- Redox property optimisation
- Synthesizability constraints
- Structural diversity
- Explicit solvent conditioning

---

## Relationship to ReSolved

ReSolved-Next builds upon **ReSolved**, an open-source solvent-aware graph neural network framework developed by Rostislav Fedorov.

ReSolved is designed to predict redox potentials of organic molecules using a message passing neural network (MPNN) architecture. A key feature of ReSolved is its explicit solvent conditioning: molecular graphs are augmented with solvent descriptors (e.g., dielectric constant ε and refractive index n), enabling the model to learn solvent-dependent electrochemical behaviour.

While ReSolved focuses on property prediction for existing molecules, ReSolved-Next extends this framework into the generative setting. Specifically, ReSolved-Next integrates the ReSolved proxy model into a Generative Flow Network (GFlowNet), allowing the direct generation of novel molecules sampled in proportion to their predicted solvent-conditioned redox properties.

In summary:

- **ReSolved** → Predicts solvent-dependent redox potentials.
- **ReSolved-Next** → Generates new molecules optimised for those properties under solvent conditioning.

This extension transforms a predictive model into a generative optimisation framework capable of exploring chemically meaningful design space under explicit solvent constraints.


## Authorship & Code Contributions

This project builds upon the upstream GFlowNet library (Recursion Pharma trunk), which provides the core generative framework and training infrastructure.

All code within the following directories represents original development work for the ReSolvedNext project:

- `gflownet/tasks/resolve_tasks/`
- `gflownet/tasks/synth_gnn/`

These components were implemented to integrate solvent-aware redox prediction, synthesizability estimation, and fragment-based molecular construction into the GFlowNet framework.

Key contributions include:

- Custom fragment-based task (`MyFragmentsResolveTask`)
- Solvent-conditioned property proxy integration (ReSolve GNN)
- Synthesizability proxy task implementation
- BRICS-compatible fragment handling and attachment logic
- Reward shaping mechanisms (Gaussian and target-based modes)
- Training configuration extensions
- Post-processing, deduplication, and evaluation pipeline
- Dependency resolution and compatibility fixes across scientific libraries

In particular, non-trivial modifications were required to ensure compatibility between BRICS-derived fragments and the fragment-building environment, as well as to resolve version and architecture-specific dependency issues.

## 🚀 Setup Instructions

Follow the steps below to set up **ReSolved-Next** locally.

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ReSolved-Next.git
cd ReSolved-Next
```

---

### 2️⃣ Create and Activate a Virtual Environment

It is recommended to use a dedicated virtual environment to avoid dependency conflicts.

```bash
# Create virtual environment
python3 -m venv resolvednext_venv

# Activate it (macOS / Linux)
source resolvednext_venv/bin/activate

# On Windows:
# resolvednext_venv\Scripts\activate
```

---

### 3️⃣ Install PyTorch

Install the exact PyTorch version used in this project:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

> ⚠️ If you are using CUDA, install the appropriate CUDA-enabled version from:  
> https://pytorch.org/get-started/locally/

---

### 4️⃣ Install PyTorch Geometric

Install the compatible PyTorch Geometric dependencies:

```bash
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.3.1+cpu.html
```

> If using CUDA, replace `+cpu` with the correct CUDA build.

---

### 5️⃣ Install Remaining Dependencies

Finally, install all remaining project requirements:

```bash
pip install -r requirements.txt
```

## Example Training Command

Run from 'src'.
```bash
python -m gflownet.tasks.resolve_tasks.run_my_fragments_gfn \
  --fragments-csv gflownet/tasks/resolve_tasks/unique_fragments.csv \
  --checkpoint gflownet/tasks/resolve_tasks/best_model.pth \
  --dielectric 78.4 \
  --refractive 1.333 \
  --target-value 3.8 \
  --steps 50000
```
---

## Training Outputs & Results

All training outputs are saved under:

```
src/logs/
```

Each training run creates a timestamped directory of the form:

```
resolve_my_fragments_YYYYMMDD_HHMMSS
```

For example:

```
src/logs/resolve_my_fragments_20260212_171022/
```

Inside this directory you will find:

```
config.yaml                         # Full training configuration
model_state.pt                      # Saved model checkpoint
train.log                           # Training log output
events.out.tfevents.*               # TensorBoard logs
train/                              # Generated molecules from training rollouts
valid/                              # Generated molecules from validation rollouts
```

---

### Generated Molecules

Both the `train/` and `valid/` directories contain generated molecules stored as SQLite databases:

```
src/logs/<run_name>/train/generated_objs_0.db
src/logs/<run_name>/valid/generated_objs_0.db
```

These databases contain sampled molecules along with their associated rewards and predicted properties.

- `train/` → Molecules generated during training
- `valid/` → Molecules generated during validation

---

## Post-Processing & Analysis

To analyse generated molecules:

1. Copy the desired database file:

```
generated_objs_0.db
```

from either:

```
src/logs/<run_name>/train/
```

or

```
src/logs/<run_name>/valid/
```

into:

```
gflownet/tasks/resolve_tasks/post_process/
```

2. From the `src/` directory, run:

```bash
python -m gflownet.tasks.resolve_tasks.post_process.run_postprocess
```

The post-processing script will:

- Load generated molecules
- Remove invalid or dummy-containing structures
- Deduplicate by SMILES
- Rank molecules by reward
- Export processed results for further analysis






