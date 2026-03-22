# ReSolved-Next

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

## Setup Instructions

Follow the steps below to set up **ReSolved-Next** locally.

---

### 1. Clone the Repository

```bash
git clone https://github.com/Izaac04/ReSolved-Next.git
cd ReSolved-Next
```

---

### 2. Create and Activate a Virtual Environment (Python 3.10 REQUIRED)

This project is **not compatible with Python 3.12**. Use Python 3.10.

```bash
# Create virtual environment (must be Python 3.10)
python3.10 -m venv resolvednext_venv

# Activate it (macOS / Linux)
source resolvednext_venv/bin/activate

# On Windows:
# resolvednext_venv\Scripts\activate
```

**Verify:**

```bash
python --version
```

**Expected output:**
`Python 3.10.x`

---

### 3. Install Build Dependencies

These fix known issues with PyTorch extension builds.

```bash
pip install --upgrade pip
pip install setuptools==69.5.1 wheel packaging
```

---

### 4. Install PyTorch (Required Version)

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```

---

### 5. Install PyTorch Geometric

Install the required PyTorch Geometric dependencies manually:

```bash
pip install --no-cache-dir --no-build-isolation \
  torch_scatter==2.1.2 \
  torch_sparse==0.6.18 \
  torch_cluster==1.6.3 \
  -f [https://data.pyg.org/whl/torch-2.1.2+cpu.html](https://data.pyg.org/whl/torch-2.1.2+cpu.html)

pip install torch_geometric==2.4.0
```

> If using CUDA, replace `+cpu` with the appropriate CUDA build.

---

### 6. Install GFlowNets

```bash
pip install git+https://github.com/Izaac04/GFlowNets.git@main
```

---

### 7. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

### Notes
* **Python 3.10** is required.
* Python 3.12 is **not supported**.
* PyTorch and PyTorch Geometric versions must match exactly.
* PyG dependencies are installed separately due to native build requirements.
## Example Training Command

Run from 'src'.
```bash
python -m ReSolvedNextTasks.run_my_fragments_gfn \
  --dielectric 78.4 \
  --refractive 1.333 \
  --target-value 3.8 \
  --best-metric sampled_reward_avg \
  --best-metric-mode max \
  --steps 50000
```
---

## Training Outputs & Results

All training outputs are saved under:

```
ReSolvedNextTasks/logs/
```

Each training run creates a timestamped directory of the form:

```
resolve_my_fragments_YYYYMMDD_HHMMSS
```

For example:

```
ReSolvedNextTasks/logs/resolve_my_fragments_20260212_171022/
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
ReSolvedNextTasks/logs/<run_name>/train/generated_objs_0.db
ReSolvedNextTasks/logs/<run_name>/valid/generated_objs_0.db
```

These databases contain sampled molecules along with their associated rewards and predicted properties.

- `train/` → Molecules generated during training
- `valid/` → Molecules generated during validation

---

## Post-Processing & Analysis

To analyse generated molecules:

1. Create a results directory inside postProcess.

2. Copy the desired database file:

```
generated_objs_0.db
```

from either:

```
ReSolvedNextTasks/logs/<run_name>/train/
```

or

```
ReSolvedNextTasks/logs/<run_name>/valid/
```

into:

```
ReSolvedNextTasks/postProcess/
```

3. From the `src/` directory, run:

```bash
python -m ReSolvedNextTasks.postProcess.run_postprocess
```

The post-processing script will:

- Load generated molecules
- Remove invalid or dummy-containing structures
- Deduplicate by SMILES
- Rank molecules by reward
- Export processed results for further analysis










