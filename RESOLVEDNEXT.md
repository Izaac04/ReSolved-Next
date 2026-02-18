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





