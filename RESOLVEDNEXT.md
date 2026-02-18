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

```bash
python -m gflownet.tasks.resolve_tasks.run_my_fragments_gfn \
  --fragments-csv gflownet/tasks/resolve_tasks/unique_fragments.csv \
  --checkpoint gflownet/tasks/resolve_tasks/best_model.pth \
  --reward-mode gaussian \
  --dielectric 78.4 \
  --refractive 1.333 \
  --steps 50000


