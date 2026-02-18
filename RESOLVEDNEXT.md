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

## Example Training Command

```bash
python -m gflownet.tasks.resolve_tasks.run_my_fragments_gfn \
  --fragments-csv gflownet/tasks/resolve_tasks/unique_fragments.csv \
  --checkpoint gflownet/tasks/resolve_tasks/best_model.pth \
  --reward-mode gaussian \
  --dielectric 78.4 \
  --refractive 1.333 \
  --steps 50000

