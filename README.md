# PI-NDF: Physics-Informed Neural Density Fields

This project implements a scalable, grid-free continuous model of macromolecular electron density using Implicit Neural Representations (INRs) and Local Relative Coordinates.

## Current Status: Phase 2 (Physics-Informed Training)
Phase 1 (Local Coordinate Framework Formulation) is complete. The current focus is Phase 2: transforming the baseline into a Physics-Informed Neural Network (PINN). 

Key active objectives:
1. **Normalization Constraint:** Enforcing electron conservation via Monte Carlo integration.
2. **Poisson Equation:** Implementing autograd Laplacian to constrain predicted electrostatic potential against charge density.
3. **$\Delta$-Learning:** Pre-training on semi-empirical (GFN2-xTB) data and predicting corrections ($\Delta\rho$) to achieve high-fidelity accuracy with minimal CCSD(T) data.

## Project Structure
- `configs/`: Configuration files
- `data/`: Dataset and dataloaders
- `src/`: Core architecture, including local coordinate processing and PINN loss functions
- `scripts/`: Training and evaluation scripts
- `test/`: Unit and integration tests

## Docker Technical Structure
The project utilizes a containerized GPU environment for dependency and execution consistency:
- **Base Image:** Built on `nvcr.io/nvidia/pytorch:26.02-py3` to leverage NVIDIA-optimized PyTorch.
- **Package Management:** Uses `uv` (via multi-stage build) targeting the system Python environment (`UV_SYSTEM_PYTHON=1`) to manage dependencies quickly while preserving the baked-in PyTorch distribution.
- **Docker Compose:** The `docker-compose.yaml` orchestrates the `app` service. It dynamically mounts the codebase (`.:/app`), allocates all available NVIDIA GPUs, utilizes host IPC (`ipc: host`), and initiates `python train.py` as the default entry point.

See `plan-all.md` and `plan-phase2.md` for detailed roadmaps.
