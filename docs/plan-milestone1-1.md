# Milestone 1: Data Generation & Pipeline Setup

This epic focuses on building the foundational data generation and ingestion pipeline for the SE(3)-invariant Neural Density Field baseline. Before building the MPNN or the continuous query engine, we need high-quality spatial density data. We will use PySCF to compute ground-truth density on 1,000 QM9 molecules, create an intelligent point sampling engine, and build the PyTorch dataloaders.

## User Review Required

> [!IMPORTANT]
> This phase establishes the data structures that the entire downstream architecture relies on. Please review the proposed components below and provide feedback, particularly on the offline vs. online sampling strategy.

## Open Questions

> [!WARNING]
> Please clarify these details so we can configure the generation scripts correctly:
> 1. **Molecule Selection**: Should we randomly select 1,000 molecules from QM9, or select based on specific criteria (e.g., max atom count, specific elements like CHON)?
> 2. **Sampling Density**: How many continuous spatial coordinates should we sample per molecule? (e.g., 100,000 points offline, to be batched dynamically online?)
> 3. **Data Format**: I propose using `HDF5` or PyG's native `.pt` format for storing the massive point cloud data. Does `HDF5` work for you?

## Proposed Changes

---

### Environment & Dependencies

We will use Docker with Python 3.12 and `uv` to manage the environment and dependencies. This ensures a reproducible, isolated execution environment for PySCF and PyTorch.

#### [NEW] `Dockerfile`
- Base image: `nvcr.io/nvidia/pytorch:26.02-py3` (NVIDIA's official PyTorch image, providing highly optimized GPU support out of the box).
- Install `uv` using a multi-stage build: `COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/`.
- Optimize `uv` package handling by utilizing Docker cache mounts (`--mount=type=cache,target=/root/.cache/uv`) and installing dependencies into the system environment (`uv pip install --system`).
- Note: The container must be run with `--gpus all` (NVIDIA Container Toolkit) to allow access to the host GPU.

#### [NEW] `pyproject.toml`
Dependencies to be managed by `uv`:
- `torch`, `torch-geometric`
- `pyscf` (for DFT calculations)
- `h5py` (for point coordinate storage)
- `numpy`, `scipy`

---

### Data Generation Pipeline

Scripts to calculate the ground-truth quantum chemistry data and sample the continuous electron density space.

#### [NEW] `src/data_gen/qm9_subset.py`
- Downloads the QM9 dataset using PyTorch Geometric (`torch_geometric.datasets.QM9`).
- Filters and subsets 1,000 molecules.
- Exports molecular structures (elements and XYZ coordinates) into a format ready for PySCF.

#### [NEW] `src/data_gen/dft_runner.py`
- Wraps `pyscf.gto.Mole` and `pyscf.dft.RKS`.
- Configures B3LYP functional with the def2-SVP basis set (`xc='B3LYP'`, `basis='def2-svp'`).
- Runs SCF (Self-Consistent Field) convergence.
- Saves the resulting converged density matrix and molecular objects to disk so DFT only needs to be run once.

#### [NEW] `src/data_gen/point_sampler.py`
- **Sampling Engine**: Takes the converged PySCF objects and computes the exact electron density at arbitrary points using `pyscf.dft.numint.eval_rho`.
- **Sampling Strategy**:
  - *Core Region*: Isotropic Gaussian distributions centered on each nucleus to capture high-frequency density changes.
  - *Valence/Tail Region*: Uniform bounding box sampling around the molecule to teach the network about empty space.
- Saves outputs (coordinates $X, Y, Z$ and density $\rho$) efficiently into HDF5 or chunked binary files.

---

### Dataloaders

#### [NEW] `src/dataset/density_dataset.py`
- PyTorch Geometric `Dataset` implementation.
- Loads the base molecular graph (atomic numbers `data.z`, atom positions `data.pos`, and bond connectivity `data.edge_index`).
- Loads the sampled spatial points and their target densities.
- **Batching Strategy**: A custom `collate_fn` or `DataLoader` that yields tuples of `(MoleculeGraph, QueryCoords)` and `TargetDensity`. Allows dynamic subsetting of points per epoch if we have generated millions of offline points.

---

## Verification Plan

### Automated Tests
- Unit test `dft_runner.py` with a simple molecule (e.g., $H_2O$) to ensure PySCF converges and returns a valid density matrix.
- Unit test `point_sampler.py` to ensure it successfully calls `eval_rho` and returns non-negative density values ($\rho \ge 0$).

### Manual Verification
- Generate a 3D scatter plot of sampled coordinates for a single molecule, color-coded by the $\rho$ value, to visually confirm that the engine is densely sampling near nuclei and that the PySCF density looks physically accurate.
- Iterate the `DensityDataset` in a dummy loop to confirm that tensor shapes are correct (e.g., `query_pos: (Batch, 3)` and `target_rho: (Batch, 1)`) and VRAM isn't spiking from data loading overhead.
