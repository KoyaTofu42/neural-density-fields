import os
import torch
import numpy as np
from torch_geometric.data import Data, Batch

from src.models.ndf_model import NeuralDensityField
from src.data.density_dataset import DensityDataset
from src.data.high_res_sampler import BoundingBoxGenerator, generate_dense_grid
from src.utils.cube_export import write_cube_file
from src.eval.ccsd_benchmark import generate_ccsd_density
from src.eval.metrics import calculate_metrics
from src.eval.nci_analysis import calculate_rdg
from pyscf.lib.parameters import BOHR

def main():
    print("="*60)
    print("PHASE 3: Gold Standard Validation (PI-NDF vs CCSD)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = NeuralDensityField(hidden_dim=128, k=5, num_gaussians=300)
    model_path = "models_checkpoints/best_ndf.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        print("✅ Loaded trained PI-NDF weights.")
    model.to(device)
    model.eval()

    dataset = DensityDataset("data/density_samples.h5", num_samples_per_mol=1)
    data = dataset[0]
    z = data.z.numpy()
    pos = data.pos.numpy()
    
    # Select target bond
    c_indices = np.where(z == 6)[0]
    idx1, idx2 = (c_indices[0], c_indices[1]) if len(c_indices) >= 2 else (0, 1)
    print(f"Targeting region between atom {idx1} and {idx2}.")

    # Generate Grid
    bbox_gen = BoundingBoxGenerator(pos, margin=1.5)
    min_coords, max_coords = bbox_gen.get_bond_bbox(idx1, idx2)
    grid_res = 30 # Use 30^3 for faster CCSD(T) calculation
    X, Y, Z, query_coords = generate_dense_grid(min_coords, max_coords, grid_res=grid_res)
    query_coords_np = query_coords.numpy()
    
    # 1. PI-NDF Prediction
    print("1. Running PI-NDF Neural Inference...")
    batch_size = 5000
    preds = []
    with torch.no_grad():
        for i in range(0, query_coords.shape[0], batch_size):
            q_batch = query_coords[i:i+batch_size]
            b = Batch.from_data_list([Data(z=torch.from_numpy(z), pos=torch.from_numpy(pos), query_pos=q_batch)]).to(device)
            density, _, _ = model(b)
            preds.append(density.cpu())
    density_pindf = torch.cat(preds, dim=0).squeeze().numpy()

    # 2. CCSD(T) Ground Truth
    print("2. Running PySCF CCSD Ground Truth calculation (this may take a minute)...")
    density_ccsd = generate_ccsd_density(z, pos, query_coords_np)

    # 3. Metrics
    print("\n--- Validation Metrics ---")
    metrics = calculate_metrics(density_pindf, density_ccsd)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # 4. NCI / RDG Analysis
    print("\n3. Performing NCI (Reduced Density Gradient) Analysis...")
    density_pindf_grid = density_pindf.reshape(grid_res, grid_res, grid_res)
    density_ccsd_grid = density_ccsd.reshape(grid_res, grid_res, grid_res)
    
    # Calculate step size in Bohr
    step_size_angstrom = (max_coords[0] - min_coords[0]) / (grid_res - 1)
    step_size_bohr = step_size_angstrom / BOHR
    
    rdg_pindf = calculate_rdg(density_pindf_grid, step_size_bohr)
    rdg_ccsd = calculate_rdg(density_ccsd_grid, step_size_bohr)

    # Export
    os.makedirs("results", exist_ok=True)
    write_cube_file("results/ccsd_ground_truth.cube", z, pos, min_coords, max_coords, grid_res, density_ccsd_grid)
    write_cube_file("results/pindf_prediction.cube", z, pos, min_coords, max_coords, grid_res, density_pindf_grid)
    write_cube_file("results/rdg_pindf.cube", z, pos, min_coords, max_coords, grid_res, rdg_pindf)
    print("\n✅ Successfully exported .cube files for side-by-side PyMOL visualization!")

if __name__ == "__main__":
    main()
