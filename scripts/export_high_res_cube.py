import os
import torch
from torch_geometric.data import Data, Batch
import numpy as np

# Adjust imports according to the project structure
from src.models.ndf_model import NeuralDensityField
from src.data.density_dataset import DensityDataset
from src.data.high_res_sampler import BoundingBoxGenerator, generate_dense_grid
from src.utils.cube_export import write_cube_file

def main():
    print("="*60)
    print("PI-NDF High-Resolution Cube Exporter")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    # 1. Load Model
    # Using num_gaussians=300 to match the checkpoint.
    model = NeuralDensityField(hidden_dim=128, k=5, num_gaussians=300)
    model_path = "models_checkpoints/best_ndf.pt"
    if os.path.exists(model_path):
        # Use strict=False since the checkpoint might be from Phase 1 (single head)
        # while the model code has moved to Phase 2 (dual head).
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        print("✅ Loaded trained weights (using strict=False to handle dual-head architecture changes).")
    else:
        print("⚠️ WARNING: Trained model not found! Using random weights.")
    model.to(device)
    model.eval()

    # 2. Fetch Molecule
    dataset = DensityDataset("data/density_samples.h5", num_samples_per_mol=1)
    data = dataset[0]
    z = data.z
    pos = data.pos
    print(f"Loaded molecule with {len(z)} atoms.")
    
    # Let's find two atoms to form a bond. For QM9, carbon is 6. Let's find the first two carbons.
    c_indices = torch.where(z == 6)[0]
    if len(c_indices) >= 2:
        idx1, idx2 = c_indices[0].item(), c_indices[1].item()
    else:
        idx1, idx2 = 0, 1 # fallback

    print(f"Targeting region between atom {idx1} (Z={z[idx1]}) and atom {idx2} (Z={z[idx2]}).")

    # 3. Generate Bounding Box and Dense Grid
    bbox_gen = BoundingBoxGenerator(pos, margin=1.5)
    min_coords, max_coords = bbox_gen.get_bond_bbox(idx1, idx2)
    
    # As requested by the user, we set a low resolution to improve speed initially
    grid_res = 40 # 40^3 = 64,000 points
    print(f"Generating dense grid of resolution {grid_res}x{grid_res}x{grid_res} ({grid_res**3} points)...")
    X, Y, Z, query_coords = generate_dense_grid(min_coords, max_coords, grid_res=grid_res)
    print(f"Grid generated.")

    # 4. Run Inference in Batches
    batch_size = 5000
    density_preds = []
    
    print("Running inference...")
    with torch.no_grad():
        for i in range(0, query_coords.shape[0], batch_size):
            q_batch = query_coords[i:i+batch_size]
            chunk_data = Data(z=z, pos=pos, query_pos=q_batch)
            chunk_batch = Batch.from_data_list([chunk_data]).to(device)
            density, potential, _ = model(chunk_batch)
            density_preds.append(density.cpu())
            
    density_flat = torch.cat(density_preds, dim=0).squeeze().numpy()
    density_grid = density_flat.reshape((grid_res, grid_res, grid_res))
    print(f"Inference complete. Max density: {density_grid.max():.6f}")

    # 5. Export to Cube
    os.makedirs("results", exist_ok=True)
    out_file = "results/high_res_bond.cube"
    print(f"Exporting density to {out_file}...")
    write_cube_file(out_file, z, pos, min_coords, max_coords, grid_res, density_grid)
    print("✅ Export complete!")

if __name__ == "__main__":
    main()
