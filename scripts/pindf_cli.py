import os
import argparse
import torch
import numpy as np
from src.models.ndf_model import NeuralDensityField
from src.data.density_dataset import DensityDataset
from src.data.high_res_sampler import BoundingBoxGenerator, generate_dense_grid
from src.utils.cube_export import write_cube_file
from src.eval.inference_utils import OptimizedInferencer

def main():
    parser = argparse.ArgumentParser(description="PI-NDF High-Resolution Cube Exporter")
    parser.add_argument("--dataset", type=str, default="data/density_samples.h5", help="Path to the HDF5 dataset")
    parser.add_argument("--molecule_idx", type=int, default=0, help="Index of the molecule in the dataset")
    parser.add_argument("--atom1", type=int, required=True, help="First atom index for the bounding box")
    parser.add_argument("--atom2", type=int, required=True, help="Second atom index for the bounding box")
    parser.add_argument("--res", type=int, default=40, help="Resolution of the 3D grid (NxNxN)")
    parser.add_argument("--margin", type=float, default=1.5, help="Margin around the atoms in Angstroms")
    parser.add_argument("--out", type=str, default="results/output.cube", help="Output .cube file path")
    args = parser.parse_args()

    print(f"PI-NDF CLI: Exporting molecule {args.molecule_idx} between atoms {args.atom1} and {args.atom2}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralDensityField(hidden_dim=128, k=5, num_gaussians=300)
    model_path = "models_checkpoints/best_ndf.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    
    inferencer = OptimizedInferencer(model, device)
    
    dataset = DensityDataset(args.dataset, num_samples_per_mol=1)
    data = dataset[args.molecule_idx]
    z = data.z.numpy()
    pos = data.pos.numpy()
    
    bbox_gen = BoundingBoxGenerator(pos, margin=args.margin)
    min_coords, max_coords = bbox_gen.get_bond_bbox(args.atom1, args.atom2)
    
    print(f"Generating {args.res}x{args.res}x{args.res} grid...")
    X, Y, Z, query_coords = generate_dense_grid(min_coords, max_coords, grid_res=args.res)
    
    print("Running optimized inference...")
    density_flat = inferencer.predict_density(data.z, data.pos, query_coords, chunk_size=50000)
    density_grid = density_flat.reshape((args.res, args.res, args.res))
    
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_cube_file(args.out, z, pos, min_coords, max_coords, args.res, density_grid)
    
    print(f"✅ Success! Wrote grid to {args.out}")

if __name__ == "__main__":
    main()
