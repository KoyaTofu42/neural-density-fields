import os
import torch
import numpy as np
import plotly.graph_objects as go
from torch_geometric.data import Data, Batch
from src.models.ndf_model import NeuralDensityField
from src.data.density_dataset import DensityDataset

def render_isosurface():
    print("="*60)
    print("Infinite Resolution 3D Electron Density Renderer")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}\n")
    
    # 1. Load best model
    print("[1] Loading trained Neural Density Field...")
    model = NeuralDensityField(hidden_dim=128, k=5, num_gaussians=50)
    model_path = "models_checkpoints/best_ndf.pt"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("  ✅ Successfully loaded trained weights.")
    else:
        print("  ⚠️ WARNING: Trained model not found! Using untrained random weights for demonstration.")
        
    model.to(device)
    model.eval()
    
    # 2. Get a single molecule from the dataset
    print("\n[2] Fetching test molecule...")
    try:
        # We don't need to sample queries, we just want the atoms
        dataset = DensityDataset("data/density_samples.h5", num_samples_per_mol=1)
        data = dataset[0] # Grab the very first molecule
        z = data.z
        pos = data.pos
        print(f"  - Loaded molecule with {z.shape[0]} atoms.")
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        return
    
    # 3. Create a dense 3D Grid
    print("\n[3] Generating dense 3D continuous query grid...")
    margin = 2.0  # Angstroms of padding around the molecule
    grid_res = 40 # 40x40x40 = 64,000 continuous points in space!
    
    x_min, y_min, z_min = pos.min(dim=0)[0] - margin
    x_max, y_max, z_max = pos.max(dim=0)[0] + margin
    
    x_lin = torch.linspace(x_min, x_max, grid_res)
    y_lin = torch.linspace(y_min, y_max, grid_res)
    z_lin = torch.linspace(z_min, z_max, grid_res)
    
    X, Y, Z_grid = torch.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
    
    # Flatten the grid into a massive list of coordinate points
    query_coords = torch.stack([X.flatten(), Y.flatten(), Z_grid.flatten()], dim=1) # [64000, 3]
    print(f"  - Generated {query_coords.shape[0]:,} spatial query points.")
    
    # 4. Predict Electron Density in Batches
    print("\n[4] Predicting electron density at infinite resolution...")
    # We chunk the queries into batches so we don't blow up the GPU VRAM
    batch_size = 5000 
    density_preds = []
    
    with torch.no_grad():
        for i in range(0, query_coords.shape[0], batch_size):
            q_batch = query_coords[i:i+batch_size]
            
            # Construct a dummy PyG Data object for this chunk
            chunk_data = Data(z=z, pos=pos, query_pos=q_batch)
            chunk_batch = Batch.from_data_list([chunk_data]).to(device)
            
            # Predict
            pred = model(chunk_batch)
            density_preds.append(pred.cpu())
            
    # Reassemble all chunks
    density_flat = torch.cat(density_preds, dim=0).squeeze().numpy()
    
    # Check max density to set dynamic visualization thresholds
    max_rho = density_flat.max()
    print(f"  - Network prediction finished! Max density found: {max_rho:.6f}")
    
    # 5. Render using Plotly
    print("\n[5] Rendering interactive 3D HTML plot...")
    
    # Create Isosurface map
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten().numpy(),
        y=Y.flatten().numpy(),
        z=Z_grid.flatten().numpy(),
        value=density_flat,
        isomin=max_rho * 0.05, # Show density cloud where rho > 5% of max
        isomax=max_rho * 0.8,
        surface_count=5,       # Render 5 nested translucent shells
        colorscale='Plasma',
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.4            # Make it glassy
    ))
    
    # Render the actual atomic nuclei as solid red spheres
    atom_x = pos[:, 0].numpy()
    atom_y = pos[:, 1].numpy()
    atom_z = pos[:, 2].numpy()
    
    fig.add_trace(go.Scatter3d(
        x=atom_x, y=atom_y, z=atom_z,
        mode='markers',
        marker=dict(size=10, color='white', line=dict(color='red', width=4)),
        name='Atomic Nuclei'
    ))
    
    # Save the output
    os.makedirs("results", exist_ok=True)
    html_path = "results/electron_density.html"
    fig.write_html(html_path)
    
    print("\n" + "="*60)
    print(f"✅ Successfully rendered infinite resolution 3D visualization!")
    print(f"File saved to: {html_path}")
    print("Simply double-click this HTML file or drag it into Chrome/Edge to spin the electron cloud!")
    print("="*60)

if __name__ == "__main__":
    render_isosurface()
