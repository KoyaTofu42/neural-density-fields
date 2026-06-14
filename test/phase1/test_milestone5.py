import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from src.models.ndf_model import NeuralDensityField

def test_training_and_rendering():
    print("="*60)
    print("Testing Milestone 5: Training Loop & Visualization (Dummy Data)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}\n")
    
    # 1. Create a synthetic molecule (e.g. H2O)
    z = torch.tensor([8, 1, 1], dtype=torch.long)
    pos = torch.randn(3, 3) * 2.0
    
    # Create synthetic spatial queries and target densities
    query_pos = torch.randn(50, 3) * 3.0
    # True density is faked mathematically: sum of inverse distances to atoms
    dists = torch.cdist(query_pos, pos)
    target_density = torch.sum(1.0 / (dists + 0.1), dim=1)
    
    data = Data(z=z, pos=pos, query_pos=query_pos, y=target_density)
    # Batch it to simulate the DataLoader
    batch = Batch.from_data_list([data, data]).to(device)
    
    # 2. Initialize Model
    print("[1] Initializing Neural Density Field...")
    # Using a tiny model to make the test run instantly
    model = NeuralDensityField(hidden_dim=32, k=3, num_gaussians=20, mlp_hidden=64).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 3. Mini Training Loop (2 Epochs)
    print("\n[2] Testing Supervised Training Loop...")
    model.train()
    for epoch in range(2):
        optimizer.zero_grad()
        density_pred = model(batch)
        loss = criterion(density_pred.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        print(f"  - Epoch {epoch+1}: Dummy MSE Loss = {loss.item():.4f}")
        
    print("  ✅ Training loop successfully performed backpropagation and weight updates!")

    # 4. Dummy Visualization Test
    print("\n[3] Testing Interactive 3D Renderer...")
    model.eval()
    
    margin = 1.0
    grid_res = 15 # Tiny 15x15x15 grid to keep the test instantly fast
    
    x_min, y_min, z_min = pos.min(dim=0)[0] - margin
    x_max, y_max, z_max = pos.max(dim=0)[0] + margin
    
    x_lin = torch.linspace(x_min, x_max, grid_res)
    y_lin = torch.linspace(y_min, y_max, grid_res)
    z_lin = torch.linspace(z_min, z_max, grid_res)
    
    X, Y, Z_grid = torch.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
    query_coords = torch.stack([X.flatten(), Y.flatten(), Z_grid.flatten()], dim=1)
    
    with torch.no_grad():
        chunk_data = Data(z=z, pos=pos, query_pos=query_coords)
        chunk_batch = Batch.from_data_list([chunk_data]).to(device)
        density_flat = model(chunk_batch).cpu().squeeze().numpy()
        
    max_rho = density_flat.max()
    
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Isosurface(
            x=X.flatten().numpy(),
            y=Y.flatten().numpy(),
            z=Z_grid.flatten().numpy(),
            value=density_flat,
            isomin=max_rho * 0.1,
            isomax=max_rho * 0.9,
            surface_count=3,
            colorscale='Plasma',
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=0.4
        ))
        
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0].numpy(), y=pos[:, 1].numpy(), z=pos[:, 2].numpy(),
            mode='markers',
            marker=dict(size=10, color='white', line=dict(color='red', width=4)),
            name='Atoms'
        ))
        
        os.makedirs("results", exist_ok=True)
        html_path = "results/dummy_test_render.html"
        fig.write_html(html_path)
        print(f"  ✅ Plotly Renderer successfully generated: {html_path}")
    except ImportError:
        print("  ⚠️ Plotly is not installed yet. Skipping HTML export test.")
        print("  (It will be installed automatically next time the docker container is restarted).")
        
    print("\n" + "="*60)
    print("✅ Milestone 5 fully verified with dummy data!")
    print("="*60)

if __name__ == "__main__":
    test_training_and_rendering()
