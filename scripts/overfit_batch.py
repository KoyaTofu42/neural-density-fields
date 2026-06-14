import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from src.data.density_dataset import DensityDataset
from src.models.ndf_model import NeuralDensityField

def test_overfit():
    print("="*60)
    print("Test: Overfit a Single Batch")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load just 1 batch from the dataset
    dataset = DensityDataset("data/density_samples.h5", num_samples_per_mol=512)
    # Get first 4 molecules
    subset = torch.utils.data.Subset(dataset, range(4))
    loader = DataLoader(subset, batch_size=4, shuffle=False)
    
    batch = next(iter(loader)).to(device)
    
    # 2. Initialize Model
    model = NeuralDensityField(hidden_dim=128, k=5, num_gaussians=300, cutoff=5.0).to(device)
    
    # 3. Optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"Target Max: {batch.y.max().item():.2f}, Mean: {batch.y.mean().item():.2f}")
    
    # 4. Train for 500 steps
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        density_pred = model(batch)
        
        # Scale for stable gradients
        pred_scaled = density_pred.squeeze() / 100.0
        target_scaled = batch.y / 100.0
        
        loss = criterion(pred_scaled, target_scaled)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        if epoch % 50 == 0:
            # Calculate true unscaled MSE
            true_mse = criterion(density_pred.squeeze(), batch.y).item()
            print(f"Step {epoch}: True MSE = {true_mse:.6f}")

if __name__ == "__main__":
    test_overfit()
