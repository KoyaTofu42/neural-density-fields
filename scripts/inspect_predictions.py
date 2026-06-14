import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch_geometric.loader import DataLoader
from src.data.density_dataset import DensityDataset
from src.models.ndf_model import NeuralDensityField
import torch.nn as nn

def inspect():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Dataset...")
    dataset = DensityDataset("data/density_samples.h5", num_samples_per_mol=10000)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, follow_batch=['query_pos'])
    batch = next(iter(loader)).to(device)
    
    print("Loading Model...")
    model = NeuralDensityField(hidden_dim=128, k=5, num_gaussians=300, cutoff=5.0).to(device)
    model.load_state_dict(torch.load("models_checkpoints/best_ndf.pt", map_location=device, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        density_pred = model(batch)
        
    pred = density_pred.squeeze()
    target = batch.y
    
    mse = nn.MSELoss()(pred, target).item()
    
    print(f"--- Analysis ---")
    print(f"MSE: {mse:.6f}")
    
    print(f"\nTarget Stats:")
    print(f"  Min: {target.min().item():.6f}")
    print(f"  Max: {target.max().item():.6f}")
    print(f"  Mean: {target.mean().item():.6f}")
    
    print(f"\nPred Stats:")
    print(f"  Min: {pred.min().item():.6f}")
    print(f"  Max: {pred.max().item():.6f}")
    print(f"  Mean: {pred.mean().item():.6f}")
    
    # Analyze spikes
    spike_mask = target > 1.0
    num_spikes = spike_mask.sum().item()
    if num_spikes > 0:
        spike_pred_mean = pred[spike_mask].mean().item()
        spike_target_mean = target[spike_mask].mean().item()
        print(f"\nCore Spikes (>1.0) Analysis (N={num_spikes}):")
        print(f"  Target Mean: {spike_target_mean:.6f}")
        print(f"  Pred Mean: {spike_pred_mean:.6f}")
        
    # Analyze empty space
    empty_mask = target < 0.01
    num_empty = empty_mask.sum().item()
    if num_empty > 0:
        empty_pred_mean = pred[empty_mask].mean().item()
        empty_target_mean = target[empty_mask].mean().item()
        print(f"\nEmpty Space (<0.01) Analysis (N={num_empty}):")
        print(f"  Target Mean: {empty_target_mean:.6f}")
        print(f"  Pred Mean: {empty_pred_mean:.6f}")

if __name__ == "__main__":
    inspect()
