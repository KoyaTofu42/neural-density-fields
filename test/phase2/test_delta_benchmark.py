import torch
import torch.nn as nn
from src.models.ndf_model import NeuralDensityField
from torch_geometric.data import Data

def run_delta_learning_step():
    print("="*60)
    print("Task 4.3: Delta-Learning Single Step Verification")
    print("="*60)
    
    # 1. Generate Dummy Target
    Q = 100
    pos = torch.zeros(1, 3)
    z = torch.tensor([6], dtype=torch.long)
    query_pos = torch.rand(Q, 3) * 5.0
    
    r = torch.norm(query_pos, dim=1)
    target_density = torch.exp(-r) + 0.1 * torch.sin(5*r)
    rho_low = torch.exp(-r)
    
    model_delta = NeuralDensityField(hidden_dim=32, k=1)
    loss_fn = nn.MSELoss()
    
    # 2. Forward Pass with Baseline
    data_delta = Data(z=z, pos=pos, query_pos=query_pos, y=target_density, rho_low=rho_low)
    
    density, _, _ = model_delta(data_delta)
    loss = loss_fn(density.squeeze(), target_density)
    
    # 3. Backward Pass
    loss.backward()
    
    has_gradients = any(p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad)) 
                        for p in model_delta.parameters())
    
    print(f"Baseline (rho_low) max: {rho_low.max().item():.4f}")
    print(f"Target max:             {target_density.max().item():.4f}")
    print(f"Predicted density max:  {density.max().item():.4f}")
    print(f"MSE Loss Step 0:        {loss.item():.4f}")
    
    assert has_gradients, "Gradients failed to flow!"
    print("-> Success! Delta-learning mathematical forward/backward sequence verified.")

if __name__ == "__main__":
    run_delta_learning_step()
