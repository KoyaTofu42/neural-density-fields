import torch
from src.models.ndf_model import NeuralDensityField
from torch_geometric.data import Data

def run_gradient_ablation():
    print("="*60)
    print("Task 4.2: Distance-Weighted Query Ablation")
    print("="*60)
    
    model = NeuralDensityField(hidden_dim=32, k=1)
    model.eval()
    
    z = torch.tensor([6], dtype=torch.long)
    pos = torch.zeros(1, 3)
    
    # We test the gradient flowing back to the MPNN (chemical representation) 
    # to isolate how much "attention" the model pays to the atom's chemistry at different distances.
    mpnn_param = next(model.encoder.parameters())
    
    # 1. Query the Core
    query_core = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float)
    data_core = Data(z=z, pos=pos, query_pos=query_core)
    _, potential_core, _ = model(data_core)
    
    grad_core = torch.autograd.grad(potential_core, mpnn_param, retain_graph=True)[0]
    grad_norm_core = torch.norm(grad_core).item()
    
    # 2. Query the Vacuum
    query_vacuum = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float)
    data_vacuum = Data(z=z, pos=pos, query_pos=query_vacuum)
    _, potential_vacuum, _ = model(data_vacuum)
    
    grad_vacuum = torch.autograd.grad(potential_vacuum, mpnn_param, retain_graph=True)[0]
    grad_norm_vacuum = torch.norm(grad_vacuum).item()
    
    print(f"Chemical Gradient Magnitude at r=0.5A (Core):   {grad_norm_core:.6f}")
    print(f"Chemical Gradient Magnitude at r=3.0A (Vacuum): {grad_norm_vacuum:.6f}")
    
    ratio = grad_norm_core / (grad_norm_vacuum + 1e-8)
    print(f"Ratio (Core/Vacuum): {ratio:.2f}x stronger")
    
    # Verify that the exponential distance-weighting trick amplified the core gradients
    assert grad_norm_core > grad_norm_vacuum * 5.0, "Exponential distance-weighting failed to amplify core gradients!"
    
    print("-> Success! Gradients are correctly amplified near nuclei.")

if __name__ == "__main__":
    run_gradient_ablation()
