import torch
from src.models.ndf_model import NeuralDensityField
from torch_geometric.data import Data

def test_delta_output_structure():
    """
    Ensure the decoder predicts a signed delta, and the main model 
    applies the ReLU(rho_low + delta) operation successfully.
    """
    model = NeuralDensityField(hidden_dim=32, k=3, num_gaussians=10)
    
    # Create dummy data
    N, Q = 5, 20
    data = Data(
        z=torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
        pos=torch.rand(N, 3),
        query_pos=torch.rand(Q, 3),
        y=torch.rand(Q), # Not used in forward directly
        rho_low=torch.rand(Q) # Low-fidelity baseline
    )
    
    # Run forward pass
    density, potential, query_pos = model(data)
    
    # Assert density is strictly non-negative (thanks to ReLU)
    assert torch.all(density >= 0.0), "Final density should be physically non-negative"
    assert density.shape == (Q, 1), f"Density shape incorrect, got {density.shape}"
    assert potential.shape == (Q, 1), "Potential shape incorrect"
    
    # Check that gradients flow to rho_low 
    # (Verifying the computational graph correctly connects the baseline to the output)
    data.rho_low.requires_grad_(True)
    density2, _, _ = model(data)
    loss = density2.sum()
    loss.backward()
    
    assert data.rho_low.grad is not None, "Gradients must flow back to rho_low input"
    
    print("[SUCCESS] Delta-Learning architecture test passed.")

if __name__ == "__main__":
    test_delta_output_structure()
