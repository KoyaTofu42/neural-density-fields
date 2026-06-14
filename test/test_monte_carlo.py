import torch
import torch.nn as nn
from src.physics.monte_carlo import UniformMonteCarloIntegrator
from src.physics.losses import NormalizationLoss

def test_monte_carlo_volume():
    """
    Test that the Monte Carlo integrator correctly calculates volumes 
    and perfectly recovers the volume when integrating a uniform density of 1.
    """
    # Create an integrator with margin=0 for predictable volume calculation
    integrator = UniformMonteCarloIntegrator(num_samples=2000, margin=0.0)
    
    # 2 molecules
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],  # Mol 0 is a 1x1x1 box -> Volume = 1.0
        [5.0, 5.0, 5.0],
        [7.0, 7.0, 7.0]   # Mol 1 is a 2x2x2 box -> Volume = 8.0
    ], dtype=torch.float)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    
    query_pos, query_batch, volumes = integrator.sample_points(pos, batch)
    
    assert torch.allclose(volumes, torch.tensor([1.0, 8.0])), f"Expected [1.0, 8.0], got {volumes}"
    
    # If density is perfectly 1.0 everywhere, the integral should exactly equal the volume
    density = torch.ones(query_pos.size(0), 1)
    N_elec_hat = integrator.integrate(density, query_batch, volumes)
    
    assert torch.allclose(N_elec_hat, torch.tensor([1.0, 8.0])), f"Expected [1.0, 8.0], got {N_elec_hat}"
    
    print("[SUCCESS] test_monte_carlo_volume passed.")

def test_normalization_loss():
    """
    Test that the loss flows gradients correctly back through the volume/box 
    calculations and the model density predictions.
    """
    integrator = UniformMonteCarloIntegrator(num_samples=100, margin=1.0)
    loss_fn = NormalizationLoss()
    
    # Pos needs gradients to test differentiability of box volume logic
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=torch.float, requires_grad=True)
    
    z = torch.tensor([1, 1], dtype=torch.long) # True electrons = 2 (H2 molecule)
    batch = torch.tensor([0, 0], dtype=torch.long)
    
    query_pos, query_batch, volumes = integrator.sample_points(pos, batch)
    
    # Dummy parameter acting like a network's output weight
    dummy_param = nn.Parameter(torch.tensor([0.5]))
    density = dummy_param * torch.ones(query_pos.size(0), 1)
    
    # Integrate and compute loss
    N_elec_hat = integrator.integrate(density, query_batch, volumes)
    loss = loss_fn(N_elec_hat, z, batch)
    
    # Backward pass
    loss.backward()
    
    assert dummy_param.grad is not None, "Gradients should flow to network parameters (dummy_param)"
    assert dummy_param.grad.item() != 0.0, "Gradient value should not be zero"
    assert pos.grad is not None, "Gradients should flow back to atomic positions"
    
    print("[SUCCESS] test_normalization_loss passed.")

if __name__ == "__main__":
    test_monte_carlo_volume()
    test_normalization_loss()
