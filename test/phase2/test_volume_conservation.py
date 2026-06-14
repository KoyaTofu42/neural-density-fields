import torch
from src.models.ndf_model import NeuralDensityField
from src.physics.monte_carlo import UniformMonteCarloIntegrator
from src.physics.losses import NormalizationLoss
from torch_geometric.data import Data

def run_volume_conservation_step():
    print("="*60)
    print("Task 4.1: Volume Conservation - Single Step Verification")
    print("="*60)
    
    # 1. Initialize components
    model = NeuralDensityField(hidden_dim=32, k=3)
    mc_integrator = UniformMonteCarloIntegrator(margin=5.0, num_samples=5000)
    norm_loss_fn = NormalizationLoss()
    
    # 2. Mock Molecule (Carbon atom)
    z = torch.tensor([6], dtype=torch.long)
    pos = torch.zeros(1, 3)
    batch = torch.zeros(1, dtype=torch.long)
    
    # 3. Sample MC points
    query_pos, query_batch, mc_volumes = mc_integrator.sample_points(pos, batch)
    data = Data(z=z, pos=pos, batch=batch, query_pos=query_pos, query_batch=query_batch)
    
    # 4. Forward Pass
    density, _, _ = model(data)
    
    # 5. Integration Step
    N_elec_hat = mc_integrator.integrate(density, query_batch, mc_volumes)
    
    # 6. Loss Step
    loss = norm_loss_fn(N_elec_hat, z, batch)
    
    # 7. Backward Step
    loss.backward()
    
    # Verification
    print(f"Predicted Total Electrons: {N_elec_hat.item():.4f}")
    print(f"Target Total Electrons:    6.0000")
    print(f"Normalization MSE Loss:    {loss.item():.4f}")
    
    has_gradients = any(p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad)) 
                        for p in model.parameters())
    
    assert has_gradients, "Gradients failed to flow backwards through the MC Integrator!"
    print("-> Success! Forward pass, integration, and backward gradient flow verified.")

if __name__ == "__main__":
    run_volume_conservation_step()
