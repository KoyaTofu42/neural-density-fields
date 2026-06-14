import torch
import math
from src.physics.losses import PoissonLoss

def test_laplacian():
    """
    Test the Laplacian computation using autograd on a known analytical function.
    V(x, y, z) = x^2 + y^2 + z^2
    Laplacian(V) = d2V/dx2 + d2V/dy2 + d2V/dz2 = 2 + 2 + 2 = 6
    """
    loss_fn = PoissonLoss()
    
    # 1. Create spatial query points
    # Needs requires_grad=True so Autograd can build the computational graph
    query_pos = torch.tensor([
        [1.0, 2.0, 3.0],
        [-1.0, 0.0, 5.0]
    ], dtype=torch.float, requires_grad=True)
    
    # 2. Analytical potential function
    potential = (query_pos ** 2).sum(dim=1, keepdim=True) # [Q, 1]
    
    # 3. Dummy density
    density = torch.ones_like(potential) # [Q, 1]
    
    # The PoissonLoss computes:
    # laplacian_V = 6.0
    # target = -4 * pi * density = -4 * pi * 1
    # loss = MSE(laplacian_V, target) = (6 - (-4*pi))^2
    
    loss = loss_fn(potential, density, query_pos)
    
    expected_laplacian = 6.0
    expected_target = -4.0 * math.pi
    expected_loss = (expected_laplacian - expected_target) ** 2
    
    assert torch.allclose(loss, torch.tensor(expected_loss, dtype=torch.float)), f"Expected {expected_loss}, got {loss.item()}"
    print("[SUCCESS] test_laplacian analytically accurate.")
    
def test_laplacian_gradient_flow():
    """
    Test that the gradient flows back through the double derivative
    and into the network's weights.
    """
    loss_fn = PoissonLoss()
    
    query_pos = torch.rand((5, 3), dtype=torch.float, requires_grad=True)
    
    # Dummy network parameter
    w1 = torch.nn.Parameter(torch.rand(3, 1))
    
    # potential = sum(query_pos^2 * w1)
    potential = torch.matmul(query_pos ** 2, w1)
    
    density = torch.rand((5, 1))
    
    loss = loss_fn(potential, density, query_pos)
    loss.backward()
    
    assert w1.grad is not None, "Gradients should flow back to network parameters"
    assert not torch.allclose(w1.grad, torch.zeros_like(w1)), "Gradients should not be zero"
    print("[SUCCESS] test_laplacian_gradient_flow successfully propagated.")

if __name__ == "__main__":
    test_laplacian()
    test_laplacian_gradient_flow()
