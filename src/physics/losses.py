import torch
import torch.nn as nn
import math
from torch_geometric.utils import scatter

class NormalizationLoss(nn.Module):
    def __init__(self):
        """
        Physics-Informed Loss that penalizes the model when the integrated 
        electron density does not equal the exact number of electrons in the system.
        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, N_elec_hat, z, batch=None):
        """
        Args:
            N_elec_hat: [B] estimated total electrons per molecule from Monte Carlo integration
            z: [N] atomic numbers of all atoms in the batch
            batch: [N] molecule indices for each atom (from PyTorch Geometric)
            
        Returns:
            loss: scalar MSE loss penalizing deviation from true electron count
        """
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            
        num_molecules = int(batch.max()) + 1
        
        # The true number of electrons is the sum of the atomic numbers 
        # (Assuming neutrally charged molecules for the QM9 dataset)
        N_elec_true = scatter(z.float(), batch, dim=0, dim_size=num_molecules, reduce="sum")
        
        # The loss is the Mean Squared Error between predicted and true electrons
        loss = self.mse(N_elec_hat, N_elec_true)
        
        return loss

class PoissonLoss(nn.Module):
    def __init__(self):
        """
        Physics-Informed Loss that enforces the Poisson Equation:
        Laplacian(V(r)) = -4 * pi * rho(r)
        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, potential, density, query_pos):
        """
        Args:
            potential: [Q, 1] predicted electrostatic potential
            density: [Q, 1] predicted electron density
            query_pos: [Q, 3] spatial query points, must have requires_grad=True
        """
        # 1. First derivative: Grad V (dV/dx, dV/dy, dV/dz)
        grad_V = torch.autograd.grad(
            outputs=potential,
            inputs=query_pos,
            grad_outputs=torch.ones_like(potential),
            create_graph=True,
            retain_graph=True
        )[0] # Returns [Q, 3]
        
        # 2. Second derivative: Laplacian V
        laplacian_V = torch.zeros_like(potential)
        for i in range(3): # x, y, z
            grad_V_i = grad_V[:, i:i+1] # [Q, 1]
            
            grad_grad_V_i = torch.autograd.grad(
                outputs=grad_V_i,
                inputs=query_pos,
                grad_outputs=torch.ones_like(grad_V_i),
                create_graph=True,
                retain_graph=True
            )[0] # [Q, 3]
            
            # Extract the i-th component (d^2V / dx_i^2)
            laplacian_V += grad_grad_V_i[:, i:i+1]
            
        # 3. Physics Target: -4 * pi * rho
        target = -4.0 * math.pi * density
        
        # 4. MSE Loss
        loss = self.mse(laplacian_V, target)
        return loss
