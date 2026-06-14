import torch
import torch.nn as nn
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
