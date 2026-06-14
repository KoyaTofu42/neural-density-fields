import torch
import torch.nn as nn
from torch_geometric.utils import scatter

class UniformMonteCarloIntegrator(nn.Module):
    def __init__(self, num_samples=10000, margin=2.0):
        """
        A Monte Carlo integrator that samples uniformly within a bounding box 
        around the molecule to estimate spatial integrals like total electron count.
        
        Args:
            num_samples (int): Number of points (M) to sample per molecule.
            margin (float): Extra space (in Angstroms) added to the bounding box beyond the outermost atoms.
        """
        super().__init__()
        self.num_samples = num_samples
        self.margin = margin

    def sample_points(self, pos, batch=None):
        """
        Samples uniform points in the bounding box of each molecule in the batch.
        
        Args:
            pos: [N, 3] atomic coordinates
            batch: [N] molecule indices for each atom (from PyTorch Geometric)
            
        Returns:
            query_pos: [B * num_samples, 3] sampled continuous coordinates
            query_batch: [B * num_samples] molecule indices for each sampled coordinate
            volumes: [B] volume of the bounding box for each molecule
        """
        device = pos.device
        num_molecules = int(batch.max()) + 1 if batch is not None else 1
        
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=device)
            
        # 1. Find bounding box per molecule
        min_pos = scatter(pos, batch, dim=0, dim_size=num_molecules, reduce="min")
        max_pos = scatter(pos, batch, dim=0, dim_size=num_molecules, reduce="max")
        
        min_pos = min_pos - self.margin
        max_pos = max_pos + self.margin
        
        # 2. Calculate volume per molecule: V = dx * dy * dz
        box_size = max_pos - min_pos
        volumes = torch.prod(box_size, dim=1) # [B]
        
        # 3. Sample points uniformly
        # min_pos is [B, 3], expand it to [B, num_samples, 3]
        min_expanded = min_pos.unsqueeze(1).expand(-1, self.num_samples, -1)
        box_expanded = box_size.unsqueeze(1).expand(-1, self.num_samples, -1)
        
        # Random uniform [0, 1]
        u = torch.rand_like(min_expanded)
        
        # Scale to bounding box
        query_pos = min_expanded + u * box_expanded
        
        # Flatten back to [B * num_samples, 3]
        query_pos = query_pos.reshape(-1, 3)
        
        # Create query_batch [B * num_samples]
        query_batch = torch.arange(num_molecules, device=device).repeat_interleave(self.num_samples)
        
        return query_pos, query_batch, volumes

    def integrate(self, density, query_batch, volumes):
        """
        Computes the integral of the predicted density over the continuous bounding box space.
        Uses the Monte Carlo estimator: Integral = V/M * sum(rho_i)
        
        Args:
            density: [B * num_samples, 1] or [B * num_samples] predicted electron density
            query_batch: [B * num_samples] molecule indices
            volumes: [B] volume of each molecule's bounding box
            
        Returns:
            N_elec_hat: [B] estimated total electrons per molecule
        """
        num_molecules = volumes.size(0)
        
        if density.dim() > 1:
            density = density.squeeze(-1)
            
        # Sum the density values for each molecule
        density_sum = scatter(density, query_batch, dim=0, dim_size=num_molecules, reduce="sum")
        
        # Multiply by Volume and divide by number of samples M
        N_elec_hat = density_sum * (volumes / self.num_samples)
        
        return N_elec_hat
