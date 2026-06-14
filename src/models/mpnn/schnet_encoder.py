import torch
import torch.nn as nn
from torch_geometric.nn.models.schnet import InteractionBlock, GaussianSmearing

def fallback_radius_graph(pos, r, batch):
    """Pure PyTorch fallback for radius_graph to avoid C++ pyg-lib dependency"""
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    
    dist = torch.cdist(pos, pos)
    # Mask distances within radius, excluding self-loops (dist > 1e-5)
    mask = (dist < r) & (dist > 1e-5)
    # Ensure edges only connect atoms within the same molecule
    batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
    mask = mask & batch_mask
    
    row, col = torch.where(mask)
    return torch.stack([row, col], dim=0)

class SchNetEncoder(nn.Module):
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=3, num_gaussians=50, cutoff=10.0):
        """
        A custom MPNN based on SchNet that returns the latent node embeddings (anchors)
        for every atom, rather than globally pooling them.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        # Atomic number embedding (handles elements up to Z=100 safely)
        self.embedding = nn.Embedding(100, hidden_channels)
        
        # Distance expansion (Continuous-filter Convolution)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Message Passing (Interaction) blocks
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)

    def forward(self, z, pos, batch=None):
        """
        Args:
            z: Atomic numbers [N]
            pos: Atomic coordinates [N, 3]
            batch: Batch indices [N] (optional)
        Returns:
            h: Node embeddings [N, hidden_channels]
        """
        # Create radius graph dynamically based on atom spatial positions using pure PyTorch
        edge_index = fallback_radius_graph(pos, r=self.cutoff, batch=batch)
        
        # Calculate pairwise scalar distances between connected atoms
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        
        # Expand distances using Gaussian smearing (RBFs)
        edge_attr = self.distance_expansion(edge_weight)

        # Initial node embeddings based purely on atomic identity
        h = self.embedding(z)

        # Message passing to incorporate chemical environment
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Return the continuous latent vector for EVERY atom. 
        # We intentionally skip global pooling here!
        return h
