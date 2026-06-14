import torch
import torch.nn as nn
from src.models.mpnn.schnet_encoder import SchNetEncoder
from src.models.query_engine.local_query import LocalQueryEngine
from src.models.decoder.mlp_decoder import ContinuousDecoder

class NeuralDensityField(nn.Module):
    def __init__(self, hidden_dim=128, k=5, num_gaussians=50, cutoff=10.0, num_interactions=3, mlp_hidden=256):
        """
        The absolute top-level Orchestrator model for the Neural Density Field.
        """
        super().__init__()
        
        # Inference caching
        self._cached_pos = None
        self._cached_h_atoms = None
        
        # 1. The MPNN that embeds the chemistry into atomic anchors
        self.encoder = SchNetEncoder(
            hidden_channels=hidden_dim, 
            num_filters=128, 
            num_interactions=num_interactions, 
            num_gaussians=num_gaussians, 
            cutoff=cutoff
        )
        
        # 2. The Local Query Engine that handles SE(3)-invariant spatial mapping
        self.query_engine = LocalQueryEngine(
            k=k, 
            num_gaussians=num_gaussians, 
            cutoff=cutoff
        )
        
        # Calculate SE(3) feature dimension dynamically based on k and gaussians
        # We extract (K * num_gaussians) RBFs, (K * (K - 1) / 2) unique angles, and K raw scalar distances
        num_angles = int(k * (k - 1) / 2)
        se3_feature_dim = (k * num_gaussians) + num_angles + k
        
        # 3. The Continuous Decoder that fuses chemistry and space to predict density
        self.decoder = ContinuousDecoder(
            hidden_dim=hidden_dim, 
            se3_feature_dim=se3_feature_dim, 
            mlp_hidden=mlp_hidden,
            k=k
        )

    def forward(self, data):
        """
        Args:
            data: A PyTorch Geometric Data object containing:
                  - z: [N] Atomic numbers
                  - pos: [N, 3] Atomic coordinates
                  - batch: [N] Graph assignments for atoms
                  - query_pos: [Q, 3] Continuous spatial query points
        Returns:
            density: [Q, 1] Predicted electron density for each query point.
        """
        # --- 1. MPNN Anchor Embedding ---
        num_molecules = getattr(data, 'num_graphs', 1)
        batch_atom = getattr(data, 'batch', None) if num_molecules > 1 else None

        # Check inference cache (if model in eval mode and atoms haven't moved, reuse embeddings)
        if not self.training and self._cached_pos is not None and data.pos is self._cached_pos:
            h_atoms = self._cached_h_atoms
        else:
            h_atoms = self.encoder(data.z, data.pos, batch_atom)
            if not self.training:
                self._cached_pos = data.pos
                self._cached_h_atoms = h_atoms
            else:
                self._cached_pos = None
                self._cached_h_atoms = None
        
        # --- 2. Query Engine Mapping ---
        # Skip creating O(Q * N) batch masks if evaluating a single molecule
        if num_molecules > 1:
            # Check if PyTorch Geometric dynamically provided the exact batch mask via `follow_batch`
            if hasattr(data, 'query_pos_batch'):
                query_batch = data.query_pos_batch
            else:
                # Fallback for perfectly uniform datasets
                queries_per_mol = data.query_pos.size(0) // num_molecules
                query_batch = torch.repeat_interleave(
                    torch.arange(num_molecules, device=data.query_pos.device), 
                    repeats=queries_per_mol
                )
        else:
            query_batch = None
        
        # Calculate SE(3) invariant spatial features
        features, neighbor_indices, distances = self.query_engine(
            data.query_pos, data.pos, batch_query=query_batch, batch_atom=batch_atom
        )
        
        # --- 3. Gather Context ---
        # Fetch the MPNN embeddings for the specific K-nearest atoms for each query point
        h_neighbors = h_atoms[neighbor_indices] # [Q, K, hidden_dim]
        
        # --- 4. Decode ---
        # Pass the neighbor embeddings, their distances, and the full geometric features
        # through the decoder to get the final density prediction!
        density_scaled = self.decoder(h_neighbors, distances, features) # [Q, 1]
        
        # Target Scaling: The network predicts a mathematically stable [0, 1] scale value.
        # We multiply by 100.0 here so the model natively outputs the true physical electron density!
        density = density_scaled * 100.0
        
        return density
