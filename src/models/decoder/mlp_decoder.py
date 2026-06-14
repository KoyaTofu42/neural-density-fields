import torch
import torch.nn as nn

class ContinuousDecoder(nn.Module):
    def __init__(self, hidden_dim=128, se3_feature_dim=260, mlp_hidden=256, k=5):
        """
        The continuous decoder that combines chemical context (MPNN anchors)
        with spatial context (SE(3) query features) to predict electron density.
        
        Args:
            hidden_dim: Dimension of the MPNN node embeddings.
            se3_feature_dim: Dimension of the SE(3) features from the LocalQueryEngine.
            mlp_hidden: Hidden dimension of the final MLP.
            k: The number of nearest neighbors tracked by the query engine.
        """
        super().__init__()
        
        # The input to the MLP is the flattened K anchor embeddings (k * hidden_dim)
        # concatenated with the geometric SE(3) features (se3_feature_dim)
        # AND the low-fidelity baseline density (1).
        in_channels = (k * hidden_dim) + se3_feature_dim + 1
        
        self.mlp_shared = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU()
        )
        
        self.density_head = nn.Sequential(
            nn.Linear(mlp_hidden, 1)
            # Softplus removed: the delta correction can be negative!
        )
        
        self.potential_head = nn.Sequential(
            nn.Linear(mlp_hidden, 1)
        )
        
        # Initialization Fix: Since we want the initial delta to be near zero, 
        # initialize the final layer to predict a tiny positive value initially 
        # so that it doesn't get trapped by the exact 0 gradient of the ReLU.
        nn.init.uniform_(self.density_head[0].weight, -1e-4, 1e-4)
        nn.init.constant_(self.density_head[0].bias, 1e-4)

    def forward(self, h_neighbors, distances, se3_features, rho_low):
        """
        Args:
            h_neighbors: [Q, K, hidden_dim] Embeddings of the K nearest atoms.
            distances: [Q, K] Exact Euclidean distances to the K neighbors.
            se3_features: [Q, se3_feature_dim] Invariant features (RBFs and angles).
            rho_low: [Q, 1] The low-fidelity baseline density.
        Returns:
            delta_density: [Q, 1] Predicted residual density correction.
            potential: [Q, 1] Predicted electrostatic potential.
        """
        # 1. Physics-based Distance Weighting (Task 4.2)
        # We strictly enforce that the neural network pays exponentially more attention
        # to atoms that are physically closer to the query point.
        # This requires zero learned parameters and is mathematically robust.
        weights = torch.exp(-distances) # [Q, K]
        
        # Expand weights for broadcasting
        weights = weights.unsqueeze(-1) # [Q, K, 1]
        
        # 2. Compute Context Vector
        # Multiply each neighbor's embedding by its distance weight, then flatten across K
        # This preserves the exact mapping between each atom's chemical identity and its specific distance
        weighted_h = h_neighbors * weights # [Q, K, hidden_dim]
        c_q = weighted_h.view(weighted_h.size(0), -1) # [Q, K * hidden_dim]
        
        # 3. Concatenate and Predict
        # We fuse the chemically-aligned context (c_q), spatial geometry (se3_features), and the baseline (rho_low)
        x = torch.cat([c_q, se3_features, rho_low], dim=1) # [Q, (K * hidden_dim) + se3_feature_dim + 1]
        
        # Pass through the shared MLP trunk
        shared_out = self.mlp_shared(x) # [Q, mlp_hidden]
        
        # Branch into delta density and potential
        delta_density = self.density_head(shared_out) # [Q, 1]
        potential = self.potential_head(shared_out) # [Q, 1]
        
        return delta_density, potential
