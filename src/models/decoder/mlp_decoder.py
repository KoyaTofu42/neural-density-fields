import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # concatenated with the geometric SE(3) features (se3_feature_dim).
        in_channels = (k * hidden_dim) + se3_feature_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
            # Softplus ensures the final predicted density is strictly >= 0 (physics constraint)
            nn.Softplus() 
        )
        
        # Initialization Fix: Softplus kills gradients if the network is forced negative.
        # 98% of the density space is empty (target=0.0). If the network starts by predicting ~69.0,
        # the empty space will forcefully drag the bias to -5.0, permanently killing the activation gradient!
        # By initializing the bias to -5.0 directly, we start near 0.0, completely avoiding the dying gradient trap!
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.constant_(self.mlp[-2].bias, -5.0)

    def forward(self, h_neighbors, distances, se3_features):
        """
        Args:
            h_neighbors: [Q, K, hidden_dim] Embeddings of the K nearest atoms.
            distances: [Q, K] Exact Euclidean distances to the K neighbors.
            se3_features: [Q, se3_feature_dim] Invariant features (RBFs and angles).
        Returns:
            density: [Q, 1] Predicted electron density.
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
        # We fuse the chemically-aligned context (c_q) with the spatial geometry (se3_features)
        x = torch.cat([c_q, se3_features], dim=1) # [Q, (K * hidden_dim) + se3_feature_dim]
        
        # Pass through the MLP to predict the final continuous scalar density
        density = self.mlp(x) # [Q, 1]
        
        return density
