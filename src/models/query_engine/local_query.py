import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.schnet import GaussianSmearing

class LocalQueryEngine(nn.Module):
    def __init__(self, k=5, num_gaussians=50, cutoff=10.0):
        super().__init__()
        self.k = k
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        
        # Maps raw scalar distances [0, cutoff] into smooth, high-dimensional vectors
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Precompute indices for unique angle pairs (upper triangle)
        idx_i, idx_j = torch.triu_indices(k, k, offset=1)
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)

    def forward(self, query_pos, atom_pos, batch_query=None, batch_atom=None):
        """
        Args:
            query_pos: [Q, 3] Query coordinates (points in empty space)
            atom_pos: [N, 3] Atom coordinates
            batch_query: [Q] Molecule index for each query point (optional if 1 graph)
            batch_atom: [N] Molecule index for each atom (optional if 1 graph)
            
        Returns:
            features: [Q, feature_dim] purely SE(3)-invariant geometric features
            neighbor_indices: [Q, k] Indices of the K nearest atoms in the atom_pos array
            d_k: [Q, k] Distances to the K nearest atoms
        """
        # 1. Compute pairwise Euclidean distances between all queries and all atoms
        # dist_mat: [Q, N]
        dist_mat = torch.cdist(query_pos, atom_pos)
        
        # Mask out atoms that don't belong to the same molecule as the query point
        # Optimization: Skip masking if batch_query/atom are None (i.e. single molecule)
        if batch_query is not None and batch_atom is not None:
            mask = (batch_query.unsqueeze(1) == batch_atom.unsqueeze(0))
            dist_mat.masked_fill_(~mask, float('inf'))
        
        # Fallback if a molecule has fewer than K atoms (like H2 which has 2)
        actual_k = min(self.k, atom_pos.size(0))
        
        # 2. Find the actual_k-Nearest Neighbors
        # d_k: [Q, actual_k], neighbor_indices: [Q, actual_k]
        d_k, neighbor_indices = torch.topk(dist_mat, k=actual_k, dim=1, largest=False)
        
        # Pad d_k and neighbor_indices up to self.k if the molecule is too small
        if actual_k < self.k:
            pad_size = self.k - actual_k
            # Pad distances with a very large number so RBF and Attention go to 0
            d_k = F.pad(d_k, (0, pad_size), value=1e4)
            # Pad indices with 0 (safe index, attention weight will be exp(-1e4) = 0 anyway)
            neighbor_indices = F.pad(neighbor_indices, (0, pad_size), value=0)
        
        # 3. Gaussian Smearing (Positional Encoding for Distances)
        # PyG's GaussianSmearing flattens the input to 1D, so we must reshape it back
        rbf = self.distance_expansion(d_k)
        rbf = rbf.view(query_pos.size(0), self.k, self.num_gaussians)
        
        # Flatten the K distance RBFs into a single long vector per query point
        # rbf_flat: [Q, k * num_gaussians]
        rbf_flat = rbf.view(-1, self.k * self.num_gaussians)
        
        # 4. Angle Computation (Cosine of angles between neighbor vectors)
        # neighbors_pos: [Q, k, 3]
        neighbors_pos = atom_pos[neighbor_indices]
        
        # Directional vectors from query to atoms: [Q, k, 3]
        vecs = neighbors_pos - query_pos.unsqueeze(1)
        
        # Normalize the vectors
        # Add epsilon to prevent division by zero if query point happens to sit EXACTLY on an atom
        norms = d_k.unsqueeze(-1) + 1e-8
        norm_vecs = vecs / norms
        
        # Compute pairwise dot products (cosines) between all K vectors
        # dot_products: [Q, k, k]
        dot_products = torch.bmm(norm_vecs, norm_vecs.transpose(1, 2))
        
        # We only care about unique pairs of neighbors (upper triangle of the matrix, no self-angles)
        # Number of unique angles is k * (k - 1) / 2
        angles = dot_products[:, self.idx_i, self.idx_j] # [Q, num_angles]
        
        # 5. Combine everything into a single completely SE(3)-invariant feature vector!
        # Standard Continuous Field architecture (e.g. NeRF): Append raw coordinates alongside positional encodings
        # to guarantee infinite resolution without being bottlenecked by the RBF bin width!
        features = torch.cat([rbf_flat, angles, d_k], dim=1)
        
        return features, neighbor_indices, d_k
