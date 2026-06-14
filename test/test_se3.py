import torch
import math
from src.models.query_engine.local_query import LocalQueryEngine

def get_random_rotation_matrix():
    """Generates a random 3D rotation matrix using Euler angles"""
    theta_x, theta_y, theta_z = torch.rand(3) * 2 * math.pi
    
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)]
    ], dtype=torch.float32)
    
    Ry = torch.tensor([
        [math.cos(theta_y), 0, math.sin(theta_y)],
        [0, 1, 0],
        [-math.sin(theta_y), 0, math.cos(theta_y)]
    ], dtype=torch.float32)
    
    Rz = torch.tensor([
        [math.cos(theta_z), -math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    return Rz @ Ry @ Rx

def run_se3_test():
    print("="*60)
    print("Testing Mathematical SE(3) Invariance of the Query Engine")
    print("="*60)
    
    engine = LocalQueryEngine(k=5, num_gaussians=50)
    
    # 1. Create a fake molecule with 6 atoms
    atom_pos = torch.randn(6, 3) * 5.0  # Spread them out
    
    # 2. Create 2 fake query points (where we want to predict density)
    query_pos = torch.randn(2, 3) * 2.0
    
    print("\n[1] Running Base Coordinates...")
    feat_base, neighbors_base, dist_base = engine(query_pos, atom_pos)
    
    # 3. Apply a random 3D Rotation and Translation (SE(3) Transformation)
    print("\n[2] Applying Random 3D Rotation & Translation to all coordinates...")
    R = get_random_rotation_matrix()
    translation = torch.tensor([100.0, -50.0, 42.0]) # Massive translation
    
    # Rotate and translate BOTH the atoms and the query points identically
    atom_pos_transformed = (atom_pos @ R.T) + translation
    query_pos_transformed = (query_pos @ R.T) + translation
    
    print("\n[3] Running Transformed Coordinates...")
    feat_trans, neighbors_trans, dist_trans = engine(query_pos_transformed, atom_pos_transformed)
    
    # 4. Compare the outputs
    print("\n[4] Verification Results:")
    
    # Check neighbors (must pick the exact same 5 atoms)
    neighbors_match = torch.all(neighbors_base == neighbors_trans)
    print(f"  - Do K-Nearest Neighbors match exactly? {neighbors_match.item()}")
    
    # Check invariant features (distances and angles)
    # Floating point math causes tiny differences, so we use torch.allclose
    max_diff = torch.max(torch.abs(feat_base - feat_trans)).item()
    features_match = torch.allclose(feat_base, feat_trans, atol=1e-4)
    print(f"  - Do Output Features match exactly? {features_match} (Max Diff: {max_diff:.6f})")
    
    if neighbors_match and features_match:
        print("\n✅ SUCCESS: The Engine is strictly SE(3) Invariant!")
        print("  Even when the molecule is completely rotated and thrown across the universe,")
        print("  the neural network will receive the exact same relative geometry!")
    else:
        print("\n❌ FAILURE: The Engine leaked coordinate data.")

if __name__ == "__main__":
    run_se3_test()
