import pytest
import torch
import numpy as np
from src.data.high_res_sampler import BoundingBoxGenerator, generate_dense_grid

def test_bounding_box_generator_bond():
    # Dummy atoms: two atoms at (0,0,0) and (2,0,0)
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    generator = BoundingBoxGenerator(pos, margin=1.0)
    
    # Test bond bbox
    min_coords, max_coords = generator.get_bond_bbox(0, 1)
    
    # Expected min: min(0,2)-1 = -1.0 for x, -1.0 for y, z
    np.testing.assert_allclose(min_coords, [-1.0, -1.0, -1.0])
    # Expected max: max(0,2)+1 = 3.0 for x, 1.0 for y, z
    np.testing.assert_allclose(max_coords, [3.0, 1.0, 1.0])

def test_bounding_box_generator_atom():
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    generator = BoundingBoxGenerator(pos, margin=2.0)
    
    min_coords, max_coords = generator.get_atom_bbox(1)
    np.testing.assert_allclose(min_coords, [0.0, -2.0, -2.0])
    np.testing.assert_allclose(max_coords, [4.0, 2.0, 2.0])

def test_generate_dense_grid():
    min_coords = np.array([-1.0, -1.0, -1.0])
    max_coords = np.array([1.0, 1.0, 1.0])
    
    grid_res = 10
    X, Y, Z, query_coords = generate_dense_grid(min_coords, max_coords, grid_res)
    
    assert X.shape == (10, 10, 10)
    assert Y.shape == (10, 10, 10)
    assert Z.shape == (10, 10, 10)
    
    assert query_coords.shape == (1000, 3)
    
    # Check boundaries
    assert torch.min(query_coords[:, 0]) == -1.0
    assert torch.max(query_coords[:, 0]) == 1.0
