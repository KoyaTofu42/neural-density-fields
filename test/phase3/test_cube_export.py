import os
import pytest
import numpy as np
from src.utils.cube_export import write_cube_file

def test_write_cube_file(tmp_path):
    filepath = os.path.join(tmp_path, "test.cube")
    
    z = np.array([6, 1])
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    
    min_coords = np.array([-1.0, -1.0, -1.0])
    max_coords = np.array([1.0, 1.0, 1.0])
    grid_res = 5
    
    # Create dummy density grid
    density_grid = np.random.rand(grid_res, grid_res, grid_res)
    
    write_cube_file(filepath, z, pos, min_coords, max_coords, grid_res, density_grid)
    
    assert os.path.exists(filepath)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    assert len(lines) > 6
    assert "PI-NDF High Resolution Density Export" in lines[0]
    
    # Atoms line: 2 atoms.
    assert lines[2].strip().startswith("2")
    
    # Grid lines
    assert lines[3].strip().startswith("5")
    assert lines[4].strip().startswith("5")
    assert lines[5].strip().startswith("5")
    
    # Atom entries
    assert lines[6].strip().startswith("6")
    assert lines[7].strip().startswith("1")
    
    # Data lines
    data_lines = lines[8:]
    assert len(data_lines) > 0
