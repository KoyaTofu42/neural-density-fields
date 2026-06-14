import pytest
import numpy as np
from src.eval.metrics import calculate_metrics
from src.eval.nci_analysis import calculate_rdg

def test_calculate_metrics():
    true = np.array([0.1, 0.2, 0.3])
    pred = np.array([0.1, 0.2, 0.25])
    
    res = calculate_metrics(pred, true)
    
    assert "MAE" in res
    assert "RMSE" in res
    assert "R2" in res
    
    np.testing.assert_allclose(res["MAE"], 0.05/3.0)

def test_calculate_rdg():
    grid_res = 10
    step_bohr = 0.5
    # Uniform field
    density = np.ones((grid_res, grid_res, grid_res))
    rdg = calculate_rdg(density, step_bohr)
    
    # Gradient of ones is zeros, so RDG is zero
    np.testing.assert_allclose(rdg, 0.0)
    
    # Linear field
    x = np.linspace(1, 2, grid_res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    density = X
    rdg = calculate_rdg(density, step_bohr)
    
    assert np.all(rdg > 0.0)
