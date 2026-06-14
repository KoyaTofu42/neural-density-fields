import pytest
import torch
import numpy as np
from src.eval.inference_utils import OptimizedInferencer
import torch.nn as nn

# Dummy model to simulate NeuralDensityField
class DummyModel(nn.Module):
    def forward(self, batch):
        # Return dummy density, potential, query_coords
        N = batch.query_pos.shape[0]
        return torch.ones((N, 1), device=batch.query_pos.device), torch.zeros((N, 1)), batch.query_pos

def test_optimized_inferencer():
    device = torch.device('cpu')
    model = DummyModel()
    
    # We can test the inferencer even without CUDA, as it falls back
    inferencer = OptimizedInferencer(model, device)
    
    # Dummy data
    z = np.array([6, 1])
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    query_coords = np.random.rand(150, 3) # 150 points
    
    # Test with chunk_size smaller than total to trigger the padding logic
    preds = inferencer.predict_density(z, pos, query_coords, chunk_size=100)
    
    assert preds.shape == (150,)
    # Our dummy model always returns 1.0
    np.testing.assert_allclose(preds, 1.0)
