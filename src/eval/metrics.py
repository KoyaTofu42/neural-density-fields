import numpy as np
from sklearn.metrics import r2_score

def calculate_metrics(pred_density, true_density):
    """
    pred_density: (Q,) numpy array
    true_density: (Q,) numpy array
    Returns dict with MAE, RMSE, R2
    """
    mae = np.mean(np.abs(pred_density - true_density))
    rmse = np.sqrt(np.mean((pred_density - true_density)**2))
    
    # Prevent R2 calculation error if variance is exactly 0 (e.g. all empty space)
    if np.var(true_density) == 0:
        r2 = 1.0 if np.allclose(pred_density, true_density) else 0.0
    else:
        r2 = r2_score(true_density, pred_density)
        
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
