import h5py
import numpy as np

def inspect_dataset():
    path = "data/density_samples.h5"
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        first_mol = f[keys[0]]
        target_density = first_mol["target_density"][:]
        
        print(f"Dataset has {len(keys)} molecules.")
        print(f"Stats for molecule '{keys[0]}':")
        print(f"  Shape: {target_density.shape}")
        print(f"  Min: {target_density.min():.6f}")
        print(f"  Max: {target_density.max():.6f}")
        print(f"  Mean: {target_density.mean():.6f}")
        print(f"  Std: {target_density.std():.6f}")
        print(f"  % > 0.01: {(target_density > 0.01).mean() * 100:.2f}%")
        print(f"  % > 1.0:  {(target_density > 1.0).mean() * 100:.2f}%")
        
        # What is the MSE if we just predict 0.0 for everything?
        mse_zero = np.mean(target_density ** 2)
        print(f"  MSE if predicting 0.0: {mse_zero:.6f}")
        
        # What is the MSE if we predict the mean?
        mse_mean = np.mean((target_density - target_density.mean()) ** 2)
        print(f"  MSE if predicting mean: {mse_mean:.6f}")

if __name__ == "__main__":
    inspect_dataset()
