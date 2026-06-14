import os
import gc
import time
import torch
import numpy as np
from src.models.ndf_model import NeuralDensityField
from src.data.density_dataset import DensityDataset
from src.data.high_res_sampler import generate_dense_grid
from src.eval.inference_utils import OptimizedInferencer

def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()

def main():
    print("="*60)
    print("PHASE 3: VRAM Profiling & Scaling Benchmark")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ Error: CUDA is not available. VRAM profiling requires an NVIDIA GPU.")
        return

    device = torch.device('cuda')
    
    # Initialize Model
    model = NeuralDensityField(hidden_dim=128, k=5, num_gaussians=300)
    model_path = "models_checkpoints/best_ndf.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        print("✅ Loaded trained PI-NDF weights.")
        
    inferencer = OptimizedInferencer(model, device)

    # Load dummy molecule
    dataset = DensityDataset("data/density_samples.h5", num_samples_per_mol=1)
    data = dataset[0]
    z = data.z
    pos = data.pos
    
    # We cap the resolution at 80 (512,000 points) so it runs easily on RTX 3060
    resolutions = [10, 20, 30, 40, 50, 60, 70, 80]
    min_c = np.array([-2.0, -2.0, -2.0])
    max_c = np.array([2.0, 2.0, 2.0])

    print(f"{'Grid Res':<10} | {'Total Points':<12} | {'Time (s)':<10} | {'Peak VRAM (MB)':<15}")
    print("-" * 55)

    for res in resolutions:
        _, _, _, query_coords = generate_dense_grid(min_c, max_c, grid_res=res)
        
        clear_vram()
        start_time = time.time()
        
        # Warmup for torch.compile
        if res == resolutions[0]:
            inferencer.predict_density(z, pos, query_coords[:1000], chunk_size=1000)
            clear_vram()
            start_time = time.time()

        # Actual Inference
        # Batch size of 50000 ensures high throughput
        inferencer.predict_density(z, pos, query_coords, chunk_size=50000)
        
        duration = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        print(f"{res:<10} | {len(query_coords):<12} | {duration:<10.3f} | {peak_vram:<15.1f}")

    print("-" * 55)
    print("✅ VRAM Profiling Complete.")
    print("If Peak VRAM remains relatively flat as Total Points increase, you have successfully beaten the Voxel Curse!")

if __name__ == "__main__":
    main()
