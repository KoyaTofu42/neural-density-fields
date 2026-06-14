import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.qm9_subset import download_and_subset_qm9
from src.data.dft_runner import run_dft_on_subset
from src.data.point_sampler import generate_density_samples

def main():
    print("="*60)
    print("Starting Neural Density Field Data Generation Pipeline")
    print("="*60)
    
    # Step 1: Data Preparation
    print("\n--- [Step 1/3] Preparing QM9 Dataset ---")
    download_and_subset_qm9()
    
    # Step 2: DFT Calculations
    print("\n--- [Step 2/3] Running PySCF DFT Computations ---")
    print("Note: This step is CPU-intensive and may take some time.")
    run_dft_on_subset()
    
    # Step 3: Density Sampling
    print("\n--- [Step 3/3] Sampling Electron Density Points ---")
    generate_density_samples()
    
    print("\n" + "="*60)
    print("Pipeline completed successfully! All data is saved in the 'data/' directory.")
    print("="*60)

if __name__ == "__main__":
    main()
