import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch_geometric.loader import DataLoader
from src.data.density_dataset import DensityDataset
from src.models.mpnn import SchNetEncoder

def run_test():
    print("="*60)
    print("Testing Dataloader and MPNN Integration")
    print("="*60)

    # 1. Load Dataset
    print("\n[1] Initializing DensityDataset...")
    try:
        dataset = DensityDataset(h5_filepath="data/density_samples.h5", num_samples_per_mol=512)
        print(f"Successfully loaded dataset with {len(dataset)} molecules.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have run the data generation pipeline first!")
        return

    # 2. Create DataLoader
    # Using batch_size=4 to ensure PyG handles batching graphs of different sizes correctly
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    
    print("\n[2] Inspected PyG Batch Object:")
    print(f"  - Number of graphs (molecules) in batch: {batch.num_graphs}")
    print(f"  - Total atoms in batch (z): {list(batch.z.shape)}")
    print(f"  - Total atom coordinates (pos): {list(batch.pos.shape)}")
    print(f"  - Total query coordinates: {list(batch.query_pos.shape)}")
    print(f"  - Total target densities: {list(batch.y.shape)}")

    # 3. Instantiate MPNN
    print("\n[3] Initializing SchNet Encoder...")
    hidden_dim = 128
    model = SchNetEncoder(hidden_channels=hidden_dim, num_interactions=3)
    
    # 4. Forward Pass
    print("\n[4] Running Forward Pass...")
    with torch.no_grad():
        # Pass atomic numbers, coordinates, and graph mapping indices
        node_embeddings = model(batch.z, batch.pos, batch.batch)
        
    print("\n[Result] Output Node Embeddings Shape:")
    print(f"  Expected shape: [{batch.z.shape[0]} (Total atoms), {hidden_dim} (Hidden dim)]")
    print(f"  Actual shape:   {list(node_embeddings.shape)}")
    
    if list(node_embeddings.shape) == [batch.z.shape[0], hidden_dim]:
        print("\n✅ SUCCESS: The MPNN correctly outputted a continuous latent vector for every single atom!")
    else:
        print("\n❌ FAILURE: Output shape mismatch.")

if __name__ == "__main__":
    run_test()
