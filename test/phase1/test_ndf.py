import torch
from torch_geometric.data import Data, Batch
from src.models.ndf_model import NeuralDensityField

def run_ndf_test():
    print("="*60)
    print("Testing Full Neural Density Field Architecture")
    print("="*60)
    
    # 1. Create a dummy graph (e.g. H2O)
    z = torch.tensor([8, 1, 1], dtype=torch.long)
    pos = torch.randn(3, 3)
    
    # 2. Create 10 dummy query points around the molecule
    query_pos = torch.randn(10, 3) * 3.0
    
    # 3. Create a PyG Data object
    data = Data(z=z, pos=pos, query_pos=query_pos)
    
    # PyG DataLoaders usually collate into a Batch object. 
    # Let's test with a batch size of 2 molecules to ensure batching logic holds.
    batch = Batch.from_data_list([data, data]) 
    
    print("\n[1] Created Dummy PyG Batch:")
    print(f"  - Number of molecules: {batch.num_graphs}")
    print(f"  - Total atoms: {batch.z.shape[0]}")
    print(f"  - Total query points: {batch.query_pos.shape[0]}")

    # 4. Instantiate the entire model
    print("\n[2] Initializing full Neural Density Field...")
    # Using k=3 because our dummy molecule only has 3 atoms
    model = NeuralDensityField(hidden_dim=64, k=3, num_gaussians=50) 
    
    # 5. Forward Pass
    print("\n[3] Running Forward Pass...")
    with torch.no_grad():
        density = model(batch)
        
    print("\n[4] Verification Results:")
    print(f"  Expected shape: [{batch.query_pos.shape[0]} (Total queries), 1 (Density scalar)]")
    print(f"  Actual shape:   {list(density.shape)}")
    
    # Check physics constraint
    is_positive = torch.all(density >= 0).item()
    print(f"  - Is the predicted density strictly >= 0? {is_positive}")
    
    if list(density.shape) == [batch.query_pos.shape[0], 1] and is_positive:
        print("\n✅ SUCCESS: The Full Architecture is strictly connected and physically valid!")
        print("  The Neural Network is officially fully constructed.")
    else:
        print("\n❌ FAILURE: Pipeline shape mismatch or negative density leaked.")

if __name__ == "__main__":
    run_ndf_test()
