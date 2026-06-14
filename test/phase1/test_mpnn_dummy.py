import torch
from src.models.mpnn import SchNetEncoder

def run_dummy_test():
    print("="*60)
    print("Testing SchNet Encoder with Dummy Data")
    print("="*60)
    
    # Create dummy data: 2 synthetic molecules
    # Molecule 0: 3 atoms (e.g. H2O)
    z_0 = torch.tensor([8, 1, 1], dtype=torch.long)
    pos_0 = torch.randn(3, 3)
    
    # Molecule 1: 5 atoms (e.g. CH4)
    z_1 = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
    pos_1 = torch.randn(5, 3)
    
    # Combine them into a single "batch"
    z = torch.cat([z_0, z_1], dim=0) # Total 8 atoms
    pos = torch.cat([pos_0, pos_1], dim=0)
    
    # Batch indices map each atom to its respective molecule (0 or 1)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    
    print("\n[1] Created Dummy Batch:")
    print(f"  - Number of molecules: 2")
    print(f"  - Total atoms (z): {list(z.shape)}")
    print(f"  - Atom coordinates (pos): {list(pos.shape)}")
    print(f"  - Batch assignments: {list(batch.shape)}")

    # Initialize MPNN
    print("\n[2] Initializing SchNet Encoder...")
    hidden_dim = 128
    model = SchNetEncoder(hidden_channels=hidden_dim, num_interactions=3)
    
    # Forward Pass
    print("\n[3] Running Forward Pass...")
    with torch.no_grad():
        node_embeddings = model(z, pos, batch)
        
    print("\n[Result] Output Node Embeddings Shape:")
    print(f"  Expected shape: [{z.shape[0]} (Total atoms), {hidden_dim} (Hidden dim)]")
    print(f"  Actual shape:   {list(node_embeddings.shape)}")
    
    if list(node_embeddings.shape) == [z.shape[0], hidden_dim]:
        print("\n✅ SUCCESS: The MPNN correctly processed the dummy data and outputted a continuous latent vector for every single atom!")
    else:
        print("\n❌ FAILURE: Output shape mismatch.")

if __name__ == "__main__":
    run_dummy_test()
