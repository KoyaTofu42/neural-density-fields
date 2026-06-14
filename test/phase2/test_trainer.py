import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.ndf_model import NeuralDensityField
from src.training.trainer import DeltaTrainer
from src.physics.losses import PoissonLoss, NormalizationLoss
from src.physics.monte_carlo import UniformMonteCarloIntegrator

def test_trainer():
    print("="*60)
    print("Testing DeltaTrainer (Stage 1 and Stage 2)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}\n")
    
    # 1. Create Dummy Dataset
    dataset = []
    for _ in range(4): # 4 dummy molecules
        z = torch.tensor([6, 1, 1], dtype=torch.long)
        pos = torch.randn(3, 3)
        
        query_pos = torch.randn(50, 3)
        r = torch.norm(query_pos, dim=1)
        
        y = torch.exp(-r) + 0.1 # High-fidelity
        rho_low = torch.exp(-r) # Low-fidelity
        
        data = Data(z=z, pos=pos, query_pos=query_pos, y=y, rho_low=rho_low)
        dataset.append(data)
        
    # PyG DataLoader with follow_batch to automatically batch query_pos into query_pos_batch
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, follow_batch=['query_pos'])
    
    # 2. Initialize Model and Trainer
    model = NeuralDensityField(hidden_dim=32, k=3)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    trainer = DeltaTrainer(model, optimizer, dataloader, device=device)
    
    # 3. Test Stage 1: Pre-training
    print("[1] Running Pre-training Epoch (Stage 1)...")
    loss_stage1 = trainer.pretrain_epoch()
    print(f"  -> Stage 1 Loss: {loss_stage1:.4f}")
    
    # 4. Test Stage 2: Fine-tuning
    print("\n[2] Running Fine-tuning Epoch (Stage 2)...")
    poisson_loss_fn = PoissonLoss().to(device)
    norm_loss_fn = NormalizationLoss().to(device)
    mc_integrator = UniformMonteCarloIntegrator(margin=2.0, num_samples=100)
    
    loss_stage2 = trainer.finetune_epoch(poisson_loss_fn, mc_integrator, norm_loss_fn)
    print(f"  -> Stage 2 Loss: {loss_stage2:.4f}")
    
    print("\n✅ SUCCESS: DeltaTrainer executed both stages perfectly on batched graphs!")

if __name__ == "__main__":
    test_trainer()
