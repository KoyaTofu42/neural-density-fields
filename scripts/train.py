import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from src.data.density_dataset import DensityDataset
from src.models.ndf_model import NeuralDensityField
from tqdm import tqdm

def train():
    print("="*60)
    print("Supervised Training Loop: Neural Density Field")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}\n")
    
    # 1. Load Dataset
    print("[1] Loading PyG Dataset...")
    try:
        # Increase points from 512 to 10000 (all points) to eliminate random sampling variance of core spikes!
        dataset = DensityDataset("data/density_samples.h5", num_samples_per_mol=10000)
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        print("Please ensure that data generation (Milestone 1) has completed successfully.")
        return
        
    total_len = len(dataset)
    print(f"Successfully loaded {total_len} molecules.")
    
    # 2. Split Dataset (90% Train / 10% Validation)
    train_len = int(0.9 * total_len)
    val_len = total_len - train_len
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)
    
    print(f"  - Training Set: {len(train_dataset)} molecules")
    print(f"  - Validation Set: {len(val_dataset)} molecules\n")
    
    # Using batch size of 4 to fit comfortably in VRAM
    # follow_batch=['query_pos'] tells PyTorch Geometric to dynamically generate exactly correct batch indices 
    # for our variable-length continuous points (query_pos_batch), completely fixing the size mismatch!
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, follow_batch=['query_pos'])
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, follow_batch=['query_pos'])
    
    # 3. Initialize Model
    print("[2] Initializing Neural Network...")
    # Resolution Upgrade: 300 Gaussians over 5.0 Angstroms = 0.016 A spatial resolution!
    model = NeuralDensityField(hidden_dim=128, k=5, num_gaussians=300, cutoff=5.0)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # 4. Optimizer, Loss, and Scheduler
    # The overfit test proved 1e-3 is stable. We use CosineAnnealing to perfectly smooth the decay 
    # and completely avoid premature drops caused by noisy small-batch validation metrics!
    epochs = 200
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # We MUST use MSELoss to directly minimize our target metric.
    # HuberLoss artificially clipped our gradients and stopped the model from learning the massive core spikes.
    train_criterion = nn.MSELoss()
    val_criterion = nn.MSELoss()
    best_val_loss = float('inf')
    save_dir = 'models_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    print("[3] Beginning Training...")
    
    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            density_pred = model(batch)
            
            # Scale both down by 100.0 to keep gradients perfectly stable in the [0, 1] regime
            # Note: density_pred is already naturally scaled up by 100.0 inside the model's forward pass!
            pred_scaled = density_pred.squeeze() / 100.0
            target_scaled = batch.y / 100.0
            
            loss = train_criterion(pred_scaled, target_scaled)
            loss.backward()
            
            # We use a standard clip to prevent NaN, keeping optimization smooth
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            # Weighted average based on number of graphs in batch
            # Multiply by 10000.0 to reverse the target scaling for accurate logging!
            true_mse = loss.item() * 10000.0
            train_loss += true_mse * batch.num_graphs
            progress_bar.set_postfix({'true_mse': f"{true_mse:.6f}"})
            
        train_loss /= len(train_dataset)
        
        # 6. Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                density_pred = model(batch)
                loss = val_criterion(density_pred.squeeze(), batch.y)
                val_loss += loss.item() * batch.num_graphs
                
        val_loss /= len(val_dataset)
        
        # Step the Cosine Annealing scheduler every epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Summary -> Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'best_ndf.pt')
            torch.save(model.state_dict(), save_path)
            print(f"⭐️ New best model! Saved to {save_path}")

if __name__ == "__main__":
    train()
