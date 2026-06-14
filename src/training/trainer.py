import torch
import torch.nn as nn
from tqdm import tqdm

class DeltaTrainer:
    def __init__(self, model, optimizer, dataloader, device="cuda"):
        """
        Trainer implementing the 2-Stage Delta-Learning and PINN strategy.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.mse_loss = nn.MSELoss()
        
    def pretrain_epoch(self):
        """
        Stage 1: Train the network to output a delta of 0 (i.e. rho_final = rho_low).
        By pre-training on the extremely cheap low-fidelity baseline, we initialize 
        the neural network with basic chemical physics logic before activating complex losses.
        """
        self.model.train()
        total_loss = 0
        
        for data in tqdm(self.dataloader, desc="Pre-training (Stage 1)"):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Predict the final density
            density, potential, query_pos = self.model(data)
            
            # Since density = ReLU(rho_low + delta), we penalize the prediction 
            # if it deviates from rho_low. This forces delta -> 0.
            loss = self.mse_loss(density, data.rho_low)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.dataloader)
        
    def finetune_epoch(self, poisson_loss_fn, mc_integrator, norm_loss_fn, lambda_poisson=0.1, lambda_norm=0.1):
        """
        Stage 2: Fine-tune using High-Fidelity targets and Physics Constraints.
        """
        self.model.train()
        total_loss = 0
        
        for data in tqdm(self.dataloader, desc="Fine-tuning (Stage 2)"):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # 1. Standard Forward Pass (Data Loss & Poisson Loss)
            density, potential, query_pos = self.model(data)
            
            loss_data = self.mse_loss(density, data.y.unsqueeze(-1))
            loss_poisson = poisson_loss_fn(potential, density, query_pos)
            
            # 2. Monte Carlo Normalization Pass
            mc_query_pos, mc_query_batch, mc_volumes = mc_integrator.sample_points(data.pos, data.batch)
            
            # We must run a second forward pass specifically on the Monte Carlo points to integrate
            # Create a mock data object for the MC queries
            # (In a real implementation, this would require querying rho_low at the MC points too)
            # For brevity in this structural template, we omit the full secondary forward pass.
            loss_norm = torch.tensor(0.0, device=self.device)
            
            # 3. Composite Objective
            loss = loss_data + (lambda_poisson * loss_poisson) + (lambda_norm * loss_norm)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.dataloader)
