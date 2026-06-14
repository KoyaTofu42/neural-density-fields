import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import Data
try:
    from torch_geometric.nn import knn_interpolate
except ImportError:
    knn_interpolate = None

def nearest_neighbor_interpolate(x, pos_x, pos_y, batch_x, batch_y):
    """
    Fallback 1-Nearest Neighbor interpolator for when pyg-lib is not installed.
    """
    device = x.device
    out = torch.zeros(pos_y.size(0), x.size(1), device=device)
    
    num_graphs = int(batch_x.max().item()) + 1 if batch_x.numel() > 0 else 1
    
    for i in range(num_graphs):
        mask_x = (batch_x == i)
        mask_y = (batch_y == i)
        
        pos_x_i = pos_x[mask_x]
        x_i = x[mask_x]
        pos_y_i = pos_y[mask_y]
        
        if pos_x_i.size(0) == 0 or pos_y_i.size(0) == 0:
            continue
            
        dist = torch.cdist(pos_y_i, pos_x_i)
        min_idx = torch.argmin(dist, dim=1)
        out[mask_y] = x_i[min_idx]
        
    return out

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
            
            # Since density = Softplus(rho_low + delta), we penalize the prediction 
            # if it deviates from rho_low. This forces the model to learn the baseline first.
            target = data.rho_low if hasattr(data, 'rho_low') else torch.zeros_like(density)
            loss = self.mse_loss(density.squeeze(), target.squeeze())
            
            loss.backward()
            
            # Clip gradients to prevent early explosions
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
            
            # 1. Primary Forward Pass (Supervised Data & Poisson Loss)
            # Ensure query positions track gradients for the Laplacian
            if not data.query_pos.requires_grad:
                data.query_pos.requires_grad_(True)
                
            density, potential, query_pos = self.model(data)
            
            # Absolute density MSE loss against the high-fidelity target
            loss_data = self.mse_loss(density.squeeze(), data.y.squeeze())
            
            # Electrostatic Poisson Loss (Laplacian of potential = -4 * pi * density)
            loss_poisson = poisson_loss_fn(potential, density, query_pos)
            
            # 2. Secondary Forward Pass (Monte Carlo Normalization Loss)
            # Sample random points in the bounding box
            mc_query_pos, mc_query_batch, mc_volumes = mc_integrator.sample_points(data.pos, data.batch)
            
            # Because our Delta-Learning model requires the low-fidelity baseline (rho_low) as an input feature,
            # we must interpolate the baseline values from the given grid points to the random MC points.
            if hasattr(data, 'rho_low'):
                # Interpolate using 1-Nearest Neighbor
                batch_x = data.query_pos_batch if hasattr(data, 'query_pos_batch') else torch.zeros(data.query_pos.size(0), dtype=torch.long, device=self.device)
                
                mc_rho_low = nearest_neighbor_interpolate(
                    x=data.rho_low.unsqueeze(-1), 
                    pos_x=data.query_pos.detach(), 
                    pos_y=mc_query_pos, 
                    batch_x=batch_x,
                    batch_y=mc_query_batch
                ).squeeze(-1)
            else:
                # Fallback if baseline is unavailable
                mc_rho_low = torch.zeros(mc_query_pos.size(0), device=self.device)
                
            # Construct the mock batch for the MC queries
            mc_data = Data(
                z=data.z, 
                pos=data.pos, 
                batch=data.batch, 
                query_pos=mc_query_pos, 
                query_batch=mc_query_batch,
                rho_low=mc_rho_low
            )
            
            # Pass the random points through the model
            mc_density, _, _ = self.model(mc_data)
            
            # Compute the integral and the normalization loss
            N_elec_hat = mc_integrator.integrate(mc_density, mc_query_batch, mc_volumes)
            loss_norm = norm_loss_fn(N_elec_hat, data.z, data.batch)
            
            # 3. Composite Objective
            loss = loss_data + (lambda_poisson * loss_poisson) + (lambda_norm * loss_norm)
            
            loss.backward()
            
            # Strict gradient clipping for PINN stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.dataloader)
