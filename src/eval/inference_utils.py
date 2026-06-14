import torch
from torch_geometric.data import Data


class OptimizedInferencer:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        # Apply torch.compile for graph fusion
        # Note: We investigated CUDAGraphs (mode='reduce-overhead') but found it incompatible 
        # with PyG's dynamic radius_graph memory allocations inside the forward pass.
        try:
            print("Applying torch.compile for kernel fusion ...")
            self.compiled_model = torch.compile(self.model)
        except Exception as e:
            print(
                f"Warning: torch.compile failed or unsupported ({e}). Falling back to standard execution."
            )
            self.compiled_model = self.model

    @torch.no_grad()
    def predict_density(self, z, pos, query_coords, chunk_size=10000):
        """
        Fast batched inference avoiding PyG overhead in the inner loop.
        z: (N,) numpy/tensor
        pos: (N, 3) numpy/tensor
        query_coords: (Q, 3) numpy/tensor
        Returns: (Q,) scalar density predictions.
        """
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.long)
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos, dtype=torch.float)
        if not isinstance(query_coords, torch.Tensor):
            query_coords = torch.tensor(query_coords, dtype=torch.float)

        preds = []

        # Pre-allocate the base graph data to avoid repeated host-to-device transfers
        z_dev = z.to(self.device)
        pos_dev = pos.to(self.device)
        
        # Pre-allocate static PyG batching attributes so memory addresses remain constant for CUDAGraphs
        batch_tensor = torch.zeros(z_dev.shape[0], dtype=torch.long, device=self.device)
        query_pos_batch_tensor = torch.zeros(chunk_size, dtype=torch.long, device=self.device)
        ptr_tensor = torch.tensor([0, z_dev.shape[0]], dtype=torch.long, device=self.device)
        # Pre-allocate the query buffer so data_ptr() never changes
        q_chunk_buffer = torch.zeros((chunk_size, 3), dtype=torch.float, device=self.device)

        total_queries = query_coords.shape[0]

        for i in range(0, total_queries, chunk_size):
            q_chunk = query_coords[i : i + chunk_size]
            actual_size = q_chunk.shape[0]

            # Copy data into the static buffer to keep memory addresses identical
            q_chunk_buffer[:actual_size].copy_(q_chunk)
            if actual_size < chunk_size:
                q_chunk_buffer[actual_size:].zero_()

            # Construct a lightweight batch using perfectly static buffers
            b = Data(
                z=z_dev, 
                pos=pos_dev, 
                query_pos=q_chunk_buffer,
                batch=batch_tensor,
                query_pos_batch=query_pos_batch_tensor,
                ptr=ptr_tensor
            )
            
            out_density, out_potential, out_query = self.compiled_model(b)
            
            # Clone and slice back to the actual valid points
            preds.append(out_density[:actual_size].clone().cpu())
            
            # CRITICAL: Delete references to CUDAGraph outputs to prevent PyTorch overwrite protection error
            del out_density
            del out_potential
            del out_query

        return torch.cat(preds, dim=0).squeeze().numpy()
