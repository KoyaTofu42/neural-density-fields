import torch
from torch_geometric.data import Data


class OptimizedInferencer:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        # Apply torch.compile for graph fusion and overhead reduction
        try:
            print("Applying torch.compile(mode='reduce-overhead') ...")
            self.compiled_model = torch.compile(self.model, mode="reduce-overhead")
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

        total_queries = query_coords.shape[0]

        for i in range(0, total_queries, chunk_size):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            q_chunk = query_coords[i : i + chunk_size].to(self.device)
            actual_size = q_chunk.shape[0]

            # Pad to chunk_size to keep static shapes for CUDA Graphs
            if actual_size < chunk_size:
                pad_size = chunk_size - actual_size
                pad_tensor = torch.zeros(
                    (pad_size, 3), dtype=q_chunk.dtype, device=self.device
                )
                q_chunk = torch.cat([q_chunk, pad_tensor], dim=0)

            # Construct a lightweight batch without using the slow Batch.from_data_list loop
            b = Data(z=z_dev, pos=pos_dev, query_pos=q_chunk)
            # PyG expects a batch attribute for scatter operations. Since we process 1 molecule, it's all 0s.
            b.batch = torch.zeros(z_dev.shape[0], dtype=torch.long, device=self.device)
            b.query_pos_batch = torch.zeros(
                chunk_size, dtype=torch.long, device=self.device
            )
            b.ptr = torch.tensor(
                [0, z_dev.shape[0]], dtype=torch.long, device=self.device
            )

            density, _, _ = self.compiled_model(b)
            # Clone and slice back to the actual valid points
            preds.append(density[:actual_size].clone().cpu())

        return torch.cat(preds, dim=0).squeeze().numpy()
