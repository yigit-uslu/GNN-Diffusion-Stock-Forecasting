import torch
import numpy as np

def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_selection_matrix(N_in, N_out):
    """
    Create a selection matrix C of shape (N_out, N_in) that selects N_out nodes from N_in nodes 
    by generating a random row permutation of the identity matrix and taking the first N_out rows in sorted order.
    """
    identity = torch.eye(N_in)
    selected_indices = torch.sort(torch.randperm(N_in)[:N_out]).values
    C = identity[selected_indices]
    
    return C, selected_indices


def get_downsampled_k_shift_matrix(C: torch.Tensor, S: torch.Tensor, k: int):

    """
    Create a downsampling matrix D that first shifts the input by k using the shift matrix S,
    then applies the selection matrix C to downsample.

    Args:
        C (torch.Tensor): Selection matrix of shape (N_out, N_in).
        S (torch.Tensor): Shift matrix of shape (N_in, N_in).
        k (int): Number of shifts to apply.

    Returns:
        S_prime (torch.Tensor): Downsampled k-shift matrix of shape (N_out, N_out).
    """
    S_k = torch.matrix_power(S, k)  # Shift matrix raised to the power of k
    S_prime = C @ S_k @ C.T  # Apply shift followed by selection
    return S_prime



# Modify the forward pass of TAGConv to use the downsampled k-shift matrix
from torch import Tensor
from torch_geometric.nn import TAGConv
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
# Convert adj to SparseTensor for efficiency
from torch_geometric.utils import to_torch_csr_tensor


class DownsampledTAGConv(TAGConv):
    def __init__(self, in_channels, out_channels, K = 3, bias = True, normalize = True, **kwargs):
        super().__init__(in_channels, out_channels, K, bias, normalize, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    improved=False, add_self_loops=False, flow=self.flow,
                    dtype=x.dtype)

            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    add_self_loops=False, flow=self.flow, dtype=x.dtype)

        out = self.lins[0](x)
        for lin in self.lins[1:]:
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            out = out + lin.forward(x)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        print("Using custom message function")
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        print("Using custom message_and_aggregate with spmm")
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')



if __name__ == "__main__":

    seed_everything(seed=42)

    N_in = 16
    downsample_factor = 2
    # N_out = 4
    x = torch.arange(0, N_in).view(N_in, 1).float()  # (N=16, F_in=1)

    # Create a random adjacency matrix for a simple undirected graph with N_in nodes
    edge_index = torch.randint(0, N_in, (2, 32))  # 32 random edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make it undirected
    adj = torch.zeros((N_in, N_in))
    adj[edge_index[0], edge_index[1]] = 1
    adj.fill_diagonal_(0)  # No self-loops


    with torch.inference_mode(True):

        conv = DownsampledTAGConv(in_channels=1, out_channels=1, K=3, normalize=False)

        # Get edge_weights from adj 
        edge_weight = adj[edge_index[0], edge_index[1]]
        
        # Convert adj to sparse tensor for efficiency
        adj_csr = to_torch_csr_tensor(edge_index = edge_index, edge_attr = edge_weight, size=(N_in, N_in))


        N_out = N_in // downsample_factor
        C_1, sel_idx_1 = get_selection_matrix(N_in, N_out)
        C_2, sel_idx_2 = get_selection_matrix(N_out, N_out // downsample_factor)

        # Subsample x by composing selection matrices
        x_downsampled = C_1 @ x  # First downsampling
        print(f"Downsampled x after first downsampling of indices {sel_idx_1}:\n", x_downsampled)
        x_downsampled = C_2 @ x_downsampled  # Second downsampling
        print(f"Downsampled x after second downsampling of indices {sel_idx_2}:\n", x_downsampled)


        # Upsample x_downsampled by transposing selection matrices
        x_upsampled_zero_padded = C_2.T @ x_downsampled  # First upsampling
        print(f"Upsampled x after first upsampling:\n", x_upsampled_zero_padded)
        x_upsampled_zero_padded = C_1.T @ x_upsampled_zero_padded  # Second upsampling
        print(f"Upsampled x after second upsampling:\n", x_upsampled_zero_padded)

        # Shif the upsampled x by k=3 using the shift matrix S and then downsample using C_1 and C_2
        k=3
        x_upsampled_zero_padded_shifted = torch.matrix_power(adj, k) @ x_upsampled_zero_padded  # Shift by 3
        print(f"y = S^{k} x): \n", x_upsampled_zero_padded_shifted)

        x_upsampled_zero_padded_shifted_downsampled = C_2 @ (C_1 @ x_upsampled_zero_padded_shifted)  # Downsample the shifted signal
        print(f"x_shifted then downsampled: \n", x_upsampled_zero_padded_shifted_downsampled)

        # Now use the downsampled k-shift matrix to achieve the same result
        S_prime_1 = get_downsampled_k_shift_matrix(C_2 @ C_1, adj, k)
        x_downsampled_shifted = S_prime_1 @ x_downsampled  # Shift the downsampled signal
        print(f"x_downsampled_then_shifted: \n", x_downsampled_shifted)

        # Compare x_shifted_downsampled with x_downsampled_shifted
        assert torch.allclose(x_upsampled_zero_padded_shifted_downsampled, x_downsampled_shifted, atol=1e-5), "Mismatch between downsampled shifted signals!"
        print("Success: Downsampled shifted signals match!")


        # Convolve upshifted zero-padded signal using sparse adjacency
        y = conv(x_upsampled_zero_padded, adj_csr)
        y = C_2 @ (C_1 @ y)  # Downsample the convolved signal
        print("y: ", y)
