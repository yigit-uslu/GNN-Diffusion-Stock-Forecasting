from typing import Tuple
from torch_geometric.data import Data
import torch
import torch.nn as nn



def downsample_time(x: torch.Tensor, time_dim: int = 2, factor: int = 2
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Downsample the input tensor by the given factor along the time dimension
    selected_indices = torch.arange(0, x.size(time_dim), factor)
    x_downsampled = x.index_select(time_dim, selected_indices)
    x_downsampled_zeropadded = torch.zeros_like(x)
    x_downsampled_zeropadded.index_copy_(time_dim, selected_indices, x_downsampled)

    # Represent downsampling with a matrix
    Tin = x.size(time_dim)
    Tout = x_downsampled.size(time_dim)

    C = torch.zeros((Tout, Tin), dtype=torch.long)
    for i, idx in enumerate(selected_indices):
        C[i, idx] = 1
    # print("Downsampling matrix D:\n", D)

    return x_downsampled, x_downsampled_zeropadded, C


def temporal_conv1d(data: Data, L: int = 2, feature_dim: int = 1, time_dim: int = 2,
                    circular_shift: bool = True,
                    inspect_node_idx: int = 0) -> torch.Tensor:
    
    # Kernel size is L+1 to include the current and L past time steps
    # Padding is L//2 to maintain the same length after convolution
    # Use groups=data.x.size(feature_dim) for depthwise convolution (separate filter for each feature)
    conv = nn.Conv1d(in_channels=data.x.size(feature_dim),
                     out_channels=data.x.size(feature_dim),
                     groups=data.x.size(feature_dim),  # Depthwise convolution
                     stride=1,
                     kernel_size=L+1,
                     padding=L,  # (L // 2, L // 2), # pad only on the left for causal convolution
                     padding_mode='circular' if circular_shift else 'zeros'
                     )
    conv.weight.data.fill_(1.0)  # Simple averaging filter
    conv.bias.data.fill_(0.0)

    Y_conv = conv(data.x)[..., :data.x.size(time_dim)]  # Shape: (N, F_in, T)
    print(f"After Conv1d, Y_conv shape: {Y_conv.shape}")
    print(f"Y_conv[{inspect_node_idx}]: ", Y_conv[inspect_node_idx].flatten())  # Print the features of the inspected node
    
    return Y_conv


def temporal_maxpool1d(y: torch.Tensor, L: int = 2, time_dim: int = 2,
                       inspect_node_idx: int = 0) -> torch.Tensor:
    # Apply 1D max pooling along the time dimension
    pool = nn.MaxPool1d(kernel_size=L+1, stride=L, padding=L // 2)
    y_pooled = pool(y)

    print(f"After MaxPool1d, y_pooled shape: {y_pooled.shape}")
    print(f"y[{inspect_node_idx}]: ", y[inspect_node_idx].flatten())  # Print the features of the inspected node
    print(f"y_pooled[{inspect_node_idx}]: ", y_pooled[inspect_node_idx].flatten())  # Print the features of the inspected node

    return y_pooled


def temporal_conv(data: Data, L: int = 2, feature_dim: int = 1, time_dim: int = 2,
                  circular_shift: bool = True,
                  proj = nn.Identity(),
                  inspect_node_idx: int = 0) -> torch.Tensor:

    Y = torch.chunk(data.x, data.x.shape[time_dim], dim=time_dim)  # List of T tensors each of shape (N, F_in, 1)
    print(f"Split data.conv.x into {len(Y)} chunks.")
    for _ in range(len(Y)):
        print(f"Chunks Y[{_}] for node {inspect_node_idx}: ", Y[_][inspect_node_idx].flatten())  # Print first chunk

    Y_stacked = torch.zeros(*data.x.shape, L+1)  # Shape: (N, F_in, T, L)
    Y_stacked[..., 0] = torch.cat(Y, dim=time_dim)  # Original data at l=0
    for l in range(1, L+1):
        if circular_shift:
            # Shift chunks by 1 time step circularly
            Y = Y[-1:] + Y[:-1]  # Circular shift
        else:
            # Shift chunks by 1 time step with zero padding
            Y = (torch.zeros_like(Y[-1]),) + Y[:-1]  # Zero padding shift

        Y_stacked[..., l] = torch.cat(Y, dim=time_dim)  # Stack along new dimension

    assert Y_stacked.shape == (*data.x.shape, L+1), "Y_stacked shape mismatch."

    for l in range(L+1):
        print(f"Y_stacked[..., {l}][{inspect_node_idx}]: ", Y_stacked[..., l][inspect_node_idx].flatten())  # Print ($inspect_node_idx)th node's features for each shift

    # Apply projection to reduce L dimension
    Y_proj = proj(Y_stacked)  # Shape: (N, F_in, T)
    assert Y_proj.shape == data.x.shape, "Projected shape mismatch."

    return Y_proj




if __name__ == "__main__":

    with torch.inference_mode(True):

        L = 2  # Number of time shifts

        x = torch.arange(24).view(2, 3, 4).float()  # (N=2, F_in=3, T=4)
        data = Data(x = x)

        inspect_node_idx = 1

        print(f"Original data.x[{inspect_node_idx}]: ", data.x[inspect_node_idx])

        Z = temporal_conv1d(data = data, L = L, time_dim = 2, circular_shift = False,
                            inspect_node_idx = inspect_node_idx)
        
        W = temporal_maxpool1d(Z, L = L, time_dim = 2, inspect_node_idx = inspect_node_idx)

        # # Apply temporal convolution
        # Z = temporal_conv(data = data, L = L, time_dim = 2, circular_shift = False,
        #                   proj = lambda x: x.sum(dim = -1, keepdim = False),
        #                   inspect_node_idx = inspect_node_idx
        #                   )
        # print(f"Time-convolved data.x[{inspect_node_idx}]: ", Z[inspect_node_idx]) # Should show combined features from time shifts

        # downsampled_1, downsampled_zeropadded_1, C_1 = downsample_time(Z, time_dim=2, factor=2)

        # print(f"Downsampled_1 data.x[{inspect_node_idx}]: ", downsampled_1[inspect_node_idx])
        # print(f"Downsampled_1 (zero-padded) data.x[{inspect_node_idx}]: ", downsampled_zeropadded_1[inspect_node_idx])
        # print("Downsampling matrix C_1:\n", C_1)

        # downsampled_2, downsampled_zeropadded_2, C_2 = downsample_time(downsampled_1, time_dim=2, factor=2)

        # print(f"Downsampled_2 data.x[{inspect_node_idx}]: ", downsampled_2[inspect_node_idx])
        # print(f"Downsampled_2 (zero-padded) data.x[{inspect_node_idx}]: ", downsampled_zeropadded_2[inspect_node_idx])
        # print("Downsampling matrix C_2:\n", C_2)


        # print("Composite downsampling matrix C_2 @ C_1:\n", C_2 @ C_1)