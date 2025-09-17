import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, List


from torch_geometric.utils import (
    add_self_loops, remove_self_loops,
    scatter, to_torch_csr_tensor
)


DEBUG_MODE = True  # Set to True to enable shape printing for debugging
INSPECT_MODE = False  # Set to True to enable detailed value inspection
INSPECT_KWS = {"graph_idx": 0,
                  "node_idx": [2, 3],  # List of node indices to inspect
                  "feature_idx": [0, 1],  # List of feature indices to inspect
                  "time_idx": [0, 1, 2, 3]  # List of time indices to inspect
                  }


def inspect_tensor(x: torch.Tensor, dim_names: List[str] = ["graph", "node", "feature", "time"]):
    if INSPECT_MODE:
        shape = x.shape
        indices = [INSPECT_KWS.get(f"{dim}_idx", list(range(size))) for dim, size in zip(dim_names, shape)]
        for i, (index, dim) in enumerate(zip(indices, dim_names)):
            if isinstance(index, list):
                temp_index = [i for i in index if i < shape[dim_names.index(dim)]]
                indices[i] = temp_index

        slices = tuple(slice(idx[0], idx[-1]+1) if isinstance(idx, list) else slice(0, size) for idx, size in zip(indices, shape))
        print(f"Inspecting tensor of shape {shape} at slices {slices}:")
        print(x[slices])


def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)



def get_pooling_selection_matrices(degrees: torch.Tensor, pool_ratio: float, depth: int) -> Tuple[List[torch.Tensor], List[int]]:

    all_selection_matrices = []
    num_pooled_nodes_list = []

    selection_matrix = torch.eye(degrees.shape[0], dtype=torch.float32)
    orig_degrees = degrees.clone()
    orig_N = degrees.shape[0]
    N = orig_N
    
    for _ in range(depth): 

        # Create a pool of nodes based on the degree
        num_pooled_nodes = int(N * pool_ratio)
        _, indices = torch.topk(degrees, num_pooled_nodes,
                                        largest = True, sorted = False)
    
        print(f"Selected node indices (top {num_pooled_nodes} by degree): {indices.tolist()}")

        sorted_new_indices = torch.sort(indices).values
        degrees = degrees[sorted_new_indices]
        print(f"Sorted selected node indices: {sorted_new_indices.tolist()}")

        # Create a matrix that selects the pooled nodes among the original nodes
        new_selection_matrix = torch.zeros((num_pooled_nodes, N))
        for i, idx in enumerate(sorted_new_indices):
            new_selection_matrix[i, idx] = 1.0

        print("Selections: ", new_selection_matrix @ torch.arange(N, dtype = torch.float32).view(-1,))
        print("Expected: ", sorted_new_indices.float())
        # print("Original degrees of selected nodes: ", orig_degrees[sorted_new_indices].float())
        assert torch.allclose(new_selection_matrix @ torch.arange(N, dtype=torch.float32).view(-1,),
                            sorted_new_indices.float()), "Selection matrix is incorrect."
        

        selection_matrix = new_selection_matrix @ selection_matrix # cumulative selections
        print("Cumulative selections: ", selection_matrix @ torch.arange(orig_N, dtype=torch.float32).view(-1,))
        # print("Cumulative expected: ", sorted_indices[sorted_new_indices].float())

        # print("Original degrees of cumulatively selected nodes: ", orig_degrees[sorted_indices[sorted_new_indices]].float())
    

        print("\n*****************************************************\n"
            + f"Repeating the pooling to verify consistency...\n" + 
            "*****************************************************\n")

        all_selection_matrices.append(new_selection_matrix)
        num_pooled_nodes_list.append(num_pooled_nodes)
        N = num_pooled_nodes


    return all_selection_matrices, num_pooled_nodes_list




def get_nn_conv1d_parameters(in_channels, out_channels, kernel_size=3):
    """
    Utility function to determine the parameters for a 1D convolutional layer so that temporal input and output dimensions are preserved.
    """
    # Assert kernel size is valid.
    while in_channels <= kernel_size:
        kernel_size = kernel_size - 2
        if kernel_size < 1:
            raise ValueError("in_timesteps is too small to apply downsampling with conv1d. Try using a smaller kernel size or a different downsampling method.")

    temp_stride = in_channels // (out_channels) # (out_channels - 1) 
    # temp_in_channels = (out_channels + 1) * temp_stride
    temp_padding = (kernel_size - 1) // 2

    if in_channels == out_channels:
        print(f"Found parameters for [L_in, L_out] = {in_channels, out_channels}: ", {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding})
        return {"kernel_size": kernel_size, "stride": 1, "padding": (kernel_size - 1) // 2}


    temp_out_channels = calc_output_length(in_channels, kernel_size, temp_stride, temp_padding, dilation=1)

    if temp_out_channels == out_channels:
        print(f"Found parameters for [L_in, L_out] = {in_channels, out_channels}: ", {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding})
        return {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding}
    
    max_num_attempts = 10
    counter = 0
    while temp_out_channels != out_channels:
        counter += 1
        if counter == max_num_attempts:
            raise ValueError(f"Could not find suitable parameters for [L_in, L_out] = {in_channels, out_channels} after {max_num_attempts} attempts. Last tried parameters: ", {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding})
        else:
            if temp_out_channels > out_channels: 
                # Reduce the padding to decrease the output length
                temp_padding = max(0, temp_padding - 1)

            else:
                # Increase the padding to increase the output length
                temp_padding += 1

    print(f"Found parameters for [L_in, L_out] = {in_channels, out_channels}: ", {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding})
    return {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding}

    

def calc_output_length(L_in, kernel_size, stride, padding, dilation):
    """
    Calculate the output length of a 1D convolutional layer.
    L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
    """
    L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
    return L_out







def augment_adj(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                    num_nodes: int):
    """
    Augment the adjacency matrix by adding 2-hop connections.
    """
        
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                num_nodes=num_nodes)
    adj = to_torch_csr_tensor(edge_index, edge_weight,
                                size=(num_nodes, num_nodes))
    adj = (adj @ adj).to_sparse_coo()
    edge_index, edge_weight = adj.indices(), adj.values()
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    
    return edge_index, edge_weight




def pool_two_neighborhood_features(x, edge_index, edge_weight=None, pool = "max", threshold: float = None):
    """
    Perform 2-hop neighborhood pooling of node features.
    
    Args:
        x: Node feature matrix of shape (N, F_in)
        edge_index: Edge index tensor of shape (2, E)
        edge_weight: Optional edge weights tensor of shape (E,)
        k: Number of hops for neighborhood aggregation

    Returns:
        Pooled node feature matrix of shape (N, F_in)
    """

    # Filter out edges with edge_weight less than threshold
    if edge_weight is not None and threshold is not None:
        mask = edge_weight >= threshold
        print("Threshold masked {} percent of edges.".format(100 * (1 - mask.float().mean().item())))
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]
        print(f"Filtered edges with weight > {threshold}. New edge count: {edge_index.shape[1]}")

    edge_index, edge_weight = augment_adj(edge_index, edge_weight, num_nodes=x.size(0))

    return pool_neighboring_features(x, edge_index, edge_weight, pool=pool, threshold=None)






def pool_neighboring_features(x, edge_index, edge_weight=None, pool = "max",
                              threshold: float = None):
    """
    Perform max pooling of neighboring node features.
    
    Args:
        x: Node feature matrix of shape (N, F_in)
        edge_index: Edge index tensor of shape (2, E)
        edge_weight: Optional edge weights tensor of shape (E,)
    
    Returns:
        Pooled node feature matrix of shape (N, F_in)
    """
    print("Performing max pooling of neighboring features...")

    # print("x.shape: ", x.shape)
    # print("edge_index.shape: ", edge_index.shape)
    # if edge_weight is not None:
        # print("edge_weight.shape: ", edge_weight.shape)
    # else:

        # print("No edge weights provided.")

    # Filter out edges with edge_weight less than threshold
    if edge_weight is not None and threshold is not None:
        mask = edge_weight < threshold
        # print("Threshold masked {} percent of edges.".format(100 * (1 - mask.float().mean().item())))
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]
        # print(f"Filtered edges with weight > {threshold}. New edge count: {edge_index.shape[1]}")


    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    row, col = edge_index
    
    # Perform max pooling using scatter_max
    # pooled_x, _ = scatter_max(x[col], row, dim=0, dim_size=x.size(0))
    pooled_x = scatter(x[row], col, dim=0, dim_size=x.size(0), reduce='max' if pool == "max" else 'mean')

    # print("x.shape:", x.shape)
    # print("pooled_x.shape:", pooled_x.shape)

   
    return pooled_x, edge_index, edge_weight