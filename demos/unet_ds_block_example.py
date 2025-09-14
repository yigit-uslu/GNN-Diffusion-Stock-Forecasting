from typing import Tuple, Union, List
import torch
import torch.nn as nn
from torch_geometric.nn import TAGConv
import numpy as np
from collections import defaultdict


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


def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_nn_conv1d_parameters(in_channels, out_channels, kernel_size=3):
    """
    Utility function to determine the parameters for a 1D convolutional layer so that temporal input and output dimensions are preserved.
    """
    # Assert kernel size is valid.
    while in_channels <= kernel_size:
        kernel_size = kernel_size - 2
        if kernel_size < 1:
            raise ValueError("in_timesteps is too small to apply downsampling with conv1d. Try using a smaller kernel size or a different downsampling method.")

    if in_channels == out_channels:
        print(f"Found parameters for [L_in, L_out] = {in_channels, out_channels}: ", {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding})
        return {"kernel_size": kernel_size, "stride": 1, "padding": (kernel_size - 1) // 2}
    

    temp_stride = in_channels // (out_channels - 1) 
    # temp_in_channels = (out_channels + 1) * temp_stride
    temp_padding = (kernel_size - 1) // 2

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


class SwapAxes(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = torch.swapaxes(x, self.dim0, self.dim1)
        return x
    


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = TAGConv(in_channels=in_channels, out_channels=out_channels, K=2, normalize=False, aggr = "mean")
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index = None, edge_weight = None, batch = None):
        x = self.conv(x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.bn(x)
        x = self.act(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_timesteps, out_timesteps,
                 hidden_channels = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T_in = in_timesteps
        self.T_out = out_timesteps
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        # self.time_embed = nn.Linear(1, self.hid)
        # self.cond_embed = nn.Linear(1, self.hidden_channels)  # Simple feature embedding for demonstration
        # print("Conv_1 with in_channels: ", in_channels, " out_channels: ", hidden_channels)

        if out_timesteps == in_timesteps // 2:
            # Testing conv1d params utility function. 
        
            conv_params = get_nn_conv1d_parameters(self.T_in, self.T_out, kernel_size=5)
            self.conv1 = nn.Conv1d(self.in_channels,
                                   self.hidden_channels,
                                   kernel_size=conv_params["kernel_size"],
                                   padding=conv_params["padding"],
                                   stride=conv_params["stride"],
                                   )
            self.conv1.weight.data.fill_(1.0)  # Initialize weights to 1.0 for easier debugging
        elif out_timesteps == in_timesteps:
            self.conv1 = nn.Sequential(SwapAxes(2, 1), nn.Linear(self.in_channels, self.hidden_channels), SwapAxes(2, 1))
        else:
            raise ValueError("out_timesteps must be either  = in_timesteps // 2 or = in_timesteps.")
        
        # self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn1 = nn.LayerNorm([self.hidden_channels, self.T_out])
        self.act = nn.ReLU()

        self.temporal_conv_block = nn.Sequential(
            self.conv1,
            self.bn1,
            self.act
        )
        

        # self.conv2 = nn.Linear(out_channels, out_channels)
        self.graph_conv_block = GraphConvLayer(in_channels = self.hidden_channels, # out_channels,
                                                out_channels = self.out_channels)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm1d(out_channels)
        # self.graph_conv_block = nn.Sequential(
        #     self.conv2,
        #     # self.bn2,
        #     self.act
        # ) 

        if self.in_channels != self.out_channels or self.T_in != self.T_out:
            self.res_connection = nn.Sequential(nn.Linear(self.T_in, self.T_out), SwapAxes(2, 1), nn.Linear(self.in_channels, self.out_channels), SwapAxes(2,1))
            # self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_connection = nn.Identity()



    def forward(self, x, edge_index = None, edge_weight = None, batch = None,
                sel_mtrx: torch.Tensor = None,
                cond = None, t = None) -> torch.Tensor:

        assert sel_mtrx is None or sel_mtrx.dim() in [2, 3], "sel_mtrx must be of shape (N_out, N_in) or (B, N_out, N_in) if provided."

        debug_print("x.shape at start of ResidualBlock: ", x.shape)  # (B*N, F_in, T)
        residual = self.res_connection(x)  # (B*N, F_in, T) -> (B*N, F_out, T) if needed
        debug_print("residual.shape after linear projection if needed: ", residual.shape)  # (B*N, F_out, T)
        
        # Temporal convolution block
        debug_print("x.shape before temporal conv block: ", x.shape)  # (B*N, F, T)
        out = self.temporal_conv_block(x)
        debug_print("x.shape after temporal conv block: ", out.shape)  # (B*N, F, T)
        

        # # Handle any of the additional time embeddings
        # if cond is not None:
        #     emb = self.cond_embed(cond.unsqueeze(1).float())  # Shape: (B, out_channels)
        #     out = out + emb.unsqueeze(-1)  # Broadcast and add time embedding

        if t is not None:
            time_emb = self.time_embed(t.unsqueeze(1).float())  # Shape: (B, out_channels)
            out = out + time_emb.unsqueeze(-1)  # Broadcast and add time embedding

        #### Graph convolution happens w.r.t. zero-padded signal ####
        F, T = out.shape[1], out.shape[2]
        B = batch.max().item() + 1
        edge_index_t, edge_weight_t, batch_t = self.repeat_graph_for_timesteps(edge_index, edge_weight, batch, T)
        
        out = out.reshape(B, -1, F, T)  # (B*N, F, T) -> (B, N, F, T)
        inspect_tensor(out, dim_names=["batch", "node", "feature", "time"])
        out = torch.moveaxis(out, source = -1, destination = 0).reshape(B*T, -1, F)  # (B*N, F, T) -> (T, B*N, F) -> (T*B, N, F)

        # Upsample out to original node dimension if sel_mtrx is provided
        if sel_mtrx is not None:
            debug_print("sel_mtrx.shape before upsampling: ", sel_mtrx.shape)  # (N_0, N/2) or (B, N_0, N/2)
            sel_mtrx = sel_mtrx.t().unsqueeze(0).expand(B*T, -1, -1)  # (N_0, N) -> (B*T, N_0, N)
            out = torch.bmm(sel_mtrx, out)  # (B*T, N_0, N) @ (B*T, N, F) -> (B*T, N_0, F)
        
        out = out.reshape(-1, F)

        debug_print("out.shape before graph conv block: ", out.shape)  # (T*B*2*N, F)        
        
        out = self.graph_conv_block(out, edge_index = edge_index_t, edge_weight = edge_weight_t, batch = batch_t)
        debug_print("out.shape after graph conv block: ", out.shape)  # (T*B*2*N, F)

        F = out.shape[-1] # F might have changed after graph conv
        out = out.reshape(B*T, -1, F)  # (B*T, 2*N, F)

        if sel_mtrx is not None:
            # C = sel_mtrx.transpose(1, 2)  # (B*T, N, 2*N)
            out = torch.bmm(sel_mtrx.transpose(1, 2), out)  # (B*T, N, 2*N) @ (B*T, 2*N, F) -> (B*T, N, F)

        # edge_index, edge_weight, batch
        out = torch.moveaxis(out.reshape(T, -1, F), source = 0, destination = 2)  # ((B*T, N, F) -> (T, B*N, F) -> (B*N, F, T)
        debug_print("out.shape after reshaping back to (B*N, F, T): ", out.shape)  # (B*N, F, T)                         
        
        out += residual
        # out = self.relu(out)
        return out
    

    # Write a method to batch-repeat edge_index and edge_weight for T timesteps
    def repeat_graph_for_timesteps(self, edge_index, edge_weight, batch, T):
        # if edge_index is None:
        #     return None, None, None

        # E = edge_index.size(1)
        # if edge_weight is None:
        #     edge_weight = torch.ones(E, device=edge_index.device)

        # edge_indices = []
        # edge_weights = []
        # batches = []

        # num_graphs = batch.max().item() + 1
        # N = batch.size(0) // num_graphs  # Number of nodes per graph

        # for t in range(T):
        #     edge_indices.append(edge_index + t * N)
        #     edge_weights.append(edge_weight)
        #     batches.append(batch + t * num_graphs)

        # edge_index_t = torch.cat(edge_indices, dim=1)
        # edge_weight_t = torch.cat(edge_weights, dim=0)
        # batch_t = torch.cat(batches, dim=0)

        # return edge_index_t, edge_weight_t, batch_t
    

        # # Parallel processing of all timesteps through graph conv
        # # Strategy: Manual batching that's compatible with all PyG versions
        # B, F, T = x.shape
        # print("B, F, T = x.shape: ", B, F, T) 
        
        # # Method 1: Manual temporal batching with proper edge index offsetting
        # x_parallel = x.permute(2, 0, 1).reshape(B * T, F)  # (B, F, T) -> (T, B, F) -> (B*T, F)
        
        # Create edge indices for each timestep's subgraph
        edge_indices_list = []
        edge_weights_list = []
        
        for t in range(T):
            # Offset edge indices for timestep t: add t*B to reference correct nodes
            edge_index_t = edge_index + t * B
            edge_indices_list.append(edge_index_t)
            
            if edge_weight is not None:
                edge_weights_list.append(edge_weight)
        
        # Combine all timesteps' edge structures
        edge_index_parallel = torch.cat(edge_indices_list, dim=1)  # Shape: (2, T*num_edges)
        edge_weight_parallel = torch.cat(edge_weights_list) if edge_weight is not None else None
        

        # Create batch indices for the parallel processing
        if batch is not None:
            # Repeat batch indices for each timestep with proper offset
            batch_list = []
            for t in range(T):
                batch_t = batch + t * (batch.max() + 1)  # Offset to avoid overlap
                batch_list.append(batch_t)
            batch_parallel = torch.cat(batch_list)  # Shape: (B*T,)
        else:
            batch_parallel = None

        return edge_index_parallel, edge_weight_parallel, batch_parallel
    

    def __repr__(self):
        return  "\n\n********************************* Residual Block Summary **********************************\n\n" + f"ResidualBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, in_timesteps={self.T_in}, out_timesteps={self.T_out}):" \
        + f"\n\tConvBlock({self.temporal_conv_block}) [{self.in_channels, self.T_in}] -> [{self.hidden_channels, self.T_out}], \
              \n\tGraphConvBlock({self.graph_conv_block}) [{self.hidden_channels, self.T_out}] -> [{self.out_channels, self.T_out}]." \
              + f"\n\n**************************************** Residual Block Summary **********************************\n\n"


class NodeUpDownSamplingBlock(nn.Module):
    def __init__(self, N_in, ds_factor, N_out = None):
        super().__init__()
        self.N_in = N_in
        self.ds_factor = ds_factor
        assert N_in % ds_factor == 0, "N_in must be divisible by ds_factor."
        self.N_out = int(N_in // ds_factor) if N_out is None else N_out
        self.node_sel_mtrx, self.selected_indices = self.get_selection_matrix(N_in, self.N_out)


    def forward(self, x: torch.Tensor, batch = None, up_or_downsample = "down"):

        if up_or_downsample == "up":
            if self.node_sel_mtrx.dim() == 2:
                C = self.node_sel_mtrx.t()  # (N_in, N_out) -> (N_out, N_in)
            elif self.node_sel_mtrx.dim() == 3:
                C = self.node_sel_mtrx.permute(0, 2, 1)  # (B, N_in, N_out) -> (B, N_out, N_in)
        else:
            C = self.node_sel_mtrx  # (N_out, N_in) or (B, N_out, N_in)

        x_sel = self.batch_select_nodes(x, batch, C)  # (B*N_out, F) or (B*N_out, F, T)
        return x_sel, C

        
    
    def batch_select_nodes(self, x: torch.Tensor, batch: torch.Tensor, C: torch.Tensor):
        """
        Apply the selection matrix C to the batched input x based on the batch assignments.
        
        Parameters:
        - x: Input tensor of shape (B*N_in, F) or (B*N_in, F, T)
        - batch: Tensor of shape (B*N_in,) indicating graph membership for each node
        - C: Selection matrix of shape (N_out, N_in)
        
        Returns:
        - x_ds: Downsampled tensor of shape (B*N_out, F) or (B*N_out, F, T)
        """
        if x.dim() == 3:
            B, F, T = x.shape  # x shape: (B*N_in, F, T)
            F_times_T = F * T
            x = x.reshape(-1, F_times_T)  # (B*N_in, F*T)
        else:
            B, F_times_T = x.shape  # x shape: (B*N_in, F*T)

        num_graphs = batch.max().item() + 1
        N_in = C.size(1)

        x = x.reshape(num_graphs, -1, *x.shape[1:])
        C = C.to(x.device).expand(num_graphs, -1, -1)  # (num_graphs, N_out, N_in)

        x_ds = torch.bmm(C, x)  # (num_graphs, N_out, F)
        x_ds = x_ds.reshape(-1, F_times_T)
        return x_ds
    

    def get_selection_matrix(self, N_in, N_out):
        """
        Create a selection matrix C of shape (N_out, N_in) that selects N_out nodes from N_in nodes 
        by generating a random row permutation of the identity matrix and taking the first N_out rows in sorted order.
        """
        identity = torch.eye(N_in)
        selected_indices = torch.sort(torch.randperm(N_in)[:N_out]).values
        C = identity[selected_indices]
        
        return C, selected_indices
    


class Block(nn.Module):
    def __init__(self, in_channels, in_nodes, in_timesteps, ds_factor = 1, out_channels = None, out_nodes = None, out_timesteps = None):
        super().__init__()
        self.in_channels = in_channels
        self.in_timesteps = in_timesteps
        self.in_nodes = in_nodes
        self.out_channels = int(in_channels // ds_factor) if out_channels is None else out_channels
        self.out_timesteps = int(in_timesteps // ds_factor) if out_timesteps is None else out_timesteps
        self.out_nodes = int(in_nodes // ds_factor) if out_nodes is None else out_nodes


class DownBlock(Block):
    def __init__(self, in_channels, in_nodes, in_timesteps, ds_factor: Union[int, float] = 2, # float for upsampling
                 out_channels = None, out_nodes = None, out_timesteps = None,
                 depth: int = None):
        super().__init__(in_channels, in_nodes, in_timesteps, ds_factor, out_channels, out_nodes, out_timesteps)
        self.depth = depth

        self.graph_pool = NodeUpDownSamplingBlock(N_in = in_nodes, ds_factor = ds_factor,
                                                 N_out = out_nodes
                                                 )

        self.res_block = ResidualBlock(self.in_channels, self.out_channels, in_timesteps = self.in_timesteps, out_timesteps = self.out_timesteps)

        print(self.__repr__())


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None,
                sel_mtrx: torch.Tensor = None, cond = None) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.res_block(x, edge_index = edge_index, edge_weight = edge_weight, batch = batch, sel_mtrx = sel_mtrx, cond = cond)
        debug_print("x.shape after residual block: ", x.shape)  # (B*N_in, F_out, T_out)
        x, C = self.graph_pool(x, batch, up_or_downsample = "down")

        x = x.reshape(x.shape[0], self.out_channels, -1)
        return x, C
    

    def __repr__(self):
        return f"********************************************************************** Depth {self.depth} DownBlock Summary *********************************************************************\n\nDownBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, in_nodes={self.in_nodes}, out_nodes={self.out_nodes}, in_timesteps={self.in_timesteps}, out_timesteps={self.out_timesteps}, ds_factor={self.in_nodes/self.out_nodes}):" \
        + f"\n\tGraphPool({self.graph_pool.__class__.__name__}) [{self.in_nodes}] -> [{self.out_nodes}], \
              \n\tResidualBlock({self.res_block.__class__.__name__}) [{self.in_channels, self.in_timesteps}] -> [{self.out_channels, self.out_timesteps}]." \
    + f"\n\n\t\t{self.res_block.__repr__()}" + f"\n\n********************************************************************** Depth {self.depth} DownBlock Summary *********************************************************************\n\n"


class UpBlock(Block):
    def __init__(self, in_channels, in_nodes, in_timesteps, us_factor: int = 2, out_nodes = None, out_channels = None, out_timesteps = None,
                 graph_pool = None, depth: int = None):
        ds_factor = int(1 / us_factor)
        super().__init__(in_channels, in_nodes, in_timesteps, ds_factor = ds_factor, out_channels = out_channels, out_nodes = out_nodes, out_timesteps = out_timesteps)
        self.depth = depth

        self.res_block = ResidualBlock(self.in_channels, self.out_channels, in_timesteps = self.out_timesteps, out_timesteps = self.out_timesteps)
        
        if graph_pool is not None:
            self.graph_pool = graph_pool
        else:
            self.graph_pool = NodeUpDownSamplingBlock(N_in = in_nodes, ds_factor = ds_factor,
                                                     N_out = out_nodes
                                                     )
            

    def zero_interleave_time(self, x: torch.Tensor):
        x_zeros = torch.zeros(*x.shape[:-1], self.out_timesteps, device=x.device)
        x_zeros[..., ::self.out_timesteps // self.in_timesteps] = x
        return x_zeros
    

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None,
                sel_mtrx: torch.Tensor = None, cond = None) -> torch.Tensor:
        x, _ = self.graph_pool(x, batch, up_or_downsample = "up")
        x = x.reshape(x.shape[0], self.in_channels, -1)

        x = self.zero_interleave_time(x)
        x = self.res_block(x, edge_index = edge_index, edge_weight = edge_weight, batch = batch, sel_mtrx = sel_mtrx, cond = cond)
        
        return x
    

    def __repr__(self):
        return f"********************************************************************** Depth {self.depth} UpBlock Summary *********************************************************************\n\nUpBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, in_nodes={self.in_nodes}, out_nodes={self.out_nodes}, in_timesteps={self.in_timesteps}, out_timesteps={self.out_timesteps}, ds_factor={self.in_nodes/self.out_nodes}):" \
              + f"\n\tResidualBlock({self.res_block.__class__.__name__}) [{self.in_channels, self.in_timesteps}] -> [{self.out_channels, self.out_timesteps}]." \
              + f"\n\tGraphPool({self.graph_pool.__class__.__name__}) [{self.in_nodes}] -> [{self.out_nodes}]" \
              f"\n\n********************************************************************** Depth {self.depth} UpBlock Summary *********************************************************************\n\n"



class MiddleBlock(nn.Module):
    def __init__(self, in_channels, in_timesteps):
        super().__init__()
        self.in_channels = in_channels
        self.in_timesteps = in_timesteps

        self.block = nn.Sequential(
            nn.Linear(in_channels * in_timesteps, in_channels * in_timesteps),
            nn.ReLU(),
            nn.Linear(in_channels * in_timesteps, in_channels * in_timesteps)
        )

    def forward(self, x: torch.Tensor):
        B, F, T = x.shape
        x = x.reshape(B, F * T)
        x = self.block(x)
        x = x.reshape(B, F, T)
        return x

    

if __name__ == "__main__":

    seed_everything(seed=42)
    B = 2
    N_in = 16
    T_in = 16
    downsample_factor = 2
    F_in = 8
    # N_out = 4
    x = torch.arange(0, N_in).view(N_in, 1).float()  # (N=16, F_in=1)
    x = x.repeat(B, F_in)  # (N=32, F_in=4)

    x = x + torch.randn_like(x) * 0.01  # Add small noise for variability

    # Create a random adjacency matrix for a simple undirected graph with N_in nodes
    edge_index = torch.randint(0, N_in, (2, B * N_in))  # 32 random edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make it undirected
    adj = torch.zeros((N_in, N_in))
    adj[edge_index[0], edge_index[1]] = 1
    adj.fill_diagonal_(0)  # No self-loops

    # Get edge index and edge weight from adjacency matrix using pyG utility
    from torch_geometric.utils import dense_to_sparse
    edge_index, edge_weight = dense_to_sparse(adj.unsqueeze(0).repeat(B, 1, 1)) 

    batch = torch.arange(B).view(-1, 1).repeat(1, N_in).view(-1)  # (B*N_in,)

    with torch.inference_mode(True):
        
        ### Test node subsampling block ###
        ns_block = NodeUpDownSamplingBlock(N_in=N_in, ds_factor=downsample_factor)
        x_tilde, x_sel_mtrx = ns_block(x, batch, up_or_downsample="down")  # (B*N_out, F_in)

        debug_print("\nx: \n", x)
        debug_print("\nx_tilde: \n", x_tilde)
        debug_print("\nx_sel_mtrx: \n", x_sel_mtrx)

        ### Add a time dimension to x for temporal convs ###
        x = x.unsqueeze(2).repeat(1, 1, T_in)  # (N, F_in, T=8)


        ############ Create down and upsampling blocks ############
        F_0, N_0, T_0 = F_in, N_in, T_in
        F_min, N_min, T_min = 2, 2, 4  # Minimum feature, node, and timestep dimensions to downsample to
        ds_blocks = nn.ModuleList()
        us_blocks = nn.ModuleList()
        for depth in range(1, 4):
            F_1 = max(F_min, F_0 // downsample_factor)
            N_1 = max(N_min, N_0 // downsample_factor)
            T_1 = max(T_min, T_0 // downsample_factor)
            ds_block = DownBlock(in_channels=F_0, in_nodes=N_0, in_timesteps=T_0, 
                                 out_channels=F_1, out_nodes=N_1, out_timesteps=T_1,
                                 ds_factor=downsample_factor, depth=depth)
            ds_blocks.append(ds_block)

            us_block = UpBlock(in_channels=F_1, out_channels=F_0,
                               in_nodes=N_1, out_nodes=N_0, in_timesteps=T_1, out_timesteps=T_0,
                               graph_pool=ds_block.graph_pool, depth=depth)  # Share the same graph pooling layer
            us_blocks.append(us_block)

            F_0 = F_1
            N_0 = N_1
            T_0 = T_1


        middle_block = MiddleBlock(in_channels=F_0, in_timesteps=T_0)
        ############ Create down and upsampling blocks ############

        x_ds = x.clone()
        
        sel_mtrx = torch.eye(N_in)  # Start with identity selection matrix
        shortcuts = defaultdict(list)
        for depth, block in enumerate(ds_blocks, 1):
            print(f"\n\n--- Testing Downblock with depth {depth} ---\n\n")
            shortcuts["C"].append(sel_mtrx.clone())

            # downsample_block = DownBlock(in_channels=F_0, in_nodes=N_0, in_timesteps=T_0, ds_factor=downsample_factor)
            x_ds, C = block(x = x_ds, edge_index = edge_index, edge_weight = edge_weight, batch = batch, sel_mtrx = sel_mtrx if depth > 1 else None)

            sel_mtrx = C @ sel_mtrx
            debug_print("\n\n\n\nComposed sel_mtrx shape: ", sel_mtrx.shape)
            debug_print("Composed sel_mtrx: \n", sel_mtrx, "\n\n\n\n")

            # F_0 = F_0 // downsample_factor
            # N_0 = N_0 // downsample_factor
            # T_0 = T_0 // downsample_factor
            print("\nOriginal x shape: ", x.shape)
            print("Downsampled x shape: ", x_ds.shape)

            shortcuts["x"].append(x_ds.clone())
            # shortcuts["C"].append(sel_mtrx.clone())


        x_ds = middle_block(x_ds)
        print("x_ds shape after middle block: ", x_ds.shape)

        # desel_mtrx = shortcuts["C"][-1] 
        for i, block in enumerate(us_blocks[::-1], 1):
            depth = len(us_blocks) - i + 1
            print(f"\n\n--- Testing Upblock with depth {depth} ---\n\n")

            # Get the corresponding shortcut and selection matrix
            x_shortcut = shortcuts["x"][-i]
            C_shortcut = shortcuts["C"][-i]   # -i because upblock at each depth first unpools/upsamples the graph dimension, then adds the skip connection

            debug_print("sel_mtrx.shape: ", sel_mtrx.shape)
            debug_print("C_shortcut shape: ", C_shortcut.shape)

            ### Add/concatenate the skip connection ###
            x_ds = x_ds + x_shortcut

            x_ds = block(x = x_ds, edge_index = edge_index, edge_weight = edge_weight, batch = batch, sel_mtrx = C_shortcut.to(x_ds.device), cond = None)

           



            # ### Upsample x_ds using the transpose of the selection matrix ###
            # B = x_ds.shape[0] // (N_in // (downsample_factor ** depth))
            # F_times_T = x_ds.shape[1] * x_ds.shape[2]
            # x_ds = x_ds.reshape(B, -1, F_times_T)  # (B*N_out, F*T)
            # C_expanded = C_shortcut.to(x_ds.device).expand(B, -1, -1)  # (B, N_in, N_out)
            # x_ds = torch.bmm(C_expanded.transpose(1, 2), x_ds)  # (B, N_in, F*T)
            # # x_ds = x_ds.reshape(-1, x_ds.shape[1], x_ds.shape[2])  # (B*N_in, F, T)
            # x_ds = x_ds.reshape(-1, F, T)


        # """"""""""""""""""""""""" Middle of the UNet """""""""""""""""""""""""
        # print("x_ds entering middle of UNet shape: ", x_ds.shape)
        # # F, T = x_ds.shape[1], x_ds.shape[2]
        # # middle_block = MiddleBlock(in_channels=F*T, in_timesteps=T)
    

        # # x_ds = middle_block(x_ds.reshape(x_ds.shape[0], -1)).reshape(x_ds.shape[0], F, T)
        # print("x_ds exiting middle of UNet shape: ", x_ds.shape)

        # ### Upsample x_ds using the transpose of the selection matrix ###
        # x_shortcut = shortcuts["x"][-1]
        # C_shortcut = shortcuts["C"][-1]
        # print("x_shortcut shape: ", x_shortcut.shape)
        # print("C_shortcut shape: ", C_shortcut.shape)

        # ### Add/concatenate the skip connection ###
        # x_ds = x_ds + x_shortcut

        # B = x_ds.shape[0] // (N_in // (downsample_factor ** depth))
        # F_times_T = x_ds.shape[1] * x_ds.shape[2]
        # x_ds = x_ds.reshape(B, -1, F_times_T)  # (B*N_out, F*T)
        # C_expanded = C_shortcut.to(x_ds.device).expand(B, -1, -1)  # (B, N_in, N_out)
        # x_ds = torch.bmm(C_expanded.transpose(1, 2), x_ds)  # (B, N_in, F*T)
        # # x_ds = x_ds.reshape(-1, x_ds.shape[1], x_ds.shape[2])  # (B*N_in, F, T)
        # x_ds = x_ds.reshape(-1, F, T)

        # """"""""""""""""""""""""" Middle of the UNet """""""""""""""""""""""""




        # print("\n************************************************\n")
        # print("************************************************\n")
        # print("x_ds before entering residual block with node upsampling: ", x_ds.shape)
        # print("************************************************\n")
        # print("************************************************\n")

        # x_zeros = torch.zeros(*x_ds.shape[:-1], x_ds.shape[-1] * downsample_factor, device=x_ds.device)
        # x_zeros[..., ::downsample_factor] = x_ds

        # res_block = ResidualBlock(in_channels=x_zeros.shape[1], out_channels=x_zeros.shape[1] * downsample_factor,
        #                           in_timesteps=x_zeros.shape[2], out_timesteps=x_zeros.shape[2]
        #                           )
        
        # x_ds = res_block(x_zeros)
        # print("\n************************************************\n")
        # print("************************************************\n")
        # print("x_ds after exiting residual block with time upsampling: ", x_ds.shape)
        # print("************************************************\n")
        # print("************************************************\n")


        # for depth in range(len(shortcuts["x"]) - 1, 0, -1):
        #     print(f"\n\n--- Testing UpsampleBlock with depth {depth} ---\n\n")
        #     # Retrieve the corresponding shortcut and selection matrix
        #     x_shortcut = shortcuts["x"][depth-1]
        #     C_shortcut = shortcuts["C"][depth-1]

        #     print("x_ds shape before upsampling: ", x_ds.shape)
        #     print("x_shortcut shape for skip connection: ", x_shortcut.shape)
        #     print("C_shortcut shape for upsampling: ", C_shortcut.shape)

        #     x_ds = x_ds + x_shortcut
        #     print("x_ds shape after adding skip connection: ", x_ds.shape)

        #     B = x_ds.shape[0] // (N_in // (downsample_factor ** depth))
        #     F, T = x_ds.shape[1], x_ds.shape[2]
        #     x_ds = x_ds.reshape(B, -1, F * T)  # (B*N_out, F*T)
        #     C_expanded = C_shortcut.to(x_ds.device).expand(B, -1, -1)  # (B, N_in, N_out)
        #     x_ds = torch.bmm(C_expanded.transpose(1, 2), x_ds)  # (B, N_in, F*T)
        #     # x_ds = x_ds.reshape(-1, x_ds.shape[1], x_ds.shape[2])  # (B*N_in, F, T)
        #     x_ds = x_ds.reshape(-1, F, T)



        #     print("\n************************************************\n")
        #     print("************************************************\n")
        #     print("Depth ", depth, " x_ds before entering residual block with node upsampling: ", x_ds.shape)
        #     print("************************************************\n")
        #     print("************************************************\n")

        #     x_zeros = torch.zeros(*x_ds.shape[:-1], x_ds.shape[-1] * downsample_factor, device=x_ds.device)
        #     x_zeros[..., ::downsample_factor] = x_ds

        #     res_block = ResidualBlock(in_channels=x_zeros.shape[1], out_channels=x_zeros.shape[1] * downsample_factor,
        #                             in_timesteps=x_zeros.shape[2], out_timesteps=x_zeros.shape[2]
        #                             )
            
        #     x_ds = res_block(x_zeros)
        #     print("\n************************************************\n")
        #     print("************************************************\n")
        #     print("Depth ", depth, " x_ds after exiting residual block with time upsampling: ", x_ds.shape)
        #     print("************************************************\n")
        #     print("************************************************\n")

            


    




        # ### Test upsampling blocks ###
        # for depth in range(len(shortcuts["x"]), 0, -1):
        #     print(f"\n\n--- Testing UpsampleBlock with depth {depth} ---\n\n")
        #     # Retrieve the corresponding shortcut and selection matrix
        #     x_shortcut = shortcuts["x"][depth - 1]
        #     C_shortcut = shortcuts["C"][depth - 1]

        #     print("x_ds shape before upsampling: ", x_ds.shape)
        #     print("x_shortcut shape for skip connection: ", x_shortcut.shape)
        #     print("C_shortcut shape for upsampling: ", C_shortcut.shape)

        #     ### Upsample x_ds using the transpose of the selection matrix ###
        #     B = x_ds.shape[0] // (N_in // (downsample_factor ** depth))
        #     F_times_T = x_ds.shape[1] * x_ds.shape[2]
        #     x_ds_reshaped = x_ds.reshape(B, -1, F_times_T)  # (B*N_out, F*T)
        #     C_expanded = C_shortcut.to(x_ds.device).expand(B, -1, -1)  # (B, N_in, N_out)
        #     x_temp = torch.bmm(C_expanded.transpose(1, 2), x_ds_reshaped)  # (B, N_in, F*T)
        #     x_temp = x_temp.reshape(-1, x_ds.shape[1], x_ds.shape[2])  # (B*N_in, F, T)

        #     # Add a simple upsampling in the temporal dimesion by adding zeros between timesteps
        #     x_upsampled = torch.zeros(*x_upsampled.shape[:-1], x_upsampled.shape[-1] * downsample_factor, device=x_temp.device)
        #     x_upsampled[..., ::downsample_factor] = x_temp

        #     print("x_upsampled shape after upsampling: ", x_upsampled.shape)

        #     ### Add/concatenate the skip connection ###
        #     x_ds = x_upsampled + x_shortcut
        #     print("x_ds shape after adding skip connection: ", x_ds.shape)

        #     num_graphs = x_ds.shape[0] // B
        #     x_ds = x_ds.reshape(num_graphs, -1, *x_ds.shape[1:])
            
        #     x_ds = torch.bmm(C_shortcut.transpose(1, 2), x_ds)  # (num_graphs, N_out, F)
        #     x_ds = x_ds.reshape(-1, F_times_T)


        #     ### Pass through a residual block to refine ### 
        #     res_block = ResidualBlock(in_channels=x_ds.shape[1], out_channels=x_ds.shape[1] * downsample_factor,
        #                               in_timesteps=x_ds.shape[2], out_timesteps=x_ds.shape[2]
        #                               )
            
        #     x_ds = res_block(x_ds)
        #     print("x_ds shape after residual block: ", x_ds.shape)
            