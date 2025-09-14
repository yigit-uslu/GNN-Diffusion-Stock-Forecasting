from collections import defaultdict
import os
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import TAGConv
from typing import Tuple, Optional, Union, List
from models.unet_embedding_utils import ConditionalEmbeddingLayer, Squeeze, SwapAxes
from models.unet_utils import get_nn_conv1d_parameters, debug_print, inspect_tensor


    

# class SwapAxes(nn.Module):
#     def __init__(self, dim0: int, dim1: int):
#         super().__init__()
#         self.dim0 = dim0
#         self.dim1 = dim1

#     def forward(self, x):
#         x = torch.swapaxes(x, self.dim0, self.dim1)
#         return x


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = TAGConv(in_channels=in_channels, out_channels=out_channels, K=2, normalize=False, aggr = "mean")
        self.norm = nn.LayerNorm(out_channels) # nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU()

        # self.conv2 = TAGConv(in_channels=out_channels, out_channels=out_channels, K=2, normalize=False, aggr = "mean")
        # self.norm = nn.LayerNorm(out_channels) # nn.BatchNorm1d(out_channels)
        # self.act = nn.LeakyReLU()

        edge_dim = 3
        self.edge_mlp = nn.Sequential(nn.Linear(edge_dim, 1), Squeeze(1), nn.SiLU())


    def forward(self, x, edge_index = None, edge_weight = None, batch = None):

        # edge_weight: [num_edges, edge_dim]
        if edge_weight is not None and edge_weight.dim() == 2:
            edge_weight = self.edge_mlp(edge_weight)  # [num_edges]
        # else:
        #     edge_weight = edge_weight  # [num_edges] or None

        x = self.conv(x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.act(x)
        x = self.norm(x)

        # x = self.conv2(x, edge_index=edge_index, edge_weight=edge_weight)
        # x = self.act(x)
        # x = self.norm(x)

        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_timesteps, out_timesteps,
                 hidden_channels = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels # this should be halved if conditioning is concatenated to the output later
        self.T_in = in_timesteps
        self.T_out = out_timesteps
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.cond_embed_strategy = "concat" # "add" or "concat"
        graph_conv_in_channels = self.hidden_channels * 2 if self.cond_embed_strategy == "concat" else self.hidden_channels

        # if out_timesteps == in_timesteps // 2:
            # Testing conv1d params utility function. 
        
        if self.T_in == self.T_out:
            # No need for convolution in time dimension if temporal dimensions are the same
            print("No temporal dimension change in ResidualBlock, using linear projection for temporal dimension.")
            if self.in_channels == self.hidden_channels:
                self.conv1 = nn.Identity()
            else:
                self.conv1 = nn.Linear(self.in_channels, self.hidden_channels)
        else:
            conv_params = get_nn_conv1d_parameters(self.T_in, self.T_out, kernel_size=5)
            self.conv1 = nn.Conv1d(self.in_channels,
                                    self.hidden_channels,
                                    kernel_size=conv_params["kernel_size"],
                                    padding=conv_params["padding"],
                                    stride=conv_params["stride"],
                                    )
            # self.conv1.weight.data.fill_(1.0)  # Initialize weights to 1.0 for easier debugging
        # elif out_timesteps == in_timesteps:
        #     self.conv1 = nn.Sequential(SwapAxes(2, 1), nn.Linear(self.in_channels, self.hidden_channels), SwapAxes(2, 1))
        # else:
        #     raise ValueError("out_timesteps must be either  = in_timesteps // 2 or = in_timesteps.")
        
        # self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn1 = nn.LayerNorm([self.hidden_channels, self.T_out])
        self.act = nn.ReLU()

        self.temporal_conv_block = nn.Sequential(
            self.conv1,
            self.act,
            self.bn1,
        )
        
        # self.conv2 = nn.Linear(out_channels, out_channels)
        self.graph_conv_block = GraphConvLayer(in_channels = graph_conv_in_channels, # out_channels,
                                                out_channels = self.out_channels)

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

        F = out.shape[-1] # F might have changed after concatenation
        out = out.reshape(-1, F)

        # Add conditioning logic here 
        if t is not None: 
            debug_print(f"t.shape: {t.shape}")  # (B * N_0, F, 1)
            assert t.shape[1] == F, f"Feature dimensions of t and out must match. {t.shape[1]}-{F}"
            
            t = t.reshape(B, -1, F, 1).repeat(1, 1, 1, T)
            t = torch.moveaxis(t, source = -1, destination=0).reshape(B*T, -1, F)  # (B*N_0, F, T)

            t = torch.bmm(sel_mtrx.transpose(2, 1), t)  # (B*T, N, N_0) @ (B*T, N_0, 1) -> (B*T, N, 1)
            t = torch.bmm(sel_mtrx, t)  # (B*T, N_0, N) @ (B*T, N, 1) -> (B*T, N_0, 1)

            debug_print(f"Adding sinusoidal time embedding t to out in the zero-padded-domain.")
            out = out + t.reshape(-1, F)  # (B*T, N_0, F) + (B*T, N_0, F)


        if cond is not None: # [B*N_0, F, T]
            debug_print(f"cond.shape: {cond.shape}")
            F_cond, T_cond = cond.shape[1], cond.shape[2]
            assert T == T_cond, f"Temporal dimensions of cond and out must match. {T_cond}-{T}"
            
            cond = cond.reshape(B, -1, F_cond, T_cond)  # (B*N_0, F, T) -> (B, N_0, F, T)
            debug_print(f"cond.shape: {cond.shape}")

            cond = torch.moveaxis(cond, source = -1, destination = 0).reshape(B*T_cond, -1, F_cond) 
            # cond = torch.bmm(sel_mtrx, cond)  # (B*T, N_0, N) @ (B*T, N, F_cond) -> (B*T, N_0, F_cond)
            # debug_print(f"cond.shape: {cond.shape} after upsampling.")
            
            # First select the relevant nodes and then zeropad back to original node dimension
            cond = torch.bmm(sel_mtrx.transpose(2, 1), cond)  # (B*T, N, N_0) @ (B*T, N_0, F_cond) -> (B*T, N, F_cond)
            cond = torch.bmm(sel_mtrx, cond)
            
            cond = cond.reshape(-1, F_cond)

            debug_print("Applying conditional embedding in the zero-padded-domain.")
            debug_print(f"cond.shape: {cond.shape}, out.shape: {out.shape}")
            if self.cond_embed_strategy == "concat":
                out = torch.cat([out, cond], dim=-1)  # Concatenate along feature dimension
            else:
                out = out + cond  # Add conditional embedding


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
        
        out =  out + residual
        # out = self.relu(out)
        return out
    

    def repeat_graph_for_timesteps(self, edge_index, edge_weight, batch, T):
        """
        Repeat the graph structure for T timesteps to create a block-diagonal adjacency matrix.
        Args:
            edge_index (torch.Tensor): The edge indices of the graph.
            edge_weight (torch.Tensor): The edge weights of the graph.
            batch (torch.Tensor): The batch indices for each node.
            T (int): The number of timesteps.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The repeated edge indices, edge weights, and batch indices.
        """
        
        # Create edge indices for each timestep's subgraph
        edge_indices_list = []
        edge_weights_list = []

        B = batch.max().item() + 1
        
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
                batch_t = batch + t * (B)  # Offset to avoid overlap
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
        # assert N_in % ds_factor == 0, "N_in must be divisible by ds_factor."
        self.N_out = int(N_in // ds_factor) if N_out is None else N_out
        self.node_sel_mtrx, self.selected_indices = self.get_selection_matrix(N_in, self.N_out)


    def forward(self, x: torch.Tensor, batch = None, up_or_downsample = "down"):

        if up_or_downsample == "up":
            if self.node_sel_mtrx.dim() == 2:
                C = self.node_sel_mtrx.t().to(x.device)  # (N_in, N_out) -> (N_out, N_in)
            elif self.node_sel_mtrx.dim() == 3:
                C = self.node_sel_mtrx.permute(0, 2, 1).to(x.device)  # (B, N_in, N_out) -> (B, N_out, N_in)
        else:
            C = self.node_sel_mtrx.to(x.device)  # (N_out, N_in) or (B, N_out, N_in)

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
    

    def get_selection_matrix(self, N_in, N_out, filepath: str = './datasets/raw/node_selection_matrices/C'):
        """
        Create a selection matrix C of shape (N_out, N_in) that selects N_out nodes from N_in nodes 
        by generating a random row permutation of the identity matrix and taking the first N_out rows in sorted order.
        """

        if N_in == N_out:
            print("N_in == N_out = {}, using identity selection matrix.".format(N_in))
            C = torch.eye(N_in)
            selected_indices = torch.arange(N_in)
            return C, selected_indices

        try: 
            C = self.load_node_selection_matrix_from_file(f"{filepath}_{N_in}_{N_out}")  # (N_in,)
            identity = torch.eye(N_in, dtype = torch.long)
            selected_indices = torch.nonzero(C).squeeze(1)
            assert C.shape == (N_out, N_in), f"Loaded selection matrix must be of shape ({N_out}, {N_in})."
            # assert torch.allclose(identity[selected_indices], C), "Loaded selection matrix must be a subset of the identity matrix."
            print(f"Loaded precomputed node selection matrix from {filepath}_{N_in}_{N_out}.npy")
        
        except FileNotFoundError as e:

            print("No valid node selection matrix file path provided or could be loaded. Using random selection matrix.")
            identity = torch.eye(N_in)
            selected_indices = torch.sort(torch.randperm(N_in)[:N_out]).values
            C = identity[selected_indices]

        # else:
        #     identity = torch.eye(N_in)
        #     C = self.load_node_selection_matrix_from_file(f"{filepath}_{N_in}_{N_out}")  # (N_in,)
        #     selected_indices = torch.nonzero(C).squeeze(1)
        #     assert C.shape == (N_out, N_in), f"Loaded selection matrix must be of shape ({N_out}, {N_in})."
        #     assert torch.allclose(identity[selected_indices], C), "Loaded selection matrix must be a subset of the identity matrix."

        return C, selected_indices
    

    def load_node_selection_matrix_from_file(self, filepath: str):
        """
        Load a precomputed node selection matrix from a .npy file.
        The file is expected to contain a numpy array of shape (N_out, N_in).
        """
        matrix = np.load(f"{filepath}.npy")

        return torch.from_numpy(matrix).float().t() # (N_out, N_in)
    


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
                sel_mtrx: torch.Tensor = None,
                t: torch.Tensor = None,
                cond: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:


        x = self.res_block(x, edge_index = edge_index, edge_weight = edge_weight, batch = batch, sel_mtrx = sel_mtrx,
                            t=t,
                            cond = cond
                            )
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

        self.cond_embed_connection = nn.Sequential(
            SwapAxes(1, 2),
            nn.Linear(self.in_channels * 2, self.in_channels),
            SwapAxes(1, 2)
        )

        self.time_embed_connection = nn.Linear(self.in_channels * 2, self.in_channels)

        self.res_block = ResidualBlock(self.in_channels, self.out_channels, in_timesteps = self.out_timesteps, out_timesteps = self.out_timesteps)
        
        if graph_pool is not None:
            self.graph_pool = graph_pool
        else:
            self.graph_pool = NodeUpDownSamplingBlock(N_in = in_nodes, ds_factor = ds_factor,
                                                     N_out = out_nodes
                                                     )
            

        print(self.__repr__())
            

    def zero_interleave_time(self, x: torch.Tensor):
        print("x.shape, self.in_timesteps, self.out_timesteps: ", x.shape, self.in_timesteps, self.out_timesteps)
        x_zeros = torch.zeros(*x.shape[:-1], self.out_timesteps, device=x.device)
        step = self.out_timesteps // self.in_timesteps
        print("time slice step: ", step)
        x_zeros[..., ::step] = x
        return x_zeros
    

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None,
                sel_mtrx: torch.Tensor = None, t: torch.Tensor = None, cond: torch.Tensor = None) -> torch.Tensor:
        
        if cond is not None:
            debug_print("cond.shape before cond_embed_connection: ", cond.shape)  # (B*N_out, F_out, T_out)
            cond = self.cond_embed_connection(cond)
            debug_print("cond.shape after cond_embed_connection: ", cond.shape)  # (B*N_out, F_out, T_out)

        if t is not None:
            debug_print("t.shape before time_embed_connection: ", t.shape)  # (B*N_out, F_out, 1)
            t = self.time_embed_connection(t.squeeze(-1)).unsqueeze(-1)  # (B*N_out, F_out, 1)
            debug_print("t.shape after time_embed_connection: ", t.shape)  # (B*N_out, F_out, 1)
        
        x, _ = self.graph_pool(x, batch, up_or_downsample = "up")
        x = x.reshape(x.shape[0], self.in_channels, -1)

        x = self.zero_interleave_time(x)
        x = self.res_block(x, edge_index = edge_index, edge_weight = edge_weight, batch = batch, sel_mtrx = sel_mtrx, t = t, cond = cond)

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
        # self.hidden_channels = hidden_channels
        self.in_timesteps = in_timesteps
        hidden_channels = in_channels * 2
        self.block = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )


    def forward(self, x: torch.Tensor):
        # MLP operates on the feature dimension, so we need to reshape/permute
        # B, F, T = x.shape
        # x = x.reshape(B, F * T)
        x = x.swapaxes(1, 2)
        x = self.block(x)
        x = x.swapaxes(1, 2)
        # x = x.reshape(B, F, T)
        return x



class UNet(nn.Module):
    """
    Placeholder UNet class.
    The actual UNet implementation should go here.
    """
    def __init__(self, F_in, N_in, T_in, downsample_factor: List[int], cond_kws: dict,
                 F_min = 2, N_min = 2, T_min = 4,
                 ):
        super().__init__()
        self.F_in = F_in
        self.N_in = N_in
        self.T_in = T_in
        self.downsample_factor_list = downsample_factor
        self.F_min = F_min
        self.N_min = N_min
        self.T_min = T_min # set T_min the same as T_in for no temporal downsampling

        # Initialize UNet layers here.
        F_0 = self.F_in
        N_0 = self.N_in
        T_0 = self.T_in

        cond_F_in = F_in if cond_kws.get("in_channels", None) is None else cond_kws["in_channels"]
        cond_T_in = T_in if cond_kws.get("in_timesteps", None) is None else cond_kws["in_timesteps"]

        ############ Create down-sampling and up-sampling blocks ############
        ds_blocks = nn.ModuleList()
        us_blocks = nn.ModuleList()
        cond_embed_layers = nn.ModuleList()
        time_embed_layers = nn.ModuleList()
        for depth, ds_factor in enumerate(self.downsample_factor_list, 1):
            F_1 = max(F_min, F_0 // ds_factor)
            N_1 = max(N_min, N_0 // ds_factor)
            T_1 = max(T_min, T_0 // ds_factor)
            ds_block = DownBlock(in_channels=F_0, in_nodes=N_0, in_timesteps=T_0,
                                 out_channels=F_1, out_nodes=N_1, out_timesteps=T_1,
                                 ds_factor=ds_factor, depth=depth)
            ds_blocks.append(ds_block)

            cond_embed_layer = ConditionalEmbeddingLayer(in_channels=cond_F_in, out_channels=F_0,
                                                         in_timesteps=cond_T_in, out_timesteps=T_1) # T_1 but F_0 because conditioning is applied after temporal DR and before graph conv
            cond_embed_layers.append(cond_embed_layer)

            time_embed_layer = nn.Sequential(nn.Linear(self.F_in, F_0))
            time_embed_layers.append(time_embed_layer)
            

            us_block = UpBlock(in_channels=F_1, out_channels=F_0,
                               in_nodes=N_1, out_nodes=N_0, in_timesteps=T_1, out_timesteps=T_0,
                               graph_pool=ds_block.graph_pool, depth=depth)  # Share the same graph pooling layer
            us_blocks.append(us_block)

            F_0 = F_1
            N_0 = N_1
            T_0 = T_1

        self.down_blocks = ds_blocks
        self.up_blocks = us_blocks[::-1]  # Reverse for upsampling order
        self.mid_block = MiddleBlock(in_channels=F_0, in_timesteps=T_0) # middle block does not change dimensions
        self.cond_embed_layers = cond_embed_layers
        self.time_embed_layers = time_embed_layers

    @property
    def depth(self):
        return len(self.downsample_factor)

        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, batch:torch.Tensor, edge_index: torch.Tensor = None, edge_weight: torch.Tensor = None,
                debug_print: bool = False
                ) -> torch.Tensor:
        """
        Forward pass of the UNet model.
        Args:
            cond (torch.Tensor): Embedded input node features.
            t (torch.Tensor): Embedded time steps.
            x (torch.Tensor): Embedded input node features.
            edge_index (torch.Tensor, optional): Edge indices for graph structure. Default is None.
            edge_weight (torch.Tensor, optional): Edge weights for graph structure. Default is None.
            debug_print (bool, optional): If True, prints debug information. Default is False.
        Returns:
            torch.Tensor: Output of the UNet model.
        """
        # Implement the forward pass of the UNet here.

        sel_mtrx = torch.eye(self.N_in).to(x.device)  # Start with identity selection matrix
        shortcuts = defaultdict(list)

        # Downsampling path of UNet
        x_ds = x.clone()
        for depth, block in enumerate(self.down_blocks, 1):
            print(f"\n\n--- Testing Downblock with depth {depth} ---\n\n") if debug_print else None
            shortcuts["C"].append(sel_mtrx.clone())

            # cond = self.cond_embed_layers[depth - 1](cond) if cond is not None else None
            cond_emb = self.cond_embed_layers[depth - 1](cond) if cond is not None else None
            time_emb = self.time_embed_layers[depth - 1](t) if t is not None else None
            # downsample_block = DownBlock(in_channels=F_0, in_nodes=N_0, in_timesteps=T_0, ds_factor=downsample_factor)
            x_ds, C = block(x = x_ds, edge_index = edge_index, edge_weight = edge_weight, batch = batch,
                            sel_mtrx = sel_mtrx, # if depth > 1 else None,
                            t = time_emb,
                            cond = cond_emb)

            sel_mtrx = C @ sel_mtrx
            print("\n\n\n\nComposed sel_mtrx shape: ", sel_mtrx.shape) if debug_print else None
            print("Composed sel_mtrx: \n", sel_mtrx, "\n\n\n\n") if debug_print else None

            print("\nOriginal x shape: ", x.shape) if debug_print else None
            print("Downsampled x shape: ", x_ds.shape) if debug_print else None

            shortcuts["x"].append(x_ds.clone())
            shortcuts["cond_embed"].append(cond_emb.clone() if cond is not None else None)
            shortcuts["time_embed"].append(time_emb.clone() if t is not None else None)
            # shortcuts["C"].append(sel_mtrx.clone())

        x_ds = self.mid_block(x_ds)
        print("x_ds shape after middle block: ", x_ds.shape) if debug_print else None

        # desel_mtrx = shortcuts["C"][-1] 
        for i, block in enumerate(self.up_blocks, 1):
            depth = len(self.up_blocks) - i + 1
            print(f"\n\n--- Testing Upblock with depth {depth} ---\n\n") if debug_print else None

            # Get the corresponding shortcut and selection matrix
            x_shortcut = shortcuts["x"][-i]
            C_shortcut = shortcuts["C"][-i]   # -i because upblock at each depth first unpools/upsamples the graph dimension, then adds the skip connection
            cond_emb_shortcut = shortcuts["cond_embed"][-i]
            t_emb_shortcut = shortcuts["time_embed"][-i]

            print("x_shortcut shape: ", x_shortcut.shape) if debug_print else None
            print("C_shortcut shape: ", C_shortcut.shape) if debug_print else None
            print("cond_emb_shortcut.shape: ", cond_emb_shortcut.shape if cond_emb_shortcut is not None else None) if debug_print else None
            print("t_emb_shortcut.shape: ", t_emb_shortcut.shape if t_emb_shortcut is not None else None) if debug_print else None


            ### Add/concatenate the skip connection ###
            x_ds = x_ds + x_shortcut
            x_ds = block(x = x_ds, edge_index = edge_index, edge_weight = edge_weight, batch = batch, sel_mtrx = C_shortcut.to(x_ds.device),
                         cond = cond_emb_shortcut,
                         t = t_emb_shortcut
                         )

        return x_ds