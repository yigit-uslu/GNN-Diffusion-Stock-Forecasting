import torch
import torch.nn as nn
from models.ResidualGNN import ResidualLayer
from models.gnn_backbone import SinusoidalTimeEmbedding
from torch_geometric.nn.models.mlp import MLP as pyg_mlp
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from typing import Optional


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        x = torch.permute(x, dims = self.dims)
        return x
    

class SwapAxes(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = torch.swapaxes(x, self.dim0, self.dim1)
        return x
    

class Squeeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.squeeze(x, dim=self.dim)
        return x


class Temporal2DConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, padding = "same", bias: bool = True):
        super(Temporal2DConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias)
        self.activation = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels


    def reset_parameters(self):
        self.conv2d.reset_parameters()


    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        # x shape: [num_graphs * num_nodes := batch_size, hidden_features, T]
        if batch is None:
            # Reshape the first dimension of x into (-1, num_nodes)
            x = x.reshape(-1, self.in_channels, *x.shape[2:])
        
        else:
            batch_size = int(batch.max()) + 1 # num of graphs in the batch
            x = x.reshape(batch_size, -1, *x.shape[2:])

        x = self.conv2d(x)
        x = self.activation(x)

        x = x.reshape(-1, *x.shape[2:])  # Flatten back to original shape

        return x, None  # Return None for norms to maintain compatibility
    

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels})'
    


class Temporal1DConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1, padding = 0, bias: bool = True,
                 pool: str = "max",
                 norm_layer: Optional[nn.Module] = None):
        super(Temporal1DConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=1, # stride,
                              dilation=1,
                              padding=kernel_size - 1, # causal time convolution
                              padding_mode='zeros',
                              bias=bias)
        if pool == "max":
            self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        elif pool in ["avg", "mean"]:
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        elif pool == "none" or pool is None:
            self.pool = nn.Identity()
        else:
            raise NotImplementedError(f"Unknown pooling type: {pool}")
        
        self.activation = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def calc_out_seq_length(self, L_in: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
        if isinstance(self.pool, nn.Identity):
            return L_in  # No change in length if no pooling is applied
        elif isinstance(self.pool, nn.MaxPool1d):
            return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        elif isinstance(self.pool, nn.AvgPool1d):
            return (L_in + 2 * padding - kernel_size) // stride + 1
        else:
            raise NotImplementedError(f"Unknown pooling type: {self.pool}")


    def reset_parameters(self):
        self.conv.reset_parameters()


    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        # x shape: [num_graphs * num_nodes := batch_size, hidden_features, T]
        print(f"Temporal1DConvLayer - Input x shape: {x.shape}")
        print(f"self.conv.in_channels: {self.conv.in_channels}, self.conv.out_channels: {self.conv.out_channels}.")
        print(f"Mean and std of x: ", x.mean().item(), x.std().item())
        y = self.conv(x)
        print(f"Temporal1DConvLayer - After conv x shape: {y.shape}")
        x = y[..., :x.size(-1)]  # Causal convolution, keep the original length T
        x = self.pool(x) # Reduce time dimensionality by pooling layer
        x = self.activation(x)
        x = self.norm_layer(x)

        return x, None  # Return None for norms to maintain compatibility


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels})'




class ResidualTemporalConvLayer(nn.Module):
    r"""
    My implementation of cascaded Residual (Graph) Conv Layer and a Temporal (2D) Conv Layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_architecture: str = "TAGConv",
        k_hops: int = 2,
        norm: str = "batch",
        mlp: Optional[nn.Module] = None,
        act: Optional[nn.Module] = nn.LeakyReLU(),
        res_connection: str = 'res+',
        dropout: float = 0.,
        normalize: bool = False,
        norm_kws: Optional[dict] = None,
        use_checkpointing: bool = False,
        return_norms: bool = False,
        num_nodes: int = 100,
        edge_dim: int = 1,
        aggr_list: Optional[list] = None,
        conv_batch_norm: bool = False,
        temporal_conv_kws: Optional[dict] = None,
    ):
        super().__init__()


        # self.temporal_conv_layer = Temporal2DConvLayer(in_channels = num_nodes,
        #                                                out_channels = num_nodes)
        self.temporal_conv_layer = Temporal1DConvLayer(in_channels=in_channels, out_channels=out_channels,
                                                       stride=temporal_conv_kws.get("stride", 1) if temporal_conv_kws is not None else 1,
                                                       kernel_size=temporal_conv_kws.get("kernel_size", 3) if temporal_conv_kws is not None else 3,
                                                       dilation=temporal_conv_kws.get("dilation", 1) if temporal_conv_kws is not None else 1,
                                                       padding=temporal_conv_kws.get("padding", 0) if temporal_conv_kws is not None else 0,
                                                       pool=temporal_conv_kws.get("pool", "max") if temporal_conv_kws is not None else "none"
                                                       )
        
        self.T_in = temporal_conv_kws.get("input_sequence_length", 20) if temporal_conv_kws is not None else 20
        self.T_out = self.temporal_conv_layer.calc_out_seq_length(L_in=self.T_in,
                                                             kernel_size=self.temporal_conv_layer.kernel_size,
                                                                stride=self.temporal_conv_layer.stride,
                                                                padding=self.temporal_conv_layer.padding,
                                                                dilation=self.temporal_conv_layer.dilation
        )
        print(f"TemporalConvLayer: Input sequence length = {self.T_in}, Output sequence length = {self.T_out}")

        self.temporal_norm_layer = nn.LayerNorm([out_channels, self.T_out])

        # if mlp is not None:
        #     assert isinstance(mlp, pyg_mlp), "mlp must be an instance of pyg_mlp"
        #     mlp = pyg_mlp(channel_list=[mlp.in_channels * self.T_out, *mlp.channel_list[1:-2], mlp.out_channels * self.T_out], act=act)

        self.res_graph_conv_layer = ResidualLayer(in_channels=self.temporal_conv_layer.out_channels, # * self.T_out,
                                                #   in_channels=hidden_channels // 4,
                                                  out_channels=out_channels, # * self.T_out,
                                                  conv_architecture=conv_architecture, edge_dim=edge_dim,
                                                  k_hops=k_hops,
                                                  norm=norm,
                                                  norm_kws=norm_kws,
                                                  mlp=mlp,
                                                  res_connection=res_connection,
                                                  dropout=dropout, # 0.0,
                                                  normalize=normalize,
                                                  use_checkpointing=use_checkpointing,
                                                  return_norms=return_norms,
                                                  aggr_list=aggr_list,
                                                  conv_batch_norm=conv_batch_norm
                                                  )


        # self.in_channels = in_channels
        # self.out_channels = out_channels

        # self.k_hops = k_hops
        # self.conv_architecture = conv_architecture
        # assert self.conv_architecture in ['LeConv', 'TAGConv', 'GCN']
        # self.norm = norm
        # assert self.norm in ['batch', 'group', 'graph', 'layer', 'none', 'instance']
        # self.act = act
        # self.res_connection = res_connection.lower()
        # assert self.res_connection in ['res+', 'res', 'dense', 'plain']
        # self.mlp = mlp
        # self.dropout = dropout
        # self.use_checkpointing = use_checkpointing

        # print(f"Using dropout rate = {self.dropout}")

        # self.return_norms = return_norms

        # ### Initialize the convolution layer
        # self.conv = ConvLayer(in_channels=in_channels, out_channels=out_channels,
        #                       architecture = self.conv_architecture, k_hops=self.k_hops,
        #                       normalize = normalize,
        #                       add_self_loops = False
        #                       )
            
        # self.mlp = nn.Identity() if self.mlp is None else self.mlp
        # self.norm = NormLayer(norm=norm, in_channels=in_channels, **norm_kws)

        # if self.in_channels != self.out_channels:
        #     self.res = nn.Linear(self.in_channels, self.out_channels, bias=False)
        # else:
        #     self.res = nn.Identity()
            

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.res_graph_conv_layer.reset_parameters()
        self.temporal_conv_layer.reset_parameters()


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None,
                debug_forward_pass: bool = False):
        
        print(f"{self.__class__.__name__} - Before temporal conv layer, x.shape: {x.shape}") if debug_forward_pass else None

        # Check x for NaNs or Infs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaNs or Infs")
        else:
            print(f"{self.__class__.__name__} - Input x is valid (no NaNs or Infs).") if debug_forward_pass else None

        x, _ = self.temporal_conv_layer(x, batch = batch)

        print(f"{self.__class__.__name__} - After temporal conv layer, x.shape: {x.shape}") if debug_forward_pass else None


        # Applying normalization across both feature channel and temporal dimensions
        x = self.temporal_norm_layer(x)
        

        # Check x for NaNs or Infs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaNs or Infs")
        else:
            print(f"{self.__class__.__name__} - Input x is valid (no NaNs or Infs).") if debug_forward_pass else None

        # F, T = x.shape[1], x.shape[2]
        # x = x.swapaxes(1, 2).reshape(x.size(0), -1)  # Flatten the last two dimensions for graph conv

        # Parallel processing of all timesteps through graph conv
        # Strategy: Manual batching that's compatible with all PyG versions
        B, F, T = x.shape
        print("B, F, T = x.shape: ", B, F, T) if debug_forward_pass else None
        
        # Method 1: Manual temporal batching with proper edge index offsetting
        x_parallel = x.permute(2, 0, 1).reshape(B * T, F)  # (B, F, T) -> (T, B, F) -> (B*T, F)
        
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
            
        print(f"{self.__class__.__name__} - Before graph conv layer, x parallel shape: {x_parallel.shape}") if debug_forward_pass else None
        print(f"{self.__class__.__name__} - Edge index parallel shape: {edge_index_parallel.shape}") if debug_forward_pass else None
        print(f"{self.__class__.__name__} - Batch parallel shape: {batch_parallel.shape if batch_parallel is not None else None}") if debug_forward_pass else None

        # Process all timesteps in parallel (single forward pass)
        h, norms = self.res_graph_conv_layer(x_parallel, edge_index_parallel, edge_weight_parallel, batch=batch_parallel)

        print("h.shape after parallel graph conv: ", h.shape) if debug_forward_pass else None

        # Validate h for NaNs or Infs
        if torch.isnan(h).any() or torch.isinf(h).any():
            raise ValueError("Output h contains NaNs or Infs after graph convolution")
        else:
            print(f"{self.__class__.__name__} - Output h is valid (no NaNs or Infs).") if debug_forward_pass else None

        # Reshape back to original temporal structure: (B*T, F_out) -> (B, F_out, T)
        F_out = h.shape[-1]  # Output feature dimension
        h = h.reshape(T, B, F_out).permute(1, 2, 0)  # (B*T, F_out) -> (T, B, F_out) -> (B, F_out, T)
        
        print(f"{self.__class__.__name__} - Final output shape after reshaping: {h.shape}") if debug_forward_pass else None
        
        return h, norms


def process_timesteps_alternative_shared_graph(x, edge_index, edge_weight, batch, graph_conv_layer, debug_forward_pass=False):
    """
    Alternative method: Process timesteps with shared graph structure.
    This treats all timesteps as using the SAME graph connectivity.
    
    This approach is often more appropriate for temporal GNNs where the graph structure
    represents relationships that are consistent across time (e.g., stock correlations).
    
    Args:
        x: Input tensor of shape (B, F, T)
        edge_index: Graph edge indices (shared across all timesteps)
        edge_weight: Graph edge weights (shared across all timesteps)
        batch: Batch indices
        graph_conv_layer: The graph convolution layer to apply
        
    Returns:
        Processed tensor of shape (B, F_out, T)
    """
    B, F, T = x.shape
    
    # Process each timestep with the same graph structure
    outputs = []
    
    for t in range(T):
        x_t = x[:, :, t]  # Shape: (B, F)
        h_t, _ = graph_conv_layer(x_t, edge_index, edge_weight, batch=batch)
        outputs.append(h_t)
    
    # Stack all timestep outputs
    h = torch.stack(outputs, dim=-1)  # Shape: (B, F_out, T)
    
    print(f"Shared graph method - Output shape: {h.shape}") if debug_forward_pass else None
    
    return h, None


def process_timesteps_with_graph_replication(x, edge_index, edge_weight, batch, graph_conv_layer, debug_forward_pass=False):
    """
    Alternative method: Explicitly replicate graph structure for each timestep.
    This creates T separate subgraphs, one for each timestep.
    
    Args:
        x: Input tensor of shape (B, F, T)
        edge_index: Graph edge indices
        edge_weight: Graph edge weights  
        batch: Batch indices
        graph_conv_layer: The graph convolution layer to apply
        
    Returns:
        Processed tensor of shape (B, F_out, T)
    """
    B, F, T = x.shape
    
    # Create separate graphs for each timestep
    edge_indices_list = []
    edge_weights_list = []
    x_list = []
    batch_list = []
    
    for t in range(T):
        # Offset node indices for each timestep's subgraph
        edge_index_t = edge_index + t * B
        edge_indices_list.append(edge_index_t)
        
        if edge_weight is not None:
            edge_weights_list.append(edge_weight)
            
        # Extract features for timestep t
        x_list.append(x[:, :, t])  # Shape: (B, F)
        
        if batch is not None:
            batch_t = batch + t * (batch.max() + 1)  # Offset batch indices
            batch_list.append(batch_t)
    
    # Concatenate all timesteps
    edge_index_combined = torch.cat(edge_indices_list, dim=1)
    edge_weight_combined = torch.cat(edge_weights_list) if edge_weight is not None else None
    x_combined = torch.cat(x_list, dim=0)  # Shape: (B*T, F)
    batch_combined = torch.cat(batch_list) if batch is not None else None
    
    print(f"Graph replication method - Combined x shape: {x_combined.shape}") if debug_forward_pass else None
    print(f"Graph replication method - Combined edge_index shape: {edge_index_combined.shape}") if debug_forward_pass else None
    
    # Process through graph conv
    h, norms = graph_conv_layer(x_combined, edge_index_combined, edge_weight_combined, batch=batch_combined)
    
    # Reshape back to (B, F_out, T)
    F_out = h.shape[-1]
    h = h.reshape(T, B, F_out).transpose(0, 2).transpose(0, 1)  # (T, B, F_out) -> (B, F_out, T)
    
    return h, norms

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'




class StockForecastTemporalConvGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, norm, num_features, num_timesteps, **kwargs):
        super(StockForecastTemporalConvGNN, self).__init__()
        self.in_channels = in_channels # number of channels in prediction target y_t, i.e., = Th
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels # number of channels in prediction target y_t, i.e., = Th
        self.num_features = num_features # number of features in conditioning input x
        self.num_timesteps = num_timesteps # number of timesteps in conditioning input x, i.e., = Tp
        self.depth = depth 
        self.norm = norm
        self.num_nodes = kwargs.get("num_nodes", 100)
        self.normalize = kwargs.get("conv_layer_normalize", False)
        self.activation = nn.LeakyReLU()
        conv_architecture = kwargs.get("conv_model", "TAGConv")
        edge_features_nb = kwargs.get("edge_features_nb", 1)
        aggr_list = kwargs.get("aggr_list", None)
        conv_batch_norm = kwargs.get("conv_batch_norm", False)
        
        if edge_features_nb > 1:
            conv_architecture = "MultiEdge" + conv_architecture
        
        self.use_res_connection_time_embed = kwargs.get("use_res_connection_time_embed", False)
        self.time_embed_strategy = kwargs.get("time_embed_strategy", "concatenate") # "add" or "multiply"
        self.cond_embed_strategy = kwargs.get("cond_embed_strategy", "concatenate") # "add" or "multiply" or "concatenate"

        in_hidden_channels_x_and_y_t_embed = hidden_channels // (1 + int(self.cond_embed_strategy == "concatenate"))
        in_hidden_channels_x_and_y_t_embed = in_hidden_channels_x_and_y_t_embed // (1 + int(self.time_embed_strategy == "concatenate"))
        in_hidden_channels_t_embed = hidden_channels // (1 + int(self.time_embed_strategy == "concatenate"))

        in_hidden_channels = hidden_channels
        if self.cond_embed_strategy == "concatenate":
            print(f"Using concatenate strategy for cond_embed_strategy. Increase in_hidden_channels = {in_hidden_channels} for each intermediate layer by {hidden_channels // 2}.")
            in_hidden_channels = in_hidden_channels + hidden_channels // 2
        if self.time_embed_strategy == "concatenate":
            print(f"Using concatenate strategy for time_embed_strategy. Increase in_hidden_channels = {in_hidden_channels} for each intermediate layer by {hidden_channels // 2}.")
            in_hidden_channels = in_hidden_channels + hidden_channels // 2

        self.total_forward_calls = 0

        # cond_in_channels = (num_features + 4) * num_timesteps  # +4 for sin/cos of two frequencies
        cond_in_channels = num_features
        self.embed_x = nn.Linear(cond_in_channels, in_hidden_channels_x_and_y_t_embed)
        self.embed_y_t = nn.Linear(1, in_hidden_channels_x_and_y_t_embed) # y_target has one dummy feature
        self.embed_t = SinusoidalTimeEmbedding(in_hidden_channels_t_embed, T_max=num_timesteps)
        self.total_seq_length = num_timesteps + in_channels


        self.layers = nn.ModuleList()
        self.layers.append(ResidualTemporalConvLayer(in_channels=hidden_channels,
                                                #   in_channels=hidden_channels // 4,
                                                  out_channels=hidden_channels,
                                                  conv_architecture=conv_architecture, edge_dim=edge_features_nb,
                                                  k_hops=kwargs.get("k_hops", 2),
                                                  norm=self.norm,
                                                  norm_kws=kwargs.get("norm_kws", dict()),
                                                  mlp=pyg_mlp(channel_list = [hidden_channels, hidden_channels], act = self.activation),
                                                  res_connection=kwargs.get("res_connection", 'res+'),
                                                  dropout=kwargs.get("dropout", 0.0), # 0.0,
                                                  normalize=self.normalize,
                                                  use_checkpointing=kwargs.get("use_checkpointing", False),
                                                  return_norms=False,
                                                  aggr_list=aggr_list,
                                                  conv_batch_norm=conv_batch_norm,
                                                  num_nodes = self.num_nodes,
                                                  temporal_conv_kws={"stride": 2, # 2
                                                                     "kernel_size": 3,
                                                                     "dilation": 1,
                                                                     "padding": 0,
                                                                     "pool": "avg", # "avg" or "max" or "none"
                                                                     "input_sequence_length": self.total_seq_length
                                                                     }
                                                  )
                                    ) 
        

        out_seq_length = self.layers[0].temporal_conv_layer.calc_out_seq_length(L_in=self.total_seq_length,
                                                                 kernel_size=self.layers[0].temporal_conv_layer.kernel_size,
                                                                 stride=self.layers[0].temporal_conv_layer.stride,
                                                                 padding=self.layers[0].temporal_conv_layer.padding,
                                                                 dilation=self.layers[0].temporal_conv_layer.dilation
                                                                 )

        y_t_time_proj_layer = nn.Linear(self.layers[0].T_in,
                                   self.layers[0].T_out) \
            if self.layers[0].T_in != self.layers[0].T_out else nn.Identity()

        y_t_time_and_feature_proj = nn.Sequential(y_t_time_proj_layer, SwapAxes(2, 1), nn.Linear(in_hidden_channels_x_and_y_t_embed, hidden_channels // 2), SwapAxes(2,1))
        self.y_t_proj_layers = nn.ModuleList([y_t_time_and_feature_proj])


        for layer_id in range(1, self.depth):
            # self.time_embed_layers.append(nn.Identity())
            # self.x_embed_layers.append(nn.Identity())
            res_temporal_conv_layer = ResidualTemporalConvLayer(in_channels=in_hidden_channels,
                                                #   in_channels=hidden_channels // 4,
                                                  out_channels=hidden_channels,
                                                  conv_architecture=conv_architecture, edge_dim=edge_features_nb,
                                                  k_hops=kwargs.get("k_hops", 2),
                                                  norm=self.norm,
                                                  norm_kws=kwargs.get("norm_kws", dict()),
                                                  mlp=pyg_mlp(channel_list = [hidden_channels, 4 * hidden_channels, hidden_channels], act = self.activation),
                                                  res_connection=kwargs.get("res_connection", 'res+'),
                                                  dropout=kwargs.get("dropout", 0.0), # 0.0,
                                                  normalize=self.normalize,
                                                  use_checkpointing=kwargs.get("use_checkpointing", False),
                                                  return_norms=False,
                                                  aggr_list=aggr_list,
                                                  conv_batch_norm=conv_batch_norm,
                                                  num_nodes = self.num_nodes,
                                                  temporal_conv_kws={"stride": 1,
                                                                     "kernel_size": 3,
                                                                     "dilation": 1,
                                                                     "padding": 0,
                                                                     "pool": "avg" if layer_id % 2 == 1 else "none" if layer_id == self.depth else "max", # "avg" or "max" or "none"
                                                                     "input_sequence_length": out_seq_length
                                                                     }
                                                  )

            out_seq_length = res_temporal_conv_layer.temporal_conv_layer.calc_out_seq_length(L_in=out_seq_length,
                                                                     kernel_size=res_temporal_conv_layer.temporal_conv_layer.kernel_size,
                                                                     stride=res_temporal_conv_layer.temporal_conv_layer.stride,
                                                                     padding=res_temporal_conv_layer.temporal_conv_layer.padding,
                                                                     dilation=res_temporal_conv_layer.temporal_conv_layer.dilation
                                                                     )
            
            # Note that the projection layer is always from the original input sequence length to the new output sequence length,
            y_t_time_proj = nn.Linear(self.layers[0].T_in, res_temporal_conv_layer.T_out) if self.layers[0].T_in != res_temporal_conv_layer.T_out else nn.Identity()
            y_t_time_and_feature_proj = nn.Sequential(y_t_time_proj, SwapAxes(2, 1), nn.Linear(in_hidden_channels_x_and_y_t_embed, hidden_channels // 2), SwapAxes(2,1))
            self.y_t_proj_layers.append(y_t_time_and_feature_proj)

            self.layers.append(res_temporal_conv_layer)


        assert out_seq_length > 0, f"Output sequence length must be positive, but got {out_seq_length}. Consider changing the convolution parameters or reducing the number of layers."
        self.out_seq_length = out_seq_length
        print(f"Total output sequence length after {self.depth} layers: {self.out_seq_length}")

        # Sequential output layers that permute the last two axis, project new last (feature) axis from hidden_channels to 1,
        # squeezes the last axis, projects the last (time window) axis down to out_channels
        # self.out_layers = nn.Sequential(
        #     SwapAxes(-1, -2),
        #     nn.Linear(hidden_channels, 1, bias = False),
        #     Squeeze(-1),
        #     nn.Linear(self.out_seq_length, self.out_channels, bias = False)
        # )
        # Sequential output layers that flatten the last two axis and project to self.out_channels.
        self.out_layers = nn.Sequential(
            nn.Flatten(start_dim = -2),
            nn.Linear(hidden_channels * self.out_seq_length, self.out_channels, bias = True)
        )

        # Apply custom weight initialization
        self.apply(self.init_weights)
        self.n_iters_trained = 0


    def init_weights(self, m):
        """ Custom weight initialization. """
        if isinstance(m, pyg_nn.TAGConv):
            # print("m: ", m)
            # Apply He initialization to the weight matrix in GraphConv
            for lin in m.lins:
                if self.activation._get_name() == "LeakyReLU":
                    torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='leaky_relu')  # He init for weight
                elif self.activation._get_name() == 'ReLU':
                    torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='relu')  # He init for weight
                else:
                    pass
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)  # Bias is initialized to zero
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier init for final layers
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


    def add_or_multiply_embeddings(self, x, cond_embed, strategy = "add"):
        if strategy == "add":
            return x + cond_embed
        elif strategy == "multiply":
            return x * cond_embed
        else:
            raise NotImplementedError(f"Unknown conditional embedding strategy: {strategy}")



    def forward(self, x: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor, edge_index, edge_weight, batch = None, return_attn_weights=False, debug_forward_pass=False):
        
        if self.normalize and edge_weight is not None:
            add_self_loops = True
            improved = add_self_loops
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(0),
                    improved=improved, add_self_loops=add_self_loops, # flow=self.conv_layers[0].flow,
                    dtype=x.dtype)
            print("GCN normalization applied to edge_weights.") if self.n_iters_trained == 0 else None

        batch_size, num_features, past_window = x.shape
        batch_size, future_window = y_t.shape

        # Validate x for NaNs or Infs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaNs or Infs after embedding and combining with y_t and t")
        

        x = self.embed_x(x.permute(0, 2, 1)).permute(0, 2, 1) # [B, F, Tp] -> [B, H or H // 2, Tp]
        y_t = self.embed_y_t(y_t.unsqueeze(-1)).permute(0, 2, 1) # [B, Th] -> [B, H or H // 2, Th]

        print(f"Input x shape after embedding: {x.shape}") if self.total_forward_calls == 0 else None
        print(f"Input y_t shape after embedding: {y_t.shape}") if self.total_forward_calls == 0 else None
        print(f"Input t shape before embedding: {t.shape}") if self.total_forward_calls == 0 else None


        # Validate t for NaNs or Infs
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError("Input t contains NaNs or Infs before time embedding")

        t = self.embed_t(t).unsqueeze(-1) # [B, 1] -> [B, H, 1]

        # Validate t for NaNs or Infs
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError("Input t contains NaNs or Infs after time embedding")
        

        assert x.shape[0] == y_t.shape[0] == t.shape[0] == batch_size, f"Batch size must be consistent across x, y_t, and t, but got {x.shape[0]}, {y_t.shape[0]}, {t.shape[0]}"
        assert x.shape[1] == y_t.shape[1], f"Feature dimension must be consistent across x, y_t, but got {x.shape[1]}, {y_t.shape[1]}"

        # Zero-pad x and y_t in the last dimension to match the maximum sequence length
        # max_seq_length = max(past_window, future_window)
        # total_seq_length = past_window + future_window

        # Validate x for NaNs or Infs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaNs or Infs after embedding and combining with y_t and t")
        
        x = torch.cat([x, torch.zeros(size = (*x.shape[:-1], self.total_seq_length - past_window), device = x.device)], dim = -1)
        y_t = torch.cat([torch.zeros(size = (*y_t.shape[:-1], self.total_seq_length - future_window), device = y_t.device), y_t], dim = -1)
        # x = F.pad(x, (0, 0, 0, max_seq_length - past_window))
        # y_t = F.pad(y_t, (0, 0, 0, max_seq_length - future_window))

        # Validate x for NaNs or Infs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaNs or Infs after embedding and combining with y_t and t")

        if self.cond_embed_strategy == "concatenate":
            # Concatenate x and y_t along the feature dimension
            x = torch.cat([x, y_t], dim=1)
        else:
            x = self.add_or_multiply_embeddings(x, y_t, strategy=self.cond_embed_strategy)

        # assert x.shape[1:] == (self.hidden_channels, self.total_seq_length), f"Expected shape {(self.hidden_channels, self.total_seq_length)}, but got {x.shape[1:]}"

        # Validate x for NaNs or Infs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaNs or Infs after embedding and combining with y_t and t")
        

        if self.time_embed_strategy == "concatenate":
            # Concatenate time embeddings along the feature dimension
            t_expanded = t.repeat(1, 1, x.size(-1))
            x = torch.cat([x, t_expanded], dim=1)  # Concatenate along feature dimension
        else:
            # Add time embeddings
            x = self.add_or_multiply_embeddings(x, t, strategy=self.time_embed_strategy)


        # Validate x for NaNs or Infs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaNs or Infs after embedding and combining with y_t and t")

        print(f"Input x shape after embedding and combining with y_t: {x.shape}") if self.total_forward_calls == 0 else None

        for i, (layer, y_t_proj) in enumerate(zip(self.layers, self.y_t_proj_layers)):
            if debug_forward_pass or self.total_forward_calls == 0:
                print(f"Layer {i} input shape: {x.shape}")
            x, norms = layer(x, edge_index, edge_weight, batch = batch, debug_forward_pass = debug_forward_pass or self.total_forward_calls == 0)

            if i == len(self.layers) - 1:
                print(f"Final layer {i} output shape (before adding y_t and t): {x.shape}") if debug_forward_pass or self.total_forward_calls == 0 else None
                print(f"We do not add conditioning y_t and t now.") if debug_forward_pass or self.total_forward_calls == 0 else None
                break

            y_t_projed = y_t_proj(y_t) # Project y_t to match the current layer's temporal dimension if needed
            assert x.size(-1) == y_t_projed.size(-1), f"Temporal dimension mismatch after projection. {x.size(-1)}-{y_t_projed.size(-1)}"
            
            if self.cond_embed_strategy == "concatenate":
                # Concatenate x and y_t along the feature dimension
                x = torch.cat([x, y_t_projed], dim=1)
            else:
                x = self.add_or_multiply_embeddings(x, y_t_projed, strategy=self.cond_embed_strategy)

            if self.use_res_connection_time_embed:
                print("Using residual connections for time embeddings.") if self.total_forward_calls == 0 else None
                assert t is not None, "Time embeddings are required for residual time connection."

                if self.time_embed_strategy == "concatenate":
                    # Concatenate time embeddings along the feature dimension
                    t_expanded = t.repeat(1, 1, x.size(-1))
                    x = torch.cat([x, t_expanded], dim=1)  # Concatenate along feature dimension
                else:
                    # x = x + t # add time embeddings to the output of each layer
                    x = self.add_or_multiply_embeddings(x, t, strategy=self.time_embed_strategy)

            if debug_forward_pass:
                print(f"Layer {i} output shape: {x.shape}")
                if norms is not None:
                    for norm in norms:
                        print(f"{norm['layer']} norm: {norm['norm']:.4f}")

        pred_noise = self.out_layers(x) # [B, H, T] -> [B, T, H] -> [B, T, 1] -> [B, T] -> [B, Th]
        assert pred_noise.shape[1] == self.out_channels, f"Output channels must match out_channels. {pred_noise.shape[1]}-{self.out_channels}"
        
        if isinstance(pred_noise, tuple):
            for i in range(len(pred_noise), 0, -1):
                if i == 3:
                    model_debug_data = pred_noise[i-1]
                elif i == 2:
                    attn_weights = pred_noise[i-1]
                elif i == 1:
                    pred_noise = pred_noise[i-1]
                else:
                    pass
        else:
            attn_weights = None
            model_debug_data = None
        
        if self.total_forward_calls == 0:
            print(f"First forward pass pred_noise.shape: {pred_noise.shape}")
            # assert pred_noise.shape == y_t.shape, f"Output shape must match y_t's shape. {pred_noise.shape}-{y_t.shape}"

        self.total_forward_calls += 1

        return pred_noise




########################################################### Res+GNN adapted for stock forecasting ######################################################################

""" We use the same class signature for GNN as the GraphUNet """
class StockForecastDiffusionTemporalConvGNN(nn.Module):
    def __init__(self, nfeatures=1, nblocks = 1, nunits=128,
                 nsteps = 100, # diffusion timesteps (useful for computing sin embeddings)
                 num_features_cond = 8,
                num_timesteps_cond = 25, 
                **kwargs):
        super(StockForecastDiffusionTemporalConvGNN, self).__init__()

        pool_ratio = kwargs.get('pool_ratio', 0.5)
        pool_layer = kwargs.get('pool_layer', 'topk')
        pool_multiplier = kwargs.get('pool_multiplier', 1.)
        sum_skip_connection = kwargs.get('sum_skip_connection', True)
        use_checkpointing = kwargs.get('use_checkpointing', False)
        use_res_connection_time_embed = kwargs.get('use_res_connection_time_embed', False)
        attn_self_loop_fill_value = kwargs.get('attn_self_loop_fill_value', "max")
        apply_gcn_norm = kwargs.get("apply_gcn_norm", False)
        norm = kwargs.get('norm', 'batch')
        conv_model = kwargs.get('conv_model', 'TAGConv')
        conv_layer_normalize = kwargs.get('conv_layer_normalize', False)
        k_hops = kwargs.get('k_hops', 2)
        dropout_rate = kwargs.get('dropout_rate', 0.0)
        norm_kws = kwargs.get('norm_kws', dict())
        edge_features_nb = kwargs.get('edge_features_nb', 1)
        aggr_list = kwargs.get("aggr_list", None)
        conv_batch_norm = kwargs.get("conv_batch_norm", False)

        self.gnn = StockForecastTemporalConvGNN(in_channels=nfeatures, # nfeatures # nunits,
                                    hidden_channels=nunits, out_channels=nfeatures,
                                    res_connection = 'res+',
                                    depth=nblocks,
                                    norm=norm,
                                    num_features=num_features_cond,
                                    num_timesteps=num_timesteps_cond,
                                    k_hops=k_hops,
                                    dropout_rate=dropout_rate,
                                    conv_model=conv_model,
                                    conv_layer_normalize=conv_layer_normalize,
                                    use_checkpointing=use_checkpointing,
                                    use_res_connection_time_embed=use_res_connection_time_embed,
                                    norm_kws=norm_kws,
                                    edge_features_nb=edge_features_nb,
                                    aggr_list=aggr_list,
                                    conv_batch_norm=conv_batch_norm
                                    )

        
        # self.gnn = StockForecastGNN(in_channels=nunits, # nfeatures,
        #                             hidden_channels=nunits, out_channels=nfeatures, # out_channels don't matter
        #                        res_connection = 'res+',
        #                        depth=nblocks,
        #                        norm=norm,
        #                        conv_layer_normalize=conv_layer_normalize,
        #                        k_hops=k_hops,
        #                        dropout_rate=dropout_rate,
        #                        use_checkpointing=use_checkpointing,
        #                        use_res_connection_time_embed=use_res_connection_time_embed,
        #                        norm_kws = norm_kws
        #                        )
        

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, data, return_attention_weights = False, debug_forward_pass = False):

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
        # Denoise noisy node embeddings
        out = self.gnn(y_t = y_t, t = t, x = x, edge_index = edge_index, edge_weight = edge_weight, batch = batch,
                        return_attn_weights = return_attention_weights, debug_forward_pass = debug_forward_pass)

        return out

    @property
    def prototype_name(self):
        return f"{self.__class__.__name__}_{self.gnn.depth}_layers"


####################################################################################################################################