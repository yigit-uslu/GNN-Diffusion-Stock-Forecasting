import torch 
import torch.nn as nn
from torch_geometric.nn import GCNConv, LEConv, TAGConv, MultiAggregation, GATv2Conv
from torch_geometric.nn.models.mlp import MLP as pyg_mlp

class Squeeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.squeeze(x, dim=self.dim)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        architecture = kwargs.get('architecture', 'GCNConv')
        k_hops = kwargs.get('k_hops', 2)
        normalize = kwargs.get('normalize', False)
        add_self_loops = kwargs.get('add_self_loops', True)

        aggr_list = kwargs.get('aggr_list', None) # ['add', 'mean', 'max'])
        if aggr_list is not None:
            msg_aggr = MultiAggregation(aggrs=aggr_list,
                                          mode="proj",
                                          mode_kwargs={"in_channels": in_channels, "out_channels": out_channels},
                                          )
            
        else:
            msg_aggr = None

        conv_batch_norm = kwargs.get('conv_batch_norm', False)

        if architecture == 'TAGConv':
            self.conv_layer = TAGConv(in_channels=in_channels, out_channels=out_channels, K=k_hops, normalize = normalize,
                                      aggr = msg_aggr)
            print(f"TAGConv is initialized with F_l={in_channels}, F_(l+1)={out_channels}, K_hops = {k_hops}, normalization = {normalize}, aggr = {msg_aggr}.")
        elif architecture == 'MultiEdgeTAGConv':
            edge_dim = kwargs.get('edge_dim', 1)
            self.conv_layer = MultiEdgeTAGConv(in_channels=in_channels, out_channels=out_channels,
                                        edge_dim=edge_dim,
                                        K=k_hops, normalize=normalize,
                                        aggr = msg_aggr
                                        )
            print(f"MultiEdgeTAGConv is initialized with F_l={in_channels}, F_(l+1)={out_channels}, edge_dim={edge_dim}, K_hops = {k_hops}, normalization = {normalize}, aggr = {msg_aggr}.")

        elif architecture == 'LEConv':
            self.conv_layer = LEConv(in_channels=in_channels, out_channels=out_channels)
        elif architecture == 'GCNConv':
            self.conv_layer = GCNConv(in_channels=in_channels, out_channels=out_channels,
                                      improved=True, add_self_loops=add_self_loops, normalize=normalize if add_self_loops is False else True)
        else:
            raise ValueError


        if conv_batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = nn.Identity()

    
    def forward(self, x, edge_index, edge_weight, batch = None):
        x = self.conv_layer(x, edge_index = edge_index, edge_weight = edge_weight)
        x = self.bn(x)
        return x
    


class MultiEdgeTAGConv(TAGConv):
    def __init__(self, in_channels, out_channels, edge_dim, K=3, bias=True, normalize = False, edge_mlp_kws=None, **kwargs):
        super().__init__(in_channels, out_channels, K, bias, normalize, **kwargs)
        
        # MLP to project edge features to scalar weights
        self.edge_mlp = nn.Sequential(pyg_mlp(channel_list=[edge_dim, 1],
                                act = nn.LeakyReLU(),
                                dropout = edge_mlp_kws.get("dropout", 0.0) if edge_mlp_kws is not None else 0.0,
                                norm = edge_mlp_kws.get("norm", None) if edge_mlp_kws is not None else None),
                                Squeeze(dim=-1)
                                
        )


    def forward(self, x, edge_index, edge_weight=None):
        # edge_weight: [num_edges, edge_dim]
        if edge_weight is not None and edge_weight.dim() == 2:
            edge_weight_scalar = self.edge_mlp(edge_weight)  # [num_edges]
        else:
            edge_weight_scalar = edge_weight  # [num_edges] or None

        return super().forward(x, edge_index, edge_weight=edge_weight_scalar)
    



class AttnLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_attn_heads = 4, **kwargs):
        super().__init__()
        self_loop_fill_value = kwargs.get('self_loop_fill_value', "max")
        self.attn_layer = GATv2Conv(in_channels=in_channels,
                                    out_channels=out_channels // num_attn_heads,
                                    heads=num_attn_heads, concat=True,
                                    edge_dim=1,
                                    add_self_loops=True,
                                    fill_value=self_loop_fill_value # "mean"
                                    )
    
    def forward(self, x, edge_index, edge_weight, batch = None):
        x = self.attn_layer(x, edge_index = edge_index, edge_attr = edge_weight.unsqueeze(-1), return_attention_weights = None)
        return x