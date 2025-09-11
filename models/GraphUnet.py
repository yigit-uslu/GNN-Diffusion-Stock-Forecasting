import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, TAGConv, LEConv, GATv2Conv, BatchNorm, LayerNorm, DiffGroupNorm, GraphNorm, InstanceNorm, MultiAggregation
from torch_geometric.typing import OptTensor
from typing import Callable, List, Union
from torch_geometric.utils.repeat import repeat
import torch.utils.checkpoint as checkpoint
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from models.ConvLayers import ConvLayer, AttnLayer
from models.NormLayers import NormLayer
from models.EmbeddingLayers import SinusoidalTimeEmbedding


from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)

    

    

    # def reset_parameters(self):
    #     # self.edge_mlp.reset_parameters()
    #     super().reset_parameters()


# # MLP-based model
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
#         super(MLP, self).__init__()
        
#         # MLP layers
#         self.layers = nn.ModuleList()
#         for i in range(num_layers):
#             if i == 0:
#                 self.layers.append(nn.Linear(input_dim, hidden_dim))
#             else:
#                 self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
#         # Output layer
#         self.out_layer = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x, graph_conditioning=None):
#         # Pass the input through the MLP layers
#         for layer in self.layers:
#             x = F.relu(layer(x))
            
#             # Apply graph conditioning if available (when conditioning is True)
#             if graph_conditioning is not None:
#                 x = self.condition_block(x, graph_conditioning)
        
#         # Output layer
#         x = self.out_layer(x)
#         return x

#     def condition_block(self, x, condition):
#         # Apply the graph condition using FiLM or concatenation
#         scale = condition.unsqueeze(0).expand_as(x)
#         shift = condition.unsqueeze(0).expand_as(x)
#         return x * scale + shift
    

# class SinusoidalEmbedding(nn.Module):
#     """
#     https://github.com/tanelp/tiny-diffusion 
#     """
#     def __init__(self, size: int, scale: float = 1.0):
#         super().__init__()
#         self.size = size
#         self.scale = scale

#     def forward(self, x: torch.Tensor):
#         x = x * self.scale
#         half_size = self.size // 2
#         emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
#         emb = torch.exp(-emb * torch.arange(half_size))
#         emb = x.unsqueeze(-1) * emb.unsqueeze(0).to(x.device)
#         emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
#         return emb

#     def __len__(self):
#         return self.size


class MyIdentity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input
    

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

    

    


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_layers: int = 2, norm: str = 'group', conv_architecture = 'GCNConv', use_checkpointing = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.norm = norm
        self.conv_architecture = conv_architecture
        self.time_channels = time_channels
        self.use_checkpointing = use_checkpointing

        self.act = nn.LeakyReLU()

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.conv_layers.append(ConvLayer(in_channels=in_channels, out_channels=out_channels, architecture = conv_architecture))
        self.norm_layers.append(NormLayer(norm=norm, in_channels=in_channels))

        for _ in range(1, self.n_layers):
            self.conv_layers.append(ConvLayer(in_channels=out_channels, out_channels=out_channels, architecture = conv_architecture))
            self.norm_layers.append(NormLayer(norm=norm, in_channels=out_channels))

        # If the number of input channels is not equal to the number of output channels we have to project the shortcut connection
        if in_channels != out_channels:
            self.res_connection = nn.Linear(in_features=in_channels, out_features=out_channels, bias = False)
        else:
            self.res_connection = nn.Identity()

        # Further linear embedding layer for time embeddings
        self.time_embed = nn.Linear(time_channels, out_channels)
        # self.time_act = Swish()
        self.time_act = nn.LeakyReLU()


    def custom_conv_embed(self, conv_embed):
        def custom_forward(*inputs):
            # print('inputs: ', inputs)
            outputs = conv_embed(inputs[0], edge_index=inputs[1], edge_weight=inputs[2])
            return outputs
        return custom_forward


    def forward(self, x: torch.Tensor, t: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None):
        h = self.norm_layers[0](x, batch = batch, batch_size = None) # batch size computation TO DO if GraphNorm is used. 
        
        if self.use_checkpointing:
            h = checkpoint.checkpoint(self.custom_conv_embed(self.conv_layers[0]), self.act(h), edge_index, edge_weight, use_reentrant=False)
        else:
            h = self.conv_layers[0](self.act(h), edge_index=edge_index, edge_weight = edge_weight)

        # add time embeddings
        tt = self.time_embed(self.time_act(t))
        h += tt

        for norm_layer, conv_layer in zip(self.norm_layers[1:], self.conv_layers[1:]):
            h = norm_layer(h, batch = batch, batch_size = None) # batch size computation TO DO if GraphNorm is used.
            h = self.act(h)

            if self.use_checkpointing:
                h = checkpoint.checkpoint(self.custom_conv_embed(conv_layer), h, edge_index, edge_weight, use_reentrant=False)
            else:
                h = conv_layer(h, edge_index, edge_weight)

        # Add residual connection
        out = h + self.res_connection(x)
        return out   
    

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_layers: int = 2, norm: str = 'group', num_attn_heads: int = 4,
                 use_checkpointing = False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.norm = norm
        self.num_attn_heads = num_attn_heads
        self.time_channels = time_channels
        self.use_checkpointing = use_checkpointing

        attn_self_loop_fill_value = kwargs.get('attn_self_loop_fill_value', "max")

        self.act = nn.LeakyReLU()

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.conv_layers.append(AttnLayer(in_channels=in_channels, out_channels=out_channels, num_attn_heads=num_attn_heads, self_loop_fill_value = attn_self_loop_fill_value))
        self.norm_layers.append(NormLayer(norm=norm, in_channels=in_channels))

        for _ in range(1, self.n_layers):
            self.conv_layers.append(AttnLayer(in_channels=out_channels, out_channels=out_channels, num_attn_heads=num_attn_heads, self_loop_fill_value = attn_self_loop_fill_value))
            self.norm_layers.append(NormLayer(norm=norm, in_channels=out_channels))

        # # If the number of input channels is not equal to the number of output channels we have to project the shortcut connection
        # if in_channels != out_channels:
        #     self.res_connection = nn.Linear(in_features=in_channels, out_features=out_channels, bias = False)
        # else:
        #     self.res_connection = nn.Identity()

        # Further linear embedding layer for time embeddings
        # self.time_embed = nn.Linear(time_channels, out_channels)
        # self.time_act = Swish()

    def custom_conv_embed(self, conv_embed):
        def custom_forward(*inputs):
            # print('inputs: ', inputs)
            outputs = conv_embed(inputs[0], edge_index=inputs[1], edge_weight=inputs[2])
            return outputs
        return custom_forward


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None):
        # h = self.norm_layers[0](x, batch = batch, batch_size = None) # batch size computation TO DO if GraphNorm is used. 
        # h = self.conv_layers[0](self.act(h), edge_index=edge_index, edge_weight = edge_weight)

        # # add time embeddings
        # tt = self.time_embed(self.time_act(t))
        # h += tt

        for norm_layer, conv_layer in zip(self.norm_layers, self.conv_layers):
            x = norm_layer(x, batch = batch, batch_size = None) # batch size computation TO DO if GraphNorm is used.
            x = self.act(x)

            if self.use_checkpointing:
                x = checkpoint.checkpoint(self.custom_conv_embed(conv_layer), x, edge_index, edge_weight, use_reentrant=False)
            else:
                x = conv_layer(x, edge_index, edge_weight)

        # # Add residual connection
        # out = h + self.res_connection(x)
        return x  
    

class DownUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_layers = 2, norm: str = 'group', conv_architecture = 'GCNConv',
                 has_attn: bool = False, use_checkpointing = False, **kwargs):
        super().__init__()

        attn_self_loop_fill_value = kwargs.get("attn_self_loop_fill_value", "max")

        n_res_layers = n_layers - 1 if has_attn else n_layers
        self.res = ResBlock(in_channels=in_channels, out_channels=out_channels, time_channels=time_channels, n_layers=n_res_layers, norm=norm,
                            conv_architecture=conv_architecture,
                            use_checkpointing=use_checkpointing)

        if has_attn:
            self.attn = AttnBlock(in_channels=out_channels, out_channels=out_channels, time_channels=time_channels, n_layers = 1, norm = norm,
                                  num_attn_heads=4,
                                  use_checkpointing=use_checkpointing,
                                  attn_self_loop_fill_value = attn_self_loop_fill_value)
        else:
            self.attn = MyIdentity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None):
        x = self.res(x, t, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x = self.attn(x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        return x

    

class MidBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int, n_layers = 3, norm: str = 'group', conv_architecture = 'GCNConv',
                 has_attn: bool = False, use_checkpointing = False, **kwargs):
        super().__init__()

        attn_self_loop_fill_value = kwargs.get("attn_self_loop_fill_value", "max")

        if has_attn:
            assert n_layers >= 3, f'Middle Block with attention enabled expected {n_layers} >= 3 n_layers.'
        else:
            assert n_layers >= 2, f"Middle Block with attention disabled expected {n_layers} >= 2 n_layers."
        
        self.pre_attn_res_layers = ResBlock(in_channels=n_channels, out_channels=n_channels, time_channels=time_channels, n_layers = n_layers - 2, norm = norm,
                                            conv_architecture=conv_architecture,
                                            use_checkpointing=use_checkpointing)
        self.post_attn_res_layer = ResBlock(in_channels=n_channels, out_channels=n_channels, time_channels=time_channels, n_layers = 1, norm = norm,
                                            conv_architecture=conv_architecture,
                                            use_checkpointing=use_checkpointing)
        # second to last layer can be a graph-attention layer
        if has_attn:
            self.attn = AttnBlock(in_channels=n_channels, out_channels = n_channels, time_channels=time_channels, n_layers=1, norm=norm,
                                  num_attn_heads=4,
                                  use_checkpointing=use_checkpointing,
                                  attn_self_loop_fill_value = attn_self_loop_fill_value
                                  )
        else:
            self.attn = MyIdentity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None):
        x = self.pre_attn_res_layers(x, t=t, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x = self.attn(x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x = self.post_attn_res_layer(x, t=t, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        return x
         


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()


# Graph UNet for graph-based conditioning
class MyGraphUNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 depth: int,
                 pool_ratio: Union[float, List[float]] = 0.5,
                 pool_layer: str = 'topk',
                 pool_multiplier: Union[float, List[float]] = 1.,
                 norm: str = 'batch',
                 sum_skip_connection: bool = True,
                 use_checkpointing: bool = False,
                 attn_self_loop_fill_value: str = "max",
                 apply_gcn_norm: bool = True
                 ):
        
        super(MyGraphUNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.pool_ratios = repeat(pool_ratio, depth)
        self.pool_layers = [pool_layer for _ in range(depth)]
        self.pool_multipliers = repeat(pool_multiplier, depth)
        self.sum_skip_connection = sum_skip_connection
        self.use_checkpointing = use_checkpointing
        
        self.attn_self_loop_fill_value = attn_self_loop_fill_value
        self.gcn_norm = apply_gcn_norm


        self.activation = nn.LeakyReLU() # F.relu
        self.norm = norm
        self.has_attn = True

        attn_layer_depth = 2
        conv_architecture = 'TAGConv' # 'GCNConv'

        self.time_embed = SinusoidalTimeEmbedding(n_channels=hidden_channels * 4)


        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        self.down_blocks.append(DownUpBlock(in_channels=in_channels, out_channels=hidden_channels, time_channels=self.time_embed.n_channels, n_layers = 1, norm = self.norm,
                                            conv_architecture=conv_architecture,
                                            use_checkpointing=self.use_checkpointing,
                                            attn_self_loop_fill_value = self.attn_self_loop_fill_value
                                            ))
        for i in range(self.depth):
            self.down_blocks.append(DownUpBlock(in_channels=hidden_channels, out_channels=hidden_channels, time_channels=self.time_embed.n_channels, n_layers = 2, norm = self.norm,
                                                conv_architecture=conv_architecture, has_attn=self.has_attn if i >= attn_layer_depth else False,
                                                use_checkpointing=self.use_checkpointing,
                                                attn_self_loop_fill_value = self.attn_self_loop_fill_value
                                                ))
            self.downsample.append(self.resolve_pooling_layer(in_channels=hidden_channels, ratio=self.pool_ratios[i], layer=self.pool_layers[i], pool_multiplier = self.pool_multipliers[i]))

        self.midblock = MidBlock(n_channels = hidden_channels, time_channels=self.time_embed.n_channels, n_layers=3, norm=self.norm,
                                 has_attn=self.has_attn,
                                 use_checkpointing=self.use_checkpointing,
                                 attn_self_loop_fill_value = self.attn_self_loop_fill_value
                                 )

        in_channels = hidden_channels if sum_skip_connection else 2 * hidden_channels

        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(self.depth - 1):
            self.up_blocks.append(DownUpBlock(in_channels=in_channels, out_channels=hidden_channels, time_channels=self.time_embed.n_channels, n_layers=3, norm=self.norm,
                                              conv_architecture=conv_architecture,
                                              has_attn=self.has_attn if i < self.depth - attn_layer_depth else False,
                                              use_checkpointing=self.use_checkpointing,
                                              attn_self_loop_fill_value = self.attn_self_loop_fill_value
                                              ))
            self.upsample.append(Upsample())
        self.up_blocks.append(DownUpBlock(in_channels=in_channels, out_channels=out_channels, time_channels=self.time_embed.n_channels, n_layers=3, norm=self.norm,
                                          conv_architecture=conv_architecture,
                                          use_checkpointing=self.use_checkpointing,
                                          attn_self_loop_fill_value = self.attn_self_loop_fill_value
                                          ))

        # Linear projection at the final layer
        # self.out_norm_layer = NormLayer(norm='batch', in_channels=out_channels)
        self.out_norm_layer = NormLayer(norm='graph', in_channels=out_channels)
        self.out_block = nn.Sequential(self.activation, nn.Linear(out_channels, self.in_channels))
    

    # def reset_parameters(self):
    #     r"""Resets all learnable parameters of the module."""
    #     for conv in self.down_blocks:
    #         conv.reset_parameters()
    #     for pool in self.pools:
    #         pool.reset_parameters()
    #     for conv in self.up_convs:
    #         conv.reset_parameters()
        
    def forward(self, x, edge_index, edge_weight=None, batch=None, t = None, return_attn_weights = False, debug_forward_pass = False):

        t_embed = self.time_embed(t) # get time embeddings [1] -> [256]

        x = self.down_blocks[0](x, t_embed, edge_index = edge_index, edge_weight = edge_weight, batch = batch)

        xs = [x]
        ts = [t_embed]
        batches = [batch]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        ### Down-resolution half of UNet ###
        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0),
                                                       remove_and_add_self_loops=False,
                                                       apply_gcn_norm=self.gcn_norm
                                                       )
            
            x, edge_index, edge_weight, batch, perm, _ = self.downsample[i-1](x, edge_index, edge_weight, batch)
            # x, t_embed = x_t_concat[..., 0:1], x_t_concat[..., 1:2]
            t_embed = t_embed[perm]
            x = self.down_blocks[i](x, t_embed, edge_index, edge_weight, batch)

            if i < self.depth:
                xs += [x]
                ts += [t_embed]
                batches += [batch]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        ### Middle Block ### 
        x = self.midblock(x, t_embed, edge_index=edge_index, edge_weight=edge_weight, batch=batch)

        ### Up-resolution half of UNet ###
        for i in range(self.depth):
            j = self.depth - 1 - i
            x_skip = xs[j]
            t_skip = ts[j]
            edge_index, edge_weight, perm, batch = edge_indices[j], edge_weights[j], perms[j], batches[j]

            up = torch.zeros_like(x_skip)
            up[perm] = x

            x = x_skip + up if self.sum_skip_connection else torch.cat((x_skip, up), dim = -1)

            x = self.up_blocks[i](x, t_skip, edge_index, edge_weight, batch)

        attn_weights = None
        if debug_forward_pass:
            debug_data = {'edge_indices': [e.detach().cpu() for e in edge_indices],
                          'edge_weights': [w.detach().cpu() for w in edge_weights],
                          'batches': [batch.detach().cpu() for batch in batches]
                          }
        else:
            debug_data = None
            
        x = self.out_norm_layer(x, batch = batch)
        out = self.out_block(x)

        return out, attn_weights, debug_data
    
    
    def augment_adj(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                    num_nodes: int, remove_and_add_self_loops = False,
                    apply_gcn_norm = False):
        
        if remove_and_add_self_loops:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                    num_nodes=num_nodes)

            
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        adj = (adj @ adj @ adj + adj @ adj + adj).to_sparse_coo() # linear combination of first three powers is said to be better
        
        thresh = 5
        # assert torch.all(adj) < thresh, f"Some edge weights are greater than {thresh} after augmentation. Max edge weight = {adj.max().item()}"

        edge_index, edge_weight = adj.indices(), adj.values()

        if remove_and_add_self_loops:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        
        if apply_gcn_norm:
            edge_weight_old = edge_weight.clone()
            # print('Gcn norm applied.')
            edge_index, edge_weight = gcn_norm(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes, add_self_loops=False)
            if not torch.all(edge_weight_old) < thresh:
                print(f"Some edge weights were greater than {thresh} before adj-augmentation with gcn_norm(). Max edge weight = {edge_weight_old.max().item()}")
                print(f"After adj-augmentation with gcn_norm(), max edge weight = {edge_weight.max().item()}")

        return edge_index, edge_weight
    

    def resolve_pooling_layer(self, in_channels, ratio, layer = 'topk', **kwargs):
        if layer == 'topk':
            multiplier = kwargs.get("pool_multiplier", 1.)
            print(f'pooling_ratio = {ratio}')
            print(f'pooling_multiplier = {multiplier}')
            pool = TopKPooling(in_channels=in_channels, ratio=ratio, multiplier=multiplier)
        elif layer == 'spectral':
            raise NotImplementedError
        elif layer == 'centrality':
            raise NotImplementedError
        else:
            raise ValueError
        
        return pool