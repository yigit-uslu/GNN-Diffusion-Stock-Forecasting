from typing import Callable, List, Union
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# from core.GraphUnet import MyGraphUNet
from models.GraphUnet import ConvLayer, DownUpBlock, MidBlock, SinusoidalTimeEmbedding, NormLayer

class GNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 depth: int,
                 norm: str = 'batch',
                 sum_skip_connection: bool = True,
                 **kwargs
                 ):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.sum_skip_connection = sum_skip_connection
        
        self.activation = nn.LeakyReLU() # F.relu
        self.norm = norm

        conv_architecture = 'TAGConv'

        self.time_embed = SinusoidalTimeEmbedding(n_channels=hidden_channels * 4)

        self.in_blocks = nn.ModuleList()
        self.in_blocks.append(DownUpBlock(in_channels=in_channels, out_channels=hidden_channels, time_channels=self.time_embed.n_channels,
                                       n_layers = 2,
                                       norm = self.norm,
                                       conv_architecture=conv_architecture
                                       )
                        )
        
        for i in range(self.depth):
            self.in_blocks.append(DownUpBlock(in_channels=hidden_channels, out_channels=hidden_channels, time_channels=self.time_embed.n_channels,
                                           n_layers = 2,
                                           norm = self.norm,
                                           conv_architecture=conv_architecture
                                        )
                            )

        self.midblock = MidBlock(n_channels = hidden_channels, time_channels=self.time_embed.n_channels,
                                 n_layers=3,
                                 norm=self.norm,
                                 conv_architecture=conv_architecture
                                 )
        
        in_channels = hidden_channels if sum_skip_connection else 2 * hidden_channels

        self.out_blocks = nn.ModuleList()
        for i in range(self.depth - 1):
            self.out_blocks.append(DownUpBlock(in_channels=in_channels, out_channels=hidden_channels, time_channels=self.time_embed.n_channels,
                                               n_layers=2,
                                               norm=self.norm,
                                               conv_architecture=conv_architecture
                                                )
                                )
            
        self.out_blocks.append(DownUpBlock(in_channels=in_channels, out_channels=out_channels, time_channels=self.time_embed.n_channels,
                                          n_layers=2,
                                          norm=self.norm,
                                          conv_architecture=conv_architecture
                                          ))
               
        # Linear projection at the final layer
        self.out_norm_layer = NormLayer(norm='graph', in_channels=out_channels)
        self.out_block = nn.Sequential(self.activation, nn.Linear(out_channels, self.in_channels))



    def forward(self, x, edge_index, edge_weight=None, batch=None, t = None, return_attn_weights = False, debug_forward_pass = False):
        t_embed = self.time_embed(t) # get time embeddings [1] -> [256]

        x = self.in_blocks[0](x, t_embed, edge_index = edge_index, edge_weight = edge_weight, batch = batch)

        xs = [x]
        ts = [t_embed]
        norms = [{"layer": 0, "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()}]
        # batches = [batch]
        # edge_indices = [edge_index]
        # edge_weights = [edge_weight]
        # perms = []

        ### Down-resolution half of UNet ###
        for i in range(1, self.depth + 1):
            # edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
            #                                            x.size(0),
            #                                            remove_and_add_self_loops=False,
            #                                            apply_gcn_norm=self.gcn_norm
            #                                            )
            
            # x, edge_index, edge_weight, batch, perm, _ = self.downsample[i-1](x, edge_index, edge_weight, batch)
            # # x, t_embed = x_t_concat[..., 0:1], x_t_concat[..., 1:2]
            # t_embed = t_embed[perm]
            x = self.in_blocks[i](x, t_embed, edge_index = edge_index, edge_weight = edge_weight, batch = batch)

            if i < self.depth:
                xs += [x]
                ts += [t_embed]
                norms += [{"layer": i, "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()}]
                # batches += [batch]
            #     edge_indices += [edge_index]
            #     edge_weights += [edge_weight]
            # perms += [perm]

        ### Middle Block ### 
        x = self.midblock(x, t_embed, edge_index=edge_index, edge_weight=edge_weight, batch=batch)

        norms += [{"layer": self.depth, "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()}]

        
        ### Up-resolution half of UNet ###
        for i in range(self.depth):
            j = self.depth - 1 - i
            x_skip = xs[j]
            t_skip = ts[j]
            # edge_index, edge_weight, perm, batch = edge_indices[j], edge_weights[j], perms[j], batches[j]

            # up = torch.zeros_like(x_skip)
            # up[perm] = x

            x = x_skip + x if self.sum_skip_connection else torch.cat((x_skip, x), dim = -1)

            x = self.out_blocks[i](x, t_skip, edge_index, edge_weight, batch)

            norms += [{"layer": -(i+1), "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()}]

        attn_weights = None
        if debug_forward_pass:
            debug_data = {
                        #   'edge_indices': [e.detach().cpu() for e in edge_indices],
                          'edge_indices': [edge_index.detach().cpu()],
                        #   'edge_weights': [w.detach().cpu() for w in edge_weights],
                          'edge_weights': [edge_weight.detach().cpu()],
                          'batches': [batch.detach().cpu()],
                          'model_layer_norms': norms
                          }
        else:
            debug_data = None
            
        x = self.out_norm_layer(x, batch = batch)
        out = self.out_block(x)

        return out, attn_weights, debug_data



class SimpleGNN(nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                depth: int,
                norm: str = 'batch',
                **kwargs
                ):
        super(SimpleGNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels 
        self.depth = depth 

        self.conv_layer_normalize = kwargs.get("conv_layer_normalize", False)
        self.k_hops = kwargs.get("k_hops", 2)

        self.activation = nn.LeakyReLU()
        self.norm = norm
        conv_architecture = "TAGConv"
        self.time_embed = SinusoidalTimeEmbedding(n_channels=hidden_channels)

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.time_embed_layers = nn.ModuleList()

        self.conv_layers.append(ConvLayer(in_channels=self.in_channels,
                                          out_channels=self.hidden_channels,
                                          architecture=conv_architecture,
                                          k_hops=self.k_hops,
                                          normalize=False # self.conv_layer_normalize
                                          )
                                )

        self.norm_layers.append(NormLayer(norm=self.norm, in_channels=self.hidden_channels))

        # self.time_embed_layers.append(nn.Sequential(nn.Linear(self.in_channels, self.hidden_channels), nn.LeakyReLU()))
        for i in range(1, self.depth):
            conv_layer = ConvLayer(in_channels=self.hidden_channels,
                                   out_channels=self.hidden_channels,
                                   architecture=conv_architecture,
                                   k_hops=self.k_hops,
                                   normalize=False # self.conv_layer_normalize
                                   )
            norm_layer = NormLayer(norm=self.norm, in_channels=self.hidden_channels)

            self.conv_layers.append(conv_layer)
            self.norm_layers.append(norm_layer)

        self.out_layers = nn.Linear(hidden_channels, self.in_channels)

        # Apply custom weight initialization
        self.apply(self.init_weights)

        self.n_iters_trained = 0


    # Manual Laplacian Normalization
    def laplacian_normalize(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0  # Avoid division by zero
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    

    def spectral_laplacian_normalize(self, edge_index, edge_weight, num_nodes):
        """Computes spectral Laplacian normalization for a batched PyG graph."""
        row, col = edge_index

        # Compute the degree matrix
        deg = degree(row, num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0  # Avoid division by zero

        # Compute normalized adjacency: A_norm = D^(-1/2) * A * D^(-1/2)
        normalized_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Compute spectral Laplacian: L = I - A_norm
        identity_weight = torch.ones(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        spectral_laplacian_weight = identity_weight - normalized_weight  # L = I - A_norm

        return spectral_laplacian_weight



    def forward(self, x, edge_index, edge_weight=None, batch=None, t=None, return_attn_weights=False, debug_forward_pass=False):

        if self.conv_layer_normalize:
            try:
                add_self_loops = True
                improved = add_self_loops
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0),
                        improved=improved, add_self_loops=add_self_loops, # flow=self.conv_layers[0].flow,
                        dtype=x.dtype)
                print("GCN normalization applied to edge_weights.") if self.n_iters_trained == 0 else None

            except:
                print("GCN normalization failed.") if self.n_iters_trained == 0 else None

        # if self.conv_layer_normalize:
        #     edge_weight = self.spectral_laplacian_normalize(edge_index, edge_weight, num_nodes=x.size(0))
        #     print("Spectral Laplacian normalization applied to edge_weights.")

        t_embed = self.time_embed(t)
        # norms = [{"layer": "0I", "norm": torch.norm(x, dim = 1).mean().item()}]
        x = self.conv_layers[0](x, edge_index=edge_index, edge_weight=edge_weight)
        norms = [{"layer": "0C", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                 {"layer": "0C-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}]
        x = self.norm_layers[0](x, batch=batch)
        norms += [{"layer": "0N", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                  {"layer": "0N-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}]
        x = x + t_embed
        norms += [{"layer": "0S", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                  {"layer": "0S-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}]
        x = self.activation(x)
        norms += [{"layer": "0O", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                  {"layer": "0O-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}]

        xs = [x]
        ts = [t_embed]

        for i, (conv_layer, norm_layer) in enumerate(zip(self.conv_layers[1:], self.norm_layers[1:])):

            x = conv_layer(x, edge_index=edge_index, edge_weight=edge_weight)
            norms += [{"layer": f"{i+1}C", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                      {"layer": f"{i+1}C-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}]
            x = norm_layer(x, batch=batch)
            norms += [{"layer": f"{i+1}N", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                      {"layer": f"{i+1}N-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}]
            x = x + t_embed
            norms += [{"layer": f"{i+1}S", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                      {"layer": f"{i+1}S-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}]
            x = self.activation(x)
            norms += [{"layer": f"{i+1}O", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                      {"layer": f"{i+1}O-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}]

            xs += [x]
            ts += [t_embed]
            
        out = self.out_layers(x)
        norms += [{"layer": -1, "norm": torch.linalg.vector_norm(out, ord = 2, dim = 1).mean().item()},
                  {"layer": "-1-var", "norm": torch.linalg.vector_norm(out, ord = 2, dim = 1).var().item()}]

        attn_weights = None

        if debug_forward_pass:
            debug_data = {
                        #   'edge_indices': [e.detach().cpu() for e in edge_indices],
                          'edge_indices': [edge_index.detach().cpu()],
                        #   'edge_weights': [w.detach().cpu() for w in edge_weights],
                          'edge_weights': [edge_weight.detach().cpu()],
                          'batches': [batch.detach().cpu()],
                          'model_layer_norms': norms
                          }
        else:
            debug_data = None

        self.n_iters_trained += 1

        return out, attn_weights, debug_data
    

    # def init_weights(self, m):
    #     """ Custom weight initialization for GNN layers. """
    #     if isinstance(m, pyg_nn.GraphConv):
    #         torch.nn.init.kaiming_uniform_(m.lin.weight, nonlinearity='relu')  # He initialization
    #         if m.lin.bias is not None:
    #             torch.nn.init.zeros_(m.lin.bias)
    #     elif isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)  # Xavier initialization for final layers
    #         if m.bias is not None:
    #             torch.nn.init.zeros_(m.bias)

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




# # Graph UNet for graph-based conditioning
# class GNN(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  hidden_channels: int,
#                  out_channels: int,
#                  depth: int,
#                  pool_ratio: Union[float, List[float]] = 0.5,
#                  pool_layer: str = 'topk',
#                  pool_multiplier: Union[float, List[float]] = 1.,
#                  norm: str = 'batch',
#                  sum_skip_connection: bool = True,
#                  use_checkpointing: bool = False,
#                  attn_self_loop_fill_value: str = "max",
#                  apply_gcn_norm: bool = True
#                  ):
        
#         super(MyGraphUNet, self).__init__()
#         self.depth = depth
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         self.pool_ratios = repeat(pool_ratio, depth)
#         self.pool_layers = [pool_layer for _ in range(depth)]
#         self.pool_multipliers = repeat(pool_multiplier, depth)
#         self.sum_skip_connection = sum_skip_connection
#         self.use_checkpointing = use_checkpointing
        
#         self.attn_self_loop_fill_value = attn_self_loop_fill_value
#         self.gcn_norm = apply_gcn_norm

#         self.activation = nn.LeakyReLU() # F.relu
#         self.norm = norm
#         self.has_attn = True

#         attn_layer_depth = 2
#         conv_architecture = 'TAGConv' # 'GCNConv'

#         self.time_embed = SinusoidalTimeEmbedding(n_channels=hidden_channels * 4)


#         self.down_blocks = nn.ModuleList()
#         self.downsample = nn.ModuleList()

#         self.down_blocks.append(DownUpBlock(in_channels=in_channels, out_channels=hidden_channels, time_channels=self.time_embed.n_channels, n_layers = 1, norm = self.norm,
#                                             conv_architecture=conv_architecture,
#                                             use_checkpointing=self.use_checkpointing,
#                                             attn_self_loop_fill_value = self.attn_self_loop_fill_value
#                                             ))
#         for i in range(self.depth):
#             self.down_blocks.append(DownUpBlock(in_channels=hidden_channels, out_channels=hidden_channels, time_channels=self.time_embed.n_channels, n_layers = 2, norm = self.norm,
#                                                 conv_architecture=conv_architecture, has_attn=self.has_attn if i >= attn_layer_depth else False,
#                                                 use_checkpointing=self.use_checkpointing,
#                                                 attn_self_loop_fill_value = self.attn_self_loop_fill_value
#                                                 ))
#             self.downsample.append(self.resolve_pooling_layer(in_channels=hidden_channels, ratio=self.pool_ratios[i], layer=self.pool_layers[i], pool_multiplier = self.pool_multipliers[i]))

#         self.midblock = MidBlock(n_channels = hidden_channels, time_channels=self.time_embed.n_channels, n_layers=3, norm=self.norm,
#                                  has_attn=self.has_attn,
#                                  use_checkpointing=self.use_checkpointing,
#                                  attn_self_loop_fill_value = self.attn_self_loop_fill_value
#                                  )

#         in_channels = hidden_channels if sum_skip_connection else 2 * hidden_channels

#         self.up_blocks = nn.ModuleList()
#         self.upsample = nn.ModuleList()
#         for i in range(self.depth - 1):
#             self.up_blocks.append(DownUpBlock(in_channels=in_channels, out_channels=hidden_channels, time_channels=self.time_embed.n_channels, n_layers=3, norm=self.norm,
#                                               conv_architecture=conv_architecture,
#                                               has_attn=self.has_attn if i < self.depth - attn_layer_depth else False,
#                                               use_checkpointing=self.use_checkpointing,
#                                               attn_self_loop_fill_value = self.attn_self_loop_fill_value
#                                               ))
#             self.upsample.append(Upsample())
#         self.up_blocks.append(DownUpBlock(in_channels=in_channels, out_channels=out_channels, time_channels=self.time_embed.n_channels, n_layers=3, norm=self.norm,
#                                           conv_architecture=conv_architecture,
#                                           use_checkpointing=self.use_checkpointing,
#                                           attn_self_loop_fill_value = self.attn_self_loop_fill_value
#                                           ))

#         # Linear projection at the final layer
#         # self.out_norm_layer = NormLayer(norm='batch', in_channels=out_channels)
#         self.out_norm_layer = NormLayer(norm='graph', in_channels=out_channels)
#         self.out_block = nn.Sequential(self.activation, nn.Linear(out_channels, self.in_channels))
    
        
#     def forward(self, x, edge_index, edge_weight=None, batch=None, t = None, return_attn_weights = False, debug_forward_pass = False):

#         t_embed = self.time_embed(t) # get time embeddings [1] -> [256]

#         x = self.down_blocks[0](x, t_embed, edge_index = edge_index, edge_weight = edge_weight, batch = batch)

#         xs = [x]
#         ts = [t_embed]
#         batches = [batch]
#         edge_indices = [edge_index]
#         edge_weights = [edge_weight]
#         perms = []

#         ### Down-resolution half of UNet ###
#         for i in range(1, self.depth + 1):
#             edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
#                                                        x.size(0),
#                                                        remove_and_add_self_loops=False,
#                                                        apply_gcn_norm=self.gcn_norm
#                                                        )
            
#             x, edge_index, edge_weight, batch, perm, _ = self.downsample[i-1](x, edge_index, edge_weight, batch)
#             # x, t_embed = x_t_concat[..., 0:1], x_t_concat[..., 1:2]
#             t_embed = t_embed[perm]
#             x = self.down_blocks[i](x, t_embed, edge_index, edge_weight, batch)

#             if i < self.depth:
#                 xs += [x]
#                 ts += [t_embed]
#                 batches += [batch]
#                 edge_indices += [edge_index]
#                 edge_weights += [edge_weight]
#             perms += [perm]

#         ### Middle Block ### 
#         x = self.midblock(x, t_embed, edge_index=edge_index, edge_weight=edge_weight, batch=batch)

#         ### Up-resolution half of UNet ###
#         for i in range(self.depth):
#             j = self.depth - 1 - i
#             x_skip = xs[j]
#             t_skip = ts[j]
#             edge_index, edge_weight, perm, batch = edge_indices[j], edge_weights[j], perms[j], batches[j]

#             up = torch.zeros_like(x_skip)
#             up[perm] = x

#             x = x_skip + up if self.sum_skip_connection else torch.cat((x_skip, up), dim = -1)

#             x = self.up_blocks[i](x, t_skip, edge_index, edge_weight, batch)

#         attn_weights = None
#         if debug_forward_pass:
#             debug_data = {'edge_indices': [e.detach().cpu() for e in edge_indices],
#                           'edge_weights': [w.detach().cpu() for w in edge_weights],
#                           'batches': [batch.detach().cpu() for batch in batches]
#                           }
#         else:
#             debug_data = None
            
#         x = self.out_norm_layer(x, batch = batch)
#         out = self.out_block(x)

#         return out, attn_weights, debug_data
    
    
#     def augment_adj(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,
#                     num_nodes: int, remove_and_add_self_loops = False,
#                     apply_gcn_norm = False):
        
#         if remove_and_add_self_loops:
#             edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
#             edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
#                                                     num_nodes=num_nodes)

            
#         adj = to_torch_csr_tensor(edge_index, edge_weight,
#                                   size=(num_nodes, num_nodes))
#         adj = (adj @ adj @ adj + adj @ adj + adj).to_sparse_coo() # linear combination of first three powers is said to be better
        
#         thresh = 5
#         # assert torch.all(adj) < thresh, f"Some edge weights are greater than {thresh} after augmentation. Max edge weight = {adj.max().item()}"

#         edge_index, edge_weight = adj.indices(), adj.values()

#         if remove_and_add_self_loops:
#             edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        
#         if apply_gcn_norm:
#             edge_weight_old = edge_weight.clone()
#             print('Gcn norm applied.')
#             edge_index, edge_weight = gcn_norm(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes, add_self_loops=False)
#             if not torch.all(edge_weight_old) < thresh:
#                 print(f"Some edge weights were greater than {thresh} before adj-augmentation with gcn_norm(). Max edge weight = {edge_weight_old.max().item()}")
#                 print(f"After adj-augmentation with gcn_norm(), max edge weight = {edge_weight.max().item()}")

#         return edge_index, edge_weight
    

#     def resolve_pooling_layer(self, in_channels, ratio, layer = 'topk', **kwargs):
#         if layer == 'topk':
#             multiplier = kwargs.get("pool_multiplier", 1.)
#             print(f'pooling_ratio = {ratio}')
#             print(f'pooling_multiplier = {multiplier}')
#             pool = TopKPooling(in_channels=in_channels, ratio=ratio, multiplier=multiplier)
#         elif layer == 'spectral':
#             raise NotImplementedError
#         elif layer == 'centrality':
#             raise NotImplementedError
#         else:
#             raise ValueError
        
#         return pool