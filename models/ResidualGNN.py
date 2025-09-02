import torch 
import torch.nn as nn
from models.GraphUnet import ConvLayer, NormLayer, SinusoidalTimeEmbedding
from torch.utils.checkpoint import checkpoint

import torch_geometric.nn as pyg_nn
from torch_geometric.nn.models.mlp import MLP as pyg_mlp
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from typing import Optional
import torch.nn.functional as F
# from torch import Tensor
# from torch.nn import Module


class ResidualLayer(nn.Module):
    r"""
    The DeepGCN with residual connections block adapted to my implementation:

    The skip connection operations from the
    `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_ and `"All You Need to Train Deeper
    GCNs" <https://arxiv.org/abs/2006.07739>`_ papers.
    The implemented skip connections includes the pre-activation residual
    connection (:obj:`"res+"`), the residual connection (:obj:`"res"`),
    the dense connection (:obj:`"dense"`) and no connections (:obj:`"plain"`).
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
        norm_kws: Optional[dict] = None,
        use_checkpointing: bool = False,
        return_norms: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.k_hops = k_hops
        self.conv_architecture = conv_architecture
        assert self.conv_architecture in ['LeConv', 'TAGConv', 'GCN']
        self.norm = norm
        assert self.norm in ['batch', 'group', 'graph', 'layer', 'none', 'instance']
        self.act = act
        self.res_connection = res_connection.lower()
        assert self.res_connection in ['res+', 'res', 'dense', 'plain']
        self.mlp = mlp
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing

        print(f"Using dropout rate = {self.dropout}")

        self.return_norms = return_norms

        ### Initialize the convolution layer
        self.conv = ConvLayer(in_channels=in_channels, out_channels=out_channels,
                              architecture = self.conv_architecture, k_hops=self.k_hops,
                              normalize = False,
                              add_self_loops = False
                              )
            
        self.mlp = nn.Identity() if self.mlp is None else self.mlp
        self.norm = NormLayer(norm=norm, in_channels=in_channels, **norm_kws)

        if self.in_channels != self.out_channels:
            self.res = nn.Linear(self.in_channels, self.out_channels, bias=False)
        else:
            self.res = nn.Identity()
            


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.norm.reset_parameters() if self.norm is not None else None
        self.mlp.reset_parameters() if self.mlp is not None else None


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None):        

        if self.res_connection == 'res+':
            h = x
            if self.norm is not None:
                h = self.norm(h, batch = batch, batch_size = None) # batch size computation TO DO if GraphNorm is used.

                if self.return_norms:
                    norms = [{"layer": "N", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).mean().item()},
                            {"layer": "N-var", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).var().item()}]
                    
                else:
                    norms = None

            if self.act is not None:
                h = self.act(h)

                if self.return_norms:
                    norms += [{"layer": "A", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).mean().item()},
                            {"layer": "A-var", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).var().item()}]
                
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_checkpointing:
                h = checkpoint.checkpoint(self.custom_conv_embed(self.conv), h, edge_index, edge_weight, use_reentrant=False)
            else:
                h = self.conv(h, edge_index=edge_index, edge_weight=edge_weight)
            
            if self.return_norms:
                norms += [{"layer": "C", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).mean().item()},
                        {"layer": "C-var", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).var().item()}]
            
            h = self.mlp(h)
            # assert h.size(1) == x.size(1), f"Hidden channels mismatch. {h.size(1)}-{x.size(1)}"
            h = h + self.res(x)  # residual connection

            if self.return_norms:
                norms += [{"layer": "O", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).mean().item()},
                        {"layer": "O-var", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).var().item()}]

            return h, norms

        else:
            
            raise NotImplementedError
        

            # if self.conv is not None and self.ckpt_grad and x.requires_grad:
            #     h = checkpoint(self.conv, x, *args, use_reentrant=True,
            #                    **kwargs)
            # else:
            #     h = self.conv(x, *args, **kwargs)
            # if self.norm is not None:
            #     h = self.norm(h)
            # if self.act is not None:
            #     h = self.act(h)

            # if self.block == 'res':
            #     h = x + h
            # elif self.block == 'dense':
            #     h = torch.cat([x, h], dim=-1)
            # elif self.block == 'plain':
            #     pass

            # return F.dropout(h, p=self.dropout, training=self.training)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(res={self.res_connection})'


class ResidualGNN(nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                depth: int,
                norm: str = 'batch',
                **kwargs
                ):
        super(ResidualGNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth 

        self.conv_layer_normalize = kwargs.get("conv_layer_normalize", False)
        self.k_hops = kwargs.get("k_hops", 2)
        self.dropout_rate = kwargs.get("dropout_rate", 0.0)
        self.res_connection = kwargs.get("res_connection", "res+")
        self.use_res_connection_time_embed = kwargs.get("use_res_connection_time_embed", False)
        self.time_embed_strategy = kwargs.get("time_embed_strategy", "add")
        self.use_checkpointing = kwargs.get("use_checkpointing", False)

        self.activation = nn.LeakyReLU(inplace=True)
        self.norm = norm
        conv_architecture = kwargs.get("conv_model", "TAGConv")

        # self.x_embed = nn.Linear(in_channels, hidden_channels // 4)
        self.x_embed = nn.Linear(in_channels, hidden_channels)
        self.time_embed = SinusoidalTimeEmbedding(n_channels=self.x_embed.out_features)


        self.res_conv_layers = nn.ModuleList()
        # self.time_embed_layers = [self.time_embed] # nn.ModuleList()
        # self.x_embed_layers = [self.x_embed]

        self.res_conv_layers.append(ResidualLayer(in_channels=self.x_embed.out_features,
                                                #   in_channels=hidden_channels // 4,
                                                  out_channels=hidden_channels,
                                                  conv_architecture=conv_architecture,
                                                  k_hops=self.k_hops,
                                                  norm=self.norm,
                                                  norm_kws=kwargs.get("norm_kws", dict()),
                                                  mlp=pyg_mlp(channel_list = [hidden_channels, 4 * hidden_channels, hidden_channels], act = self.activation),
                                                  res_connection=self.res_connection,
                                                  dropout=self.dropout_rate, # 0.0,
                                                  use_checkpointing=self.use_checkpointing,
                                                  return_norms=False
                                                  )
                                    )
        
        # hidden_channels = self.res_conv_layers[-1].out_channels
        # print(f"Hidden channels = {self.hidden_channels} increased from 1st layer to: ", hidden_channels)

        # self.norm_layers.append(NormLayer(norm=self.norm, in_channels=self.hidden_channels))

        # self.time_embed_layers.append(nn.Sequential(nn.Linear(self.in_channels, self.hidden_channels), nn.LeakyReLU()))
        for i in range(1, self.depth):
            # self.time_embed_layers.append(nn.Identity())
            # self.x_embed_layers.append(nn.Identity())

            res_conv_layer = ResidualLayer(in_channels=hidden_channels,
                                           out_channels=hidden_channels,
                                           conv_architecture=conv_architecture,
                                           k_hops=self.k_hops,
                                           norm=self.norm,
                                           norm_kws=kwargs.get("norm_kws", dict()),
                                           mlp=pyg_mlp(channel_list = [hidden_channels, 4 * hidden_channels, hidden_channels], act = self.activation),
                                           res_connection=self.res_connection,
                                           dropout=self.dropout_rate, # 0.0,
                                           use_checkpointing=self.use_checkpointing
                                           ) 

            self.res_conv_layers.append(res_conv_layer)

        # assert len(self.res_conv_layers) == len(self.time_embed_layers) == len(self.x_embed_layers) == self.depth, \
        #     f"Mismatch in the number of layers. {len(self.res_conv_layers)}-res-conv-layers, {len(self.time_embed_layers)}-time-embed-layers, {len(self.x_embed_layers)}-x-embed-layers."
        
        self.out_layers = nn.Linear(hidden_channels, self.out_channels)

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
    

    def add_or_multiply_time_embeddings(self, x, t_embed):
        if self.time_embed_strategy == "add":
            return x + t_embed
        elif self.time_embed_strategy == "multiply":
            return x * t_embed
        else:
            raise NotImplementedError(f"Unknown time embedding strategy: {self.time_embed_strategy}")
        

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

        all_norms = []

        x = self.x_embed(x)

        if t is not None:
            assert x.size(0) == t.size(0), f"Batch size mismatch. {x.size(0)}-{t.size(0)}"
            t = self.time_embed(t)
            assert x.size(1) == t.size(1), f"Hidden channels mismatch. {x.size(1)}-{t.size(1)}" 

            # x = x + t # add time embeddings only at the input layer
            x = self.add_or_multiply_time_embeddings(x, t)
        
        
        ### Start with single convolution layer similar to DeepGCN architecture ###
        # x, norms = self.res_conv_layers[0](x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x = self.res_conv_layers[0].conv(x, edge_index=edge_index, edge_weight=edge_weight)
        norms = [{"layer": "C", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                 {"layer": "C-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}] if self.res_conv_layers[0].return_norms else None
        ### Start with single convolution layer similar to DeepGCN architecture ###

        if debug_forward_pass and norms is not None:
            # Compute individual layer statistics and label them by layer_id.
            for norm in norms:
                norm["layer"] = f"0{norm['layer']}"
            all_norms += norms

        for layer_id, conv_layer in enumerate(self.res_conv_layers[1:], start = 1):
            
            # if layer_id == 0:
            #     x = x_embed_layer(x)
            #     t = t_embed_layer(t)
            #     assert x.size(1) == t.size(1), f"Hidden channels mismatch. {x.size(1)}-{t.size(1)}" 
            # # assert t.size(1) == self.hidden_channels, f"Hidden channels mismatch. {t.size(1)}-{self.hidden_channels}"

            # x = x + (layer_id == 0) * t # add time embeddings only to the input layer

            x, norms = conv_layer(x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)

            if self.use_res_connection_time_embed:
                print("Using residual connections for time embedddings.") if self.n_iters_trained == 0 else None
                assert t is not None, "Time embeddings are required for residual time connection."
                assert x.size(1) == t.size(1), f"Hidden channels mismatch. {x.size(1)}-{t.size(1)}"

                x = self.add_or_multiply_time_embeddings(x, t)

            if debug_forward_pass and norms is not None:
                # Compute individual layer statistics and label them by layer_id.
                for norm in norms:
                    norm["layer"] = f"{layer_id}{norm['layer']}"
                all_norms += norms

        ### New part to the forward pass ###
        x = self.res_conv_layers[0].act(self.res_conv_layers[0].norm(x, batch = batch, batch_size = None))
        ### New part to the forward pass ###
        
        out = self.out_layers(x)

        if debug_forward_pass:
            all_norms += [{"layer": "-1", "norm": torch.linalg.vector_norm(out, ord = 2, dim = 1).mean().item()},
                    {"layer": "-1-var", "norm": torch.linalg.vector_norm(out, ord = 2, dim = 1).var().item()}]

        attn_weights = None

        if debug_forward_pass:
            debug_data = {
                        #   'edge_indices': [e.detach().cpu() for e in edge_indices],
                          'edge_indices': [edge_index.detach().cpu()],
                        #   'edge_weights': [w.detach().cpu() for w in edge_weights],
                          'edge_weights': [edge_weight.detach().cpu()],
                          'batches': [batch.detach().cpu()],
                          'model_layer_norms': all_norms
                          }
        else:
            debug_data = None

        # print(f"Pred_noise.shape: {out.shape}")

        self.n_iters_trained += 1
        return out, attn_weights, debug_data
    

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






class ResidualGNNwithConditionalEmbeddings(ResidualGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, norm, cond_in_channels, **kwargs):
        super(ResidualGNNwithConditionalEmbeddings, self).__init__(in_channels=in_channels,
                                                                  hidden_channels=hidden_channels,
                                                                  out_channels=out_channels,
                                                                  depth=depth,
                                                                  norm=norm,
                                                                  **kwargs)
        self.cond_in_channels = cond_in_channels

        self.cond_embedding_layer = nn.Sequential(
            nn.Linear(cond_in_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.cond_embed_strategy = kwargs.get("cond_embed_strategy", "multiply") # "add" or "multiply"


    def add_or_multiply_cond_embeddings(self, x, cond_embed):
        if self.cond_embed_strategy == "add":
            return x + cond_embed
        elif self.cond_embed_strategy == "multiply":
            return x * cond_embed
        else:
            raise NotImplementedError(f"Unknown conditional embedding strategy: {self.cond_embed_strategy}")


    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, t: torch.Tensor = None, edge_index = None, edge_weight = None, batch=None,
                return_attn_weights=False, debug_forward_pass=False):

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

        all_norms = []

        x = self.x_embed(x)

        if t is not None:
            assert x.size(0) == t.size(0), f"Batch size mismatch. {x.size(0)}-{t.size(0)}"
            t = self.time_embed(t)
            assert x.size(1) == t.size(1), f"Hidden channels mismatch. {x.size(1)}-{t.size(1)}" 

            x = self.add_or_multiply_time_embeddings(x, t)


        if cond is not None:
            print("Adding or multiplying conditional embeddings.") if self.n_iters_trained == 0 else None
            assert cond.size(0) == x.size(0), f"Batch size mismatch. {cond.size(0)}-{x.size(0)}"
            cond_embed = self.cond_embedding_layer(cond)
            assert x.size(1) == cond_embed.size(1), f"Hidden channels mismatch. {x.size(1)}-{cond_embed.size(1)}"

            x = self.add_or_multiply_cond_embeddings(x, cond_embed)

        ### Start with single convolution layer similar to DeepGCN architecture ###
        # x, norms = self.res_conv_layers[0](x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x = self.res_conv_layers[0].conv(x, edge_index=edge_index, edge_weight=edge_weight)
        norms = [{"layer": "C", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).mean().item()},
                 {"layer": "C-var", "norm": torch.linalg.vector_norm(x, ord = 2, dim = 1).var().item()}] if self.res_conv_layers[0].return_norms else None
        ### Start with single convolution layer similar to DeepGCN architecture ###

        if debug_forward_pass and norms is not None:
            # Compute individual layer statistics and label them by layer_id.
            for norm in norms:
                norm["layer"] = f"0{norm['layer']}"
            all_norms += norms

        for layer_id, conv_layer in enumerate(self.res_conv_layers[1:], start = 1):
            
            # if layer_id == 0:
            #     x = x_embed_layer(x)
            #     t = t_embed_layer(t)
            #     assert x.size(1) == t.size(1), f"Hidden channels mismatch. {x.size(1)}-{t.size(1)}" 
            # # assert t.size(1) == self.hidden_channels, f"Hidden channels mismatch. {t.size(1)}-{self.hidden_channels}"

            # x = x + (layer_id == 0) * t # add time embeddings only to the input layer

            x, norms = conv_layer(x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)

            if self.use_res_connection_time_embed:
                print("Using residual connections for time embeddings.") if self.n_iters_trained == 0 else None
                assert t is not None, "Time embeddings are required for residual time connection."
                assert x.size(1) == t.size(1), f"Hidden channels mismatch. {x.size(1)}-{t.size(1)}"
                x = x + t # add time embeddings to the output of each layer
                x = self.add_or_multiply_time_embeddings(x, t)

                # We use residual connections for cond embeddings if residual time embeddings are also utilized.
                print("Using residual connections also for cond embeddings.") if self.n_iters_trained == 0 else None
                assert cond_embed is not None, "Conditional embeddings are required for residual cond connection."
                assert x.size(1) == cond_embed.size(1), f"Hidden channels mismatch. {x.size(1)}-{cond_embed.size(1)}"
                x = self.add_or_multiply_cond_embeddings(x, cond_embed)

            if debug_forward_pass and norms is not None:
                # Compute individual layer statistics and label them by layer_id.
                for norm in norms:
                    norm["layer"] = f"{layer_id}{norm['layer']}"
                all_norms += norms

        ### New part to the forward pass ###
        x = self.res_conv_layers[0].act(self.res_conv_layers[0].norm(x, batch = batch, batch_size = None))
        ### New part to the forward pass ###
        
        out = self.out_layers(x)

        if debug_forward_pass:
            all_norms += [{"layer": "-1", "norm": torch.linalg.vector_norm(out, ord = 2, dim = 1).mean().item()},
                    {"layer": "-1-var", "norm": torch.linalg.vector_norm(out, ord = 2, dim = 1).var().item()}]

        attn_weights = None

        if debug_forward_pass:
            debug_data = {
                        #   'edge_indices': [e.detach().cpu() for e in edge_indices],
                          'edge_indices': [edge_index.detach().cpu()],
                        #   'edge_weights': [w.detach().cpu() for w in edge_weights],
                          'edge_weights': [edge_weight.detach().cpu()],
                          'batches': [batch.detach().cpu()],
                          'model_layer_norms': all_norms
                          }
        else:
            debug_data = None

        # print(f"Pred_noise.shape: {out.shape}")

        self.n_iters_trained += 1
        return out, attn_weights, debug_data