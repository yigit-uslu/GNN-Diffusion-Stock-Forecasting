import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.conv import TransformerConv, GATv2Conv, LEConv, TAGConv
from torch_geometric.transforms import GDC


class SinusoidalEmbedding(nn.Module):
    """
    https://github.com/tanelp/tiny-diffusion 
    """
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0).to(x.device)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size
    


# Graph Transformer v2 for graph-attention and FiLM-based conditioning
class GraphTransformerv2(nn.Module):
    # def __init__(self, num_in, graph_embed_num_in, num_out, nsteps, activation, nlayers, **kwargs):
    def __init__(self, nsteps, nlayers, in_channels_mlp, in_channels_gnn, hidden_channels_mlp, hidden_channels_gnn, timestep_embed_channels, **kwargs):
        super(GraphTransformerv2, self).__init__()
        self.nsteps = nsteps
        self.nlayers = nlayers
        self.activation = kwargs.get('activation', F.relu)
        self.attn_num_heads = kwargs.get('attn_num_heads', 4)

        timestep_embed = SinusoidalEmbedding(size=timestep_embed_channels, scale=1.)
        self.time_embed = nn.Sequential(*[timestep_embed, nn.Linear(timestep_embed_channels, 2), nn.Flatten(start_dim=-2)])

        
        self.conv_layers = nn.ModuleList([TransformerConv(in_channels=in_channels_gnn,
                                                          out_channels=hidden_channels_gnn,
                                                          heads=self.attn_num_heads,
                                                          concat=False,
                                                          edge_dim=0
                                                          )
                                        ])
        self.mlp_layers = nn.ModuleList([nn.Linear(in_channels_mlp, hidden_channels_mlp)])

        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels_gnn)])

        for i in range(self.nlayers):

            conv_layer = TransformerConv(in_channels=hidden_channels_gnn,
                                         out_channels=hidden_channels_gnn,
                                         heads=self.attn_num_heads,
                                         concat=False,
                                         edge_dim=0
                                         )
            self.conv_layers.append(conv_layer)

            mlp_layer = nn.Linear(hidden_channels_mlp, hidden_channels_mlp)
            self.mlp_layers.append(mlp_layer)

            bn_layer = BatchNorm(hidden_channels_gnn)
            self.bn_layers.append(bn_layer)
                
        self.conv_layers.append(TransformerConv(in_channels=hidden_channels_gnn,
                                                out_channels=2,
                                                heads=self.attn_num_heads,
                                                concat=False,
                                                edge_dim=0
                                                ))
        self.mlp_layers.append(nn.Linear(hidden_channels_mlp, in_channels_mlp))

        self.bn_layers.append(nn.Identity())


    def conditioning_block(self, x, condition):
        scale = condition[..., 0:1].view_as(x)
        shift = condition[..., 1:2].view_as(x)
        return x * scale + shift
    

    def convs(self, x, edge_index, edge_weight):
        for i, (layer, bn_layer) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = layer(x, edge_index = edge_index, edge_attr = edge_weight.unsqueeze(-1), return_attention_weights = None)
            if i < len(self.conv_layers) - 1:
                x = self.activation(x)
                x = bn_layer(x)
        return x
    
    def mlps(self, x):
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            if i < len(self.mlp_layers) - 1:
                x = self.activation(x)
        return x 
        
        
    def forward(self, x, edge_index, edge_weight=None, batch=None, t = None):
        # Add self-loops to the edge index if not already present
        # edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0)

        x_fc_embed = self.mlps(x.view(-1, self.mlp_layers[0].in_features).clone()) # [B, n]
        x_fc_embed = F.relu(x_fc_embed)

        if t is not None:
            # t = t / (self.nsteps - 1)
            t_embed = self.time_embed(t)

            x_fc_embed = self.conditioning_block(x=x_fc_embed, condition=t_embed)
            # print('t.shape: ', t.shape)
            # print('t_embed.shape: ', t_embed.shape)

        if edge_index is not None and edge_weight is not None:
            if self.conv_layers[0].in_channels == 1:
                x_graph_attn_embed = self.convs(x, edge_index, edge_weight)
            elif self.conv_layers[0].in_channels == 2: # feed time as a node signal too
                # print('x.shape: ', x.shape)
                # print('t.shape: ', t.shape)
                # print('[x, t]: ', torch.cat([x, t], dim = -1).shape)
                x_graph_attn_embed = self.convs(torch.cat([x, t], dim = -1), edge_index, edge_weight)
            else:
                raise ValueError
            x_graph_attn_embed = torch.moveaxis(input=x_graph_attn_embed, source=-1, destination=0) # [Bxn, 2] -> [2, Bxn]
            x_graph_attn_embed = x_graph_attn_embed.view(-1, *x_fc_embed.shape) # [2, B, n]
            x_graph_attn_embed = torch.moveaxis(input=x_graph_attn_embed, source=0, destination=-1) # [B, n, 2]

            out = self.conditioning_block(x=x_fc_embed, condition=x_graph_attn_embed) # [B, n]

        else:
            out = x_fc_embed

        out = out.view_as(x)
       
        return out
    



# Graph Transformer v3 for graph-attention and FiLM-based conditioning
class GraphTransformerv3(nn.Module):
    # def __init__(self, num_in, graph_embed_num_in, num_out, nsteps, activation, nlayers, **kwargs):
    def __init__(self, nsteps, nlayers, in_channels_mlp, in_channels_gnn, hidden_channels_mlp, out_channels_gnn, out_channels_mlp, hidden_channels_gnn, timestep_embed_channels, **kwargs):
        super(GraphTransformerv3, self).__init__()
        self.nsteps = nsteps
        self.nlayers = nlayers
        self.activation = kwargs.get('activation', F.relu)
        self.attn_num_heads = kwargs.get('attn_num_heads', 4)

        self.out_channels_gnn = out_channels_gnn
        self.out_channels_mlp = out_channels_mlp # out_channels_gnn * (in_channels_mlp // in_channels_gnn)

        timestep_embed = SinusoidalEmbedding(size=timestep_embed_channels, scale=1.)
        self.time_embed = nn.Sequential(*[timestep_embed, nn.Linear(timestep_embed_channels, out_channels_gnn), nn.Flatten(start_dim=-2)])

        
        self.conv_layers = nn.ModuleList([TransformerConv(in_channels=in_channels_gnn,
                                                          out_channels=hidden_channels_gnn,
                                                          heads=self.attn_num_heads,
                                                          concat=False,
                                                          edge_dim=0
                                                          )
                                        ])
        self.mlp_layers = nn.ModuleList([nn.Linear(in_channels_mlp, hidden_channels_mlp)])

        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels_gnn)])

        for i in range(self.nlayers):

            conv_layer = TransformerConv(in_channels=hidden_channels_gnn,
                                         out_channels=hidden_channels_gnn,
                                         heads=self.attn_num_heads,
                                         concat=False,
                                         edge_dim=0
                                         )
            self.conv_layers.append(conv_layer)

            mlp_layer = nn.Linear(hidden_channels_mlp, hidden_channels_mlp)
            self.mlp_layers.append(mlp_layer)

            bn_layer = BatchNorm(hidden_channels_gnn)
            self.bn_layers.append(bn_layer)
                
        self.conv_layers.append(TransformerConv(in_channels=hidden_channels_gnn,
                                                out_channels=out_channels_gnn,
                                                heads=self.attn_num_heads,
                                                concat=False,
                                                edge_dim=0
                                                ))
        
        self.mlp_layers.append(nn.Linear(hidden_channels_mlp, self.out_channels_mlp))

        self.bn_layers.append(nn.Identity())


    def conditioning_block(self, x, condition):
        scale = condition[..., 0:1].view_as(x)
        shift = condition[..., 1:2].view_as(x)
        return x * scale + shift
    

    def convs(self, x, edge_index, edge_weight):
        for i, (layer, bn_layer) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = layer(x, edge_index = edge_index, edge_attr = edge_weight.unsqueeze(-1), return_attention_weights = None)
            if i < len(self.conv_layers) - 1:
                x = self.activation(x)
                x = bn_layer(x)
        return x
    
    def mlps(self, x):
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            if i < len(self.mlp_layers) - 1:
                x = self.activation(x)
        return x 
        
        
    def forward(self, x, y, edge_index, edge_weight=None, batch=None, t = None):
        # Add self-loops to the edge index if not already present
        # edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0)


        # x [B, n, C_in]

        B, n, C_in = x.shape

        x_fc_embed = self.mlps(x.view(-1, self.mlp_layers[0].in_features).clone()) # [B, n x C_out]
        x_fc_embed = x_fc_embed.view(B, n, -1) # [B, n, C_out]
        C_out = x_fc_embed.shape[-1]
        x_fc_embed = F.relu(x_fc_embed)

        if t is not None:
            # t = t / (self.nsteps - 1)
            t_embed = self.time_embed(t)
            t_embed = torch.moveaxis(t_embed, -1, 0)
            t_embed = t_embed.view(-1, *x_fc_embed.shape[0:2])
            t_embed = torch.moveaxis(t_embed, 0, -1).view(*x_fc_embed.shape, -1) # [B, n, C_out, 2]

            x_fc_embed = self.conditioning_block(x=x_fc_embed, condition=t_embed)
            # print('t.shape: ', t.shape)
            # print('t_embed.shape: ', t_embed.shape)

        if edge_index is not None and edge_weight is not None:

            if self.conv_layers[0].in_channels == 1:
                x_graph_attn_embed = self.convs(y, edge_index, edge_weight)
            elif self.conv_layers[0].in_channels == 2: # feed time as a node signal too
                # print('x.shape: ', x.shape)
                # print('t.shape: ', t.shape)
                # print('[x, t]: ', torch.cat([x, t], dim = -1).shape)
                x_graph_attn_embed = self.convs(torch.cat([y, t], dim = -1), edge_index, edge_weight)
            else:
                raise ValueError
            
            x_graph_attn_embed = torch.moveaxis(input=x_graph_attn_embed, source=-1, destination=0) # [Bxn, 2 * C] -> [2 * C, Bxn]
            # C_out = x_graph_attn_embed.shape[-1] // 2
            x_graph_attn_embed = x_graph_attn_embed.view(-1, *x_fc_embed.shape[0:2]) # [2 * C_out, B, n]
            x_graph_attn_embed = torch.moveaxis(input=x_graph_attn_embed, source=0, destination=-1) # [B, n, 2 * C_out]
            # B = x_graph_attn_embed.shape[0]
            # x_graph_attn_embed = torch.cat([x_graph_attn_embed[..., :C_out].view(B, -1, 1), x_graph_attn_embed[..., C_out:].view(B, -1, 1)], dim = -1) # [B, nxC, 2]
            x_graph_attn_embed = x_graph_attn_embed.view(B, n, C_out, 2)

            out = self.conditioning_block(x=x_fc_embed, condition=x_graph_attn_embed) # [B, n, C]

        else:
            out = x_fc_embed

        # out = out.view(-1, C_out)
       
        return out
    

class ThresholdEdgeWeights(nn.Module):
    def __init__(self, thresh_levels = [0.5, 0.2, 0.1, 0.01]):
        super(ThresholdEdgeWeights, self).__init__()
        self.thresh_levels = thresh_levels

    def sparsify_edge_weights(self, thresh, edge_weights):
        # Sparsify the graph for attention learning
        edge_weights_thresh = edge_weights
        edge_weights_thresh[edge_weights < thresh] = 0.
        return edge_weights_thresh
    
    def forward(self, edge_weights):
        edge_weights_thresh = [edge_weights]
        for thresh in self.thresh_levels:
            edge_weights_thresh.append(self.sparsify_edge_weights(thresh, edge_weights))

        edge_weights_thresh = torch.cat(edge_weights_thresh, dim = -1)
        return edge_weights_thresh
        
    

class EdgeAttrEmbedding(nn.Module):
    def __init__(self, in_channels = 1, hidden_channels = 32, out_channels = 16, n_layers = 2):
        super(EdgeAttrEmbedding, self).__init__()
        self.in_channels = 1
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.embed = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(),
                                   nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                                   nn.Linear(hidden_channels, out_channels)
                                   )
        
    def forward(self, edge_weights):
        edge_weights = self.embed(edge_weights)
        return edge_weights


# Graph Transformer v4 for graph-attention and FiLM-based conditioning
class GraphTransformerv4(nn.Module):
    # def __init__(self, num_in, graph_embed_num_in, num_out, nsteps, activation, nlayers, **kwargs):
    def __init__(self, nsteps, nlayers, in_channels_gnn, hidden_channels, out_channels, **kwargs):
        super(GraphTransformerv4, self).__init__()
        self.nsteps = nsteps
        self.nlayers = nlayers
        activation = kwargs.get('activation', 'relu')
        self.resolve_activation(activation)

        self.attn_num_heads = kwargs.get('attn_num_heads', 8)

        attn_edge_dim = kwargs.get('attn_edge_dim', 8)

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.in_channels_mlp = out_channels
        self.in_channels_gnn = in_channels_gnn

        ### Learnable edge_weights ###
        if not attn_edge_dim == 1:
            self.edgeWeightsThresh = ThresholdEdgeWeights()
            self.edge_weight_embed = EdgeAttrEmbedding(in_channels=len(self.edgeWeightsThresh.thresh_levels) + 1, out_channels=attn_edge_dim)

        else:
            self.edgeWeightsThresh = None
            self.edge_weight_embed = None


        self.timestep_embed_channels = 2 * self.out_channels

        timestep_embed = SinusoidalEmbedding(size=self.timestep_embed_channels, scale=1.)
        self.time_embed = nn.Sequential(*[timestep_embed, nn.Flatten(start_dim=-2)])

   
        self.x_embed_mlp_layers = nn.Sequential(nn.Linear(self.in_channels_mlp, self.hidden_channels), self.activation,
                                                nn.Linear(self.hidden_channels, self.hidden_channels), self.activation,
                                                nn.Linear(self.hidden_channels, self.out_channels), self.activation
                                                )

        self.bn_layers = nn.ModuleList([])
        self.conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])

        for i in range(self.nlayers):
            if i == 0:
                in_channels = self.in_channels_gnn
                out_channels = self.hidden_channels // self.attn_num_heads
                bn_layer = BatchNorm(in_channels)

                # conv_layer = LEConv(in_channels, self.hidden_channels)
                conv_layer = TAGConv(in_channels=in_channels, out_channels=self.hidden_channels, K = 2, normalize=False)
            
            elif i == self.nlayers - 1: # final layer with two main heads
                in_channels = self.hidden_channels

                if self.attn_num_heads is None or self.attn_num_heads < 2: # GCN
                    conv_layer = TAGConv(in_channels=in_channels, out_channels=self.out_channels * 2, K=2, normalize=False)
                else:
                    out_channels = self.out_channels // (self.attn_num_heads // 2)
                    conv_layer = GATv2Conv(in_channels=in_channels, out_channels=out_channels, heads=self.attn_num_heads, concat=True, dropout=0.0, add_self_loops=True, fill_value=1., edge_dim=attn_edge_dim)
                
                bn_layer = BatchNorm(in_channels) # nn.Identity()
                
            else:
                in_channels = self.hidden_channels

                if self.attn_num_heads is None or self.attn_num_heads < 2: # GCN
                    conv_layer = TAGConv(in_channels=in_channels, out_channels=self.hidden_channels, K=2, normalize=False)
                else:
                    out_channels = self.hidden_channels // self.attn_num_heads
                    conv_layer = GATv2Conv(in_channels=in_channels, out_channels=out_channels, heads=self.attn_num_heads, concat=True, dropout=0.0, add_self_loops=True, fill_value=1., edge_dim=attn_edge_dim)
                
                bn_layer = BatchNorm(in_channels) # nn.Identity()
            # bn_layer = BatchNorm(in_channels)
            self.bn_layers.append(bn_layer)
            self.res_layers.append(nn.Linear(in_channels, out_channels * self.attn_num_heads))
            self.conv_layers.append(conv_layer)

        self.conv_embed_mlp_layers = nn.ModuleList([])
        for i in range(len(self.conv_layers) - 1):
            if i == 0:
                in_channels = self.conv_layers[i].out_channels
            else:
                in_channels = self.conv_layers[i].out_channels * self.attn_num_heads
            out_channels = self.conv_layers[i+1].in_channels

            self.conv_embed_mlp_layers.append(nn.Linear(in_channels, out_channels))

        self.conv_embed_mlp_layers.append(nn.Identity())


    def conditioning_block(self, x, condition):
        scale = condition[..., 0:1].view_as(x)
        shift = condition[..., 1:2].view_as(x)
        return x * scale + shift
    

    def convs(self, x, edge_index, edge_weight, return_attention_weights = None):
        attn_weights_all = []
        for i, (bn_layer, conv_layer, res_layer, linear_layer) in enumerate(zip(self.bn_layers, self.conv_layers, self.res_layers, self.conv_embed_mlp_layers)):
            x = bn_layer(x)
            if not isinstance(conv_layer, GATv2Conv): # GCN
                x = conv_layer(x, edge_index = edge_index, edge_weight = edge_weight[..., 0])
            else: # GAT
                y = 0 * res_layer(x)
                x, attn_weights = conv_layer(x, edge_index = edge_index, edge_attr = edge_weight, return_attention_weights = True)
                attn_weights = None if not return_attention_weights else attn_weights
                attn_weights_all.append((f'conv_layer_{i}', attn_weights))
                x += y
            x = linear_layer(x)
            if i < len(self.conv_layers) - 1:
                x = self.activation(x)
                # x = bn_layer(x)
        return x, attn_weights_all
    
    
    def mlps(self, x):
        x = 0 * self.x_embed_mlp_layers(x)
        return x 
    

    def resolve_activation(self, activation):
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky-relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError
        
        
    def forward(self, x, y, edge_index, edge_weight=None, batch=None, t = None, return_attention_weights = None):

        x_fc_embed = self.mlps(x)  # [Bxn, ...] -> [Bxn, C]
        condition = self.time_embed(t) # [Bxn, 1] -> [Bxn, 2*C]

        if edge_index is not None and edge_weight is not None:

            if self.edgeWeightsThresh is not None:
                edge_weight = self.edgeWeightsThresh(edge_weights=edge_weight)

            if self.edge_weight_embed is not None:
                edge_weight = self.edge_weight_embed(edge_weight)

            x_graph_attn_embed, attn_weights_list = self.convs(y, edge_index, edge_weight, return_attention_weights) # [...] -> [2C]
            condition += x_graph_attn_embed

        C = condition.shape[-1] // 2
        condition = torch.stack((condition[..., :C], condition[..., C:]), dim = -1)
        out = self.conditioning_block(x=x_fc_embed, condition=condition) # [C]

        out += x # residual connection
        out = self.activation(out)
       
        return out, attn_weights_list