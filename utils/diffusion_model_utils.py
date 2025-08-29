import torch
import torch.nn as nn
from torch_geometric.nn import TAGConv, LEConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torch_scatter import scatter
from models.GraphTransformerv2 import GraphTransformerv2, GraphTransformerv3, GraphTransformerv4
from models.ResidualGNN import ResidualGNN

from models.gnn_backbone import get_conv_model
from torch_geometric.utils import to_dense_adj

from models.GraphUnet import MyGraphUNet
from models.GraphTransformer import GraphTransformer
from models.GNN import GNN, SimpleGNN

from core.config import NUM_NODES

device = 'cpu'


def compute_snr(alphas_cumprod):
    """Compute Signal-to-Noise Ratio (SNR) for a given noise schedule."""
    snr = alphas_cumprod / (1 - alphas_cumprod + 1e-3)

    return snr


def snr_weighting(alphas_cumprod, mode="log_snr"):
    """Compute SNR-dependent weights based on different strategies."""
    snr = compute_snr(alphas_cumprod)

    if mode == "snr":  # Direct SNR weighting
        weights = snr
    elif mode == "log_snr":  # Log-SNR weighting
        weights = torch.log(1 + snr)
    elif mode == "snr_squared":  # Squared SNR
        weights = snr ** 2
    elif mode == "inv_snr":  # Inverse SNR weighting
        weights = 1 / (snr + 1e-5)  # Avoid division by zero
    else:
        raise ValueError("Invalid mode for SNR weighting")

    return weights


## Unconditional FCNN diffusion model
class DiffusionBlock(nn.Module):
    def __init__(self, nunits):
        super(DiffusionBlock, self).__init__()
        self.linear = nn.Linear(nunits, nunits)
        
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = nn.functional.relu(x)
        return x
        
    
class DiffusionModel(nn.Module):
    def __init__(self, nfeatures: int, nblocks: int = 2, nunits: int = 64):
        super(DiffusionModel, self).__init__()
        
        self.inblock = nn.Linear(nfeatures+1, nunits)
        self.midblocks = nn.ModuleList([DiffusionBlock(nunits) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, nfeatures)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        val = torch.hstack([x, t])  # Add t to inputs
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val)
        val = self.outblock(val)
        return val
    

## Conditional Linear diffusion model
import torch.nn.functional as F

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps, n_labels):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.time_embed = nn.Embedding(n_steps, num_out)
        self.time_embed.weight.data.uniform_()

        self.y_embed = nn.Linear(in_features=n_labels, out_features=num_out, bias=True)
        # self.y_embed.weight.data.uniform_()

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):
        out = self.lin(x) # [B, num_out]
        gamma = self.time_embed(t).view(-1, self.num_out) # [B, 1, num_out] -> [B, num_out]

        if y is not None:
            gamma_y = self.y_embed(y).view(-1, self.num_out) # [B, num_out] -> [B, num_out]
            gamma += gamma_y

        out = gamma * out

        out = F.softplus(out)
        return out
    
        
class ConditionalLinearModel(nn.Module):
    def __init__(self, nsteps, nlabels, nfeatures: int, nblocks: int = 2, nunits: int = 128):
        super(ConditionalLinearModel, self).__init__()
        self.nsteps = nsteps
        self.nlabels = nlabels
        self.inblock = ConditionalLinear(nfeatures, nunits, nsteps, nlabels)
        # self.lin2 = ConditionalLinear(128, 128, nsteps)
        # self.lin3 = ConditionalLinear(128, 128, nsteps)
        self.midblock = nn.ModuleList([ConditionalLinear(nunits, nunits, nsteps, nlabels) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, nfeatures)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):
        val = self.inblock(x, t, y)
        for midblock in self.midblock:
            val = midblock(val, t, y)
        val = self.outblock(val)
        return val
    


# class gnn_backbone(nn.Module):
#     def __init__(self, num_in, num_out, k_hops = 1):
#         super(gnn_backbone, self).__init__()
#         self.num_in = num_in
#         self.num_out = num_out
#         self.k_hops = k_hops

#         self.layers = nn.Sequential(*[nn.Linear(num_in, num_out) for _ in range(self.k_hops)])
#         self.final_layer = nn.Linear(in_features=NUM_NODES * self.num_out, out_features=self.num_out)

#         self.A_layer = nn.Linear(in_features=NUM_NODES*NUM_NODES, out_features=self.num_out)


#     def forward(self, x: torch.Tensor, A: torch.Tensor):
#         # out = 0.
#         # for k, layer in enumerate(self.layers):
#         #     signal_in = torch.bmm(torch.linalg.matrix_power(A, k), x) # [B, NUM_NODES, num_in]
#         #     signal_out = layer(signal_in) # [B, NUM_NODES, num_out]
#         #     out += signal_out

#         # out = out.reshape(out.shape[0], -1)
#         # out = self.final_layer(out)
#         # out = F.softplus(out)

#         out = A.reshape(A.shape[0], -1)
#         out = self.A_layer(out)

#         return out


### Conditional Graph-Augmented Diffusion Model ###
class ConditionalGraphAugmentedLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps, n_labels):
        super(ConditionalGraphAugmentedLinear, self).__init__()
        self.aggr = lambda x: torch.mean(x, dim = -2)
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.time_embed = nn.Embedding(n_steps, num_out)
        self.time_embed.weight.data.uniform_()

        # self.y_embed = nn.Linear(in_features=n_labels, out_features=num_out, bias=True)
        self.y_embed = gnn_backbone(num_in=1, num_out=num_out, k_hops=3)
        # self.y_embed.weight.data.uniform_()

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None, z: torch.Tensor = None):
        # if len(x.size()) == 2:
        #     x = x.unsqueeze(-1)
        if z is None:
            out = self.lin(x.squeeze(-1)) # [B, NUM_NODES] -> [B, num_out]
        else:
            out = self.lin(z) # [B, F_in] -> [B, F_out]

        # gamma = self.time_embed(t).repeat(1, x.shape[1], 1) # [B, 1, num_out] -> [B, NUM_NODES, num_out]
        gamma = self.time_embed(t).view(-1, self.num_out) # [B, num_out]

        if y is not None:
            gamma_y = self.y_embed(x, y) # [B, NUM_NODES, num_out]
            gamma_y = self.aggr(gamma_y) # [B, num_out]
            gamma += gamma_y

        out = gamma * out
        out = F.softplus(out)
        return out
    
        
class ConditionalGraphAugmentedLinearModel(nn.Module):
    def __init__(self, nsteps, nlabels, nfeatures: int = 1, nblocks: int = 2, nunits: int = 128):
        super(ConditionalGraphAugmentedLinearModel, self).__init__()
        self.nsteps = nsteps
        self.nlabels = nlabels
        self.inblock = ConditionalGraphAugmentedLinear(nfeatures, nunits, nsteps, nlabels)
        
        # self.lin2 = ConditionalLinear(128, 128, nsteps)
        # self.lin3 = ConditionalLinear(128, 128, nsteps)
        self.midblock = nn.ModuleList([ConditionalGraphAugmentedLinear(nunits, nunits, nsteps, nlabels) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, nfeatures)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None, normalize_adjacency = True):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)

        if normalize_adjacency and y is not None:
            pass
            # y = normalize_GSO(y)
            
        val = self.inblock(x, t, y) # [B, num_out]
        for midblock in self.midblock:
            val = midblock(x, t, y, val)
        val = self.outblock(val)
        # val = val.squeeze(-1)
        return val
    



### Conditional Graph-Augmented Diffusion Model ###
# class ConditionalGNN(nn.Module):
#     def __init__(self, num_in, num_out, n_steps, n_labels):
#         super(ConditionalGNN, self).__init__()
#         self.num_out = num_out
#         self.lin = nn.Linear(num_in, num_out)
#         self.time_embed = nn.Embedding(n_steps, num_out)
#         self.time_embed.weight.data.uniform_()

#         # self.y_embed = nn.Linear(in_features=n_labels, out_features=num_out, bias=True)
#         self.y_embed = gnn_backbone(num_in=1, num_out=num_out, k_hops=3)
#         # self.y_embed.weight.data.uniform_()

#     def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):

#         out = self.lin(x)
#         # gamma = self.time_embed(t).repeat(1, x.shape[1], 1) # [B, 1, num_out] -> [B, NUM_NODES, num_out]
#         gamma = self.time_embed(t).view(-1, self.num_out) # [B, num_out]

#         if y is not None:
#             gamma_y = self.y_embed(torch.ones_like(x[..., :NUM_NODES].unsqueeze(-1)), y) # [B, num_out]
#             gamma += gamma_y

#         out = gamma * out
#         out = F.softplus(out)
#         return out
    
        
# class ConditionalGNNModel(nn.Module):
#     def __init__(self, nsteps, nlabels, nfeatures: int = 1, nblocks: int = 2, nunits: int = 128):
#         super(ConditionalGNNModel, self).__init__()
#         self.nsteps = nsteps
#         self.nlabels = nlabels
#         self.inblock = ConditionalGNN(nfeatures, nunits, nsteps, nlabels)
        
#         # self.lin2 = ConditionalLinear(128, 128, nsteps)
#         # self.lin3 = ConditionalLinear(128, 128, nsteps)
#         self.midblock = nn.ModuleList([ConditionalGNN(nunits, nunits, nsteps, nlabels) for _ in range(nblocks)])
#         self.outblock = nn.Linear(nunits, nfeatures)
#         # self.outblock = lambda x: torch.mean(x, dim = -1) # mean aggregation  # nn.Linear(nunits, nfeatures)
    
#     def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None, normalize_adjacency = True):
#         # if len(x.size()) == 2:
#         #     x = x.unsqueeze(-1)

#         if normalize_adjacency and y is not None:
#             pass
#             # y = normalize_GSO(y)
            
#         val = self.inblock(x, t, y) # [B, num_out]
#         for midblock in self.midblock:
#             val = midblock(val, t, y)
#         val = self.outblock(val)

#         return val
    

# class py_gnn_backbone(nn.Module):
#     def __init__(self, num_in, num_out, k_hops = 1, n_gnn_layers = 1):
#         super(py_gnn_backbone, self).__init__()
#         self.num_in = num_in
#         self.num_out = num_out
#         self.k_hops = k_hops
#         self.n_gnn_layers = n_gnn_layers

#         self.node_hidden_size = num_out // 2
#         # self.node_embedding = nn.Linear(in_features=num_in, out_features=self.node_hidden_size)
#         self.node_embedding = nn.Embedding(num_in, self.node_hidden_size, device = device)
#         # Initialize the embeddings with small random values
#         nn.init.normal_(self.node_embedding.weight, std=1.)

#         num_features_list = [self.node_hidden_size] + [num_out] * (n_gnn_layers - 1) + [num_out]
        
#         layers = []
#         for i in range(self.n_gnn_layers):
#             # layer = GCNConv(in_channels=num_features_list[i], out_channels=num_features_list[i+1], add_self_loops=False)
#             layer = TAGConv(in_channels=num_features_list[i], out_channels=num_features_list[i+1], K = self.k_hops)
#             layers.append(layer)

#         self.gnn_layers = nn.Sequential(*layers)
            

#     def forward(self, data: torch.Tensor):
      
#         x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weights
#         # x = torch.randn_like(x)
#         x = self.node_embedding.weight.repeat(x.shape[0], *x.shape[1:])
#         # x = self.node_embedding(x) # [B * n, 1] -> [B * n, H]

#         for i, gnn_layer in enumerate(self.gnn_layers):
#             x = gnn_layer(x = x, edge_index = edge_index, edge_weight = edge_weights)
#             x = F.tanh(x) if i < len(self.gnn_layers) - 1 else x

#         x = scatter(x, data.batch, dim = 0, reduce = 'mean')

#         return x
    

### Conditional pygNN Model ###
# class ConditionalDiffusionGNN(nn.Module):
#     def __init__(self, num_in, num_out, n_steps, n_labels):
#         super(ConditionalDiffusionGNN, self).__init__()
#         self.num_out = num_out
#         self.lin = nn.Linear(num_in, num_out)
#         self.time_embed = nn.Embedding(n_steps, num_out)
#         self.time_embed.weight.data.uniform_()

#         # self.y_embed = nn.Linear(in_features=n_labels, out_features=num_out, bias=True)
#         self.y_embed = py_gnn_backbone(num_in=1, num_out=num_out, k_hops=2, n_gnn_layers=1)
#         # self.y_embed.weight.data.uniform_()


#     def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):

#         out = self.lin(x)
#         # gamma = self.time_embed(t).repeat(1, x.shape[1], 1) # [B, 1, num_out] -> [B, NUM_NODES, num_out]
#         gamma = self.time_embed(t).view(-1, self.num_out) # [B, num_out]

#         if y is not None:
#             # gamma_y = self.y_embed(y).view(out.shape[0], -1, self.num_out) # [B x NUM_NODES, num_out] -> [B, NUM_NODES, num_out]
#             gamma_y = self.y_embed(y)
#             # gamma_y = torch.mean(gamma_y, dim = -2) # aggregate node features
#             gamma += gamma_y

#         out = gamma * out
#         out = F.softplus(out)
#         return out
    
        
# class ConditionalDiffusionGNNModel(nn.Module):
#     def __init__(self, nsteps, nlabels, nfeatures: int = 1, nblocks: int = 2, nunits: int = 128):
#         super(ConditionalDiffusionGNNModel, self).__init__()
#         self.nsteps = nsteps
#         self.nlabels = nlabels
#         self.inblock = ConditionalDiffusionGNN(nfeatures, nunits, nsteps, nlabels)
        
#         # self.lin2 = ConditionalLinear(128, 128, nsteps)
#         # self.lin3 = ConditionalLinear(128, 128, nsteps)
#         self.midblock = nn.ModuleList([ConditionalDiffusionGNN(nunits, nunits, nsteps, nlabels) for _ in range(nblocks)])
#         self.outblock = nn.Linear(nunits, nfeatures)
#         # self.outblock = lambda x: torch.mean(x, dim = -1) # mean aggregation  # nn.Linear(nunits, nfeatures)
    
#     def forward(self, x: torch.Tensor, t: torch.Tensor, y = None):
#         # if len(x.size()) == 2:
#         #     x = x.unsqueeze(-1)

#         # if y is not None:
#         #     data_list = []
#         #     for i in range(x.shape[0]):
#         #         data_x = x[i].unsqueeze(-1)
#         #         data_x = torch.ones_like(data_x)

#         #         edge_idx = torch.stack(torch.where(y[i] > 0), dim = 0)
#         #         edge_w = y[i][torch.where(y[i] > 0)]
#         #         data = Data(x=data_x,
#         #                     edge_index=edge_idx,
#         #                     edge_weights=edge_w
#         #                     )
                
#         #         data_list.append(data)

#         #     temp_loader = DataLoader(dataset=data_list, batch_size=x.shape[0], shuffle=False)
#         #     y_batch = next(iter(temp_loader))
#         # else:
#         #     y_batch = None
            
#         val = self.inblock(x, t, y) # [B, num_out]
#         for midblock in self.midblock:
#             val = midblock(val, t, y)
#         val = self.outblock(val)

#         return val


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
    

### GNN backbone ###
class gnn_backbone(torch.nn.Module):
    def __init__(self, num_in, num_out, nsteps, activation, nlayers, **kwargs):
        super(gnn_backbone, self).__init__()
        self.nlayers = nlayers
        self.nsteps = nsteps
        self.num_in = num_in
        self.num_out = num_out
        self.activation = activation

        sinusoidal_time_embed = kwargs.get('sinusoidal_time_embed', False)

        k_hops = kwargs.get('k_hops', 2)
        batch_norm = kwargs.get('batch_norm', False)
        conv_model_architecture = kwargs.get('conv_model', 'LeConv')
        aggregation = kwargs.get('aggregation', None)
        num_heads = kwargs.get('num_heads', 2)
        dropout_rate = kwargs.get('dropout_rate', 0.0)

        self.condition_by_summation_weight = kwargs.get('condition_by_summation_weight', None)

        if sinusoidal_time_embed:
            print('Sinusoidal time embeddings are used.')
            self.time_embed = SinusoidalEmbedding(size=num_out, scale=1.)
        else:
            self.time_embed = nn.Embedding(nsteps, num_out)
            self.time_embed.weight.data.uniform_()

        if nlayers == 1:
            self.gnn_embed = nn.ModuleList([get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=num_in,
                                                            num_out=num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                                            )
                                                            ])
            # self.gnn_embed = nn.ModuleList([LEConv(num_in, num_out)])
        else:
            self.gnn_embed = nn.ModuleList([get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=num_in,
                                                            num_out=num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                            )] + \
                                            [get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=num_out,
                                                            num_out=num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                                            )] * (self.nlayers - 1))
            # self.gnn_embed = nn.ModuleList([LEConv(num_in, num_out)] + [LEConv(num_out, num_out)] * (self.nlayers - 1))

        if batch_norm:
            self.bn = BatchNorm(in_channels=num_out)
        else:
            self.bn = nn.Identity() # LayerNorm(in_channels=num_out)
            # self.bn = nn.Identity()
        

    def convs(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.gnn_embed):
            x = layer(x, edge_index, edge_weight)
            if i < self.nlayers - 1:
                x = self.activation(x)
        return x
    

    def forward(self, x, t, edge_index, edge_weight):
        out = self.convs(x, edge_index, edge_weight) # [..., nunits]
        # out = self.bn(out)
        gamma = self.time_embed(t).view(-1, self.num_out) # [..., nunits]

        if self.condition_by_summation_weight is None:
            out = out * gamma
        else:
            out = self.condition_by_summation_weight * out + gamma * (1 - self.condition_by_summation_weight)

        out = self.activation(out)
        out = self.bn(out)
        return out
    

### Conditional MLP backbone ###
class mlp_conditional_mlp_backbone(torch.nn.Module):
    def __init__(self, num_in, graph_embed_num_in, num_out, nsteps, activation, nlayers, **kwargs):
        super(mlp_conditional_mlp_backbone, self).__init__()
        self.nlayers = nlayers
        self.nsteps = nsteps
        self.num_in = num_in
        self.graph_embed_num_in = graph_embed_num_in
        self.num_out = num_out
        self.activation = activation

        sinusoidal_time_embed = kwargs.get('sinusoidal_time_embed', False)
        batch_norm = kwargs.get('batch_norm', False)

        self.condition_by_summation_weight = kwargs.get('condition_by_summation_weight', None)

        if sinusoidal_time_embed:
            print('Sinusoidal time embeddings are used.')
            self.time_embed = SinusoidalEmbedding(size=num_out, scale=1.)
        else:
            self.time_embed = nn.Embedding(nsteps, num_out)
            self.time_embed.weight.data.uniform_()

        if nlayers == 1:
            self.x_embed = nn.ModuleList([nn.Linear(num_in, num_out)])
            self.graph_embed = nn.ModuleList([nn.Linear(graph_embed_num_in, num_out)])
        else:
            self.x_embed = nn.ModuleList([nn.Linear(num_in, num_out)] + [nn.Linear(num_out, num_out)] * (self.nlayers - 1))
            self.graph_embed = nn.ModuleList([nn.Linear(graph_embed_num_in, num_out)] + [nn.Linear(num_out, num_out)] * (self.nlayers - 1))
            

        if batch_norm:
            self.bn = BatchNorm(in_channels=num_out)
        else:
            self.bn = nn.Identity() # LayerNorm(in_channels=num_out)
            # self.bn = nn.Identity()
        

    def mlps(self, x, g):
        for i, layer in enumerate(self.x_embed):
            x = layer(x)
            if i < self.nlayers - 1:
                x = self.activation(x)

        for i, layer in enumerate(self.graph_embed):
            g = layer(g)
            if i < self.nlayers - 1:
                g = self.activation(g)
        
        return x, g
    

    def forward(self, x, t, g):
        out_x, out_g = self.mlps(x, g) # [..., nunits]
        # out = self.bn(out)
        gamma = self.time_embed(t).view(-1, self.num_out) # [..., nunits]

        if self.condition_by_summation_weight is None:
            out = out_x * (gamma + out_g)
        else:
            out = self.condition_by_summation_weight * out_x + (1 - self.condition_by_summation_weight) * (gamma + out_g)

        # out = self.bn(out)
        out = self.activation(out)
        out = self.bn(out)
        return out

    


class mlp_conditional_gnn_backbone(torch.nn.Module):
    def __init__(self, num_in, gnn_num_in, hidden_dim, nsteps, activation, nlayers, **kwargs):
        super(mlp_conditional_gnn_backbone, self).__init__()
        
        self.nlayers = nlayers
        self.nsteps = nsteps
        self.num_in = num_in
        self.gnn_num_in = gnn_num_in
        self.hidden_dim = hidden_dim
        self.num_out = num_in
        self.gnn_num_out = gnn_num_in
        self.activation = activation

        sinusoidal_time_embed = kwargs.get('sinusoidal_time_embed', False)
        batch_norm = kwargs.get('batch_norm', False)


        k_hops = kwargs.get('k_hops', 2)
        conv_model_architecture = kwargs.get('conv_model', 'LeConv')
        aggregation = kwargs.get('aggregation', None)
        num_heads = kwargs.get('num_heads', 2)
        dropout_rate = kwargs.get('dropout_rate', 0.0)

        self.condition_by_summation_weight = kwargs.get('condition_by_summation_weight', None)

        if sinusoidal_time_embed:
            print('Sinusoidal time embeddings are used.')
            self.time_embed = SinusoidalEmbedding(size=self.num_out, scale=1.)
        else:
            self.time_embed = nn.Embedding(nsteps, self.num_out)
            self.time_embed.weight.data.uniform_()

        if nlayers == 1:
            self.x_embed = nn.ModuleList([nn.Linear(num_in, self.num_out)])
            self.gnn_embed = nn.ModuleList([get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.gnn_num_in,
                                                            num_out=self.gnn_num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                                            )
                                                            ])
        else:
            self.x_embed = nn.ModuleList([nn.Linear(self.num_in, self.hidden_dim)] + [nn.Linear(self.hidden_dim, self.hidden_dim)] * (self.nlayers - 2) + [nn.Linear(self.hidden_dim, self.num_out)])
            self.gnn_embed = nn.ModuleList([get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.gnn_num_in,
                                                            num_out=self.hidden_dim,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                            )] + \
                                            [get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.hidden_dim,
                                                            num_out=self.hidden_dim,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                                            )] * (self.nlayers - 2) + \
                                            [get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.hidden_dim,
                                                            num_out=self.gnn_num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                            )])
            

        if batch_norm:
            self.bn = BatchNorm(in_channels=self.num_out)
        else:
            self.bn = nn.Identity() # LayerNorm(in_channels=num_out)
            # self.bn = nn.Identity()


    def mlps(self, x):
        for i, layer in enumerate(self.x_embed):
            x = layer(x)
            if i < len(self.x_embed) - 1:
                x = self.activation(x)
        return x
    

    def convs(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.gnn_embed):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.gnn_embed) - 1:
                x = self.activation(x)
        return x
    

    def forward(self, x, t, y = None, edge_index = None, edge_weight = None):
        out_x = self.mlps(x)
        gamma = self.time_embed(t).view(-1, self.gnn_num_out) # [..., nunits]

        if y is not None and edge_index is not None and edge_weight is not None:
            conv_out = self.convs(y, edge_index, edge_weight) # y can be a fixed signal
            try:
                gamma += conv_out
            except:
                print('gamma: ', gamma.shape)
                print('conv_out: ', conv_out.shape)
                print('edge_index: ', edge_index.shape)

        gamma = gamma.view_as(out_x)

        if self.condition_by_summation_weight is None:
            out = out_x * gamma
        else:
            out = self.condition_by_summation_weight * out_x + (1 - self.condition_by_summation_weight) * gamma

        # out = self.bn(out)
        out = self.activation(out)
        out = self.bn(out)
        return out
    



class gnn_conditional_gnn_backbone(torch.nn.Module):
    def __init__(self, x_gnn_num_in, cond_gnn_num_in, num_out, nsteps, activation, nlayers, **kwargs):
        super(gnn_conditional_gnn_backbone, self).__init__()
        self.nlayers = nlayers
        self.nsteps = nsteps
        self.num_in = x_gnn_num_in
        self.cond_gnn_embed_num_in = cond_gnn_num_in
        self.num_out = num_out
        self.activation = activation

        sinusoidal_time_embed = kwargs.get('sinusoidal_time_embed', False)
        batch_norm = kwargs.get('batch_norm', False)

        k_hops = kwargs.get('k_hops', 2)
        conv_model_architecture = kwargs.get('conv_model', 'LeConv')
        aggregation = kwargs.get('aggregation', None)
        num_heads = kwargs.get('num_heads', 2)
        dropout_rate = kwargs.get('dropout_rate', 0.0)

        self.condition_by_summation_weight = kwargs.get('condition_by_summation_weight', None)

        if sinusoidal_time_embed:
            print('Sinusoidal time embeddings are used.')
            self.time_embed = SinusoidalEmbedding(size=num_out, scale=1.)
        else:
            self.time_embed = nn.Embedding(nsteps, num_out)
            self.time_embed.weight.data.uniform_()

        if nlayers == 1:
            self.x_gnn_embed = nn.ModuleList([get_conv_model(conv_model_architecture=conv_model_architecture,
                                                         num_in=self.num_in,
                                                         num_out=self.num_out,
                                                         aggregation=aggregation,
                                                         k_hops=k_hops,
                                                         num_heads=num_heads,
                                                         dropout_rate=dropout_rate
                                                         )
                                                         ])
            
            self.cond_gnn_embed = nn.ModuleList([get_conv_model(conv_model_architecture=conv_model_architecture,
                                                           num_in=self.cond_gnn_embed_num_in,
                                                           num_out=self.num_out,
                                                           aggregation=aggregation,
                                                           k_hops=k_hops,
                                                           num_heads=num_heads,
                                                           dropout_rate=dropout_rate
                                                           )
                                                           ])
    
        else:
            self.x_gnn_embed = nn.ModuleList([get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.num_in,
                                                            num_out=self.num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                            )] + \
                                            [get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.num_out,
                                                            num_out=self.num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                                            )] * (self.nlayers - 2) + \
                                            [get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.num_out,
                                                            num_out=self.num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                            )])

            self.cond_gnn_embed = nn.ModuleList([get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.cond_gnn_embed_num_in,
                                                            num_out=self.num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                            )] + \
                                            [get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.num_out,
                                                            num_out=self.num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                                            )] * (self.nlayers - 2) + \
                                            [get_conv_model(conv_model_architecture=conv_model_architecture,
                                                            num_in=self.num_out,
                                                            num_out=self.num_out,
                                                            aggregation=aggregation,
                                                            k_hops=k_hops,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate
                                            )])
            

        if batch_norm:
            self.bn = BatchNorm(in_channels=num_out)
        else:
            self.bn = nn.Identity() # LayerNorm(in_channels=num_out)
            # self.bn = nn.Identity()
        

    def convs(self, x, y, edge_index_list, edge_weights_list):
        for i, layer in enumerate(self.x_gnn_embed):
            x = layer(x, edge_index_list[0], edge_weights_list[0])
            if i < len(self.x_gnn_embed) - 1:
                x = self.activation(x)

        if edge_index_list[1] is not None and edge_weights_list[1] is not None:
            for i, layer in enumerate(self.cond_gnn_embed):
                y = layer(y, edge_index_list[1], edge_weights_list[1])
                if i < len(self.cond_gnn_embed) - 1:
                    y = self.activation(y)
        else:
            y = None
        
        return x, y
    

    def forward(self, x, t, y, edge_index_list, edge_weights_list):
        conv_out_x, conv_out_y = self.convs(x, y, edge_index_list=edge_index_list, edge_weights_list=edge_weights_list)
        gamma = self.time_embed(t).view(-1, self.num_out) # [..., nunits]

        if conv_out_y is not None:
            gamma += conv_out_y

        if self.condition_by_summation_weight is None:
            out = conv_out_x * gamma
        else:
            out = self.condition_by_summation_weight * conv_out_x + (1 - self.condition_by_summation_weight) * gamma

        # out = self.bn(out)
        out = self.activation(out)
        out = self.bn(out)
        return out



# class DiffusionGNN(torch.nn.Module):
#     def __init__(self, nfeatures, nunits = 64, nblocks = 2, nsteps = 100, nsubblocks = 1, **kwargs):
#         super(DiffusionGNN, self).__init__()
#         self.nlayers = nblocks
#         self.activation = F.softplus # F.leaky_relu
#         self.inblock = gnn_backbone(nfeatures, nunits, nsteps, self.activation, nlayers = nsubblocks, **kwargs) # nlayers = 1
#         self.midblock = nn.ModuleList([gnn_backbone(nunits, nunits, nsteps, self.activation, nlayers = nsubblocks, **kwargs) for _ in range(nblocks)])
#         self.outblock = nn.Linear(nunits, 1, bias = True)

#     def forward(self, x: torch.Tensor, t: torch.Tensor, data):
#         val = self.inblock(x, t, data.edge_index_l, data.edge_weight_l)
#         for midblock in self.midblock:
#             val = midblock(val, t, data.edge_index_l, data.edge_weight_l)
#         Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean')
#         y = self.outblock(Tx_embeddings)
#         return y
    
#     @property
#     def prototype_name(self):
#         return f"DiffusionGNN_{self.nlayers}_layers_gnn_backbone_{[midblock.nlayers for midblock in self.midblock]}_layers"



##################################################### GraphUNet ############################################################################

class DiffusionGraphUNet(torch.nn.Module):
    def __init__(self, nfeatures=1, nblocks = 1, nunits=128, nsteps = 100, **kwargs):
        super(DiffusionGraphUNet, self).__init__()
        pool_ratio = kwargs.get('pool_ratio', 0.5)
        pool_layer = kwargs.get('pool_layer', 'topk')
        pool_multiplier = kwargs.get('pool_multiplier', 1.)
        sum_skip_connection = kwargs.get('sum_skip_connection', True)
        use_checkpointing = kwargs.get('use_checkpointing', False)
        attn_self_loop_fill_value = kwargs.get('attn_self_loop_fill_value', "max")
        apply_gcn_norm = kwargs.get("apply_gcn_norm", False)
        norm = kwargs.get('norm', 'batch')

        self.unet = MyGraphUNet(in_channels=nfeatures,
                                hidden_channels=nunits,
                                out_channels=nunits // 4,
                                depth = nblocks,
                                pool_ratio=pool_ratio,
                                pool_layer=pool_layer,
                                pool_multiplier=pool_multiplier,
                                norm=norm,
                                sum_skip_connection=sum_skip_connection,
                                use_checkpointing=use_checkpointing,
                                attn_self_loop_fill_value=attn_self_loop_fill_value,
                                apply_gcn_norm=apply_gcn_norm
                                )

    def forward(self, x: torch.Tensor, t: torch.Tensor, data, return_attention_weights = False, debug_forward_pass = False):
        # print(f'Number of graphs in data.batch = {len(torch.unique(data.batch))}')
        # print(f'Number of transmitters in data.transmitters_index = {len(torch.unique(data.transmitters_index))}')
        
        # Map transmitter embeddings to node embeddings
        x = torch.index_select(input=x, dim = 0, index = data.transmitters_index)
        # Denoise noisy node embeddings
        val = self.unet(x, t = t, edge_index = data.edge_index_l, edge_weight = data.edge_weight_l, batch = data.batch,
                        return_attn_weights = return_attention_weights, debug_forward_pass = debug_forward_pass)

        if isinstance(val, tuple):
            if len(val) == 3:
                debug_data = val[2]
            attn_weights = val[1]
            val = val[0]
        else:
            attn_weights = None
            debug_data = None
        # Map node embeddings to transmitter embeddings
        Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean') # reduction doesn't matter as each index is unique
        return Tx_embeddings, attn_weights, debug_data
    
    @property
    def prototype_name(self):
        return f"DiffusionGraphUNet_depth_{self.unet.depth}"
    
####################################################################################################################################


    
########################################################### GNN ######################################################################
""" We use the same class signature for GNN as the GraphUNet """
class DiffusionGNN(nn.Module):
    def __init__(self, nfeatures=1, nblocks = 1, nunits=128, nsteps = 100, **kwargs):
        super(DiffusionGNN, self).__init__()
        pool_ratio = kwargs.get('pool_ratio', 0.5)
        pool_layer = kwargs.get('pool_layer', 'topk')
        pool_multiplier = kwargs.get('pool_multiplier', 1.)
        sum_skip_connection = kwargs.get('sum_skip_connection', True)
        use_checkpointing = kwargs.get('use_checkpointing', False)
        attn_self_loop_fill_value = kwargs.get('attn_self_loop_fill_value', "max")
        apply_gcn_norm = kwargs.get("apply_gcn_norm", False)
        norm = kwargs.get('norm', 'batch')

        self.gnn = GNN(in_channels=nfeatures,
                       hidden_channels=nunits,
                       out_channels=nunits // 4,
                       depth = nblocks,
                    #    pool_ratio=pool_ratio,
                    #    pool_layer=pool_layer,
                    #    pool_multiplier=pool_multiplier,
                       norm=norm,
                       sum_skip_connection=sum_skip_connection,
                    #    use_checkpointing=use_checkpointing,
                    #    attn_self_loop_fill_value=attn_self_loop_fill_value,
                    #    apply_gcn_norm=apply_gcn_norm
                       )
        

    def forward(self, x: torch.Tensor, t: torch.Tensor, data, return_attention_weights = False, debug_forward_pass = False):

        # Map transmitter embeddings to node embeddings
        x = torch.index_select(input=x, dim = 0, index = data.transmitters_index)

        # Denoise noisy node embeddings
        val = self.gnn(x, t = t, edge_index = data.edge_index_l, edge_weight = data.edge_weight_l, batch = data.batch,
                        return_attn_weights = return_attention_weights, debug_forward_pass = debug_forward_pass)

        if isinstance(val, tuple):
            if len(val) == 3:
                debug_data = val[2]
            attn_weights = val[1]
            val = val[0]
        else:
            attn_weights = None
            debug_data = None
        # Map node embeddings to transmitter embeddings
        Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean') # reduction doesn't matter as each index is unique
        # print(f'y = GNN.forward(x, G), x.shape = {x.shape}\ty.shape = {val.shape}\tTx_embed.shape = {Tx_embeddings.shape}')

        return Tx_embeddings, attn_weights, debug_data
        
    
    @property
    def prototype_name(self):
        return f"DiffusionGNN_{self.gnn.depth * 2 + 1}_layers"
    
####################################################################################################################################





########################################################### Simple-GNN ######################################################################
""" We use the same class signature for GNN as the GraphUNet """
class DiffusionSimpleGNN(nn.Module):
    def __init__(self, nfeatures=1, nblocks = 1, nunits=128, nsteps = 100, **kwargs):
        super(DiffusionSimpleGNN, self).__init__()
        pool_ratio = kwargs.get('pool_ratio', 0.5)
        pool_layer = kwargs.get('pool_layer', 'topk')
        pool_multiplier = kwargs.get('pool_multiplier', 1.)
        sum_skip_connection = kwargs.get('sum_skip_connection', True)
        use_checkpointing = kwargs.get('use_checkpointing', False)
        attn_self_loop_fill_value = kwargs.get('attn_self_loop_fill_value', "max")
        apply_gcn_norm = kwargs.get("apply_gcn_norm", False)
        norm = kwargs.get('norm', 'batch')
        conv_layer_normalize = kwargs.get('conv_layer_normalize', False)
        k_hops = kwargs.get('k_hops', 2)

        self.gnn = SimpleGNN(in_channels=nfeatures,
                       hidden_channels=nunits,
                       out_channels=nunits // 4,
                       depth = nblocks,
                    #    pool_ratio=pool_ratio,
                    #    pool_layer=pool_layer,
                    #    pool_multiplier=pool_multiplier,
                       norm=norm,
                       conv_layer_normalize=conv_layer_normalize,
                       k_hops=k_hops
                    #    use_checkpointing=use_checkpointing,
                    #    attn_self_loop_fill_value=attn_self_loop_fill_value,
                    #    apply_gcn_norm=apply_gcn_norm
                       )
        

    def forward(self, x: torch.Tensor, t: torch.Tensor, data, return_attention_weights = False, debug_forward_pass = False):

        # Map transmitter embeddings to node embeddings
        x = torch.index_select(input=x, dim = 0, index = data.transmitters_index)
        # Denoise noisy node embeddings
        val = self.gnn(x, t = t, edge_index = data.edge_index_l, edge_weight = data.edge_weight_l, batch = data.batch,
                        return_attn_weights = return_attention_weights, debug_forward_pass = debug_forward_pass)

        if isinstance(val, tuple):
            if len(val) == 3:
                debug_data = val[2]
            attn_weights = val[1]
            val = val[0]
        else:
            attn_weights = None
            debug_data = None

        # Map node embeddings to transmitter embeddings
        Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean') # reduction doesn't matter as each index is unique
        # print(f'y = GNN.forward(x, G), x.shape = {x.shape}\ty.shape = {val.shape}\tTx_embed.shape = {Tx_embeddings.shape}')
        return Tx_embeddings, attn_weights, debug_data
    
    @property
    def prototype_name(self):
        return f"DiffusionSimpleGNN_{self.gnn.depth}_layers"
    
####################################################################################################################################



########################################################### Res+GNN ######################################################################

""" We use the same class signature for GNN as the GraphUNet """
class DiffusionResPlusGNN(nn.Module):
    def __init__(self, nfeatures=1, nblocks = 1, nunits=128, nsteps = 100, **kwargs):
        super(DiffusionResPlusGNN, self).__init__()
        pool_ratio = kwargs.get('pool_ratio', 0.5)
        pool_layer = kwargs.get('pool_layer', 'topk')
        pool_multiplier = kwargs.get('pool_multiplier', 1.)
        sum_skip_connection = kwargs.get('sum_skip_connection', True)
        use_checkpointing = kwargs.get('use_checkpointing', False)
        use_res_connection_time_embed = kwargs.get('use_res_connection_time_embed', False)
        attn_self_loop_fill_value = kwargs.get('attn_self_loop_fill_value', "max")
        apply_gcn_norm = kwargs.get("apply_gcn_norm", False)
        norm = kwargs.get('norm', 'batch')
        conv_layer_normalize = kwargs.get('conv_layer_normalize', False)
        k_hops = kwargs.get('k_hops', 2)
        dropout_rate = kwargs.get('dropout_rate', 0.0)
        norm_kws = kwargs.get('norm_kws', dict())

        
        self.gnn = ResidualGNN(in_channels=nfeatures, hidden_channels=nunits, out_channels=nfeatures, # out_channels don't matter
                               res_connection = 'res+',
                               depth=nblocks,
                               norm=norm,
                               conv_layer_normalize=conv_layer_normalize,
                               k_hops=k_hops,
                               dropout_rate=dropout_rate,
                               use_checkpointing=use_checkpointing,
                               use_res_connection_time_embed=use_res_connection_time_embed,
                               norm_kws = norm_kws
                               )
        

    def forward(self, x: torch.Tensor, t: torch.Tensor, data, return_attention_weights = False, debug_forward_pass = False):

        if "transmitters_index" not in data or data.transmitters_index is None:
            data.transmitters_index = torch.arange(x.shape[0], device=x.device)

        # Assert that transmitters_index doesn't have duplicate values
        assert len(data.transmitters_index) == len(torch.unique(data.transmitters_index)), \
            f"Duplicate values found in transmitters_index: {data.transmitters_index}"

        # Assert that transmitters_index is in the range of x
        assert torch.all(data.transmitters_index < x.shape[0]), \
            f"Transmitters index out of bounds: {data.transmitters_index} for x of shape {x.shape}"

        # If edge_index_t and edge_weight_t are not present, use edge_index_l and edge_weight_l
        # This is to ensure compatibility with different data formats

        if "edge_index_t" in data and "edge_weight_t" in data:
            edge_index, edge_weight = data.edge_index_t, data.edge_weight_t
        else:
            edge_index, edge_weight = data.edge_index_l, data.edge_weight_l
            
        
        # Map transmitter embeddings to node embeddings
        x = torch.index_select(input=x, dim = 0, index = data.transmitters_index)
        
        # Denoise noisy node embeddings
        val = self.gnn(x, t = t, edge_index = edge_index, edge_weight = edge_weight, batch = data.batch,
                        return_attn_weights = return_attention_weights, debug_forward_pass = debug_forward_pass)

        if isinstance(val, tuple):
            if len(val) == 3:
                debug_data = val[2]
            attn_weights = val[1]
            val = val[0]
        else:
            attn_weights = None
            debug_data = None

        # Map node embeddings to transmitter embeddings
        Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean') # reduction doesn't matter as each index is unique
        # print(f'y = GNN.forward(x, G), x.shape = {x.shape}\ty.shape = {val.shape}\tTx_embed.shape = {Tx_embeddings.shape}')
        return Tx_embeddings, attn_weights, debug_data
    
    @property
    def prototype_name(self):
        return f"DiffusionResPlusGNN_{self.gnn.depth}_layers"
    

####################################################################################################################################
    
    

from torch_geometric.nn.conv import TransformerConv

class ReadInBlock(torch.nn.Module):
    def __init__(self, x_in_channels = 1, x_embed_channels = 16, timestep_embed_channels = 16, edge_weight_embed_channels = 16, num_timestep_embed = 1):
        super(ReadInBlock, self).__init__()
        self.x_embed_channels = x_embed_channels
        self.edge_weight_embed_channels = edge_weight_embed_channels
        self.timestep_embed_channels = timestep_embed_channels
        self.num_timestep_embed = num_timestep_embed

        self.edge_weight_embed = nn.Linear(1, edge_weight_embed_channels)
        self.x_embed = TransformerConv(in_channels=x_in_channels,
                                        out_channels=x_embed_channels,
                                        heads = 1,
                                        concat=False,
                                        dropout=0.,
                                        # edge_dim=0
                                        edge_dim = edge_weight_embed_channels
                                        )
        # self.x_embed = LEConv(x_in_channels, x_embed_channels) # nn.Linear(x_in_channels, x_embed_channels)

        self.t_embed = nn.Sequential(SinusoidalEmbedding(size=timestep_embed_channels * 2, scale=1.), nn.Linear(timestep_embed_channels * 2, timestep_embed_channels * num_timestep_embed), nn.Flatten(start_dim=-2))

    def forward(self, x, t, edge_index, edge_weight):

        edge_weight_embed = self.edge_weight_embed(edge_weight.unsqueeze(-1))
        x_embed = self.x_embed(x, edge_index = edge_index, edge_attr = edge_weight_embed, return_attention_weights = None)
        t_embed = self.t_embed(t)

        # if not self.num_timestep_embed == 1:
        t_embed = t_embed.view(*t_embed.shape[:-1], -1, self.num_timestep_embed)
        t_embed = torch.moveaxis(t_embed, source=-1, destination=0)

        return x_embed, edge_weight_embed, t_embed
    

# class DiffusionGraphTransformer(torch.nn.Module):
#     def __init__(self, nfeatures=1, nblocks = 3, nunits=16, nsteps = 100, **kwargs):
#         super(DiffusionGraphTransformer, self).__init__()
#         self.nsteps = nsteps
#         self.inblock = ReadInBlock(x_in_channels = nfeatures, x_embed_channels = nunits, timestep_embed_channels = nunits, edge_weight_embed_channels=1, num_timestep_embed=nblocks)
#         self.midblock = nn.ModuleList([GraphTransformer(in_channels=self.inblock.x_embed_channels + self.inblock.timestep_embed_channels,
#                                                         hidden_channels=nunits,
#                                                         out_channels=self.inblock.x_embed_channels,
#                                                         attn_edge_dim = self.inblock.edge_weight_embed_channels,
#                                                         **kwargs) for _ in range(nblocks)])
#         # self.bn = BatchNorm(in_channels=self.graph_unet.conv1.in_channels)
#         self.outblock = nn.Linear(self.midblock[-1].out_channels, nfeatures, bias = True)

#     def forward(self, x: torch.Tensor, t: torch.Tensor, data):
#         # x = self.bn(x)
#         val, edge_weight_embed, t_embed = self.inblock(x, t, edge_index = data.edge_index_l, edge_weight = data.edge_weight_l)
#         # val = x_embed.clone()
#         for i, midblock in enumerate(self.midblock):
#             # val = midblock(val, t = t_embed[i], edge_index = data.edge_index_l, edge_weight = data.edge_weight_l, batch = data.batch)
#             val = midblock(val, t = t_embed[i], edge_index = data.edge_index_l, edge_weight = edge_weight_embed, batch = data.batch)
        
#         val = self.outblock(val)
#         Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean')
#         return Tx_embeddings
    
#     @property
#     def prototype_name(self):
#         return f"DiffusionGraphTransformer"
    
class DiffusionGraphTransformer(torch.nn.Module):
    def __init__(self, nfeatures=1, nblocks = 3, nsubblocks = 3, nunits=16, nsteps = 100, **kwargs):
        super(DiffusionGraphTransformer, self).__init__()
        self.nsteps = nsteps
        self.nblocks = nblocks
        self.nsubblocks = nsubblocks
        self.inblock = nn.Linear(nfeatures, nunits)

        self.midblock = nn.ModuleList([GraphTransformerv4(nsteps=nsteps, nlayers=self.nsubblocks,
                                                          in_channels_gnn=nfeatures, hidden_channels=nunits * 4, out_channels=nunits,
                                                          **kwargs) for _ in range(nblocks)])
        # self.bn = BatchNorm(in_channels=self.graph_unet.conv1.in_channels)
        self.outblock = nn.Linear(nunits, nfeatures, bias = True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, data, return_attention_weights = None):
        attn_weights_list_list = []
        y = x.clone()
        val = self.inblock(x)
        for i, midblock in enumerate(self.midblock):
            val, attn_weights_list = midblock(x = val, y = y, t = t, edge_index = data.edge_index_l, edge_weight = data.edge_weight_l.unsqueeze(-1), batch = data.batch, return_attention_weights = return_attention_weights)
            # val = midblock(val, t = t, edge_index = data.edge_index_l, edge_weight = edge_weight_embed, batch = data.batch)
            attn_weights_list_list.append(attn_weights_list)
        
        val = self.outblock(val)
        Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean')
        return Tx_embeddings, attn_weights_list_list
    
    @property
    def prototype_name(self):
        return f"DiffusionGraphTransformer"
    

class GraphTransformerConditionalDiffusionMLP(torch.nn.Module):
    # def __init__(self, nfeatures=[NUM_NODES, 1], nblocks = 1, nsubblocks = 2, nunits=[128, 128], nsteps = 100, **kwargs):
    #     super(GraphTransformerConditionalDiffusionMLP, self).__init__()
    #     self.nsteps = nsteps
    #     self.nlayers = nblocks
        
    #     self.blocks = nn.ModuleList([])
    #     for _ in range(self.nlayers):
    #         block = GraphTransformerv2(nsteps=nsteps,
    #                                    nlayers=nsubblocks,
    #                                    in_channels_mlp=nfeatures[0],
    #                                    in_channels_gnn=nfeatures[1],
    #                                    hidden_channels_mlp=nunits[0],
    #                                    hidden_channels_gnn=nunits[1],
    #                                    timestep_embed_channels=2 * nunits[0])
    #         self.blocks.append(block)

    #     self.outblock = nn.Identity()


    def __init__(self, nfeatures=[NUM_NODES, 1], nblocks = 1, nsubblocks = 2, nunits=[128, 128], nsteps = 100, **kwargs):
        super(GraphTransformerConditionalDiffusionMLP, self).__init__()
        self.nsteps = nsteps
        self.nlayers = nblocks
        
        self.blocks = nn.ModuleList([])
        for i in range(self.nlayers):
            block = GraphTransformerv3(nsteps=nsteps,
                                       nlayers=nsubblocks,
                                       in_channels_mlp=nfeatures[0] if i == 0 else self.blocks[-1].out_channels_mlp,
                                       in_channels_gnn=nfeatures[1],
                                       hidden_channels_mlp=nunits[0],
                                       hidden_channels_gnn=nunits[1],
                                       out_channels_gnn=nunits[1],
                                       out_channels_mlp=(nunits[1] // 2) * nfeatures[0],
                                       timestep_embed_channels=2 * nunits[0])
            self.blocks.append(block)

        self.outblock = nn.Linear(self.blocks[-1].out_channels_gnn // 2, 1, bias = True)
    

    def forward(self, x: torch.Tensor, t: torch.Tensor, data, conditioning = True):
        if not conditioning:
            edge_index_l, edge_weight_l, batch = None, None, None
        else:
            edge_index_l, edge_weight_l, batch = data.edge_index_l, data.edge_weight_l, data.batch

        n = data.num_nodes // data.num_graphs
        val = x.clone().view(-1, n, x.shape[-1])
        for block in self.blocks:
            y = x.clone()
            val = block(val, y = y, t = t, edge_index = edge_index_l, edge_weight = edge_weight_l, batch = batch)

        # x = scatter(x, data.transmitters_index, dim = 0, reduce='mean')
        # Tx_embeddings = self.outblock(x)
    
        val = self.outblock(val).view(-1, 1) # [Bxn, 1]
        Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean')
        # y = self.outblock(Tx_embeddings)
        return Tx_embeddings
    
    @property
    def prototype_name(self):
        return f"DiffusionMLP_{self.nlayers}_layers_graph_transformer_conditional_backbone_{[block.nlayers for block in self.blocks]}_layers"
    

class MLPConditionalDiffusionGraphTransformer(torch.nn.Module):
    def __init__(self, nfeatures=1, nblocks = 1, nunits=128, nsteps = 100, **kwargs):
        super(MLPConditionalDiffusionGraphTransformer, self).__init__()
        self.nsteps = nsteps
        self.graph_transformer = DiffusionGraphTransformer(nfeatures=nfeatures, nsteps=nsteps, nblocks=nblocks, nunits=nunits, outfeatures=2)
        self.mlp = nn.ModuleList([nn.Linear(2 * NUM_NODES, 128), nn.ReLU()])
        for _ in range(nblocks):
            self.mlp.append(nn.Linear(128, 128))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(128, NUM_NODES))
        self.mlp = nn.Sequential(*self.mlp)
       

    def forward(self, x: torch.Tensor, t: torch.Tensor, data, conditioning = True):
        x_t_mlp_embed = torch.cat((x.view(-1, NUM_NODES), (t / (self.nsteps-1)).view(-1, NUM_NODES)), dim = -1)
        y = self.mlp(x_t_mlp_embed).view(-1, 1)

        if conditioning:
            condition = self.graph_transformer(x, t, data)
            y = y * condition[..., 0:1] + condition[..., 1:2]

        return y
    
    @property
    def prototype_name(self):
        return f"DiffusionGraphTransformer_mlp_conditional_backbone" 

    


class MLPConditionalDiffusionMLP(torch.nn.Module):
    def __init__(self, nfeatures, nunits = 64, nblocks = 2, nsteps = 100, nsubblocks = 1, **kwargs):
        super(MLPConditionalDiffusionMLP, self).__init__()
        self.nlayers = nblocks
        self.activation = F.softplus # F.leaky_relu
        self.inblock = mlp_conditional_mlp_backbone(num_in=nfeatures[0], graph_embed_num_in=nfeatures[1], num_out=nunits, nsteps=nsteps, activation=self.activation, nlayers=nsubblocks, **kwargs) # nlayers = 1
        self.midblock = nn.ModuleList([mlp_conditional_mlp_backbone(num_in=nunits, graph_embed_num_in=nfeatures[1], num_out=nunits, nsteps=nsteps, activation=self.activation, nlayers=nsubblocks, **kwargs) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, nfeatures[0], bias = True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, data, conditioning = True):
        x_unflattened = x.view(-1, self.inblock.num_in)
        t_unflattened = t.view(-1, self.inblock.num_in)[:, 0:1]

        adjacencies = to_dense_adj(edge_index=data.edge_index_l, batch=data.batch, edge_attr=data.edge_weight_l)
        a_flattened = adjacencies.view(-1, self.inblock.graph_embed_num_in)

        val = self.inblock(x_unflattened, t_unflattened, a_flattened)
        for midblock in self.midblock:
            val = midblock(val, t_unflattened, a_flattened)
        # Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean')
        val = self.outblock(val).view(-1, 1)
        Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean')
        return Tx_embeddings
    
    @property
    def prototype_name(self):
        return f"DiffusionMLP_{self.nlayers}_layers_conditional_mlp_backbone_{[midblock.nlayers for midblock in self.midblock]}_layers"
    
    

class MLPConditionalDiffusionGNN(torch.nn.Module):
    def __init__(self, nfeatures, nunits = 64, nblocks = 2, nsteps = 100, nsubblocks = 1, **kwargs):
        super(MLPConditionalDiffusionGNN, self).__init__()
        self.nlayers = nblocks
        self.activation = F.softplus # F.leaky_relu
        self.inblock = mlp_conditional_gnn_backbone(num_in=nfeatures[0], gnn_num_in=nfeatures[1], hidden_dim=nunits, nsteps=nsteps, activation=self.activation, nlayers=nsubblocks, **kwargs)
        self.midblock = nn.ModuleList([mlp_conditional_gnn_backbone(num_in=nfeatures[0], gnn_num_in=nfeatures[1], hidden_dim=nunits, nsteps=nsteps, activation=self.activation, nlayers=nsubblocks, **kwargs) for _ in range(nblocks)])
        # self.outblock = nn.Linear(nunits, nfeatures[0], bias = True)
        # self.outblock = nn.Linear(nfeatures[0], nfeatures[0])
        self.outblock = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, data, conditioning = True):
        # y = torch.ones_like(x) # constant node signal
        y = x.clone()
        # y = torch.randn_like(x) # random node signal

        x = x.view(-1, self.inblock.num_in)
        t = t.view(-1, self.inblock.num_in)[:, 0:1]

        if not conditioning:
            edge_index_l, edge_weight_l = None, None
        else:
            edge_index_l, edge_weight_l = data.edge_index_l, data.edge_weight_l

        val = self.inblock(x, t, y, edge_index_l, edge_weight_l)
        for midblock in self.midblock:
            val = midblock(val, t, y, edge_index_l, edge_weight_l)

        val = self.outblock(val).view(-1, 1)
        Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean')
        return Tx_embeddings
        # Tx_embeddings = val
        # # Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean')
        # y = self.outblock(Tx_embeddings)
        # y = y.view(-1, 1)
        # return y

    @property
    def prototype_name(self):
        return f"DiffusionGNN_{self.nlayers}_layers_conditional_mlp_backbone_{[midblock.nlayers for midblock in self.midblock]}_layers"
    



class GNNConditionalDiffusionGNN(torch.nn.Module):
    def __init__(self, nfeatures, nunits = 64, nblocks = 2, nsteps = 100, nsubblocks = 1, **kwargs):
        super(GNNConditionalDiffusionGNN, self).__init__()
        self.nlayers = nblocks
        self.activation = F.softplus # F.leaky_relu
        self.inblock = gnn_conditional_gnn_backbone(x_gnn_num_in=nfeatures, cond_gnn_num_in=nfeatures, num_out=nunits, nsteps=nsteps, activation=self.activation, nlayers=nsubblocks, **kwargs)
        self.midblock = nn.ModuleList([gnn_conditional_gnn_backbone(x_gnn_num_in=nunits, cond_gnn_num_in=nfeatures, num_out=nunits, nsteps=nsteps, activation=self.activation, nlayers=nsubblocks, **kwargs) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, nfeatures, bias = True)


    def forward(self, x: torch.Tensor, t: torch.Tensor, data, conditioning = True):

        if conditioning:
            edge_index_list = [data.avg_graph_edge_index_l, data.edge_index_l]
            edge_weight_list = [data.avg_graph_edge_weight_l, data.edge_weight_l]
        else:
            edge_index_list = [data.avg_graph_edge_index_l, None]
            edge_weight_list = [data.avg_graph_edge_weight_l, None]

        # y = torch.ones_like(x) # constant node signal
        y = x.clone() # clone the input for skip-connection
        # y = torch.randn_like(x) # random node signal

        val = self.inblock(x, t, y, edge_index_list, edge_weight_list)
        for midblock in self.midblock:
            val = midblock(val, t, y, edge_index_list, edge_weight_list)

        Tx_embeddings = scatter(val, data.transmitters_index, dim=0, reduce='mean')
        y = self.outblock(Tx_embeddings)
        return y
      

    @property
    def prototype_name(self):
        return f"DiffusionGNN_{self.nlayers}_layers_conditional_gnn_backbone_{[midblock.nlayers for midblock in self.midblock]}_layers"
    



def save_cd_train_chkpt(save_path, model, optimizer, lr_sched, **kwargs):
    experiment_name = kwargs.get('experiment_name', "")
    epoch = kwargs.get("epoch", None)
    loss = kwargs.get("loss", None)
    is_best_model = kwargs.get("is_best_model", False)
    accelerator = kwargs.get('accelerator', None)
    global_chkpt_step = kwargs.get('global_chkpt_step', None)

    experiment_name = "_" + experiment_name if len(experiment_name) and experiment_name is not None else experiment_name
    if epoch is not None:
        experiment_name = experiment_name + f"_epoch_{epoch}"

    # save checkpoint
    chkpt = {"model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": lr_sched.state_dict(),
             "epoch": epoch,
             "loss": loss,
             "is_best_model": is_best_model,
             "global_chkpt_step": global_chkpt_step
             }
    

    print("Disabled printing of checkpoint for disk memory management.")
    torch.save(chkpt, f'{save_path}/cd_train_chkpt{experiment_name}.pt')

    
    
    # if accelerator is None:
    #     torch.save(chkpt, f'{save_path}/cd_train_chkpt{experiment_name}.pt')
    # else:
    #     accelerator.save(chkpt, f'{save_path}/cd_train_chkpt{experiment_name}.pt')

    # accelerator.save_state()



def get_diffusion_model_architecture(args, n_features, num_nodes, diffusion_steps):
    print(f"Creating diffusion model with architecture: {args.gnn_backbone}, n_features = {n_features}, num_nodes = {num_nodes}, diffusion_steps = {diffusion_steps}")

    if args.gnn_backbone == 'mlp-conditional-mlp':
        n_features = [num_nodes] + [num_nodes**2]
        model = MLPConditionalDiffusionMLP(nfeatures=n_features, nunits=args.hidden_dim, nblocks=args.n_layers, nsubblocks=args.n_sublayers, nsteps=diffusion_steps,
                                           batch_norm = args.batch_norm, sinusoidal_time_embed = args.sinusoidal_time_embed,
                                           condition_by_summation_weight = args.condition_by_summation_weight
                                           )

    elif args.gnn_backbone == 'mlp-conditional-gnn':
        n_features = [num_nodes] + [n_features]
        model = MLPConditionalDiffusionGNN(nfeatures=n_features, nunits=args.hidden_dim, nblocks=args.n_layers, nsubblocks=args.n_sublayers, nsteps=diffusion_steps,
                                        conv_model_architecture = args.model, batch_norm=args.batch_norm,
                                        k_hops = args.k_hops, dropout_rate = args.dropout_rate,
                                        sinusoidal_time_embed = args.sinusoidal_time_embed,
                                        condition_by_summation_weight = args.condition_by_summation_weight
                                        )
        
    elif args.gnn_backbone == 'gnn-conditional-gnn':
        model = GNNConditionalDiffusionGNN(nfeatures=n_features, nunits=args.hidden_dim, nblocks=args.n_layers, nsubblocks=args.n_sublayers, nsteps=diffusion_steps,
                                           conv_model_architecture = args.model, batch_norm=args.batch_norm,
                                           k_hops = args.k_hops, dropout_rate = args.dropout_rate,
                                           sinusoidal_time_embed = args.sinusoidal_time_embed,
                                           condition_by_summation_weight = args.condition_by_summation_weight
                                           )
        
    elif args.gnn_backbone == 'gnn':
        model = DiffusionGNN(nfeatures=n_features, nunits=args.hidden_dim, nblocks=args.n_layers, nsteps = diffusion_steps,
                             norm = args.norm_layer,
                            #  pool_ratio = args.pool_ratio,
                            #  pool_layer = args.pool_layer,
                            #  pool_multiplier = args.pool_multiplier,
                             sum_skip_connection = args.sum_skip_connection,
                            #  use_checkpointing = args.use_checkpointing,
                            #  attn_self_loop_fill_value = args.attn_self_loop_fill_value,
                            #  apply_gcn_norm=args.apply_gcn_norm
                            )

    elif args.gnn_backbone == 'simple-gnn':
        model = DiffusionSimpleGNN(nfeatures=n_features, nunits=args.hidden_dim, nblocks=args.n_layers, nsteps = diffusion_steps,
                             norm = args.norm_layer, conv_layer_normalize = args.conv_layer_normalize, k_hops = args.k_hops
                            #  pool_ratio = args.pool_ratio,
                            #  pool_layer = args.pool_layer,
                            #  pool_multiplier = args.pool_multiplier,
                            #  sum_skip_connection = args.sum_skip_connection,
                            #  use_checkpointing = args.use_checkpointing,
                            #  attn_self_loop_fill_value = args.attn_self_loop_fill_value,
                            #  apply_gcn_norm=args.apply_gcn_norm
                            )
        
    elif args.gnn_backbone == 'resplus-gnn':
        model = DiffusionResPlusGNN(nfeatures=n_features, nunits=args.hidden_dim, nblocks=args.n_layers, nsteps = diffusion_steps,
                             norm = args.norm_layer, conv_layer_normalize = args.conv_layer_normalize, k_hops = args.k_hops,
                             use_res_connection_time_embed = args.use_res_connection_time_embed,
                             dropout_rate = args.dropout_rate,
                             norm_kws = {'layer_norm_mode': args.layer_norm_mode}
                            #  pool_ratio = args.pool_ratio,
                            #  pool_layer = args.pool_layer,
                            #  pool_multiplier = args.pool_multiplier,
                            #  sum_skip_connection = args.sum_skip_connection,
                            #  use_checkpointing = args.use_checkpointing,
                            #  attn_self_loop_fill_value = args.attn_self_loop_fill_value,
                            #  apply_gcn_norm=args.apply_gcn_norm
                            )
        
        
    elif args.gnn_backbone == 'gnn-unet':
        model = DiffusionGraphUNet(nfeatures=n_features, nunits=args.hidden_dim, nblocks=args.n_layers, nsteps=diffusion_steps,
                                   norm = args.norm_layer,
                                   pool_ratio = args.pool_ratio,
                                   pool_layer = args.pool_layer,
                                   pool_multiplier = args.pool_multiplier,
                                   sum_skip_connection = args.sum_skip_connection,
                                   use_checkpointing = args.use_checkpointing,
                                   attn_self_loop_fill_value = args.attn_self_loop_fill_value,
                                   apply_gcn_norm=args.apply_gcn_norm
                                   )

    elif args.gnn_backbone == 'graph-transformer':
        model = DiffusionGraphTransformer(nfeatures=n_features, nblocks=args.n_layers, nsubblocks=args.n_sublayers, nunits=args.hidden_dim, nsteps=diffusion_steps,
                                          dropout_rate = args.dropout_rate,
                                          activation = args.activation, # nn.LeakyReLU()
                                          res_connection = args.res_connection,
                                          norm_layer = args.norm_layer,
                                          use_checkpointing = args.use_checkpointing,
                                          attn_num_heads = args.attn_num_heads
                                          )

    elif args.gnn_backbone == 'mlp-conditional-graph-transformer':
        model = MLPConditionalDiffusionGraphTransformer(nfeatures=1, nblocks=args.n_layers, nunits=args.hidden_dim, nsteps=diffusion_steps)

    elif args.gnn_backbone == 'graph-transformer-conditional-mlp':
        # n_features = [NUM_NODES] + [n_features]
        n_features = [NUM_NODES] + [2]
        hiddendims = [args.hidden_dim * (n_features[0] // n_features[1])] + [args.hidden_dim]
        model = GraphTransformerConditionalDiffusionMLP(nfeatures=n_features,
                                                        nblocks=args.n_layers,
                                                        nunits=hiddendims,
                                                        nsubblocks=args.n_sublayers,
                                                        nsteps=diffusion_steps
                                                        )
    else:
        raise NotImplementedError(f"Model architecture {args.gnn_backbone} is not implemented.")
    

    return model


    

def create_cd_model(accelerator, args, n_features = 1, diffusion_steps = 100, device = 'cpu', num_nodes = 100):
    '''
    Create and initialize the (conditional) diffusion model.
    '''
    # from utils.model_utils import PrimalGNN

    # model = PrimalGNN(conv_model=args.model, P_max=0.01, num_features_list=[n_features] + args.num_features_list, k_hops = args.k_hops, batch_norm = args.batch_norm, dropout_rate = args.dropout_rate)
    
    # model = DiffusionGNN(nfeatures=n_features, nunits=128, nblocks = 4, nsteps=diffusion_steps, nsubblocks=3)


    model = get_diffusion_model_architecture(args, n_features=n_features, num_nodes=num_nodes, diffusion_steps=diffusion_steps)
    
    is_trained = False
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = accelerator.device
    model = model.to(device)

    if hasattr(args, 'load_model_chkpt_path') and args.load_model_chkpt_path is not None:
        try:
            # with accelerator.main_process_first():
            state_dict = torch.load(args.load_model_chkpt_path)
            # Print model's state_dict
            accelerator.print("Loaded state_dict:")
            for param_tensor in state_dict:
                accelerator.print(param_tensor, "\t", state_dict[param_tensor].size())

            print("Model's state_dict:")
            for param_tensor in model.state_dict():
                accelerator.print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            
            # useful to unwrap only if you use the load function after making your model go through prepare()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(args.load_model_chkpt_path))

            accelerator.print('Loading pretrained diffusion model weights is successful.')
            is_trained = True
        except:
            accelerator.print(f'Loading model state dict from path {args.load_model_chkpt_path} failed.\nDefault weight initialization is applied.')

    return model, is_trained
