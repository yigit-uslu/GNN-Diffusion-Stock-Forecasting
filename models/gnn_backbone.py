import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LEConv, TAGConv, GINConv, GINEConv, GATv2Conv, aggr, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.norm import BatchNorm, LayerNorm, GraphNorm, DiffGroupNorm
from torch_scatter import scatter
from torch_geometric.transforms import normalize_features
from torch_geometric.utils import add_self_loops
import copy

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    

class SinusoidalTimeEmbedding(nn.Module):
    """
    https://nn.labml.ai/diffusion/ddpm/unet.html 
    """
    def __init__(self, n_channels: int, act: nn.Module = Swish(), T_max: float = 10000):
        super().__init__()
        self.n_channels = n_channels
        self.act = act
        self.T_max = T_max

        self.lin_embed = nn.Sequential(nn.Flatten(start_dim=-2),
                                       nn.Linear(self.n_channels // 4, self.n_channels),
                                       self.act,
                                       nn.Linear(self.n_channels, self.n_channels)
                                       )
        
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = torch.log(torch.Tensor([self.T_max])) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = t.device) * -emb.to(t.device))
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim = -1)

        emb = self.lin_embed(emb)
        return emb
    

class NormLayer(nn.Module):
    def __init__(self, norm, in_channels, **kwargs):
        super().__init__()
        self.norm = norm
        self.in_channels = in_channels
        self.n_groups = kwargs.get('n_groups', 8)
        self.resolve_norm_layer(norm = self.norm, in_channels=in_channels, **kwargs)


    def resolve_norm_layer(self, norm, in_channels, **kwargs):
        n_groups = kwargs.get('n_groups', 8)

        if norm == 'batch':
            self.norm_layer = BatchNorm(in_channels)
        elif norm == 'layer':
            self.norm_layer = LayerNorm(in_channels)
        elif norm == 'group':
            self.norm_layer = DiffGroupNorm(in_channels=in_channels, groups=n_groups)
        elif norm == 'graph':
            self.norm_layer = GraphNorm(in_channels=in_channels)
        elif norm == 'none' or norm is None:
            self.norm_layer = nn.Identity()


    def forward(self, x, batch = None, batch_size = None):
        if self.norm in ['batch', 'layer', 'group', 'none', None]:
            return self.norm_layer(x)
        elif self.norm in ['graph']:
            # print('x.shape: ', x.shape)
            # print('batch.shape: ', batch.shape)
            return self.norm_layer(x, batch = batch)
        else:
            raise NotImplementedError
        

class ResGraphConvBlock(nn.Module):
    """ 
    Residual GCN block.
    """
    def __init__(self, conv_layer, norm_layer, activation, dropout_rate, res_connection, layer_ord = ['conv', 'norm', 'act', 'dropout']):
        super().__init__()
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation = activation
        self.res_connection = res_connection
        self.dropout_rate = dropout_rate
        self.layer_ord = layer_ord


    def forward(self, y: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None):
        if self.layer_ord == ['conv', 'norm', 'act', 'dropout']:

            if any([isinstance(self.conv_layer, _) for _ in [LEConv, TAGConv]]):
                h = self.conv_layer(y, edge_index=edge_index, edge_weight=edge_weight)
            else:
                h = self.conv_layer(y, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))

            h = self.norm_layer(h, batch = batch) # identity layer if no batch/graph normalization is used.
            h = self.activation(h)
            h = F.dropout(h, p = self.dropout_rate, training=self.training)
            out = self.res_connection(y) + h

        else:
            raise NotImplementedError
        
        return out
    


def get_conv_model(conv_model_architecture, num_in, num_out, **kwargs):
    aggregation = kwargs.get('aggregation', None)
    k_hops = kwargs.get('k_hops', 2)
    num_heads = kwargs.get('num_heads', 2)
    dropout_rate = kwargs.get('dropout_rate', 0.0)
    conv_layer_normalize = kwargs.get('conv_layer_normalize', False)

    if conv_model_architecture == 'LeConv':
        if aggregation is not None:
            conv_layer = LEConv(num_in, num_out, aggr = copy.deepcopy(aggregation))
        else:
            conv_layer = LEConv(num_in, num_out)

    elif conv_model_architecture == 'TAGConv':
        if aggregation is not None:
            conv_layer = TAGConv(in_channels=num_in, out_channels=num_out, K=k_hops, normalize=False, aggr = copy.deepcopy(aggregation))
        else:
            conv_layer = TAGConv(in_channels=num_in, out_channels=num_out, K=k_hops, normalize=False)

    elif conv_model_architecture == 'GINEConv':
        eps_0 = 0.1
        train_eps = True
        if aggregation is not None:
            conv_layer = GINEConv(nn=nn.Linear(in_features=num_in, out_features=num_out), eps = eps_0, train_eps=train_eps, aggr = copy.deepcopy(aggregation))
        else:
            conv_layer = GINEConv(nn=nn.Linear(in_features=num_in, out_features=num_out), eps = eps_0, train_eps=train_eps)

    elif conv_model_architecture == 'GATv2Conv':
        add_self_loops = False
        heads = num_heads
        dropout = 0.
        if aggregation is not None:
            conv_layer = GATv2Conv(in_channels=num_in,
                                    out_channels=num_out // heads,
                                    heads=heads,
                                    dropout=dropout,
                                    add_self_loops=add_self_loops,
                                    edge_dim=1,
                                    aggr = copy.deepcopy(aggregation)
                                    )
        else:
            conv_layer = GATv2Conv(in_channels=num_in,
                                    out_channels=num_out // heads,
                                    heads=heads,
                                    dropout=dropout,
                                    edge_dim=1,
                                    add_self_loops=add_self_loops
                                    )
    
    else:
        raise NotImplementedError

    return conv_layer   


# backbone GNN class
class gnn_backbone(torch.nn.Module):
    def __init__(self, conv_model_architecture, num_features_list, **kwargs):
        super(gnn_backbone, self).__init__()
        k_hops = kwargs.get('k_hops', 2)
        num_layers = len(num_features_list)
        activation = kwargs.get('activation', 'leaky_relu')
        aggregation = kwargs.get('aggregation', None)

        self.dropout_rate = kwargs.get('dropout_rate', 0.0)
        num_heads = kwargs.get('num_heads', 2)
        norm = kwargs.get('norm_layer', 'batch')
        global_pooling = kwargs.get('global_pooling', None)

        # Define activation functions
        if activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'softplus':
            self.activation = F.softplus
        else:
            raise NotImplementedError
        
        # Define optional pooling layers after the last conv + (batch norm + nonlinearity + dropout) layer
        self.global_pooling_layer = None
        if global_pooling is not None:
            if global_pooling == 'max':
                self.global_pooling_layer = global_max_pool
            elif global_pooling == 'mean':
                self.global_pooling_layer = global_mean_pool
            elif global_pooling == 'add':
                self.global_pooling_layer = global_add_pool
        

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.res_connections = nn.ModuleList()

        for i in range(num_layers - 1):
            conv_layer = get_conv_model(conv_model_architecture=conv_model_architecture,
                                        num_in=num_features_list[i],
                                        num_out=num_features_list[i+1],
                                        aggregation=aggregation,
                                        k_hops=k_hops,
                                        num_heads=num_heads,
                                        dropout_rate=self.dropout_rate
                                        )
            self.conv_layers.append(conv_layer)

            norm_layer = NormLayer(norm=norm, in_channels=num_features_list[i+1])
            self.norm_layers.append(norm_layer)

            # If the number of input channels is not equal to the number of output channels we have to project the shortcut connection
            if num_features_list[i] != num_features_list[i+1]:
                res_connection = nn.Linear(in_features=num_features_list[i], out_features=num_features_list[i+1], bias = False)
            else:
                res_connection = nn.Identity()
            self.res_connections.append(res_connection)


            
    def forward(self, y, edge_index, edge_weight, batch = None):
        # Apply normalization or get sinusoidal embeddings of lambdas before passing through graph-conv layers.
        # pos_embedding_scaling = 50 / LAMBDAS_MAX

        for i, (norm_layer, conv_layer, res_connection) in enumerate(zip(self.norm_layers, self.conv_layers, self.res_connections)):
            
            if any([isinstance(conv_layer, _) for _ in [LEConv, TAGConv]]):
                y = conv_layer(y, edge_index=edge_index, edge_weight=edge_weight)
            else:
                y = conv_layer(y, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))

            # if i < len(self.conv_layers)-1:
            y = norm_layer(y, batch = batch) # identity layer if no batch normalization is used.
            y = self.activation(y)
            y = F.dropout(y, p = self.dropout_rate, training=self.training)

        if self.global_pooling_layer is not None and batch is not None:
            y = self.global_pooling_layer(y, batch)
            
        return y
    


# residual backbone GNN class with layer ordering (Conv -> Norm -> Nonlinearity -> Dropout) + Residual connection
class res_gnn_backbone(torch.nn.Module):
    def __init__(self, conv_model_architecture, num_features_list, **kwargs):
        super(res_gnn_backbone, self).__init__()

        self.layer_ord = ['conv', 'norm', 'act', 'dropout']

        k_hops = kwargs.get('k_hops', 2)
        num_layers = len(num_features_list)
        activation = kwargs.get('activation', 'leaky_relu')
        aggregation = kwargs.get('aggregation', None)

        self.dropout_rate = kwargs.get('dropout_rate', 0.0)
        num_heads = kwargs.get('num_heads', 2)
        norm = kwargs.get('norm_layer', 'batch')
        global_pooling = kwargs.get('global_pooling', None)

        # Define activation functions
        if activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'softplus':
            self.activation = F.softplus
        else:
            raise NotImplementedError
        
        # Define optional pooling layers after the last conv + (batch norm + nonlinearity + dropout) layer
        self.global_pooling_layer = None
        if global_pooling is not None:
            if global_pooling == 'max':
                self.global_pooling_layer = global_max_pool
            elif global_pooling == 'mean':
                self.global_pooling_layer = global_mean_pool
            elif global_pooling == 'add':
                self.global_pooling_layer = global_add_pool
        

        self.blocks = nn.ModuleList()
        # self.norm_layers = nn.ModuleList()
        # self.res_connections = nn.ModuleList()

        for i in range(num_layers - 1):
            conv_layer = get_conv_model(conv_model_architecture=conv_model_architecture,
                                        num_in=num_features_list[i],
                                        num_out=num_features_list[i+1],
                                        aggregation=aggregation,
                                        k_hops=k_hops,
                                        num_heads=num_heads,
                                        dropout_rate=self.dropout_rate
                                        )
            # self.conv_layers.append(conv_layer)

            norm_layer = NormLayer(norm=norm,
                                   in_channels=num_features_list[i+1] if self.layer_ord.index('norm') > self.layer_ord.index('conv') else num_features_list[i])
            # self.norm_layers.append(norm_layer)

            # If the number of input channels is not equal to the number of output channels we have to project the shortcut connection
            if num_features_list[i] != num_features_list[i+1]:
                res_connection = nn.Linear(in_features=num_features_list[i], out_features=num_features_list[i+1], bias = False)
            else:
                res_connection = nn.Identity()
            # self.res_connections.append(res_connection)
                
            res_block = ResGraphConvBlock(conv_layer=conv_layer, norm_layer=norm_layer, activation=self.activation,
                                          dropout_rate=self.dropout_rate, res_connection=res_connection,
                                          layer_ord=self.layer_ord
                                          )
            self.blocks.append(res_block)


            
    def forward(self, y, edge_index, edge_weight, batch = None):
        # Apply normalization or get sinusoidal embeddings of lambdas before passing through graph-conv layers.
        # pos_embedding_scaling = 50 / LAMBDAS_MAX

        # for i, (norm_layer, conv_layer, res_connection) in enumerate(zip(self.norm_layers, self.conv_layers, self.res_connections)):
            
        #     if any([isinstance(conv_layer, _) for _ in [LEConv, TAGConv]]):
        #         y = conv_layer(y, edge_index=edge_index, edge_weight=edge_weight)
        #     else:
        #         y = conv_layer(y, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))

        #     # if i < len(self.conv_layers)-1:
        #     y = norm_layer(y, batch = batch) # identity layer if no batch normalization is used.
        #     y = self.activation(y)
        #     y = F.dropout(y, p = self.dropout_rate, training=self.training)

        for block in self.blocks:
            y = block(y=y, edge_index=edge_index, edge_weight=edge_weight, batch=batch)

        if self.global_pooling_layer is not None and batch is not None:
            y = self.global_pooling_layer(y, batch)
            
        return y