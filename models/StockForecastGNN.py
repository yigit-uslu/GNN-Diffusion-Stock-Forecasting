import torch
import torch.nn as nn
from models.ResidualGNN import ResidualGNN


class StockForecastGNN(ResidualGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, norm, **kwargs):
        super(StockForecastGNN, self).__init__(in_channels = in_channels,
                                               hidden_channels = hidden_channels,
                                               out_channels = out_channels,
                                               depth = depth,
                                               norm = norm,
                                               **kwargs
                                               )
        
        n_features = 8
        past_window = 25

        self.concat_embeddings = True

        embed_channels = in_channels // 2 if self.concat_embeddings else in_channels

        self.x_embed_layer = nn.Sequential(nn.Linear(n_features * past_window, embed_channels), nn.LeakyReLU(), nn.Linear(embed_channels, embed_channels))
        self.y_embed_layer = nn.Sequential(nn.Linear(1, embed_channels), nn.LeakyReLU(), nn.Linear(embed_channels, embed_channels))


    def forward(self, x: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor, edge_index, edge_weight, batch = None, return_attn_weights=False, debug_forward_pass=False):

        print(f"y_t.shape: {y_t.shape}\tx.shape: {x.shape}\tt.shape: {t.shape}")
        
        y_t_embed = self.y_embed_layer(y_t)  # Embed y_t to match x's shape [batch_size, in_channels]
        x_embed = self.x_embed_layer(x)  # Embed x to match y_t's shape [batch_size, in_channels]
        print(f"x_embed.shape: {x_embed.shape}\ty_embed.shape: {y_t_embed.shape}")

        if self.concat_embeddings:
            x_and_y_t = torch.cat([x_embed, y_t_embed], dim=-1)
        else:
            x_and_y_t = x_embed + y_t_embed

        # Implement the forward pass logic here
        return super().forward(x = x_and_y_t, t = t, edge_index = edge_index, edge_weight = edge_weight, batch = batch,
                        return_attn_weights = return_attn_weights, debug_forward_pass = debug_forward_pass)


########################################################### Res+GNN adapted for stock forecasting ######################################################################

""" We use the same class signature for GNN as the GraphUNet """
class StockForecastDiffusionGNN(nn.Module):
    def __init__(self, nfeatures=1, nblocks = 1, nunits=128, nsteps = 100, **kwargs):
        super(StockForecastDiffusionGNN, self).__init__()
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

        
        self.gnn = StockForecastGNN(in_channels=nunits, # nfeatures,
                                    hidden_channels=nunits, out_channels=nfeatures, # out_channels don't matter
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
        

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, data, return_attention_weights = False, debug_forward_pass = False):

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = x.reshape(y_t.shape[0], -1)  # Reshape x to match y_t's shape [batch_size, num_features * past_window]
        
        # Denoise noisy node embeddings
        val = self.gnn(y_t = y_t, t = t, x = x, edge_index = edge_index, edge_weight = edge_weight, batch = batch,
                        return_attn_weights = return_attention_weights, debug_forward_pass = debug_forward_pass)

        if isinstance(val, tuple):
            if len(val) == 3:
                debug_data = val[2]
            attn_weights = val[1]
            val = val[0]
        else:
            attn_weights = None
            debug_data = None
  
        return val, attn_weights, debug_data
    
    @property
    def prototype_name(self):
        return f"DiffusionResPlusGNN_{self.gnn.depth}_layers"
    

####################################################################################################################################