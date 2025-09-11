import torch
import torch.nn as nn
from models.ResidualGNN import ResidualGNNwithConditionalEmbeddings


# class StockForecastGNN(ResidualGNN):
#     def __init__(self, in_channels, hidden_channels, out_channels, depth, norm, **kwargs):
#         super(StockForecastGNN, self).__init__(in_channels = in_channels,
#                                                hidden_channels = hidden_channels,
#                                                out_channels = out_channels,
#                                                depth = depth,
#                                                norm = norm,
#                                                **kwargs
#                                                )
        
#         self.total_forward_calls = 0
        
#         n_features = 8
#         past_window = 25

#         self.concat_embeddings = True

#         embed_channels = in_channels // 2 if self.concat_embeddings else in_channels

#         self.x_embed_layer = nn.Sequential(nn.Linear(n_features * past_window, embed_channels), nn.LeakyReLU(), nn.Linear(embed_channels, embed_channels))
#         self.y_embed_layer = nn.Sequential(nn.Linear(1, embed_channels), nn.LeakyReLU(), nn.Linear(embed_channels, embed_channels))




#     def forward(self, x: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor, edge_index, edge_weight, batch = None, return_attn_weights=False, debug_forward_pass=False):

#         x = x.reshape(y_t.shape[0], -1)  # Reshape x to match y_t's shape [batch_size, num_features * past_window]

#         if self.total_forward_calls == 0:
#             print(f"First forward pass y_t.shape: {y_t.shape}\tx.shape: {x.shape}\tt.shape: {t.shape}")

#         y_t_embed = self.y_embed_layer(y_t)  # Embed y_t to match x's shape [batch_size, in_channels]
#         x_embed = self.x_embed_layer(x)  # Embed x to match y_t's shape [batch_size, in_channels]

#         if self.total_forward_calls == 0:
#             print(f"First forward pass x_embed.shape: {x_embed.shape}\ty_embed.shape: {y_t_embed.shape}")

#         if self.concat_embeddings:
#             x_and_y_t = torch.cat([x_embed, y_t_embed], dim=-1)
#         else:
#             x_and_y_t = x_embed + y_t_embed


#         pred_noise = super().forward(x = x_and_y_t, t = t, edge_index = edge_index, edge_weight = edge_weight, batch = batch,
#                         return_attn_weights = return_attn_weights, debug_forward_pass = debug_forward_pass)
        
#         if isinstance(pred_noise, tuple):
#             for i in range(len(pred_noise), 0, -1):
#                 if i == 3:
#                     model_debug_data = pred_noise[i-1]
#                 elif i == 2:
#                     attn_weights = pred_noise[i-1]
#                 elif i == 1:
#                     pred_noise = pred_noise[i-1]
#                 else:
#                     pass
#         else:
#             attn_weights = None
#             model_debug_data = None
        
#         if self.total_forward_calls == 0:
#             print(f"First forward pass pred_noise.shape: {pred_noise.shape}")

#         self.total_forward_calls += 1

#         return pred_noise
    


class StockForecastGNN(ResidualGNNwithConditionalEmbeddings):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, norm, num_features, num_timesteps, **kwargs):
        cond_in_channels = (num_features + 4) * num_timesteps  # +4 for sin/cos of two frequencies
        super(StockForecastGNN, self).__init__(in_channels, hidden_channels, out_channels, depth, norm, cond_in_channels, **kwargs)

        # Additional layers or modifications specific to stock forecasting can be added here

        self.total_forward_calls = 0
        
        # self.concat_embeddings = True

        # embed_channels = in_channels // 2 if self.concat_embeddings else in_channels

        # self.x_embed_layer = nn.Sequential(nn.Linear(cond_in_channels, embed_channels), nn.LeakyReLU(), nn.Linear(embed_channels, embed_channels))
        # self.y_embed_layer = nn.Sequential(nn.Linear(1, embed_channels), nn.LeakyReLU(), nn.Linear(embed_channels, embed_channels))


    def preprocess_x(self, x: torch.Tensor) -> torch.Tensor:

        # Scale by past_window to create a fundamental frequency that completes one cycle over the time window
        # Also add a higher frequency component for finer temporal resolution
        batch_size, num_features, past_window = x.shape
        t_x = torch.arange(past_window, device=x.device).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, past_window)
        t_x_scaled_1 = t_x * 2 * torch.pi / past_window  # Fundamental frequency (1 cycle per window)
        t_x_scaled_2 = t_x * 2 * torch.pi / (past_window // 4 + 1)  # Higher frequency (4 cycles per window)
        t_x_embed = torch.cat([
            torch.sin(t_x_scaled_1), torch.cos(t_x_scaled_1),  # Low frequency components
            torch.sin(t_x_scaled_2), torch.cos(t_x_scaled_2)   # High frequency components
        ], dim=1)  # Sinusoidal time embeddings: [batch_size, 4, past_window]
        x = torch.cat([x, t_x_embed], dim=1)  # [batch_size, num_features + 4, past_window]

        return x



    def forward(self, x: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor, edge_index, edge_weight, batch = None, return_attn_weights=False, debug_forward_pass=False):

        x = self.preprocess_x(x)
        x = x.reshape(y_t.shape[0], -1)  # Reshape x to match y_t's shape [batch_size, (num_features + 4) * past_window]

        # if self.total_forward_calls == 0:
        #     print(f"First forward pass y_t.shape: {y_t.shape}\tx.shape: {x.shape}\tt.shape: {t.shape}")

        # y_t_embed = self.y_embed_layer(y_t)  # Embed y_t to match x's shape [batch_size, in_channels]
        # x_embed = self.x_embed_layer(x)  # Embed x to match y_t's shape [batch_size, in_channels]

        # if self.total_forward_calls == 0:
        #     print(f"First forward pass x_embed.shape: {x_embed.shape}\ty_embed.shape: {y_t_embed.shape}")

        # if self.concat_embeddings:
        #     x_and_y_t = torch.cat([x_embed, y_t_embed], dim=-1)
        # else:
        #     x_and_y_t = x_embed + y_t_embed

        pred_noise = super().forward(x = y_t, cond = x, t = t, edge_index = edge_index, edge_weight = edge_weight, batch = batch,
                        return_attn_weights = return_attn_weights, debug_forward_pass = debug_forward_pass)
        
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
            assert pred_noise.shape == y_t.shape, f"Output shape must match y_t's shape. {pred_noise.shape}-{y_t.shape}"

        self.total_forward_calls += 1

        return pred_noise




########################################################### Res+GNN adapted for stock forecasting ######################################################################

""" We use the same class signature for GNN as the GraphUNet """
class StockForecastDiffusionGNN(nn.Module):
    def __init__(self, nfeatures=1, nblocks = 1, nunits=128,
                 nsteps = 100, # diffusion timesteps (useful for computing sin embeddings)
                 num_features_cond = 8,
                num_timesteps_cond = 25, 
                **kwargs):
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
        conv_model = kwargs.get('conv_model', 'TAGConv')
        conv_layer_normalize = kwargs.get('conv_layer_normalize', False)
        k_hops = kwargs.get('k_hops', 2)
        dropout_rate = kwargs.get('dropout_rate', 0.0)
        norm_kws = kwargs.get('norm_kws', dict())
        edge_features_nb = kwargs.get('edge_features_nb', 1)
        aggr_list = kwargs.get("aggr_list", None)
        conv_batch_norm = kwargs.get("conv_batch_norm", False)

        self.gnn = StockForecastGNN(in_channels=nfeatures, # nfeatures # nunits,
                                    hidden_channels=nunits, out_channels=nfeatures, # out_channels don't matter
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