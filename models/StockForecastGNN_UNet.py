import torch.nn as nn
import torch
from typing import Union, List
from models.unet_embedding_utils import UNetEmbeddingBlock, SwapAxes, Squeeze, ConditionalEmbeddingLayer
from models.unet import UNet


class GUNet(nn.Module):
    def __init__(self, depth: int = 3, sampling_factor: Union[int, List[int]] = 2,
                 num_nodes: int = 100,
                 T_diffusion: int = 500,
                 cond_in_channels: int = 8,  # Features per node
                 cond_in_timesteps: int = 20, # past window
                 hidden_channels: int = 64, # initial hidden feature dimension in the UNet
                 hidden_timesteps: int = None, # initial hidden temporal dimension in the UNet
                 in_channels: int = 1, # input node feature dimension
                 out_channels: int = 5): # Same as future window
        super(GUNet, self).__init__()

        self.out_channels = out_channels

        if isinstance(sampling_factor, int):
            sampling_factor = [sampling_factor] * depth
        assert len(sampling_factor) == depth, "Length of sampling_factor list must match depth."

        # T_in = future_window = out_channels
        past_window = cond_in_timesteps
        in_timesteps = out_channels

        self.T_diffusion = T_diffusion # diffusion steps
        self.depth = depth
        self.sampling_factor = sampling_factor if isinstance(sampling_factor, list) else [sampling_factor] * depth


        # Embedding only for y_t and t, x (the conditional info) is handled by each DownBlock in UNet.
        self.global_embedding_layer = UNetEmbeddingBlock(in_channels = in_channels,
                                                         in_timesteps = in_timesteps,
                                                         hidden_channels = hidden_channels,
                                                         hidden_timesteps = hidden_timesteps,
                                                         T_diffusion = T_diffusion,
                                                         )
        
        if hidden_timesteps is None:
            # UNet's temporal dimension is the same as input's if not specified
            hidden_timesteps = T_min = in_timesteps
            
        else:
            T_min = 4 # Minimum temporal dimension after downsampling in UNet

        self.net = UNet(F_in = hidden_channels, N_in = num_nodes, T_in = hidden_timesteps, T_min=T_min,
                        downsample_factor=sampling_factor,
                        cond_kws = {"in_channels": cond_in_channels, "in_timesteps": cond_in_timesteps, "hidden_channels": hidden_channels, "hidden_timesteps": hidden_timesteps},
                        # cond_F_in=cond_in_channels, cond_T_in=cond_in_timesteps,
                        )
        
        self.projection_head = nn.Sequential(
            SwapAxes(1, 2),
            nn.Linear(hidden_channels, 1),
            Squeeze(-1)
        )

        self.total_forward_calls = 0



    def forward(self, y_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor,
                return_attn_weights = False, debug_forward_pass = False):

        x, t, y_t, edge_index, edge_weight = self.global_embedding_layer(x, t, y_t, edge_index, edge_weight,
                                                                         debug_print = debug_forward_pass or self.total_forward_calls == 0)

        pred_noise = self.net(x = y_t, t = t,
                              cond = x, #None,
                              edge_index = edge_index, edge_weight = edge_weight, batch = batch)

        pred_noise = self.projection_head(pred_noise)

        # raise ValueError("GUNet forward pass is not fully implemented yet.")
    
        # pred_noise = None # self.out_layers(x) # [B, H, T] -> [B, T, H] -> [B, T, 1] -> [B, T] -> [B, Th]

        print(f"pred_noise shape: {pred_noise.shape}") if debug_forward_pass or self.total_forward_calls == 0 else None
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
        





class StockForecastDiffusionGUNet(nn.Module):
    def __init__(self, depth: int = 3, sampling_factor: Union[int, List[int]] = 2):
        super(StockForecastDiffusionGUNet, self).__init__()
        self.gunet = GUNet(depth = depth, sampling_factor = sampling_factor, hidden_channels = 64)


    def forward(self, y_t: torch.Tensor, t: torch.Tensor, data, return_attention_weights = False, debug_forward_pass = False):

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
        # Denoise noisy node embeddings
        out = self.gunet(y_t = y_t, t = t, x = x, edge_index = edge_index, edge_weight = edge_weight, batch = batch,
                        return_attn_weights = return_attention_weights, debug_forward_pass = debug_forward_pass)

        return out
    

    @property
    def prototype_name(self):
        return f"{self.__class__.__name__}_{self.gunet.depth}_layers"
