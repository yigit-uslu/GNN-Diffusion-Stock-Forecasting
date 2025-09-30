import torch 
import torch.nn as nn
from typing import List, Tuple, Optional

from models.TemporalConvLayers import TemporalConvLayer
from models.unet_utils import get_nn_conv1d_parameters, resolve_conv_type


class Squeeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.squeeze(x, dim=self.dim)
        return x
    

class SwapAxes(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = torch.swapaxes(x, self.dim0, self.dim1)
        return x
    


class Chop(nn.Module):
    def __init__(self, chop_size: int, dim: int = -1, chop_from_end: bool = True):
        super().__init__()
        self.chop_size = chop_size
        self.dim = dim
        self.chop_from_end = chop_from_end

    def forward(self, x):
        if self.chop_from_end:
            x = torch.index_select(x, self.dim, torch.arange(x.size(self.dim) - self.chop_size).to(x.device))
        else:
            x = torch.index_select(x, self.dim, torch.arange(self.chop_size, x.size(self.dim)).to(x.device))
        return x
    

class SinusoidalTimeEmbedding(nn.Module):
    """
    https://nn.labml.ai/diffusion/ddpm/unet.html 
    """
    def __init__(self, n_channels: int, act: nn.Module = nn.SiLU(), T_max: float = 10000):
        super().__init__()
        self.n_channels = n_channels
        self.act = act
        self.T_max = T_max

        self.lin_embed = nn.Sequential(nn.Flatten(start_dim=-2),
                                       nn.Linear(self.n_channels // 4, self.n_channels),
                                       self.act,
                                       nn.LayerNorm(self.n_channels)
                                    #    nn.Linear(self.n_channels, self.n_channels)
                                       )
        
        
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = torch.log(torch.Tensor([self.T_max])) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = t.device) * -emb.to(t.device))
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim = -1)

        emb = self.lin_embed(emb)
        return emb
    

class ConditionalEmbeddingLayer(nn.Module):
    def __init__(self, in_channels: int, in_timesteps: int, out_channels: int, out_timesteps: int,
                 cond_embed_kws: dict = None):
        super(ConditionalEmbeddingLayer, self).__init__()


        # Override default kws if provided
        self.cond_embed_kws = self.default_cond_embed_kws.copy()
        if cond_embed_kws is not None:
            print("Updating cond_embed_kws with provided kws: {}".format(cond_embed_kws))
            self.cond_embed_kws.update(cond_embed_kws)

        # Define any layers or parameters needed for the conditional embedding here.
        if in_timesteps is None or out_timesteps is None:
            # or in_timesteps == out_timesteps:
            print(f"Projecting only feature dimension for conditional embedding to F_0 = {out_channels}.")
            # If temporal dimensions are the same, only project feature dimension
            self.layer = nn.Sequential(
                            SwapAxes(1, 2),
                            nn.Linear(in_channels, out_channels),
                            nn.LeakyReLU(),
                            nn.LayerNorm(out_channels),
                            SwapAxes(1, 2)
                        )
            
        else:
            # Project both feature and temporal dimensions
            print(f"Projecting both feature and temporal dimensions for conditional embedding to F_0 = {out_channels}, T_0 = {out_timesteps}.")
            conv_type = resolve_conv_type(self.cond_embed_kws["convolution_type"])


            if conv_type in ["mlp", "conv1d"]:
                # conv = self.get_temporal_conv_layer()
                # # self.bn1 = nn.BatchNorm1d(out_channels)
                # norm_layer = nn.LayerNorm([self.hidden_channels, self.T_out])
                # act = nn.ReLU()

                # self.temporal_conv_block = nn.Sequential(
                #     conv,
                #     act,
                #     norm_layer,
                # )
                kernel_size = self.cond_embed_kws["kernel_size"]
                dilation = 1
                conv_params = get_nn_conv1d_parameters(in_channels=in_timesteps, out_channels=out_timesteps, kernel_size=kernel_size)
                
                self.layer = nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_params["kernel_size"],
                            padding=conv_params["padding"], stride=conv_params["stride"], dilation=dilation),
                    nn.LeakyReLU(),
                    nn.LayerNorm([out_channels, out_timesteps]),
                )


            elif conv_type == "causal-conv1d":
                print("Using causal conv1d for cond-embeddding: temporal conv block.")
                kernel_size = self.cond_embed_kws["kernel_size"]
                dilation = 1
                stride = int(in_timesteps // out_timesteps)
                kernel_size = 2 * stride + 1
                padding = (kernel_size - 1) * dilation // 2 + 1 # For causal conv1d
                print("Cond-embed: Causal conv1d overwrites kernel_size from {} to {}".format(self.cond_embed_kws["kernel_size"], kernel_size))

                conv = nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size,
                                padding = padding, dilation = dilation, stride = stride)


                chop = Chop(dim = -1, chop_size = 1)  # Remove extra padding to maintain causality

                norm_layer = nn.LayerNorm([out_channels, out_timesteps])
                act = nn.LeakyReLU()

                # if self.hidden_channels != self.in_channels:
                #     res_connection = nn.Conv1d(self.in_channels, self.hidden_channels, kernel_size=1)
                # else:
                #     res_connection = nn.Identity()

                # self.temporal_conv_block = TemporalConvLayer(
                #     conv,
                #     chop,
                #     act,
                #     norm_layer,
                #     # res_connection
                # )
                self.layer = nn.Sequential(
                    conv,
                    chop,
                    act,
                    norm_layer,
                )


            elif conv_type == "gated-causal-conv1d":
                print("Using gated causal conv1d for cond-embeddding: temporal conv block.")
                kernel_size = self.cond_embed_kws["kernel_size"]
                dilation = 1
                stride = int(in_timesteps // out_timesteps)
                kernel_size = 2 * stride + 1
                padding = (kernel_size - 1) * dilation // 2 + 1 # For causal conv1d
                print("Cond-embed: Causal conv1d overwrites kernel_size from {} to {}".format(self.cond_embed_kws["kernel_size"], kernel_size))

                conv = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=kernel_size,
                                 padding=padding, dilation=dilation, stride=stride)

                chop = Chop(dim=-1, chop_size=1)  # Remove extra padding to maintain causality

                act = nn.GLU(dim=-2)  # Gated Linear Unit activation
                norm_layer = nn.LayerNorm([out_channels, out_timesteps])


                self.layer = nn.Sequential(
                    conv,
                    chop,
                    act,
                    norm_layer,
                )
        

            else:
                raise ValueError(f"Unsupported convolution type: {conv_type}")
    

            # kernel_size = 5
            # conv_params = get_nn_conv1d_parameters(in_channels=in_timesteps, out_channels=out_timesteps, kernel_size=kernel_size)
            
            # self.layer = nn.Sequential(
            #     nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_params["kernel_size"],
            #               padding=conv_params["padding"], stride=conv_params["stride"], dilation=1),
            #     nn.LeakyReLU(),
            #     nn.LayerNorm([out_channels, out_timesteps]),
            # )


    def forward(self, cond):
        # Implement the forward pass for the conditional embedding.
        return self.layer(cond)
    

    @property
    def default_cond_embed_kws(self):
        return {
            'kernel_size': 5,
            'convolution_type': 'conv1d'
        }


class UNetEmbeddingBlock(nn.Module):
    """
    Embedding block for the UNet model.
    This block embeds the input features, time steps, and target features.
    """
    def __init__(self, in_channels: int, in_timesteps: int,
                  hidden_channels: int, hidden_timesteps: int, T_diffusion: int):
        super(UNetEmbeddingBlock, self).__init__()
        # Initialize any layers or parameters needed for the embedding block here.
        # self.embed_x = nn.Sequential(nn.Linear(cond_in_channels, hidden_channels), nn.LeakyReLU(), nn.Linear(hidden_channels, hidden_channels))
        # Time projections of X (i.e., conditioning info) is handled by each DownBlock in UNet.

        self.concatenate_embeddings = True
        out_channels = hidden_channels // 2 if self.concatenate_embeddings else hidden_channels

        self.time_proj_before_concat = nn.Linear(hidden_channels, out_channels) if self.concatenate_embeddings else nn.Identity() # Project time embedding to match feature dimension if concatenating

        if hidden_timesteps is None:
            # If not specified, keep the same temporal dimension and project only feature dimension
            print("No hidden_timesteps specified, projecting only feature dimension for y_t embedding.")
            self.embed_y_t = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LeakyReLU(),
                nn.LayerNorm(out_channels),
                # nn.Linear(hidden_channels, out_channels),
                # nn.LeakyReLU(),
                # nn.LayerNorm(out_channels),
                SwapAxes(-1, -2),
                )
        else:
            print(f"Projecting both feature and temporal dimensions for y_t embedding to hidden_timesteps = {hidden_timesteps}.")
            self.embed_y_t = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LeakyReLU(),
                nn.LayerNorm(out_channels),
                SwapAxes(-1, -2),
                nn.Linear(in_timesteps, hidden_timesteps),
                nn.LeakyReLU(),
                nn.LayerNorm(hidden_timesteps),
                )
            
        self.embed_t = SinusoidalTimeEmbedding(hidden_channels, T_max=T_diffusion)


    def forward(self, x: torch.Tensor, t: torch.Tensor, y_t: torch.Tensor, edge_index: torch.Tensor = None, edge_weight: torch.Tensor = None,
                debug_print: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Embeds the input features x, time steps t, and target features y_t.
        Args:
            x (torch.Tensor): Input node features of shape [batch_size, num_features, past_window].
            t (torch.Tensor): Time steps of shape [batch_size].
            y_t (torch.Tensor): Target node features of shape [batch_size, future_window].
            edge_index (torch.Tensor, optional): Edge indices for graph structure. Default is None.
            edge_weight (torch.Tensor, optional): Edge weights for graph structure. Default is None.
            debug_print (bool, optional): If True, prints debug information. Default is False.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                - Embedded x of shape [batch_size, hidden_channels, past_window].
                - Embedded t of shape [batch_size, hidden_channels, 1].
                - Embedded y_t of shape [batch_size, hidden_channels, future_window].
                - edge_index (torch.Tensor, optional): Unchanged edge indices.
                - edge_weight (torch.Tensor, optional): Unchanged edge weights.
        """
        # batch_size, num_features, past_window = x.shape
        # batch_size, future_window = y_t.shape

        # Validate x for NaNs or Infs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaNs or Infs after embedding and combining with y_t and t")

        # print(f"Input x shape before embedding: {x.shape}") if debug_print else None
        # x_emb = self.embed_x(x.permute(0, 2, 1)).permute(0, 2, 1) # [B, Tp, F] -> [B, Tp, H] -> [B, H, Tp]
        x_emb = nn.Identity()(x) # No embedding for x, just pass through
        y_t_emb = self.embed_y_t(y_t.unsqueeze(-1)) # [B, Th] -> [B, H or H // 2, T_unet_in]

        # print(f"Input x shape after embedding: {x.shape}") if debug_print else None
        print(f"Input y_t shape before vs after embedding: {y_t.shape} vs {y_t_emb.shape}") if debug_print else None
        t_emb = self.embed_t(t) # .unsqueeze(-1) # [B, H] -> [B, H]

        print(f"Input t shape before vs after embedding: {t.shape} vs {t_emb.shape}") if debug_print else None

        if self.concatenate_embeddings:
            y_t_emb = torch.cat([y_t_emb,
                self.time_proj_before_concat(t_emb).unsqueeze(-1).expand(-1, -1, y_t_emb.shape[-1])], dim=1) # [B, H // 2, T_unet_in] + [B, H // 2, T_unet_in] -> [B, H, T_unet_in]
            print("Input y_t after concatenation with time information: ", y_t_emb.shape) if debug_print else None

        if edge_index is not None and edge_weight is not None:
            print(f"Edge index shape: {edge_index.shape}") if debug_print else None
            print(f"Edge weight shape: {edge_weight.shape}") if debug_print else None

        return x_emb, t_emb, y_t_emb, edge_index, edge_weight
    