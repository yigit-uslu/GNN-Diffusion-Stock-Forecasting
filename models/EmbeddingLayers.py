import torch.nn as nn
import torch


from core.config import SINUSOIDAL_TIME_EMBED_MAX_T


class SinusoidalTimeEmbedding(nn.Module):
    """
    https://nn.labml.ai/diffusion/ddpm/unet.html 
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        # self.act = Swish()
        self.act = nn.LeakyReLU()

        self.lin_embed = nn.Sequential(nn.Flatten(start_dim=-2),
                                       nn.Linear(self.n_channels // 4, self.n_channels),
                                       self.act,
                                       nn.Linear(self.n_channels, self.n_channels)
                                       )
        
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = torch.log(torch.Tensor([SINUSOIDAL_TIME_EMBED_MAX_T])) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = t.device) * -emb.to(t.device))
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim = -1)

        emb = self.lin_embed(emb)
        return emb