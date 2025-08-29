import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LayerNorm
from torch_geometric.nn.conv import TransformerConv
import torch.utils.checkpoint as checkpoint

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

# Graph Transformer for graph-based conditioning
class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super(GraphTransformer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        activation = kwargs.get('activation', 'tanh') # F.relu
        self.resolve_activation(activation=activation)

        self.heads = kwargs.get('attn_num_heads', 4)
        dropout_rate = kwargs.get('dropout_rate', 0.0)
        attn_edge_dim = kwargs.get('attn_edge_dim', 1)
        norm_layer = kwargs.get('norm_layer', None)
        res_connection = kwargs.get('res_connection', 'skip')
        self.use_checkpointing = kwargs.get('use_checkpointing', False)
        
        # self.conv_embed = TransformerConv(in_channels=in_channels,
        #                                   out_channels=hidden_channels,
        #                                   heads = self.heads,
        #                                   concat=False,
        #                                   dropout=0.,
        #                                   edge_dim=attn_edge_dim
        #                                   )
        
        self.conv_embed = nn.ModuleList([TransformerConv(in_channels=in_channels,
                                          out_channels=hidden_channels,
                                          heads = self.heads,
                                          concat=False,
                                          dropout=0.,
                                          edge_dim=attn_edge_dim
                                          )
                                        #   ,
                                        #   TransformerConv(in_channels=hidden_channels,
                                        #   out_channels=hidden_channels,
                                        #   heads = self.heads,
                                        #   concat=False,
                                        #   dropout=0.,
                                        #   edge_dim=attn_edge_dim
                                        #   )
                                        ]
                                          )
        
        self.mlp = nn.Sequential(nn.Linear(hidden_channels, 2 * out_channels), self.activation)
        self.dropout = nn.Identity() # nn.Dropout(dropout_rate)

        # self.x_mlp_embed = nn.Sequential(nn.Linear(out_channels, 8 * out_channels), self.activation, nn.Linear(8 * out_channels, 8*out_channels), self.activation, nn.Linear(8 * out_channels, out_channels))
        # self.x_mlp_embed = nn.Identity()
        self.resolve_res_connection(res_connection=res_connection)

        # self.norm = BatchNorm(in_channels)
        self.resolve_norm(norm=norm_layer, in_channels=in_channels)


    def conditioning_block(self, x, condition):
        scale = condition[..., 0:1].view_as(x)
        shift = condition[..., 1:2].view_as(x)
        return x * scale + shift
    
    def resolve_activation(self, activation):
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky-relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError
    
    def resolve_norm(self, norm, in_channels):
        if norm == 'layer':
            self.norm = LayerNorm(in_channels)
        elif norm == 'batch':
            self.norm = BatchNorm(in_channels)
        elif norm is None or norm == '':
            self.norm = nn.Identity()
        else:
            raise ValueError
        
    def resolve_res_connection(self, res_connection):

        if res_connection == 'mlp':
            self.x_mlp_embed = nn.Sequential(nn.Linear(self.out_channels, 2 * self.out_channels), self.activation, nn.Linear(2 * self.out_channels, 2*self.out_channels), self.activation, nn.Linear(2 * self.out_channels, self.out_channels))
        elif res_connection == 'skip':
            self.x_mlp_embed = nn.Identity()
        else:
            raise ValueError
        
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward
    
    def custom_conv_embed(self, conv_embed):
        def custom_forward(*inputs):
            # print('inputs: ', inputs)
            outputs = conv_embed(inputs[0], edge_index=inputs[1], edge_attr=inputs[2], return_attention_weights = None)
            return outputs
        return custom_forward
        
    def forward(self, x, t, edge_index, edge_weight=None, batch=None):
     
        # x = self.norm(x)
        y = torch.cat([x, x * t], dim = -1)
        y = self.norm(y)

        if not self.use_checkpointing:
            # y = self.conv_embed(y, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1), return_attention_weights = None)
            for conv_embed in self.conv_embed:
                # y = self.activation(y)
                y = conv_embed(y, edge_index=edge_index, edge_attr=edge_weight, return_attention_weights = None)
                y = self.activation(y)
        else:
            # y = checkpoint.checkpoint(self.custom_conv_embed(), y, edge_index, edge_weight.unsqueeze(-1), use_reentrant=False)
            for conv_embed in self.conv_embed:
                # y = self.activation(y)
                y = checkpoint.checkpoint(self.custom_conv_embed(conv_embed), y, edge_index, edge_weight, use_reentrant=False)
                y = self.activation(y)
        y = self.mlp(y) 
        y = self.dropout(y)

        if not self.use_checkpointing:
            x = self.x_mlp_embed(x)
        else:
            x = checkpoint.checkpoint(self.custom(self.x_mlp_embed), x, use_reentrant=False)

        C_out = y.shape[-1] // 2
        condition = torch.stack((y[..., :C_out], y[..., C_out:]), dim = -1)
        out = self.conditioning_block(x=x, condition=condition)

        # out = y + x
       
        return out