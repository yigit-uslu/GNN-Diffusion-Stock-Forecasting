import torch.nn as nn
from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm, InstanceNorm, DiffGroupNorm



class NormLayer(nn.Module):
    def __init__(self, norm, in_channels, **kwargs):
        super().__init__()
        self.norm = norm
        self.in_channels = in_channels
        self.n_groups = kwargs.get('n_groups', 8)
        self.resolve_norm_layer(norm = self.norm, in_channels=in_channels, **kwargs)


    def resolve_norm_layer(self, norm, in_channels, **kwargs):
        n_groups = kwargs.get('n_groups', 8)
        layer_norm_mode = kwargs.get('layer_norm_mode', 'node')

        assert layer_norm_mode in ['node', 'graph'], f"Layer norm mode should be either 'node' or 'graph'. Got {layer_norm_mode}."

        if norm == 'batch':
            self.norm_layer = BatchNorm(in_channels)
        elif norm == 'layer':
            self.norm_layer = LayerNorm(in_channels, mode=layer_norm_mode)
        elif norm == 'group':
            self.norm_layer = DiffGroupNorm(in_channels=in_channels, groups=n_groups)
        elif norm == 'graph':
            self.norm_layer = GraphNorm(in_channels=in_channels)
        elif norm == 'instance':
            self.norm_layer = InstanceNorm(in_channels=in_channels)
        elif norm == 'none' or norm is None:
            self.norm_layer = nn.Identity()

        print(f"Norm layer initialized with {self.norm_layer}")


    def forward(self, x, batch = None, batch_size = None):
        if self.norm in ['batch', 'group', 'none', None]:
            return self.norm_layer(x)
        elif self.norm in ['graph', 'layer', 'instance']:
            # print('x.shape: ', x.shape)
            # print('batch.shape: ', batch.shape)
            return self.norm_layer(x, batch = batch, batch_size = None)
        

        else:
            raise NotImplementedError