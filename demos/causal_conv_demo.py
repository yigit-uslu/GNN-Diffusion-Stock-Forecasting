import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(CausalConv1d, self).__init__()
        # Define a simple causal Conv1d layer
        self.dilation = 1
        self.padding = (kernel_size - 1) * self.dilation // 2 + 1  # For causal conv1d
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              dilation=self.dilation,
                              padding=0,
                              stride=stride)  # Padding to ensure causality

    def forward(self, x):

        x_padded = F.pad(x, (self.padding, 0))  # Apply padding for causality
        print("x.shape afeter pad: ", x_padded.shape)
        out = self.conv(x_padded)
        print("out.shape after conv: ", out.shape)

        return out[..., :x.size(-1)]  # Trim the output to match input length for causality


if __name__ == "__main__":

    with torch.no_grad():

        N = 20
        stride = 4
        kernel_size = 2 * stride + 1
        dilation = 1
        padding = ((kernel_size - 1) * dilation) // 2 + 1 # For causal conv1d

        input = torch.arange(1, N+1).view(1, 1, N).float()

        conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding, device="cpu")
        conv.weight.data.fill_(1.0 / kernel_size) # Averaging filter
        conv.bias.data.fill_(0.0)

        output = conv(input)
        print("Input:", input)
        print("Conv1d output:", output)


        conv = CausalConv1d(kernel_size=kernel_size, stride=stride)
        conv.conv.weight.data.fill_(1.0 / kernel_size) # Averaging filter
        conv.conv.bias.data.fill_(0.0)

        output = conv(input)
        print("Input:", input)
        print("CausalConv1d output:", output)
