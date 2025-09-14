import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, List


DEBUG_MODE = True  # Set to True to enable shape printing for debugging
INSPECT_MODE = False  # Set to True to enable detailed value inspection
INSPECT_KWS = {"graph_idx": 0,
                  "node_idx": [2, 3],  # List of node indices to inspect
                  "feature_idx": [0, 1],  # List of feature indices to inspect
                  "time_idx": [0, 1, 2, 3]  # List of time indices to inspect
                  }


def inspect_tensor(x: torch.Tensor, dim_names: List[str] = ["graph", "node", "feature", "time"]):
    if INSPECT_MODE:
        shape = x.shape
        indices = [INSPECT_KWS.get(f"{dim}_idx", list(range(size))) for dim, size in zip(dim_names, shape)]
        for i, (index, dim) in enumerate(zip(indices, dim_names)):
            if isinstance(index, list):
                temp_index = [i for i in index if i < shape[dim_names.index(dim)]]
                indices[i] = temp_index

        slices = tuple(slice(idx[0], idx[-1]+1) if isinstance(idx, list) else slice(0, size) for idx, size in zip(indices, shape))
        print(f"Inspecting tensor of shape {shape} at slices {slices}:")
        print(x[slices])


def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)




def get_nn_conv1d_parameters(in_channels, out_channels, kernel_size=3):
    """
    Utility function to determine the parameters for a 1D convolutional layer so that temporal input and output dimensions are preserved.
    """
    # Assert kernel size is valid.
    while in_channels <= kernel_size:
        kernel_size = kernel_size - 2
        if kernel_size < 1:
            raise ValueError("in_timesteps is too small to apply downsampling with conv1d. Try using a smaller kernel size or a different downsampling method.")

    temp_stride = in_channels // (out_channels) # (out_channels - 1) 
    # temp_in_channels = (out_channels + 1) * temp_stride
    temp_padding = (kernel_size - 1) // 2

    if in_channels == out_channels:
        print(f"Found parameters for [L_in, L_out] = {in_channels, out_channels}: ", {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding})
        return {"kernel_size": kernel_size, "stride": 1, "padding": (kernel_size - 1) // 2}


    temp_out_channels = calc_output_length(in_channels, kernel_size, temp_stride, temp_padding, dilation=1)

    if temp_out_channels == out_channels:
        print(f"Found parameters for [L_in, L_out] = {in_channels, out_channels}: ", {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding})
        return {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding}
    
    max_num_attempts = 10
    counter = 0
    while temp_out_channels != out_channels:
        counter += 1
        if counter == max_num_attempts:
            raise ValueError(f"Could not find suitable parameters for [L_in, L_out] = {in_channels, out_channels} after {max_num_attempts} attempts. Last tried parameters: ", {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding})
        else:
            if temp_out_channels > out_channels: 
                # Reduce the padding to decrease the output length
                temp_padding = max(0, temp_padding - 1)

            else:
                # Increase the padding to increase the output length
                temp_padding += 1

    print(f"Found parameters for [L_in, L_out] = {in_channels, out_channels}: ", {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding})
    return {"kernel_size": kernel_size, "stride": temp_stride, "padding": temp_padding}

    

def calc_output_length(L_in, kernel_size, stride, padding, dilation):
    """
    Calculate the output length of a 1D convolutional layer.
    L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
    """
    L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
    return L_out