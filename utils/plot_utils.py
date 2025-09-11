from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from torch_geometric.data import Data


def plot_stock_correlation_distribution(fundamentals_corr, save_path):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(fundamentals_corr.values.flatten(), bins=50, color='skyblue', edgecolor='black')

        ax.set_title('Fundamentals Correlation Distribution')
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Frequency')

    plt.savefig(save_path, dpi = 300, bbox_inches='tight')
    plt.close(fig)



def plot_batched_data(data: Data, 
                      stocks_idx: Union[np.ndarray, list, None], 
                      plot_target: str = "close_price",
                      save_dir: str = None,
                      save_ext: str = ".pdf") -> None:
    """
    Test batched Data and plot time evolution of closing prices for 4 example stocks.
    """

    print("Data.x.shape: ", data.x.shape) # [num_stocks x num_timestamps, num_features, past_window]
    print("Data.edge_index.shape: ", data.edge_index.shape)
    print("Data.edge_weight.shape: ", data.edge_weight.shape)
    print("Data.y.shape: ", data.y.shape)
    print("Data.close_price.shape: ", data.close_price.shape) # [num_stocks x num_timestamps, past_window]
    print("Data.close_price_y.shape: ", data.close_price_y.shape) # [num_stocks x num_timestamps, 1]

    num_timestamps = len(data.ptr) - 1
    num_stocks = data.x.shape[0] // num_timestamps
    
    num_features, past_window = data.x.shape[1], data.x.shape[2]  # num_features, past_window

    print(f"Number of timestamps in the batch: {num_timestamps}")
    print(f"Number of stocks in the batch: {num_stocks}")
    print(f"Number of features in x: {num_features}")
    print(f"Past window size in x: {past_window}")

    if stocks_idx is None:
        stocks_idx = np.random.choice(num_stocks, 4, replace = False)

    if plot_target == "Closing_Price":
        target = data.close_price.reshape(num_timestamps, num_stocks, -1)

    elif plot_target == "Closing_Price_y":
        target = data.close_price_y.reshape(num_timestamps, -1)

    elif plot_target == "y":
        target = data.y.reshape(num_timestamps, num_stocks, -1)

    elif isinstance(plot_target, tuple) and plot_target[0] == "y":
        target = data.y.reshape(num_timestamps, num_stocks, -1)
        plot_target = f"{plot_target[1]} (y)"

    elif isinstance(plot_target, tuple) and plot_target[0].startswith('x[') and plot_target[0].endswith(']'):
        try:
            feature_idx = int(plot_target[0][2:-1])  # Extract the index from 'x[idx]'
            if 0 <= feature_idx < num_features:
                target = data.x[:, feature_idx, :].reshape(num_timestamps, num_stocks, -1)
                plot_target = f"{plot_target[1]} (Feature {feature_idx})"
            else:
                raise ValueError(f"Feature index {feature_idx} is out of range [0, {num_features-1}]")
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"Invalid feature index format in '{plot_target}'. Expected format: 'x[idx]' where idx is an integer.")
            else:
                raise e

    else:
        raise ValueError(f"Invalid plot_target '{plot_target}'. Must be one of: 'Closing_Price', 'Closing_Price_y', 'y', or 'x[idx]' where idx is a feature index.")

    print("Target.shape: ", target.shape)


    if save_dir is not None:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))

            for idx, stock_idx in enumerate(stocks_idx):
                ax = axs[idx // 2, idx % 2]
                ax.plot(target[:, stock_idx].detach().numpy(), label=None) 
                ax.set_title(f"Stock {stock_idx}")
                ax.set_xlabel("Timestamp (Date)")
                ax.set_ylabel(plot_target)
                # ax.legend()

            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/batched_data_{plot_target}" + save_ext, dpi=300, bbox_inches='tight')
            plt.close(fig)




def plot_regression(preds, target, observations, timestamps = None, fig = None, axs = None, **kwargs):
    """
    Plot regression for the specified stock over a given time period.
    Input:
    - observations: np.ndarray of shape [batch_size, past_window]
    - timestamps: np.ndarray of shape [batch_size, past_window + future_window] (optional)
    - preds: np.ndarray of shape [batch_size, future_window, nsamples]
    - target: np.ndarray of shape [batch_size, future_window]
    """

    metric = kwargs.get("metric", "Log Return")
    stock_idx = kwargs.get("stocks_idx", 0)

    B, future_window, nsamples = preds.shape
    past_window = observations.shape[1]

    assert observations.shape[0] == B, "Mismatch in batch size between observations and preds."
    assert target.shape[0] == B, "Mismatch in batch size between target and preds."
    assert target.shape[1] == future_window, "Mismatch in future window size between target and preds."

    if timestamps is None:
        timestamps = np.arange(past_window + future_window).repeat(B, axis=0)  # [B, past_window + future_window]


    with sns.axes_style("darkgrid"):
        if fig is None or axs is None:
            fig, axs = plt.subplots(nrows = int(np.ceil(B/ 2)), ncols = 2, figsize = (12, 6), squeeze = False)

        assert B <= len(axs.flatten()), "Number of timestamps to plot exceeds number of subplots available."

        for idx in range(B):
            ax = axs[idx // 2, idx % 2]

            target_cts = np.concatenate([observations[idx, -1:].reshape(1), target[idx].reshape(-1)])  # [future_window + 1]
            preds_cts = np.concatenate([np.repeat(observations[idx, -1:].reshape(1, 1), repeats=nsamples, axis=1), preds[idx]], axis=0)  # [future_window + 1, nsamples]

            ax.plot(timestamps[idx, :past_window], observations[idx], label="Observations", color='black')
            ax.plot(timestamps[idx, -future_window-1:], preds_cts, linestyle = '--', alpha = 0.7, label="Predictions" if nsamples == 1 else None)
            ax.plot(timestamps[idx, -future_window-1:], target_cts, marker = 'd', markersize = 4, markevery = 10, label="Target")

            # # Connect last plotted point of observations to first plotted point of predictions
            # ax.plot(timestamps[idx, past_window - 1:past_window + 1], 
            #         [observations[idx, -1].detach().cpu().numpy(), preds[idx, 0].detach().cpu().numpy()],
            #         linestyle = '-', color='black', alpha = 0.7)
        
            ax.set_title(f"{metric} of Stock {stock_idx}")
            ax.set_xlabel("Timestamp (Date)")
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend()

        # fig.tight_layout()

    return fig, axs





def plot_regression_errors_deprecated(preds, target, metric, stocks_idx = None, fig = None, axs = None,
                           segment_timestamps = True,
                           confidence_band_kws = None):
    """
    Plot regression errors for the specified stocks.
    Input:
    - preds: Tensor of shape [num_timestamps, num_stocks, future_window, nsamples]
    - target: Tensor of shape [num_timestamps, num_stocks, future_window, 1]
    - metric: str 
    """
    print("Plotting regression errors...")
    print(f"preds.shape: {preds.shape}\tTarget.shape: {target.shape}.")

    if confidence_band_kws is not None:
        confidence_interval = confidence_band_kws.get("confidence_interval", 0.95)

    if stocks_idx is None:
        stocks_idx = np.random.choice(preds.shape[1], 4, replace = False) # 4 random stocks

    with sns.axes_style("darkgrid"):
        if fig is None or axs is None:
            fig, axs = plt.subplots(nrows = int(np.ceil(len(stocks_idx) / 2)), ncols = 2, figsize = (15, 10))

        assert len(stocks_idx) <= len(axs.flatten()), "Number of stocks to plot exceeds number of subplots available."

        for idx, stock_idx in enumerate(stocks_idx):
            ax = axs[idx // 2, idx % 2]

            if preds.shape[-2] == 1:
                num_segments = 1
            elif segment_timestamps:
                num_segments = preds.shape[0] // preds.shape[-2]
            else:
                num_segments = preds.shape[0]
            # num_segments = 1 if preds.shape[-2] == 1 else preds.shape[0] // preds.shape[-2]

            if num_segments > 1:
                segment_size = preds.shape[-2]
                for ii in range(num_segments):
                    time_start = ii * segment_size
                    time_end = (ii + 1) * segment_size # if ii < num_segments - 1 else preds.shape[0]

                    preds_start_idx = time_start if segment_timestamps else ii

                    ax.plot(np.arange(time_start, time_end), preds[preds_start_idx, stock_idx].detach().cpu().numpy(),
                                linestyle = '--', alpha = 0.7, label = "Predictions" if preds.shape[-1] == 1 and ii == 0 else None)

                    ax.axvline(time_end, linestyle=':', color='black', alpha = 0.5, label=r'$t = T_p + T_h$' if ii == 0 else None)

                # target_timestamps = np.arange(0, target.shape[0], segment_size)
                ax.plot(target[:, stock_idx, :, 0].flatten().detach().cpu().numpy(), marker = 'd', markersize = 4, markevery = 10, label="Target")


            else:
                ax.plot(preds[:, stock_idx, 0].detach().cpu().numpy(), linestyle = '--', alpha = 0.7, label="Predictions" if preds.shape[-1] == 1 else None)
                # target_timestamps = np.arange(target.shape[0])
                ax.plot(target[:, stock_idx, 0].detach().cpu().numpy(), marker = 'd', markersize = 4, markevery = 10, label="Target")

            ax.set_title(f"{metric} of Stock {stock_idx}" + r" $(T_h = {})$".format(preds.shape[-2]))
            ax.set_xlabel("Timestamp (Date)")
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend()
 
    return fig, axs



def create_repeating_colormap(n_points, n_distinct_colors=10):
    """
    Create a colormap with distinct colors that repeat after n_distinct_colors.
    
    Parameters:
    -----------
    n_points : int
        Total number of points to generate colors for
    n_distinct_colors : int
        Maximum number of distinct colors before repeating
        
    Returns:
    --------
    colors : array
        Array of RGBA colors
    """
    # Generate distinct colors using a colormap
    cmap = plt.get_cmap('tab10' if n_distinct_colors <= 10 else 
                         'tab20' if n_distinct_colors <= 20 else 
                         'viridis')
    
    # Create distinct colors
    distinct_colors = [cmap(i/n_distinct_colors) for i in range(n_distinct_colors)]
    
    # Repeat colors as needed
    colors = [distinct_colors[i % n_distinct_colors] for i in range(n_points)]
    
    return np.array(colors)