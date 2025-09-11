from collections import OrderedDict
import copy
import os
from typing import List, Tuple
import torch
from tqdm import tqdm
import wandb
import abc

from core.GRW import GeometricRandomWalk as GRW
from datasets.utils import daily_log_returns_to_closing_prices, daily_log_returns_to_log_returns, gather_target_preds_observations_and_timestamps, run_correlated_geometric_random_walk_baseline, run_geometric_random_walk_baseline
from models.eval import get_probabilistic_errors, get_regression_errors
from models.utils import log_plot_regression, log_regression_errors
from utils.diffusion_model_utils import save_cd_train_chkpt


import copy
import io
import os
import time
from typing import Union
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns
import numpy as np
import torch 
from tqdm import tqdm
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
import wandb 
import random
from collections import defaultdict
from torch.utils.data import TensorDataset, Subset
from utils.data_utils import LambdasDataset, WirelessDataset, batch_from_data_list, create_data_list_from_batch
from utils.diffusion_model_utils import save_cd_train_chkpt
from utils.distance_utils import wasserstein_per_dimension
from utils.ema import EMA
from utils.general_utils import debug_print
from core.config import MAX_LOGGED_CLIENTS, MAX_LOGGED_NETWORKS

from core.Diffusion import ConditionalDiffusionLearner


def make_validation_criteria_fnc(min_epoch, validate_freq, max_loss):

    def thunk(epoch, loss):
        validate = False
        if epoch >= min_epoch and loss <= max_loss and (epoch + 1) % validate_freq == 0:
            validate = True
            print("Evaluation criterion met. Validating the model...")

        return validate
    return thunk



# Make evaluation function.
def make_eval_function(eval_keys):

    def thunk(variables):
        metrics = {}
        for key in [key for key in variables if key in eval_keys]:
            # metrics[key].append(variables[key])
            metrics[key] = variables[key]
            

        return metrics
    return thunk


class StockPriceForecastDiffusionLearner(ConditionalDiffusionLearner):
    """
    A derived class from Conditional Diffusion Learner
    """ 

    def __init__(self, config, device='cpu', **kwargs):
        super().__init__(config, device)

        self.global_chkpt_step = 0
        self.current_train_loss = np.Inf


    def init_optimizer_and_lr_scheduler(self, model, optimizer = None, lr_sched = None):

        if optimizer is None:
            lr = self.config.lr if hasattr(self.config, "lr") and self.config.lr is not None else 1e-2 # 1e-4
            weight_decay = self.config.weight_decay if hasattr(self.config, "weight_decay") and self.config.weight_decay is not None else 1e-6
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if lr_sched is None:
            # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma = 0.9)
            T_0=10 # 10
            T_mult=2
            gamma=self.config.lr_sched_gamma if hasattr(self.config, "lr_sched_gamma") and self.config.lr_sched_gamma is not None else 0.9
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max = n_epochs // 10, eta_min=lr * 1e-2)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0, T_mult=T_mult, eta_min=lr * 1e-3)
            warmup_period = T_0

            # Define LambdaLR to apply decay to restart values
            def decay_lambda(epoch):
                if epoch < warmup_period:
                    return (epoch + 1) / warmup_period
                else:
                    epoch -= warmup_period
                    restart_count = 0
                    T_cur = T_0  # Current restart period
                    while epoch >= T_cur:
                        epoch -= T_cur
                        T_cur *= T_mult
                        restart_count += 1
                    return gamma ** restart_count  # Apply decay factor per restart

            # Wrap the scheduler
            decay_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lambda)
            lr_sched = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[cosine_scheduler, decay_scheduler], milestones=[0])


        return optimizer, lr_sched



    def make_validation_criteria_fnc(self):
        # return make_validation_criteria_fnc(min_epoch=20, validate_freq=20, max_loss = 0.90)
        return make_validation_criteria_fnc(min_epoch=100, validate_freq=200, max_loss = 0.50)
    

    def model_trained_criterion(self):
        # Check if the model has been trained for at least 10 epochs and last recorded loss is low enough.
        if self.global_chkpt_step >= 10 and self.current_train_loss < 0.2: # 10, 0.3
            return True
        else:
            return False


    def validate_model(self, accelerator, model, val_dataloader, device = "cpu"):

        val_metrics = defaultdict(list)
        model.eval()

        n_val_iters = 10 
        with torch.inference_mode(True):
            for iter in range(n_val_iters):
                for data in val_dataloader:
                    data = data.to(device)
                    # Generate random diffusion timesteps
                    timesteps = self.generate_random_timesteps(num_steps = self.diffusion_steps,
                                                                size = (data.num_graphs, 1), 
                                                                device = device,
                                                                weights = None
                                                                )
                            # accelerator.print("Timesteps.shape: ", timesteps.shape             
                    timesteps = timesteps.repeat_interleave(repeats = data.num_nodes // data.num_graphs, dim = 0)
                    assert timesteps.shape[0] == data.y.shape[0], f"Shape mismatch: {timesteps.shape[0]} != {data.y.shape[0]}"

                    # Add random noise to clean samples
                    y_0 = data.y.clone()
                    noisy_y_t, eps = self.noise(y_0, timesteps)

                    # Predict noise
                    pred_noise = model(noisy_y_t, timesteps, data)

                    # Compute the diffusion loss as the validation metric.
                    loss = self.loss_fn(pred_noise, eps).mean()
                    val_metrics["loss"].append(loss)


        val_metrics["loss"] = sum(val_metrics["loss"]) / len(val_metrics["loss"]) if val_metrics["loss"] else 0.0

        return val_metrics


    def batch_from_data_list(self, data_list, exclude_keys, follow_batch = ["y"]):
        data_batch = Batch.from_data_list(data_list = data_list,
                                          follow_batch=follow_batch,
                                          exclude_keys=exclude_keys
                                          )

        if data_batch.num_graphs == len(data_list):
            print("data_batch.num_graphs: ", data_batch.num_graphs)
            # Store the actual number of graphs in a custom attribute
            actual_num_graphs = len(data_list) * data_list[0].num_graphs
            data_batch._actual_num_graphs = actual_num_graphs
            print("data_batch._actual_num_graphs: ", data_batch._actual_num_graphs)

        for key in data_batch.keys():
            if key in ["Features", "Target"]:
                if data_batch[key] is not None and isinstance(data_batch[key][0], list):
                    # Concatenate lists of strings
                    data_batch[key] = sum(data_batch[key], []) 

        return data_batch


    def accelerated_sample_ddpm(self, model, data, nsamples = 100, device = "cpu"):

        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)""" 

        data_batch = self.batch_from_data_list(data_list = [data.clone() for _ in range(nsamples)],
                                               exclude_keys = None)
        
        print(f"Data: ", data, "\tData_batch: ", data_batch)
        # Use the actual number of graphs if available
        num_graphs = getattr(data_batch, '_actual_num_graphs', data_batch.num_graphs)
        print(f"Using num_graphs: {num_graphs}")

        try:
            y_batch_debug_idx = torch.arange(0, len(data_batch.y_batch), step = (len(data.ptr) - 1) // 2).to(data_batch.y_batch.device)
            print("Data_batch.y_batch[debug_idx]: ", torch.stack((y_batch_debug_idx, data_batch.y_batch[y_batch_debug_idx]), dim = 1))
        except Exception as e:
            print("Error occurred while stacking tensors: ", e)

        data_batch = data_batch.to(device)

        # data = data.to(device)
        model.eval()
        with torch.inference_mode(True):
            # if isinstance(data.y, torch.Tensor):
            sample_size = (data_batch.y.shape[0], *data_batch.y.shape[1:])
            # else:
            #     sample_size = (nsamples, *data.y[0].shape)

            y = self.sample_prior(sample_size=sample_size).to(device) # [n_samples, n_features]

            print(f"y_T.shape: {y.shape}.")

            # xt = [x.cpu()]
            yt = [y.clone()]
            scoret = []

            for t in tqdm(range(self.diffusion_steps-1, 0, -1)):

                predicted_noise = model(y, torch.full(size = (y.shape[0], 1), fill_value=t, device=y.device), data_batch)

                if t == self.diffusion_steps - 1:
                    print("Predicted_noise.shape: ", predicted_noise.shape)

                # See DDPM paper between equations 11 and 12
                y = 1 / (self.alphas[t] ** 0.5) * (y - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * predicted_noise)
                if t > 1:
                    # See DDPM paper section 3.2.
                    # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                    variance = self.betas[t]
                    std = variance ** (0.5)
                    y += std * self.sample_prior(sample_size=y.shape).to(device)
                # xt += [x.cpu()]
                yt += [y.clone()]

                score, _ = self.score_fnc_to_noise_pred(timestep=t, score = None, noise_pred=predicted_noise)
                # scoret += [score.cpu()]
                scoret += [score.clone()]


        return y, {'x_t': yt,
                    'score_t': scoret,
                    'data_batch': data_batch.cpu() # move data_batch to cpu
                    }
    


    def accelerated_sample(self, model, data, sampler = "ddpm", nsamples = 100, device = "cpu", *kwargs):

        if sampler == 'ddpm':
            return self.accelerated_sample_ddpm(model, data, nsamples = nsamples, device = device)
        elif sampler == 'ddpm_x0':
            return self.accelerated_sample_ddpm_x0(model, data, nsamples = nsamples, device = device)
        elif sampler == 'ddim':
            return self.accelerated_sample_ddim(model, data, nsamples = nsamples, device = device, eta=kwargs.get('eta', 0.2))
        else:
            raise ValueError(f"Sampler {sampler} is not implemented.")
    


    def eval(self, accelerator, model: torch.nn.Module, data: Data, nsamples = 100, device = "cpu", sampler = "ddpm") -> defaultdict:
        
        eval_metrics = {}
        
        model = model.to(device)
        model.eval()
        data = data.to(device)

        # num_timestamps = len(data.ptr) - 1
        # num_stocks = data.x.shape[0] // num_timestamps
        # num_features, past_window = data.x.shape[1], data.x.shape[2]  # num_features, past_window
        # target = data.y.reshape(len(data.ptr) - 1, num_stocks, -1) # num_timestamps, num_stocks, future_window

        ygen, ygen_hist = self.accelerated_sample(model = model, data=data, sampler=sampler, nsamples=nsamples, device=device)

        timestamps_gen = ygen_hist["data_batch"].timestamp.to(device)
        y_batch_gen = ygen_hist["data_batch"].y_batch.to(device)
        batch_gen = ygen_hist["data_batch"].batch.to(device)
        stocks_index_gen = ygen_hist["data_batch"].stocks_index.to(device)


        preds, target, observations, obs_and_pred_timestamps = gather_target_preds_observations_and_timestamps(data = data,
                                                                                                               ygen = ygen,
                                                                                                               timestamps = timestamps_gen,
                                                                                                               stocks_index = stocks_index_gen,
                                                                                                               debug_assert = self.global_chkpt_step < 2,
                                                                                                               debug_print = self.global_chkpt_step < 2
                                                                                                               )
        num_timestamps, num_stocks, future_window, nsamples = preds.shape
        past_window = observations.shape[2]

        target_metric = data.info["Target"] if isinstance(data.info["Target"], str) else data.info["Target"][0]


        # Get daily log returns and convert to log returns
        pred_log_returns = daily_log_returns_to_log_returns(preds, time_dim = 2)
        target_log_returns = daily_log_returns_to_log_returns(target, time_dim = 2)
        observations_log_returns = daily_log_returns_to_log_returns(observations, time_dim = 2)

        # print("Pred log returns: ", pred_log_returns)
        # print("Target log returns: ", target_log_returns)
        # print("Observations log returns: ", observations_log_returns)

        target_log_returns = target_log_returns + observations_log_returns[:, :, -1:].unsqueeze(-1).expand_as(target_log_returns)
        pred_log_returns = pred_log_returns + observations_log_returns[:, :, -1:].unsqueeze(-1).expand_as(pred_log_returns)


        ################ Baseline model: Uncorrelated Geometric Random Walk #################
        # Fit a Geometric Brownian Motion to the past observations and plot its predictions
        grw_pred_log_returns = run_geometric_random_walk_baseline(observations = observations,
                                                                  obs_and_pred_timestamps = obs_and_pred_timestamps,
                                                                  nsamples = nsamples
                                                                  )
        grw_pred = grw_pred_log_returns - torch.cat([observations_log_returns[:, :, -1:].unsqueeze(3).repeat(1, 1, 1, grw_pred_log_returns.shape[3]), grw_pred_log_returns[:, :, :-1, :]], dim = 2)

        # grw_pred_log_returns = grw_pred_log_returns + observations_log_returns[:, :, -1:].unsqueeze(-1).expand_as(grw_pred_log_returns)
        
        ################ Baseline model: Uncorrelated Geometric Random Walk #################





        ################ Baseline model: Correlated Geometric Random Walk #################
        # Fit a Correlated Geometric Brownian Motion to the past observations and plot its predictions
        # cgrw_pred_log_returns = run_correlated_geometric_random_walk_baseline(observations = observations,
        #                                                                      obs_and_pred_timestamps = obs_and_pred_timestamps,
        #                                                                      nsamples = nsamples
        #                                                                      )
        
        # cgrw_pred = cgrw_pred_log_returns - torch.cat([torch.zeros_like(cgrw_pred_log_returns[:, :, 0:1, :]), cgrw_pred_log_returns[:, :, :-1, :]], dim = 2)

        # cgrw_pred_log_returns = cgrw_pred_log_returns + observations_log_returns[:, :, -1:].unsqueeze(-1).expand_as(cgrw_pred_log_returns)
        ################ Baseline model: Correlated Geometric Random Walk #################






        # preds = ygen.reshape(-1, num_stocks, *target.shape[2:]) # [num_timestamps * nsamples, num_stocks, future_window]    
        
        # preds_stocks_index = stocks_index_gen.reshape(-1, num_stocks, 
        #                                             #   *target.shape[2:-1],
        #                                                 1) # [num_timestamps * nsamples, num_stocks, 1]
        # assert torch.all(preds_stocks_index[:, 0] == 0), "Stocks index should be 0 for all samples."
        # assert torch.all(preds_stocks_index[:, 2] == 2), "Stocks index should be 2 for all samples."
        # assert torch.all(preds_stocks_index[:, -1] == num_stocks - 1), f"Stocks index should be {num_stocks - 1} for all samples."

        # # Split ygen into nsamples chunks and stack them across the last dimension
        # preds = torch.stack(torch.chunk(preds, nsamples, dim = 0), dim = -1) # [num_timestamps, num_stocks, future_window, nsamples]

        # preds_timestamps = timestamps_gen.reshape(-1, num_stocks, 
        #                                          #   *target.shape[2:-1],
        #                                             1) # [num_timestamps * nsamples, num_stocks, 1]
        # preds_timestamps = torch.stack(torch.chunk(preds_timestamps, nsamples, dim = 0), dim = -1) # [num_timestamps, num_stocks, 1, nsamples]
        # assert torch.all(preds_timestamps[0] == preds_timestamps[0, 0, 0, 0]), f"Timestamps index should be {preds_timestamps[0, 0, 0, 0]} for all samples but it is {preds_timestamps[0]}."
        # assert torch.all(preds_timestamps[2] == preds_timestamps[2, 0, 0, 0]), f"Timestamps index should be {preds_timestamps[2, 0, 0, 0]} for all samples but it is {preds_timestamps[2]}."
        # assert torch.all(preds_timestamps[-1] == preds_timestamps[-1, 0, 0, 0]), f"Timestamps index should be {preds_timestamps[-1, 0, 0, 0]} for all samples but it is {preds_timestamps[-1]}."

        # assert torch.all(preds_timestamps[3, 0] == preds_timestamps[3, 0, 0, 0]), f"Timestamps index should be {preds_timestamps[3, 0, 0, 0]} for all samples but it is {preds_timestamps[3, 0]}."
        # assert torch.all(preds_timestamps[3, 2] == preds_timestamps[3, 2, 0, 0]), f"Timestamps index should be {preds_timestamps[3, 2, 0, 0]} for all samples but it is {preds_timestamps[3, 2]}."
        # assert torch.all(preds_timestamps[3, -1] == preds_timestamps[3, -1, 0, 0]), f"Timestamps index should be {preds_timestamps[3, -1, 0, 0]} for all samples but it is {preds_timestamps[3, -1]}."

        # target.unsqueeze_(-1) # [num_timestamps, num_stocks, future_window, 1]
        # assert preds.shape[:3] == target.shape[:3], f"Shape mismatch: {preds.shape} != {target.shape}"



        error_dict = log_regression_errors(preds = preds,
                                           target = target,
                                           grw_pred=grw_pred,
                                           cgrw_pred=None,
                                           pred_log_returns=pred_log_returns,
                                           target_log_returns=target_log_returns,
                                           grw_pred_log_returns=grw_pred_log_returns,
                                           cgrw_pred_log_returns=None,
                                           target_metric = target_metric
                                           )

        eval_metrics.update(error_dict)


        plot_stocks_idx = np.random.choice(num_stocks, size = 4, replace = False)
        plot_timestamps_idx = torch.randint(0, len(obs_and_pred_timestamps), size = (2,), device=obs_and_pred_timestamps.device)

        for stock_idx in plot_stocks_idx:
            # accelerator.print(f"preds_timestamps for stock {stock_idx}: ", preds_timestamps[:, stock_idx, 0, 0])
            # accelerator.print(f"Target.shape for stocks {stock_idx}: ", target[:, stock_idx, :, 0].shape)
            
            # target_column_idx = 1
            # observations = data.x[:, target_column_idx, :].reshape(-1, num_stocks, past_window)
            # future_window = target.shape[2]
            # obs_and_pred_timestamps = torch.arange(-past_window, future_window, device = preds_timestamps.device).view(1, -1).repeat(num_timestamps, 1) # [num_timestamps, past_window + future_window]
            # obs_and_pred_timestamps = obs_and_pred_timestamps + preds_timestamps[:, stock_idx, 0, 0].view(-1, 1) # Shift by the prediction start timestamp


            # ## Assert shifted observations match the target start and end values
            # delta_obs_time = obs_and_pred_timestamps[1,0] - obs_and_pred_timestamps[0,0]
            # print(f"Delta observation time: {delta_obs_time}")
            # print("Observations[0:2, :]: ", observations[0:2, stock_idx, :])
            # print("Target[0:2, stock]: ", target[0:2, stock_idx, :, 0])

            
            temp_dict = log_plot_regression(preds = preds[plot_timestamps_idx, stock_idx, :, :].detach().cpu(), 
                                target = target[plot_timestamps_idx, stock_idx, :, 0].detach().cpu(), 
                                observations = observations[plot_timestamps_idx, stock_idx].detach().cpu(),
                                timestamps = obs_and_pred_timestamps[plot_timestamps_idx].detach().cpu(),
                                metric = target_metric,
                                stocks_idx = stock_idx,
                                )

            # eval_metrics.update({f"{k}-stock-{stock_idx}": v for k, v in temp_dict.items()})
            eval_metrics.update(temp_dict)


            # # Get daily log returns and convert to log returns for plotting
            # pred_log_returns = daily_log_returns_to_log_returns(preds[plot_timestamps_idx, stock_idx, :, :].detach().cpu(), time_dim = 1)
            # target_log_returns = daily_log_returns_to_log_returns(target[plot_timestamps_idx, stock_idx, :, 0].detach().cpu(), time_dim = 1)
            # observations_log_returns = daily_log_returns_to_log_returns(observations[plot_timestamps_idx, stock_idx].detach().cpu(), time_dim = 1)

            # print("Pred log returns: ", pred_log_returns)
            # print("Target log returns: ", target_log_returns)
            # print("Observations log returns: ", observations_log_returns)


            # target_log_returns = target_log_returns + observations_log_returns[:, -1].view(-1, 1)
            # pred_log_returns = pred_log_returns + observations_log_returns[:, -1].view(-1, 1, 1)

            temp_dict = log_plot_regression(preds = pred_log_returns[plot_timestamps_idx, stock_idx, :, :].detach().cpu(),
                                target = target_log_returns[plot_timestamps_idx, stock_idx, :, 0].detach().cpu(),
                                observations = observations_log_returns[plot_timestamps_idx, stock_idx].detach().cpu(),
                                timestamps = obs_and_pred_timestamps[plot_timestamps_idx].detach().cpu(),
                                metric = "Log returns",
                                stocks_idx = stock_idx,
                                )
            eval_metrics.update({f"{k}_gdm": v for k, v in temp_dict.items()})


        
            ### Plot the GRW baselines ###
            temp_dict = log_plot_regression(preds = grw_pred_log_returns[plot_timestamps_idx, stock_idx, :, :].detach().cpu(),
                                target = target_log_returns[plot_timestamps_idx, stock_idx, :, 0].detach().cpu(),
                                observations = observations_log_returns[plot_timestamps_idx, stock_idx].detach().cpu(),
                                timestamps = obs_and_pred_timestamps[plot_timestamps_idx].detach().cpu(),
                                metric = "Log returns",
                                stocks_idx = stock_idx,
                                )
            eval_metrics.update({f"{k}_grw": v for k, v in temp_dict.items()})

            
            # temp_dict = log_plot_regression(preds = cgrw_pred_log_returns[plot_timestamps_idx, stock_idx, :, :].detach().cpu(),
            #                     target = target_log_returns[plot_timestamps_idx, stock_idx, :, 0].detach().cpu(),
            #                     observations = observations_log_returns[plot_timestamps_idx, stock_idx].detach().cpu(),
            #                     timestamps = obs_and_pred_timestamps[plot_timestamps_idx].detach().cpu(),
            #                     metric = "Log returns",
            #                     stocks_idx = stock_idx,
            #                     )
            # eval_metrics.update({f"{k}_cgrw": v for k, v in temp_dict.items()})


        # raise ValueError("Stopping here for debugging...")


        # eval_metrics.update(log_plot_regression_errors(preds, target,
        #                                                metric = data.info["Target"] if isinstance(data.info["Target"], str) else data.info["Target"][0],
        #                                                stocks_idx=plot_stocks_idx,
        #                                                segment_timestamps = False,
        #                                                )
        #                                                )

        # # Pred and target are DailyLogReturns, map them to log returns
        # target_log_returns = torch.flatten(torch.permute(target, (0, 2, 1, 3)), start_dim = 0, end_dim = 1) # [num_timestamps * future_window, num_stocks, 1]
        # target_log_returns = daily_log_returns_to_log_returns(target_log_returns)
        # target_log_returns = torch.permute(target_log_returns.reshape(num_timestamps, -1, num_stocks, 1), (0, 2, 1, 3)) # [num_timestamps, num_stocks, future_window, 1]

        # pred_log_returns = daily_log_returns_to_log_returns(preds, time_dim = 2) # Take cumulative sum over prediction window
        # pred_log_returns = pred_log_returns + target_log_returns[:, :, :1, :] - preds[:, :, :1, :] # Adjust the first log return to match the target's first log return

        # # pred_log_returns = torch.flatten(torch.permute(preds, (0, 2, 1, 3)), start_dim = 0, end_dim = 1) # [num_timestamps * future_window, num_stocks, nsamples]
        # # pred_log_returns = torch.permute(pred_log_returns.reshape(num_timestamps, -1, num_stocks, nsamples), (0, 2, 1, 3)) # [num_timestamps, num_stocks, future_window, nsamples]


        # eval_metrics.update(log_plot_regression_errors(pred_log_returns, target_log_returns,
        #                                                metric = "Log returns",
        #                                                stocks_idx=plot_stocks_idx,
        #                                                segment_timestamps = False
        #                                                )
        #                                                )

        # features_info = data.info['Features']
        # if isinstance(features_info, list) and isinstance(features_info[0], list):
        #     features_info = features_info[0]
        #     if 'Close' in features_info:
        #         closing_prices_col_idx = features_info.index('Close')
        #     else:
        #         closing_prices_col_idx = 0

        # init_closing_prices = data.x[:, closing_prices_col_idx, -1]

        # accelerator.print(f"Init closing prices: ", init_closing_prices)
        # init_closing_prices = init_closing_prices.reshape(num_timestamps, num_stocks).unsqueeze(-1).unsqueeze(-1) # [num_timestamps, num_stocks, 1, 1]
        # pred_closing_prices = daily_log_returns_to_closing_prices(preds, init_closing_prices = init_closing_prices)
        # target_closing_prices = daily_log_returns_to_closing_prices(target, init_closing_prices = init_closing_prices)

        # eval_metrics.update(log_plot_regression_errors(pred_closing_prices, target_closing_prices, metric = "Closing prices", stocks_idx=plot_stocks_idx))

        return eval_metrics





    def train(self, accelerator, model, dataloader, optimizer, lr_sched, n_epochs = [100], device = 'cpu', log_dict = {}, **kwargs):   

        assert model is not None, "Model must be provided for training."
        assert dataloader is not None, "Dataloader must be provided for training."  
        eval_criterion = self.make_validation_criteria_fnc()

        if optimizer is None or lr_sched is None:
            optimizer, lr_sched = self.init_optimizer_and_lr_scheduler(model, optimizer, lr_sched)

        if accelerator is not None:
            accelerator.print(f'Accelerated {self.__class__.__name__} training...')
            accelerator.print("Accelerator is running on device: ", accelerator.device)
            model, optimizer, dataloader['train'], lr_sched = accelerator.prepare(model, optimizer, dataloader['train'], lr_sched)
            for phase in dataloader:
                if phase not in ['train']:
                    dataloader[phase] = accelerator.prepare(dataloader[phase])

            model = model.to(device)

        else:
            print("Accelerator is not running...")


        accelerator.print(f"Dataloader['train'].dataset[0]: {dataloader['train'].dataset[0]}")


        ema = EMA(parameters=model.parameters(),
                  ema_max_decay=0.99,
                  ema_inv_gamma=1.0
                  )

        chkpt_step = self.global_chkpt_step # 0
        # prev_chkpt_step = 0
        epoch_pbar = tqdm(range(*n_epochs), leave=True)
        for epoch in epoch_pbar:
        # for epoch in tqdm.tqdm(range(*n_epochs)):
            epoch_loss = epoch_pgrad_norm = 0.0
            num_batches = 0

            for phase in dataloader:
                if phase not in ['train']:
                    continue

                if phase in ['train']:
                    model.train()
                else:
                    model.eval()

                for batch_idx, data in enumerate(dataloader[phase]):

                    # #### Clone the graph data ####
                    # data = self.batch_from_data_list(data_list = [datum.clone() for _ in range(self.config.batch_size)],
                    #                                     exclude_keys=["close_price", "close_price_y", "stocks_index", "timestamp"],
                    #                                     follow_batch=None
                    #                                     )

                    accelerator.print(f"Batch idx: {batch_idx}\tData: ", data)
                    accelerator.print(f"Data.num_graphs: {data.num_graphs}")

                    # Use the actual number of graphs if available
                    # num_graphs = getattr(data, '_actual_num_graphs', data.num_graphs)
                    # print(f"Using num_graphs: {num_graphs}")

                    # assert data._actual_num_graphs == datum.num_graphs * self.config.batch_size, f"Data.num_graphs: {data._actual_num_graphs} != datum.num_graphs * batch_size: {datum.num_graphs * self.config.batch_size}"
                    # assert data.num_nodes == datum.num_nodes * self.config.batch_size, f"Data.num_nodes: {data.num_nodes} != datum.num_nodes * batch_size: {datum.num_nodes * self.config.batch_size}"

                    if epoch == 0 and phase == 'train':
                        # Log batched data for reference in the first epoch only.
                        accelerator.print("Logging batched data for reference...")
                        for key, value in data.items():
                            accelerator.print(f"{key}: {value}")


                    with torch.set_grad_enabled(phase == 'train'):
                        with accelerator.accumulate(model):

                            ##################### Training logic ##################### 
                            model.zero_grad()

                            data = data.to(device)

                            # Generate random diffusion timesteps
                            timesteps = self.generate_random_timesteps(num_steps = self.diffusion_steps,
                                                                        size = (data.num_graphs, 1), 
                                                                        device = device,
                                                                        weights = None
                                                                        )
                                    # accelerator.print("Timesteps.shape: ", timesteps.shape             
                            timesteps = timesteps.repeat_interleave(repeats = data.num_nodes // data.num_graphs, dim = 0)
                            assert timesteps.shape[0] == data.y.shape[0], f"Shape mismatch: {timesteps.shape[0]} != {data.y.shape[0]}"

                            # Add random noise to clean samples
                            y_0 = data.y.clone()
                            noisy_y_t, eps = self.noise(y_0, timesteps)

                            # Predict the added noise (conditioning info is in the data object)
                            pred_noise = model(noisy_y_t, timesteps, data)

                            # # Assert pred_noise does not contain NaNs or Infs
                            # assert not torch.isnan(pred_noise).any(), "pred_noise contains {} many NaNs".format(torch.isnan(pred_noise).sum().item())
                            # assert not torch.isinf(pred_noise).any(), "pred_noise contains {} many Infs".format(torch.isinf(pred_noise).sum().item())

                            # Compute the diffusion loss
                            all_loss_terms = self.loss_fn(pred_noise, eps).mean(dim = -1)
                            loss = all_loss_terms.mean()

                        if phase == 'train':
                            optimizer.zero_grad()

                            if accelerator is not None:
                                accelerator.backward(loss)
                            else:
                                loss.backward()

                            # Clip the gradients
                            if self.config.grad_clipping_constant is not None:
                                if self.config.clip_grad_by == 'norm':
                                    accelerator.clip_grad_norm_(model.parameters(), max_norm=self.config.grad_clipping_constant)
                                elif self.config.clip_grad_by == 'value':
                                    accelerator.clip_grad_value_(model.parameters(), clip_value=self.config.grad_clipping_constant)
                                else:
                                    raise ValueError(f"Invalid clip_grad_by value: {self.config.clip_grad_by}. Choose 'norm' or 'value'.")


                            params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                            pgrad_norm = np.sqrt(np.sum([p.grad.norm().item()**2 for p in params]))
                            
                            optimizer.step()

                            if ema is not None:
                                # Update the exponential moving average
                                ema.step(model.parameters())
                                # ema.update(model)

                            model.zero_grad()

                            accelerator.print(f'Epoch: {epoch}\tPhase: {phase}\tBatch: {batch_idx}\tLoss: {loss.item()}')


                    epoch_loss += loss.item()
                    epoch_pgrad_norm += pgrad_norm.item()
                    num_batches += 1

                    # Update progress bar with current batch loss
                    epoch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", 
                                            "epoch_loss": f"{epoch_loss/num_batches:.4f}"})

                                ##################### Training logic ##################### 

                    if batch_idx >= 2:
                        # We want to process two batches per epoch
                        break
             
                epoch_loss /= num_batches
                epoch_pgrad_norm /= num_batches

                lr_sched.step()
                current_lr = optimizer.param_groups[0]['lr']

                self.current_train_loss = epoch_loss

                log_dict.update({f"{self.__class__.__name__}-Training/{phase}-loss": epoch_loss,
                                 f"{self.__class__.__name__}-Training/model-trained-flag": float(self.model_trained_criterion()),
                            f"{self.__class__.__name__}-Training/pgrad_norm": epoch_pgrad_norm,
                            f"{self.__class__.__name__}-Training/lr": current_lr,
                            "train-step": epoch,
                            # "DualSampler-Training/buffer-replacement-percentage": 100 * (self.lambdas_buffer.buffer_adds / self.lambdas_buffer.buffer_size - 1),
                            })
                
                if phase == "train" and eval_criterion(epoch=epoch, loss=epoch_loss):
                    accelerator.print(f"Validation criteria met at epoch {epoch}, loss {epoch_loss}")
                    val_metrics = self.validate_model(accelerator = accelerator,
                                                        model = model,
                                                        val_dataloader = dataloader['val'],
                                                        device = device
                                                        )
                    log_dict.update({f"{self.__class__.__name__}-Evaluation/val-{k}": v for k, v in val_metrics.items()})


                    # Sample from the trained model and regress
                    for eval_phase in ["train-val", "val"]:
                        nsamples = 10 # 50
                        eval_metrics = self.eval(accelerator=accelerator, model = model, data = next(iter(dataloader[eval_phase])), nsamples = nsamples, device = device, sampler = "ddpm")
                        log_dict.update({f"{self.__class__.__name__}-Evaluation/{eval_phase}-{k}": v for k, v in eval_metrics.items()})

                    chkpt_step += 1


                elif not eval_criterion(epoch=epoch, loss=epoch_loss) and phase == "train":
                    accelerator.print(f"Validation criteria not met at epoch {epoch}, for loss {epoch_loss}. Skipping validation...")

                else:
                    pass
                    

        self.global_chkpt_step = chkpt_step
        return model, optimizer, lr_sched, log_dict


    # def train(self, accelerator, model, dataloader, optimizer, lr_sched, eval_criterion, loggers=None, device='cpu'):
    # def train(self, accelerator, model, dataloader, optimizer, lr_sched, n_epochs = [100], device = 'cpu', log_dict = {}, **kwargs):   

    #     assert model is not None, "Model must be provided for training."
    #     assert dataloader is not None, "Dataloader must be provided for training."  
    #     eval_criterion = self.make_validation_criteria_fnc()

    #     if optimizer is None or lr_sched is None:
    #         optimizer, lr_sched = self.init_optimizer_and_lr_scheduler(model, optimizer, lr_sched)

    #     if accelerator is not None:
    #         accelerator.print(f'Accelerated {self.__class__.__name__} training...')
    #         accelerator.print("Accelerator is running on device: ", accelerator.device)
    #         model, optimizer, dataloader['train'], lr_sched = accelerator.prepare(model, optimizer, dataloader['train'], lr_sched)
    #         for phase in dataloader:
    #             if phase not in ['train']:
    #                 dataloader[phase] = accelerator.prepare(dataloader[phase])

    #         model = model.to(device)

    #     else:
    #         print("Accelerator is not running...")


    #     accelerator.print(f"Dataloader['train'].dataset.data_list[0]: {dataloader['train'].dataset.data_list[0]}")


    #     ema = EMA(parameters=model.parameters(),
    #               ema_max_decay=0.99,
    #               ema_inv_gamma=1.0
    #               )

    #     chkpt_step = self.global_chkpt_step # 0
    #     # prev_chkpt_step = 0
    #     epoch_pbar = tqdm(range(*n_epochs), leave=True)
    #     for epoch in epoch_pbar:
    #     # for epoch in tqdm.tqdm(range(*n_epochs)):
    #         epoch_loss = epoch_pgrad_norm = 0.0
    #         num_batches = 0

    #         for phase in dataloader:
    #             if phase not in ['train']:
    #                 continue

    #             if phase in ['train']:
    #                 model.train()
    #             else:
    #                 model.eval()

    #             all_variables = defaultdict(list)
    #             all_sample_idx = []
    #             all_variables["loss"] = torch.zeros((len(dataloader[phase].dataset),)).to(device)

    #             for batch_idx, (data, sample_idx) in enumerate(dataloader[phase]):

    #                 accelerator.print(f"Batch idx: {batch_idx}\tSample idx: {sample_idx}")
    #                 accelerator.print("Data: ", data)
    #                 num_x_samples = self.config.x_batch_size # 64 # Number of copies of the same graph to sample from.
    #                 print("num_x_samples: ", num_x_samples)

    #                 # return model, optimizer, lr_sched, log_dict
    #                 with torch.set_grad_enabled(phase == 'train'):
    #                     with accelerator.accumulate(model):
    #                         model.zero_grad()
    #                         sample_idx = sample_idx.to(device)

    #                         # Make sure data.network_id is the same as sample_idx
    #                         assert torch.allclose(data.network_id.to(sample_idx.device), sample_idx), f"Data network_id {data.network_id} and sample_idx {sample_idx} are not the same."


    #                         if num_x_samples > 1:
    #                             # # NOTE TO MYSELF: Batch_size overwrites number of iterable keys.
    #                             # data_list = create_data_list_from_batch(data = data,
    #                             #                                         keys_to_copy="all",
    #                             #                                         keys_to_iterate=['y'],
    #                             #                                         batch_size = n_lambdas,
    #                             #                                         batch_dim = 0,
    #                             #                                         batch_size_overwrite_num_iterable_keys = True
    #                             #                                         )
                                
    #                             # assert len(data_list) == n_lambdas, f"Data list length {len(data_list)} is not equal to n_lambdas {n_lambdas}."
    #                             # data_batch = self.batch_from_data_list(data_list = data_list, exclude_keys=["x", "x_l"])
                                
    #                             # Measure the time it takes to create the data list.
    #                             start_time = time.time()
    #                             data_list = self.create_data_list_from_batch(data = data,
    #                                                                     keys_to_copy=['edge_index_l', 'edge_weight_l', 'x_l', 'transmitters_index', 'num_nodes', 'network_id', 'm', 'n', 'batch'],
    #                                                                     keys_to_iterate=['x'],
    #                                                                     batch_size = num_x_samples,
    #                                                                     batch_dim = 0,
    #                                                                     batch_size_overwrite_num_iterable_keys=True,
    #                                                                     shuffle_keys_to_iterate=True,
    #                                                                     # efficient_cloning=True
    #                                                                     )
    #                             end_time = time.time()
    #                             accelerator.print(f"[Experimental method]\tTime taken to create data_list: {end_time - start_time:.4f} seconds")
                                
    #                             # start_time = time.time()
    #                             # data_list = self.create_data_list_from_batch(data = data,
    #                             #                                         keys_to_copy=['edge_index_l', 'edge_weight_l', 'x_l', 'transmitters_index', 'num_nodes', 'network_id', 'm', 'n', 'batch'],
    #                             #                                         keys_to_iterate=['x'],
    #                             #                                         batch_size = num_x_samples,
    #                             #                                         batch_dim = 0,
    #                             #                                         batch_size_overwrite_num_iterable_keys=True,
    #                             #                                         shuffle_keys_to_iterate=True
    #                             #                                         )
    #                             # end_time = time.time()
    #                             # accelerator.print(f"[Original method]\tTime taken to create data_list: {end_time - start_time:.4f} seconds")
                                
    #                             data_batch = self.batch_from_data_list(data_list=data_list,
    #                                                                     exclude_keys=None
    #                                                                     )
                            

    #                             sample_idx = sample_idx.repeat(num_x_samples,).to(device)


    #                         data_batch = data_batch.to(device)
    #                         # lambdas = data_batch.x # diffusion variable is the lambdas
    #                         # print("lambdas: ", lambdas.shape)
    #                         n = data_batch.num_nodes // data_batch.num_graphs

    #                         X_batch = data_batch.x.view(-1, n).to(device)
    #                         # X_batch = data_batch.x.to(device)
    #                         # X_batch = torch.stack(X_batch[torch.randint(0, X_batch.shape[0], (n_lambdas,))], dim=0) # Randomly sample n_lambdas from the batch.
    #                         # X_batch = X_batch.view(-1, n)

    #                         accelerator.print(f'X.shape: {X_batch.shape}')
    #                         timesteps = self.generate_random_timesteps(num_steps = self.diffusion_steps,
    #                                                                         size = (X_batch.shape[0], 1), 
    #                                                                         device = device,
    #                                                                         weights = None
    #                                                                         )
                            
    #                         # accelerator.print("Timesteps.shape: ", timesteps.shape)
    #                         timesteps = timesteps.repeat_interleave(repeats=X_batch.shape[1], dim=0)
    #                         # accelerator.print("Timesteps (repeated).shape: ", timesteps.shape)
    #                         X_batch = X_batch.view(-1, 1)
    #                         # accelerator.print("X_batch.shape: ", X_batch.shape)

    #                         noised, eps = self.noise(X_batch, timesteps)

    #                         # # We don't need the transmitters index for the diffusion model.
    #                         # transmitters_index = data_batch.transmitters_index
    #                         # data_batch.transmitters_index = None

    #                         predicted_noise = model(noised.to(device), timesteps.to(device), data_batch)
    #                         # Plug the transmitters index back in.
    #                         # data_batch.transmitters_index = transmitters_index

    #                         if isinstance(predicted_noise, tuple):
    #                             for i in range(len(predicted_noise), 0, -1):
    #                                 if i == 3:
    #                                     model_debug_data = predicted_noise[i-1]
    #                                 elif i == 2:
    #                                     attn_weights = predicted_noise[i-1]
    #                                 elif i == 1:
    #                                     predicted_noise = predicted_noise[i-1]
    #                                 else:
    #                                     pass
    #                         else:
    #                             attn_weights = None
    #                             model_debug_data = None

    #                         all_loss_terms = self.loss_fn(predicted_noise.view(-1, n), eps.view(-1, n).to(device)).mean(dim = -1)
    #                         loss = all_loss_terms.mean()

    #                         scatter(src=all_loss_terms, index=data_batch.network_id, out=all_variables["loss"], dim=0, reduce="mean")

    #                         # all_variables["loss"].extend(loss_per_network.detach().cpu().tolist())
    #                         all_variables["sample-idx"].extend(sample_idx.detach().cpu().tolist())
    #                         all_variables["network-idx"].extend(data.network_id.detach().cpu().tolist())


    #                     if phase == 'train':
    #                         optimizer.zero_grad()

    #                         if accelerator is not None:
    #                             accelerator.backward(loss)
    #                         else:
    #                             loss.backward()

    #                         accelerator.clip_grad_norm_(model.parameters(), max_norm=10.0)

    #                         params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    #                         pgrad_norm = np.sqrt(np.sum([p.grad.norm().item()**2 for p in params]))
                            
    #                         optimizer.step()

    #                         if ema is not None:
    #                             # Update the exponential moving average
    #                             ema.step(model.parameters())
    #                             # ema.update(model)

    #                         model.zero_grad()

    #                         accelerator.print(f'Epoch: {epoch}\tPhase: {phase}\tBatch: {batch_idx}\tLoss: {loss.item()}')

    #                 epoch_loss += loss.item()
    #                 epoch_pgrad_norm += pgrad_norm.item()
    #                 num_batches += 1

    #                 # Update progress bar with current batch loss
    #                 epoch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", 
    #                                         "epoch_loss": f"{epoch_loss/num_batches:.4f}"})
                    
    #                 # del data_batch
    #                 # torch.cuda.empty_cache()
    #             epoch_loss /= num_batches
    #             epoch_pgrad_norm /= num_batches

    #             lr_sched.step()
    #             current_lr = optimizer.param_groups[0]['lr']

    #             self.current_train_loss = epoch_loss

    #             log_dict.update({f"{self.__class__.__name__}-Training/{phase}-loss": epoch_loss,
    #                              f"{self.__class__.__name__}-Training/model-trained-flag": float(self.model_trained_criterion()),
    #                         f"{self.__class__.__name__}-Training/pgrad_norm": epoch_pgrad_norm,
    #                         f"{self.__class__.__name__}-Training/lr": current_lr,
    #                         "train-step": epoch,
    #                         # "DualSampler-Training/buffer-replacement-percentage": 100 * (self.lambdas_buffer.buffer_adds / self.lambdas_buffer.buffer_size - 1),
    #                         })
                

    #             for key in all_variables:
    #                 # Reorder values corresponding to the key based on ascending order of sample_idx key.
    #                 if key not in ['sample-idx', 'network-idx']:
    #                     # all_variables[key] = [all_variables[key][idx] for idx in np.argsort(all_variables['network-idx'])]
    #                     all_variables[key] = all_variables[key].tolist()
    #                 else:
    #                     all_variables[key] = sorted(all_variables[key])

    #             # Log the variables to the log_dict
    #             for key in all_variables:
    #                 if isinstance(all_variables[key], list) and len(all_variables[key]) == len(all_variables['network-idx']) and key not in ["network-idx", "sample-idx"]:
    #                     for graph_id, value in zip(all_variables["network-idx"], all_variables[key]):
    #                         log_dict.update({f"{self.__class__.__name__}-Training/{phase}/{key}-graph-{graph_id}": value})


    #             if eval_criterion(epoch=epoch, loss=epoch_loss) and phase == 'train':
    #                 accelerator.print("Validation criterion met. Evaluating the model...")

    #                 # eval_graphs_idx = torch.arange(0, min(len(dataloader[phase].dataset), MAX_LOGGED_NETWORKS))
    #                 eval_graphs_idx = None # For now, we will evaluate all graphs in the validation set.

    #                 if eval_graphs_idx is not None:
    #                     # From dataloader[phase], create a new dataloader with only the eval_graphs_idx
    #                     # eval_dataset = copy.deepcopy(Subset(dataloader[phase].dataset, eval_graphs_idx))
    #                     eval_dataset = Subset(dataloader[phase].dataset, eval_graphs_idx)
    #                     accelerator.print("Eval_dataset.dataset: ", eval_dataset.dataset)
    #                     accelerator.print(f'Creating eval dataset with {len(eval_dataset)} graphs.')
    #                     eval_dataloader = DataLoader(eval_dataset,
    #                                                  batch_size=MAX_LOGGED_NETWORKS,
    #                                                  shuffle=True,
    #                                                  collate_fn=dataloader[phase].collate_fn
    #                                                  )
                        
    #                     accelerator.print("Passing eval_dataloader to self.accelerated_eval().")
                        
    #                 else:
    #                     eval_dataloader = dataloader["val"]
    #                     accelerator.print("Passing dataloader['val'] to self.accelerated_eval().")

    #                 # if loggers is not None:
    #                 #     # Filter out the loggers that are not in the eval_graphs_idx.
    #                 #     eval_loggers = [logger for logger in loggers[phase] if not (hasattr(logger, 'network_id') and logger.network_id not in eval_graphs_idx.tolist())]
    #                 # else:
    #                 #     eval_loggers = None
    #                 eval_loggers = None

    #                 accelerator.print("Passing eval_loggers to self.accelerated_eval().")
    #                 graph_eval_keys = ['diffusion-sampler', 'wasserstein-dist', 'mean-rates', 'min-rates', 'ergodic-rates'] # per graph/network keys
    #                 global_eval_keys = [] # global metrics keys
    #                 # Take the union of two kinds of keys
    #                 eval_keys = list(set(graph_eval_keys + global_eval_keys))
    #                 eval_metrics = self.accelerated_eval(
    #                                     accelerator=accelerator,
    #                                     model=model,
    #                                     dataloader=eval_dataloader,
    #                                     nsamples=self.config.diffuse_n_samples,
    #                                     device=device,
    #                                     loggers=eval_loggers,
    #                                     sampler='ddpm',
    #                                     eta=0.2, # effective only when sampler is ddim.
    #                                     eval_fn = make_eval_function(eval_keys=eval_keys)
    #                                     )
                    
    #                 print("Eval_metrics.keys(): ", eval_metrics.keys())
                    
    #                 # log_dict.update({f"{self.__class__.__name__}-Evaluation-{phase}/{k}": v for k, v in eval_metrics.items()})

    #                 r_min = self.constraints.r_min if self.constraints is not None and hasattr(self.constraints, 'r_min') else None

    #                 for key in eval_keys:
    #                     log_dict = self.update_eval_log_dict(log_dict=log_dict, eval_metrics=eval_metrics['eval_fn'], eval_key=key, graph_id=None, wandb_log_prefix = f"{self.__class__.__name__}-Evaluation-{phase}",
    #                                                             r_min=r_min)

    #                 # lambdas_max = eval_dataset.dataset.lambdas_sampler.lambdas_max
    #                 lambdas_max = self.lambdas_max # 1.0
    #                 accelerator.print(f"lambdas_max: {lambdas_max}")
    #                 for key in graph_eval_keys:
    #                     for graph_id in range(len(eval_dataloader.dataset)):
    #                         log_dict = self.update_eval_log_dict(log_dict=log_dict, eval_metrics=eval_metrics['eval_fn'], eval_key=key,
    #                                                                 graph_id=graph_id,
    #                                                                 node_pair_list=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)] if key in ['diffusion-sampler'] and graph_id in [0, 1] else None,
    #                                                                 wandb_log_prefix = f"{self.__class__.__name__}-Evaluation-{phase}",
    #                                                                 xrange = [1.2 * (0-lambdas_max/2), 1.2 * (lambdas_max-lambdas_max/2)] if key in ['diffusion-sampler'] and self.lambdas_max is not None else None,
    #                                                                 yrange = [1.2 * (0-lambdas_max/2), 1.2 * (lambdas_max-lambdas_max/2)] if key in ['diffusion-sampler'] and self.lambdas_max is not None else None
    #                                                                 )
    #                     # log_dict = self.update_eval_log_dict(log_dict=log_dict, eval_metrics=eval_metrics['eval_fn'], eval_key='wasserstein-dist', graph_id=graph_id)


    #                 chkpt_step += 1

    #                 batch_size = len(eval_dataloader.dataset) // len(eval_dataloader)
    #                 if eval_graphs_idx is not None:
    #                     ### Update the dataloader with the recent samples. ###
    #                     # batch_size = len(dataloader[phase].dataset) // len(dataloader[phase])
    #                     dataloader[phase] = self.update_dataloader(accelerator, eval_metrics=eval_metrics["eval_fn"], dataloader = dataloader[phase],
    #                                                         phase = phase, batch_size = batch_size)
                        
    #                 else:
    #                     ### Update the dataloader with the recent samples. ###
    #                     # batch_size = len(dataloader["val"].dataset) // len(dataloader["val"])
    #                     dataloader["val"] = self.update_dataloader(accelerator, eval_metrics=eval_metrics["eval_fn"], dataloader = dataloader["val"],
    #                                                         phase = "val", batch_size = batch_size)

                    
    #             # accelerator.log(log_dict)
    #             elif not eval_criterion(epoch=epoch, loss=epoch_loss) and phase == 'train':
    #                 accelerator.print(f"Validation criterion not met. Skipping evaluation for epoch {epoch}.")

    #             else:
    #                 pass


    #     self.global_chkpt_step = chkpt_step

    #     return model, optimizer, lr_sched, log_dict
    


    def eval_model_and_update_dataloader(self, accelerator, eval_dataloader, model, device, phase = 'test'):

        # if loggers is not None:
        #     # Filter out the loggers that are not in the eval_graphs_idx.
        #     eval_loggers = [logger for logger in loggers[phase] if not (hasattr(logger, 'network_id') and logger.network_id not in eval_graphs_idx.tolist())]
        # else:
        #     eval_loggers = None
        eval_loggers = None

        accelerator.print("Passing eval_loggers to self.accelerated_eval().")
        graph_eval_keys = ['diffusion-sampler', 'wasserstein-dist', 'mean-rates', 'min-rates', 'ergodic-rates'] # per graph/network keys
        global_eval_keys = [] # global metrics keys
        # Take the union of two kinds of keys
        eval_keys = list(set(graph_eval_keys + global_eval_keys))
        eval_metrics = self.accelerated_eval(
                            accelerator=accelerator,
                            model=model,
                            dataloader=eval_dataloader,
                            nsamples=self.config.diffuse_n_samples,
                            device=device,
                            loggers=eval_loggers,
                            sampler='ddpm',
                            eta=0.2, # effective only when sampler is ddim.
                            eval_fn = make_eval_function(eval_keys=eval_keys)
                            )
                    
        print("Eval_metrics.keys(): ", eval_metrics.keys())

        logging = False
        
        if logging:
            r_min = self.constraints.r_min if self.constraints is not None and hasattr(self.constraints, 'r_min') else None

            for key in eval_keys:
                log_dict = self.update_eval_log_dict(log_dict=log_dict, eval_metrics=eval_metrics['eval_fn'], eval_key=key, graph_id=None, wandb_log_prefix = f"{self.__class__.__name__}-Evaluation-{phase}",
                                                        r_min=r_min)

            # lambdas_max = eval_dataset.dataset.lambdas_sampler.lambdas_max
            lambdas_max = self.lambdas_max # 1.0
            accelerator.print(f"lambdas_max: {lambdas_max}")
            for key in graph_eval_keys:
                for graph_id in range(len(eval_dataloader.dataset)):
                    log_dict = self.update_eval_log_dict(log_dict=log_dict, eval_metrics=eval_metrics['eval_fn'], eval_key=key,
                                                            graph_id=graph_id,
                                                            node_pair_list=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)] if key in ['diffusion-sampler'] and graph_id in [0, 1] else None,
                                                            wandb_log_prefix = f"{self.__class__.__name__}-Evaluation-{phase}",
                                                            xrange = [1.2 * (0-lambdas_max/2), 1.2 * (lambdas_max-lambdas_max/2)] if key in ['diffusion-sampler'] and self.lambdas_max is not None else None,
                                                            yrange = [1.2 * (0-lambdas_max/2), 1.2 * (lambdas_max-lambdas_max/2)] if key in ['diffusion-sampler'] and self.lambdas_max is not None else None
                                                            )
                # log_dict = self.update_eval_log_dict(log_dict=log_dict, eval_metrics=eval_metrics['eval_fn'], eval_key='wasserstein-dist', graph_id=graph_id)


            # chkpt_step += 1

        batch_size = len(eval_dataloader.dataset) // len(eval_dataloader)
        
        eval_dataloader = self.update_dataloader(accelerator, eval_metrics=eval_metrics["eval_fn"], dataloader = eval_dataloader,
                                            phase = phase, batch_size = batch_size)

        return eval_dataloader



    def update_dataloader(self, accelerator, eval_metrics, dataloader, phase, batch_size = 2) -> DataLoader:
        """
        Update the dataloader with the recent samples based on the eval_metrics.
        """
        
        Xgen_list = eval_metrics.get("diffusion-sampler", None)
        if Xgen_list is None:
            accelerator.print("No generated samples found in eval_metrics. Returning the original dataloader.")
            return dataloader 
        
        data_list = dataloader.dataset.data_list

        assert isinstance(Xgen_list, list) and len(Xgen_list) == len(data_list), \
            f"Xgen_list must be a list of the same length as data_list. Got {len(Xgen_list)} and {len(data_list)}."
        

        accelerator.print(f"Data_list already has data.xgen attribute but will be updated." if hasattr(data_list[0], "xgen") else "Data_list does not have data.xgen attribute, we will set it for the first time.")
        accelerator.print("Updating the dataloader with the recent samples based on the eval_metrics.")

        for Xorig_and_Xgen, data in zip(Xgen_list, data_list):
            _, Xgen = Xorig_and_Xgen
            accelerator.print(f"Data: {data}")
            accelerator.print(f"Data.network_id: {data.network_id}")
            accelerator.print(f"type(Xgen): {type(Xgen)}")
            accelerator.print(f"Xgen.shape: {Xgen.shape}" if hasattr(Xgen, "shape") else f"len(Xgen): {len(Xgen)}")

            rand_idx = torch.randint(low = 0, high = Xgen.shape[0], size = (len(data.x),), device=Xgen.device).tolist()

            data_xgen = [Xgen[idx].view_as(data.x[0]).cpu() for idx in rand_idx]

            assert len(data_xgen) == len(data.x), f"Length of data_xgen {len(data_xgen)} must be equal to length of data.x {len(data.x)}."
            assert data_xgen[0].shape == data.x[0].shape, f"Shape of data_xgen {data_xgen[0].shape} must be equal to shape of data.x {data.x[0].shape}."
            assert data_xgen[0].device == data.x[0].device, f"Device of data_xgen {data_xgen[0].device} must be equal to device of data.x {data.x[0].device}."
            
            data.xgen = data_xgen

        dataloader = DataLoader(WirelessDataset(data_list),
                                shuffle = (phase == "train"),
                                batch_size = batch_size,
                                num_workers = 4 if phase == "train" else 0,
                                collate_fn=dataloader.collate_fn,
                                )

        return dataloader





class StockPriceForecastDiffusionLearnerWrapper(abc.ABC):
    def __init__(self, config, diffusion_config, device = 'cpu'):
        self.config = config # stock config
        self.device = device

        self.cd_learner = StockPriceForecastDiffusionLearner(config = diffusion_config, device = device)


    def train_chkpt_criterion(self, epoch, total_num_epochs, freq = 1/20):
        """
        Define the criterion for saving the training checkpoint.
        This can be customized based on the training requirements.
        """
        # # Example criterion: save every 5 epochs after 80% of the total epochs
        # if epoch % 20 == 0 and epoch > 0.8 * total_num_epochs:
        #     return True
        if self.cd_learner.global_chkpt_step >= 1 and (epoch + 1) % (total_num_epochs * freq) == 0:
            return True
        
        return False
    


    def resume_training_from_chkpt(self, accelerator, cd_model, cd_optimizer, cd_lr_sched,
                                          load_cd_train_chkpt_path, n_epochs) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, List[int]]:
        """
        Resume the DiffusionModel training from the checkpoint.
        """

        accelerator.print(f"Resuming the {self.__class__.__name__} training from the checkpoint {load_cd_train_chkpt_path}.")

        if cd_optimizer is None or cd_lr_sched is None:

            cd_optimizer, cd_lr_sched = self.cd_learner.init_optimizer_and_lr_scheduler(
                model=cd_model, optimizer=cd_optimizer, lr_sched=cd_lr_sched
            )
            accelerator.print("Initialized the optimizer and lr scheduler for the interference model.")


        # Load the checkpoint
        chkpt = torch.load(load_cd_train_chkpt_path, map_location=accelerator.device, weights_only=False)

        checkpoint = chkpt['model_state_dict']
        # Remove the prefix "_orig_mod." from the keys in the checkpoint
        # This is needed for compatibility with the current model structure
        # as the checkpoint was saved with a different prefix.
        temp_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            prefix = "_orig_mod."
            if key.startswith(prefix):
                temp_checkpoint[key[len(prefix):]] = value
        
        incompatible_keys = cd_model.load_state_dict(temp_checkpoint, strict=True)
        print("Incompatible keys: ", incompatible_keys)

        # cd_model.load_state_dict(chkpt['model_state_dict'])
        cd_optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        cd_lr_sched.load_state_dict(chkpt['scheduler_state_dict'])
        epoch = chkpt['epoch']

        accelerator.print(f"Loaded cd_model, optimizer and lr_sched state dictionaries at epoch {epoch}.")


        if "global_chkpt_step" in chkpt and chkpt["global_chkpt_step"] is not None:
            self.cd_learner.global_chkpt_step = chkpt["global_chkpt_step"]
            accelerator.print(f"Loaded global_chkpt_step: {self.cd_learner.global_chkpt_step} from the checkpoint.")
        else:
            self.cd_learner.global_chkpt_step = 0
            accelerator.print("No global_chkpt_step found in the checkpoint. Setting it to 0.")


        if isinstance(n_epochs, list) and len(n_epochs) == 1:
            n_epochs = [0] + n_epochs

        accelerator.print("Updating n_epochs from {} to {}.".format(n_epochs, [epoch + 1] + n_epochs[1:] if isinstance(n_epochs, list) else epoch + 1))
        n_epochs = [epoch + 1] + n_epochs[1:] if isinstance(n_epochs, list) else epoch + 1
          
        accelerator.print(f"Resuming {self.__class__.__name__} training from the checkpoint with preloaded cd-model, cd-optimizer and cd-lr-scheduler at epoch {epoch}.")
        return cd_model, cd_optimizer, cd_lr_sched, n_epochs
    


    def train(self, accelerator, args, arg_groups, cd_model, cd_optimizer, cd_lr_sched, cd_dataloaders,
              n_epochs = [100], n_iters_cd_per_epoch = 100, device = 'cpu',
               **kwargs):
        
        save_train_chkpt_path = kwargs.get('save_train_chkpt_path', None)
        load_train_chkpt_path = kwargs.get('load_train_chkpt_path', None)
        save_test_metrics_path = kwargs.get('save_test_metrics_path', None)
    

        if wandb.run is not None:
            # wandb.define_metric(name="my_wandb_histogram", summary=None)
            wandb.define_metric(name="train-step")
            wandb.define_metric(name="train-epoch", step_metric="train-step", summary=None)
            wandb.define_metric(name=f"{self.cd_learner.__class__.__name__}-Training/*", step_metric="train-epoch", summary=None)
            # wandb.define_metric(name="StateAugmentation-Training/*", step_metric="epoch", summary="min")

        # cd_optimizer, cd_lr_sched = None, None
        if cd_optimizer is None or cd_lr_sched is None:
            accelerator.print("Creating the optimizer and lr scheduler for the interference model if they do not exist.")
            cd_optimizer, cd_lr_sched = self.cd_learner.init_optimizer_and_lr_scheduler(
                model=cd_model, optimizer=cd_optimizer, lr_sched=cd_lr_sched
            )


        if load_train_chkpt_path not in [None, "None", "", 'none']:

            cd_model, cd_optimizer, cd_lr_sched, n_epochs = self.resume_training_from_chkpt(accelerator = accelerator,
                                                    cd_model = cd_model, 
                                                    cd_optimizer = cd_optimizer, 
                                                    cd_lr_sched = cd_lr_sched,
                                                    load_cd_train_chkpt_path = load_train_chkpt_path,
                                                    n_epochs = n_epochs,
                                                    # dr_dataloader = dr_dataloader
                                                    )
            n_iters = n_epochs[0] * n_iters_cd_per_epoch


        accelerator.print("n_epochs: ", n_epochs)
        n_iters = 0
        epoch_pbar = tqdm(range(*n_epochs), leave=True, initial=n_epochs[0], total=n_epochs[-1])
        for epoch in epoch_pbar:
            accelerator.print(f"Epoch {epoch} of {n_epochs[-1]}...")

            n_iters_cd = [n_iters, n_iters + n_iters_cd_per_epoch]
            cd_model, cd_optimizer, cd_lr_sched, log_dict = self.cd_learner.train(accelerator=accelerator, model = cd_model, dataloader=cd_dataloaders,
                                                                                    n_epochs = n_iters_cd, optimizer=cd_optimizer, lr_sched=cd_lr_sched,
                                                                                    device=device, log_dict={}
                                                                                    )
            log_dict.update({"train-epoch": epoch})
            n_iters = n_iters_cd[-1]

            # epoch_pbar.update()
            accelerator.log(log_dict)
            # self.log(accelerator, log_dict)


            if save_train_chkpt_path is not None and self.train_chkpt_criterion(epoch=epoch, total_num_epochs=n_epochs[-1]):
                # Save the CD model checkpoint every 1/100th of total training epochs.
                    
                accelerator.print(f"Saving the cd model weights to path {save_train_chkpt_path}.")

                os.makedirs(save_train_chkpt_path, exist_ok=True)

                save_cd_train_chkpt(save_path=save_train_chkpt_path,
                                    model=accelerator.unwrap_model(cd_model),
                                    optimizer=accelerator.unwrap_model(cd_optimizer),
                                    lr_sched=accelerator.unwrap_model(cd_lr_sched),
                                    epoch=epoch,
                                    global_chkpt_step=self.cd_learner.global_chkpt_step,
                )

                
            #     if sa_model is not None:
            #         ### Evaluate the model on the validation set and log the results ###
            #         accelerator.print("Evaluating the model on the validation set...")

            #         test_phases = ['val']
            #         sa_learner = StateAugmentedDualRegressionLearner(
            #             config=arg_groups['SA-train-algo'],
            #             channel_config = arg_groups['RRM'],
            #             dr_config = arg_groups['DR-train-algo'],
            #             device = device,
            #         )

            #         test_algos = ['SA', 'SA-star', 'GDM']
            #         sa_model_dict = {}
            #         for algo in test_algos:
            #             if algo == "GDM":
            #                 T0_overwrite = 1
            #                 accelerator.print(f"Using the interference model to create a GDM model sampler with T_0 = {T0_overwrite}.")
            #                 sa_model_dict[algo] = self.cd_learner.make_cd_model_sampler_fn(P_max = sa_model.P_max, T0_overwrite=T0_overwrite)
            #             else:
            #                 sa_model_dict[algo] = sa_model
                     

            #         temp, test_log_dict = sa_learner.test(accelerator=accelerator, sa_model=sa_model_dict,
            #                     lambda_init_fn = sa_learner.make_custom_dual_dynamics_init_fn(lambda_model=None, lambda_transform = None,
            #                                                                                   sa_star_init_with_y_l=True),
            #                     dataloader={k: v for k, v in channel_dataloader.items() if k in test_phases},
            #                     algos = test_algos,
            #                     save_path = save_test_metrics_path,
            #                     epoch = epoch,
            #                     custom_dual_dynamics_step_fn = None,
            #                     T_range = range(0, 500), # range(0, 200),
            #                     deduce_phase_from_lr_dual=False,
            #                     log_dict = {},
            #                     test_log = [self.cd_learner.make_test_log_fn(P_max = sa_model.P_max), sa_learner.test_log]
            #                     )
                    
            #         test_log_dict = {f"{self.__class__.__name__}-Test/{k}": v for k, v in test_log_dict.items()}

            #         accelerator.print("Logging test results to wandb.")
            #         accelerator.log(test_log_dict)


        return cd_model
    


    def test(self, accelerator, args, arg_groups, cd_model, cd_optimizer, cd_lr_sched, channel_dataloader,
              n_epochs = [100], n_iters_cd_per_epoch = 100, device = 'cpu', loggers = None, sa_model = None,
               **kwargs):
        
        """ 
        This function has the same signature as the train function, but it is used for testing the model.
        It will not train the model, but it will evaluate the model on the test set and log the results.
        It will also save the test metrics to the specified path.
        The test metrics will be saved in a dictionary format with keys as the metric names and values as the metric values.
        The test metrics will be logged to wandb if wandb is enabled.
        """

        save_train_chkpt_path = kwargs.get('save_train_chkpt_path', None)
        load_train_chkpt_path = kwargs.get('load_train_chkpt_path', None)
        save_test_metrics_path = kwargs.get('save_test_metrics_path', None)
        sweep_train_chkpts = kwargs.get('sweep_train_chkpts', False)
    

        if wandb.run is not None:
            # wandb.define_metric(name="my_wandb_histogram", summary=None)
            wandb.define_metric(name="train-step")
            wandb.define_metric(name="train-epoch", step_metric="train-step", summary=None)
            wandb.define_metric(name=f"{self.cd_learner.__class__.__name__}-Training/*", step_metric="train-epoch", summary=None)
            # wandb.define_metric(name="StateAugmentation-Training/*", step_metric="epoch", summary="min")

        # cd_optimizer, cd_lr_sched = None, None
        if cd_optimizer is None or cd_lr_sched is None:
            accelerator.print("Creating the optimizer and lr scheduler for the interference model if they do not exist.")
            cd_optimizer, cd_lr_sched = self.cd_learner.init_optimizer_and_lr_scheduler(
                model=cd_model, optimizer=cd_optimizer, lr_sched=cd_lr_sched
            )


        n_epochs_orig = copy.deepcopy(n_epochs)
        
        if load_train_chkpt_path not in [None, "None", "", 'none'] and sweep_train_chkpts:
            # Find all the chkpt paths in the same directory as the load_train_chkpt_path where epoch number is greater or equal to that of the load_train_chkpt_path
            load_train_chkpt_paths = os.listdir(os.path.dirname(load_train_chkpt_path))
            load_train_chkpt_paths = [os.path.join(os.path.dirname(load_train_chkpt_path), f) for f in load_train_chkpt_paths if f.endswith('.pt')]
            
            load_train_chkpt_paths = [f for f in load_train_chkpt_paths if int(f.split('_')[-1].split('.')[0]) >= int(load_train_chkpt_path.split('_')[-1].split('.')[0])]
            load_train_chkpt_paths = sorted(load_train_chkpt_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        else:
            pass

        accelerator.print(f"Found {len(load_train_chkpt_paths)} training checkpoints in the directory {os.path.dirname(load_train_chkpt_path)} with chkpt-epoch larger than {int(load_train_chkpt_path.split('_')[-1].split('.')[0])}.")
        if len(load_train_chkpt_paths) == 0:
            accelerator.print(f"No training checkpoints found in the directory {os.path.dirname(load_train_chkpt_path)} with chkpt-epoch larger than {int(load_train_chkpt_path.split('_')[-1].split('.')[0])}.")
            load_train_chkpt_paths = [load_train_chkpt_path]
        else:
            pass
        
        
        for load_train_chkpt_path in load_train_chkpt_paths:
            if load_train_chkpt_path not in [None, "None", "", 'none']:

                cd_model, cd_optimizer, cd_lr_sched, n_epochs = self.resume_training_from_chkpt(accelerator = accelerator,
                                                        cd_model = cd_model, 
                                                        cd_optimizer = cd_optimizer, 
                                                        cd_lr_sched = cd_lr_sched,
                                                        load_cd_train_chkpt_path = load_train_chkpt_path,
                                                        n_epochs = n_epochs_orig,
                                                        # dr_dataloader = dr_dataloader
                                                        )
                n_iters = n_epochs[0] * n_iters_cd_per_epoch


            epoch = n_epochs[0] - 1
            # self.test_train_chkpt(accelerator=accelerator, args=args, arg_groups=arg_groups,
            #                     cd_model=cd_model, dataloader=channel_dataloader,
            #                     sa_model=sa_model, device=device,
            #                     save_test_metrics_path=save_test_metrics_path, epoch=epoch, phase='test')




    # def test_train_chkpt(self, accelerator, args, arg_groups, cd_model, dataloader, sa_model, device,
    #                save_test_metrics_path, epoch, phase = 'test'):

        
    #     eval_dataloader = dataloader[phase]
    #     eval_dataloader = self.cd_learner.eval_model_and_update_dataloader(accelerator=accelerator,
    #                                                                   eval_dataloader=eval_dataloader,
    #                                                                   model=cd_model,
    #                                                                   device=device,
    #                                                                   phase=phase
    #                                                               )
    #     accelerator.print(f"{phase}-dataloader has been updated with generated samples.")


    #     if sa_model is not None:
    #         ### Evaluate the model on the validation set and log the results ###
    #         accelerator.print(f"Evaluating the model on the {phase} set...")

    #         sa_learner = StateAugmentedDualRegressionLearner(
    #             config=arg_groups['SA-train-algo'],
    #             channel_config = arg_groups['RRM'],
    #             dr_config = arg_groups['DR-train-algo'],
    #             device = device,
    #         )

    #         test_algos = ['SA', 'SA-star', 'GDM']
    #         sa_model_dict = {}
    #         for algo in test_algos:
    #             if algo == "GDM":
    #                 T0_overwrite = 1
    #                 accelerator.print(f"Using the interference model to create a GDM model sampler with T_0 = {T0_overwrite}.")
    #                 sa_model_dict[algo] = self.cd_learner.make_cd_model_sampler_fn(P_max = sa_model.P_max, T0_overwrite=T0_overwrite)
    #             else:
    #                 sa_model_dict[algo] = sa_model
                

    #         temp, test_log_dict = sa_learner.test(accelerator=accelerator, sa_model=sa_model_dict,
    #                     lambda_init_fn = sa_learner.make_custom_dual_dynamics_init_fn(lambda_model=None, lambda_transform = None,
    #                                                                                     sa_star_init_with_y_l=True),
    #                     dataloader={phase: eval_dataloader},
    #                     algos = test_algos,
    #                     save_path = save_test_metrics_path,
    #                     epoch = epoch,
    #                     custom_dual_dynamics_step_fn = None,
    #                     T_range = range(0, 500), # range(0, 200),
    #                     deduce_phase_from_lr_dual=False,
    #                     log_dict = {},
    #                     test_log = [self.cd_learner.make_test_log_fn(P_max = sa_model.P_max), sa_learner.test_log]
    #                     )
            
    #         test_log_dict = {f"{self.__class__.__name__}-Test/{k}": v for k, v in test_log_dict.items()}

    #         accelerator.print("Logging test results to wandb.")
    #         accelerator.log(test_log_dict)





    def test_transferability(self, accelerator, args, arg_groups, cd_model, cd_optimizer, cd_lr_sched, channel_dataloader,
              n_epochs = [100], n_iters_cd_per_epoch = 100, device = 'cpu', loggers = None, sa_model = None,
               **kwargs):
        
        """ 
        This function has the same signature as the train function, but it is used for testing transferability of the model.
        It will not train the model, but it will evaluate the model on the transferability set and log the results.
        It will also save the transferability metrics to the specified path.
        The transferability metrics will be saved in a dictionary format with keys as the metric names and values as the metric values.
        The transferability metrics will be logged to wandb if wandb is enabled.
        """

        save_train_chkpt_path = kwargs.get('save_train_chkpt_path', None)
        load_train_chkpt_path = kwargs.get('load_train_chkpt_path', None)
        save_test_metrics_path = kwargs.get('save_test_metrics_path', None)

        sweep_train_chkpts = kwargs.get('sweep_train_chkpts', False)
    

        if wandb.run is not None:
            # wandb.define_metric(name="my_wandb_histogram", summary=None)
            wandb.define_metric(name="train-step")
            wandb.define_metric(name="train-epoch", step_metric="train-step", summary=None)
            wandb.define_metric(name=f"{self.cd_learner.__class__.__name__}-Training/*", step_metric="train-epoch", summary=None)
            # wandb.define_metric(name="StateAugmentation-Training/*", step_metric="epoch", summary="min")

        # cd_optimizer, cd_lr_sched = None, None
        if cd_optimizer is None or cd_lr_sched is None:
            accelerator.print("Creating the optimizer and lr scheduler for the interference model if they do not exist.")
            cd_optimizer, cd_lr_sched = self.cd_learner.init_optimizer_and_lr_scheduler(
                model=cd_model, optimizer=cd_optimizer, lr_sched=cd_lr_sched
            )


        n_epochs_orig = copy.deepcopy(n_epochs)
        
        if load_train_chkpt_path not in [None, "None", "", 'none'] and sweep_train_chkpts:
            # Find all the chkpt paths in the same directory as the load_train_chkpt_path where epoch number is greater or equal to that of the load_train_chkpt_path
            load_train_chkpt_paths = os.listdir(os.path.dirname(load_train_chkpt_path))
            load_train_chkpt_paths = [os.path.join(os.path.dirname(load_train_chkpt_path), f) for f in load_train_chkpt_paths if f.endswith('.pt')]
            
            load_train_chkpt_paths = [f for f in load_train_chkpt_paths if int(f.split('_')[-1].split('.')[0]) >= int(load_train_chkpt_path.split('_')[-1].split('.')[0])]
            load_train_chkpt_paths = sorted(load_train_chkpt_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        else:
            pass

        accelerator.print(f"Found {len(load_train_chkpt_paths)} training checkpoints in the directory {os.path.dirname(load_train_chkpt_path)} with chkpt-epoch larger than {int(load_train_chkpt_path.split('_')[-1].split('.')[0])}.")
        if len(load_train_chkpt_paths) == 0:
            accelerator.print(f"No training checkpoints found in the directory {os.path.dirname(load_train_chkpt_path)} with chkpt-epoch larger than {int(load_train_chkpt_path.split('_')[-1].split('.')[0])}.")
            load_train_chkpt_paths = [load_train_chkpt_path]
        else:
            pass
        
        
        for load_train_chkpt_path in load_train_chkpt_paths:
            if load_train_chkpt_path not in [None, "None", "", 'none']:

                cd_model, cd_optimizer, cd_lr_sched, n_epochs = self.resume_training_from_chkpt(accelerator = accelerator,
                                                        cd_model = cd_model, 
                                                        cd_optimizer = cd_optimizer, 
                                                        cd_lr_sched = cd_lr_sched,
                                                        load_cd_train_chkpt_path = load_train_chkpt_path,
                                                        n_epochs = n_epochs_orig,
                                                        # dr_dataloader = dr_dataloader
                                                        )
                n_iters = n_epochs[0] * n_iters_cd_per_epoch


            epoch = n_epochs[0] - 1

            for transferability_phase in [phase for phase in channel_dataloader.keys() if phase.startswith('transferability_test')]:
                self.test_train_chkpt(accelerator=accelerator, args=args, arg_groups=arg_groups,
                                    cd_model=cd_model, dataloader=channel_dataloader,
                                    sa_model=sa_model, device=device,
                                    save_test_metrics_path=save_test_metrics_path, epoch=epoch, phase=transferability_phase)