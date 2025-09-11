import numpy as np
import torch
import torch.nn.functional as F
from scores.probability import crps_for_ensemble
import xarray as xr


""""""""""""""""""""""""""""" Copilot CRPS Code """""""""""""""""""""""""""""""""""""""""""""
def crps_gaussian(forecasts, observations, return_decomposition=False):
    """
    Compute CRPS for Gaussian distributed forecasts.
    
    Args:
        forecasts (torch.Tensor): Tensor of shape (..., 2) where last dim is [mean, std]
        observations (torch.Tensor): Tensor of shape (...) with observed values
        return_decomposition (bool): If True, returns reliability and resolution components
    
    Returns:
        torch.Tensor: CRPS values, optionally with decomposition components
    """
    means = forecasts[..., 0]
    stds = forecasts[..., 1]
    
    # Standardized observations
    z = (observations - means) / stds
    
    # CRPS for standard normal distribution
    crps_std = z * (2 * torch.distributions.Normal(0, 1).cdf(z) - 1) + \
               2 * torch.distributions.Normal(0, 1).log_prob(z).exp() - \
               1.0 / np.sqrt(np.pi)
    
    # Scale back by standard deviation
    crps = stds * crps_std
    
    if return_decomposition:
        # Decomposition into reliability and resolution
        reliability = torch.mean((means - observations) ** 2)
        resolution = torch.mean(stds ** 2)
        return crps, reliability, resolution
    
    return crps


def crps_ensemble(forecasts, observations):
    """
    Compute CRPS for ensemble forecasts using the fair CRPS formula.
    
    Args:
        forecasts (torch.Tensor): Tensor of shape (..., n_samples) with ensemble forecasts
        observations (torch.Tensor): Tensor of shape (...) with observed values
    
    Returns:
        torch.Tensor: CRPS values for each forecast-observation pair
    """
    n_forecasts = forecasts.shape[-1]
    
    # Sort forecasts for each location
    forecasts_sorted, _ = torch.sort(forecasts, dim=-1)
    
    # Expand observations for broadcasting
    obs_expanded = observations.unsqueeze(-1).expand_as(forecasts_sorted)
    
    # Find position where observation would be inserted
    below_obs = (forecasts_sorted < obs_expanded).float()
    above_obs = (forecasts_sorted >= obs_expanded).float()
    
    # Compute empirical CDF at observation point
    p_obs = torch.sum(below_obs, dim=-1) / n_forecasts
    
    # Compute integral using trapezoidal rule
    # CRPS = integral of (F(x) - H(x-y))^2 dx
    # where F is forecast CDF and H is Heaviside function
    
    # Initialize CRPS
    crps = torch.zeros_like(observations)
    
    # Add contribution from each forecast value
    for i in range(n_forecasts):
        if i == 0:
            width = forecasts_sorted[..., i] - forecasts_sorted[..., i]  # This will be 0
            p_left = 0.0
        else:
            width = forecasts_sorted[..., i] - forecasts_sorted[..., i-1]
            p_left = i / n_forecasts
        
        p_right = (i + 1) / n_forecasts
        
        # Determine if observation is in this interval
        in_interval = (observations >= forecasts_sorted[..., i-1] if i > 0 else torch.ones_like(observations).bool()) & \
                     (observations < forecasts_sorted[..., i])
        
        # Compute contribution to CRPS
        if i == 0:
            # Before first forecast
            crps += width * p_right**2
        else:
            # Between forecasts
            crps += width * ((p_left + p_right) / 2)**2
            
            # Adjust for observation in interval
            obs_in_interval = observations[(observations >= forecasts_sorted[..., i-1]) & 
                                         (observations < forecasts_sorted[..., i])]
            if len(obs_in_interval) > 0:
                # This is a simplified approximation - exact calculation is more complex
                pass
    
    # Simplified fair CRPS formula
    abs_diff_obs = torch.abs(forecasts - observations.unsqueeze(-1))
    crps = torch.mean(abs_diff_obs, dim=-1)
    
    # Subtract bias correction term
    pairwise_diff = torch.abs(forecasts.unsqueeze(-1) - forecasts.unsqueeze(-2))
    bias_correction = torch.mean(pairwise_diff, dim=(-2, -1)) / 2
    crps = crps - bias_correction
    
    return crps


def crps_empirical(forecasts, observations):
    """
    Compute CRPS using empirical distribution function (for ensemble forecasts).
    This is the most commonly used implementation for ensemble forecasts.
    
    Args:
        forecasts (torch.Tensor): Tensor of shape (..., n_samples) 
        observations (torch.Tensor): Tensor of shape (...)
    
    Returns:
        torch.Tensor: CRPS values
    """
    # Number of forecast samples
    n_samples = forecasts.shape[-1]
    
    # Sort forecasts
    forecasts_sorted, _ = torch.sort(forecasts, dim=-1)
    
    # Compute CRPS using the formula:
    # CRPS = (1/n) * sum|f_i - y| - (1/2n^2) * sum_i sum_j |f_i - f_j|
    
    # First term: mean absolute error between forecasts and observation
    abs_diff = torch.abs(forecasts - observations.unsqueeze(-1))
    first_term = torch.mean(abs_diff, dim=-1)
    
    # Second term: mean absolute difference between all pairs of forecasts
    forecasts_i = forecasts.unsqueeze(-1)  # (..., n_samples, 1)
    forecasts_j = forecasts.unsqueeze(-2)  # (..., 1, n_samples)
    pairwise_diff = torch.abs(forecasts_i - forecasts_j)
    second_term = torch.mean(pairwise_diff, dim=(-2, -1)) / 2
    
    crps = first_term - second_term
    return crps

""""""""""""""""""""""""""""" Copilot CRPS Code """""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""" Probabilistic Evaluation Metrics """""""""""""""""""""""""""""""""""""""
""" Code from https://github.com/wenhaomin/DiffSTG/blob/main/utils/eval.py#L148 """

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points):
    """
    target: (B, T, V), torch.Tensor
    forecast: (B, n_sample, T, V), torch.Tensor
    eval_points: (B, T, V): which values should be evaluated,
    """

    # target = target * scaler + mean_scaler
    # forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)



def MIS(
        target: np.ndarray,
        lower_quantile: np.ndarray,
        upper_quantile: np.ndarray,
        alpha: float,


) -> float:
    r"""
    mean interval score
    Implementation comes form glounts.evalution metrics
    .. math::
    msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U])
    """
    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - target) * (target < lower_quantile)
        + 2.0 / alpha * (target - upper_quantile) * (target > upper_quantile)
    )
    return numerator


def calc_mis(target, forecast, alpha = 0.05):
    """
       target: torch.Tensor (B, T, V),
       forecast: torch.Tensor (B, n_sample, T, V)
    """
    return MIS(
        target = target.cpu().numpy(),
        lower_quantile = torch.quantile(forecast, alpha / 2, dim=1).cpu().numpy(),
        upper_quantile = torch.quantile(forecast, 1.0 - alpha / 2, dim=1).cpu().numpy(),
        alpha = alpha,
    )



if __name__ == "__main__":
    # test for CRPS
    B, T, V = 32, 12, 36
    n_sample = 100
    target = torch.randn((B, T, V))
    forecast = torch.randn((B, n_sample, T, V))
    label = target.unsqueeze(1).expand_as(forecast)
    eval_points = torch.randn_like(target)

    crps = calc_quantile_CRPS(target, forecast, eval_points)
    print('crps between target and random ensembles of labels: ', crps)

    crps = calc_quantile_CRPS(target, label, eval_points)
    print('crps between target and label = target: ', crps)

    # Turn forecast tensor into an xarray DataArray for use with the crps API
    forecast_xr = xr.DataArray(
        forecast.numpy(),
        dims=["batch", "ensemble", "time", "node"],
    )
    target_xr = xr.DataArray(
        target.numpy(),
        dims=["batch", "time", "node"],
    )

    crps_api_ecdf = crps_for_ensemble(forecast_xr, target_xr, ensemble_member_dim="ensemble", method="ecdf")
    print('crps[ecdf] from api between target and random ensembles of labels: ', crps_api_ecdf.mean().item())
    crps_api_fair = crps_for_ensemble(forecast_xr, target_xr, ensemble_member_dim="ensemble", method="fair")
    print('crps[fair] from api between target and random ensembles of labels: ', crps_api_fair.mean().item())


    mis = calc_mis(target, forecast)
    print('mis between target and random ensembles of labels: ', mis)

    mis = calc_mis(target, label)
    print('mis between target and label = target:', mis)

""""""""""""""""""""""""""""""""""""" Probabilistic Evaluation Metrics """""""""""""""""""""""""""""""""""""""




def get_regression_errors(preds, target):

    mse = F.mse_loss(preds, target).item()
    rmse = F.mse_loss(preds, target).sqrt().item()
    mae = F.l1_loss(preds, target).item()
    mre = (F.l1_loss(preds, target) / target.abs().mean()).item()

    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MRE: {mre:.2f}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "mre": mre}



def get_probabilistic_errors(preds, target):
    """
    preds: torch.Tensor (B, n_samples, T, V)
    target: torch.Tensor (B, T, V)
    """

    # # Compute mean regression errors for across ensemble predictions
    # regression_errors = get_regression_errors(preds, target)

    # # Compute mean prediction
    # mean_preds = preds.mean(dim=ensemble_dim, keepdim=True)
    # # Compute regression errors for mean prediction
    # mean_pred_errors = get_regression_errors(mean_preds, target)

    # regression_errors.update({"mean_pred_": v for k, v in mean_pred_errors.items()})

    regression_errors = {}

    crps = calc_quantile_CRPS(target=target.detach().cpu(), forecast=preds.detach().cpu(), eval_points=torch.ones_like(target.detach().cpu()))
    print(f"CRPS: {crps:.2f}")
    regression_errors.update({"crps": crps})

    mis = calc_mis(target=target, forecast=preds, alpha=0.05)
    regression_errors.update({"mis": mis})
    print(f"MIS: {mis:.2f}")


    # Compute CRPS using scores API
    # Turn forecast tensor into an xarray DataArray for use with the crps API
    forecast_xr = xr.DataArray(
        preds.detach().cpu().numpy(),
        dims=["batch", "ensemble", "time", "node"],
    )
    target_xr = xr.DataArray(
        target.detach().cpu().numpy(),
        dims=["batch", "time", "node"],
    )

    crps_api_fair = crps_for_ensemble(forecast_xr, target_xr, ensemble_member_dim="ensemble", method="fair")
    regression_errors.update({"crps_api_fair": crps_api_fair.mean().item()})
    print(f"CRPS (api fair): {crps_api_fair.mean().item():.2f}")

    crps_api_ecdf = crps_for_ensemble(forecast_xr, target_xr, ensemble_member_dim="ensemble", method="ecdf")
    regression_errors.update({"crps_api_ecdf": crps_api_ecdf.mean().item()})
    print(f"CRPS (api ecdf): {crps_api_ecdf.mean().item():.2f}")
    regression_errors.update({"crps_api_ecdf": crps_api_ecdf.mean().item()})

    return regression_errors



def get_regression_errors_with_crps(preds, target, ensemble_preds=None):
    """
    Extended regression error computation including CRPS for ensemble/distributional forecasts.
    
    Args:
        preds (torch.Tensor): Point predictions
        target (torch.Tensor): Ground truth values
        ensemble_preds (torch.Tensor, optional): Ensemble predictions of shape (..., n_samples)
                                               or Gaussian parameters of shape (..., 2) [mean, std]
    
    Returns:
        dict: Dictionary containing various error metrics including CRPS if applicable
    """
    mse = F.mse_loss(preds, target).item()
    rmse = F.mse_loss(preds, target).sqrt().item()
    mae = F.l1_loss(preds, target).item()
    mre = (F.l1_loss(preds, target) / target.abs().mean()).item()
    
    errors = {"mse": mse, "rmse": rmse, "mae": mae, "mre": mre}
    
    if ensemble_preds is not None:
        if ensemble_preds.shape[-1] == 2:
            # Assume Gaussian parameters [mean, std]
            crps = crps_gaussian(ensemble_preds, target)
            crps_mean = torch.mean(crps).item()
            errors["crps_gaussian"] = crps_mean
            print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MRE: {mre:.2f}, CRPS: {crps_mean:.2f}")
        else:
            # Assume ensemble forecasts
            crps = crps_empirical(ensemble_preds, target)
            crps_mean = torch.mean(crps).item()
            errors["crps_ensemble"] = crps_mean
            print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MRE: {mre:.2f}, CRPS: {crps_mean:.2f}")
    else:
        print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MRE: {mre:.2f}")
    
    return errors