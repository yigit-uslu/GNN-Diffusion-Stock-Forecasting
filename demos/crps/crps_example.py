"""
CRPS (Continuous Ranked Probability Score) Implementation and Examples
====================================================================

This file demonstrates the implementation and usage of CRPS for evaluating
time-series forecasting models, particularly useful for diffusion models
that generate distributional forecasts.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats


def crps_gaussian(forecasts, observations, return_decomposition=False):
    """
    Compute CRPS for Gaussian distributed forecasts.
    
    This is the most efficient method when your model outputs Gaussian parameters.
    
    Args:
        forecasts (torch.Tensor): Tensor of shape (..., 2) where last dim is [mean, std]
        observations (torch.Tensor): Tensor of shape (...) with observed values
        return_decomposition (bool): If True, returns reliability and resolution components
    
    Returns:
        torch.Tensor: CRPS values, optionally with decomposition components
    
    Mathematical formula:
    For a Gaussian N(μ, σ²), CRPS(F, y) = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    where z = (y - μ)/σ, Φ is CDF, φ is PDF of standard normal.
    """
    means = forecasts[..., 0]
    stds = forecasts[..., 1]
    
    # Standardized observations
    z = (observations - means) / stds
    
    # Standard normal distribution
    normal = torch.distributions.Normal(0, 1)
    
    # CRPS for standard normal distribution
    crps_std = z * (2 * normal.cdf(z) - 1) + \
               2 * normal.log_prob(z).exp() - \
               1.0 / np.sqrt(np.pi)
    
    # Scale back by standard deviation
    crps = stds * crps_std
    
    if return_decomposition:
        # Decomposition into reliability and resolution
        reliability = torch.mean((means - observations) ** 2)
        resolution = torch.mean(stds ** 2)
        return crps, reliability, resolution
    
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
    
    Formula:
    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent copies of the forecast distribution
    """
    # Number of forecast samples
    n_samples = forecasts.shape[-1]
    
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


def demonstrate_crps_properties():
    """
    Demonstrate key properties of CRPS with examples.
    """
    print("=" * 60)
    print("CRPS PROPERTIES DEMONSTRATION")
    print("=" * 60)
    
    # Create synthetic data
    n_samples = 1000
    true_values = torch.randn(n_samples)
    
    # Example 1: Perfect forecast (deterministic)
    print("\n1. Perfect Forecast (deterministic):")
    perfect_forecast = true_values.unsqueeze(-1).repeat(1, 50)  # Ensemble of identical values
    crps_perfect = crps_empirical(perfect_forecast, true_values)
    print(f"   CRPS (should be ~0): {torch.mean(crps_perfect):.6f}")
    
    # Example 2: Biased forecast
    print("\n2. Biased Forecast:")
    biased_forecast = (true_values + 1.0).unsqueeze(-1).repeat(1, 50)  # Add bias
    crps_biased = crps_empirical(biased_forecast, true_values)
    print(f"   CRPS (should be ~1): {torch.mean(crps_biased):.6f}")
    
    # Example 3: Unbiased but overly uncertain
    print("\n3. Unbiased but Overly Uncertain:")
    uncertain_forecast = true_values.unsqueeze(-1) + torch.randn(n_samples, 50) * 2.0
    crps_uncertain = crps_empirical(uncertain_forecast, true_values)
    print(f"   CRPS (higher due to spread): {torch.mean(crps_uncertain):.6f}")
    
    # Example 4: Underconfident (too narrow)
    print("\n4. Underconfident (too narrow):")
    narrow_forecast = true_values.unsqueeze(-1) + torch.randn(n_samples, 50) * 0.1
    crps_narrow = crps_empirical(narrow_forecast, true_values)
    print(f"   CRPS: {torch.mean(crps_narrow):.6f}")
    
    # Compare with MAE for deterministic forecasts
    print("\n5. Comparison with MAE:")
    point_forecast = true_values + torch.randn(n_samples) * 0.5
    mae = torch.mean(torch.abs(point_forecast - true_values))
    # For deterministic forecast, CRPS = MAE
    crps_det = crps_empirical(point_forecast.unsqueeze(-1), true_values)
    print(f"   MAE: {mae:.6f}")
    print(f"   CRPS (deterministic): {torch.mean(crps_det):.6f}")
    print("   Note: For deterministic forecasts, CRPS ≈ MAE")


def demonstrate_gaussian_vs_empirical():
    """
    Compare Gaussian CRPS vs Empirical CRPS for Gaussian forecasts.
    """
    print("\n" + "=" * 60)
    print("GAUSSIAN vs EMPIRICAL CRPS COMPARISON")
    print("=" * 60)
    
    # Generate true observations
    n_obs = 500
    true_values = torch.randn(n_obs)
    
    # Generate Gaussian forecast parameters
    forecast_means = true_values + torch.randn(n_obs) * 0.2  # Slightly biased
    forecast_stds = torch.ones(n_obs) * 0.8  # Slightly underconfident
    
    # Method 1: Gaussian CRPS (analytical)
    gaussian_params = torch.stack([forecast_means, forecast_stds], dim=-1)
    crps_gaussian_analytical = crps_gaussian(gaussian_params, true_values)
    
    # Method 2: Empirical CRPS (sample from Gaussian)
    n_samples = 1000
    samples = torch.normal(
        forecast_means.unsqueeze(-1).expand(-1, n_samples),
        forecast_stds.unsqueeze(-1).expand(-1, n_samples)
    )
    crps_gaussian_empirical = crps_empirical(samples, true_values)
    
    print(f"Gaussian CRPS (analytical): {torch.mean(crps_gaussian_analytical):.6f}")
    print(f"Gaussian CRPS (empirical):  {torch.mean(crps_gaussian_empirical):.6f}")
    print(f"Difference: {torch.mean(torch.abs(crps_gaussian_analytical - crps_gaussian_empirical)):.6f}")
    print("Note: Analytical should be nearly identical to empirical with enough samples")


def stock_forecasting_example():
    """
    Example of using CRPS for stock return forecasting with diffusion models.
    """
    print("\n" + "=" * 60)
    print("STOCK FORECASTING EXAMPLE")
    print("=" * 60)
    
    # Simulate stock returns (log returns)
    n_stocks = 100
    n_days = 252  # 1 year
    
    # True returns (slightly trending with volatility clustering)
    torch.manual_seed(42)
    true_returns = torch.cumsum(torch.randn(n_stocks, n_days) * 0.02, dim=-1)
    
    # Simulate three different forecasting models
    
    # Model 1: Simple point forecast (baseline)
    print("\n1. Simple Point Forecast Model:")
    point_forecast = true_returns + torch.randn_like(true_returns) * 0.05
    mae_simple = torch.mean(torch.abs(point_forecast - true_returns))
    crps_simple = crps_empirical(point_forecast.unsqueeze(-1), true_returns)
    print(f"   MAE: {mae_simple:.6f}")
    print(f"   CRPS: {torch.mean(crps_simple):.6f}")
    
    # Model 2: Gaussian diffusion model (predicts mean and variance)
    print("\n2. Gaussian Diffusion Model:")
    forecast_mean = true_returns + torch.randn_like(true_returns) * 0.03
    forecast_std = torch.ones_like(true_returns) * 0.04  # Learned uncertainty
    gaussian_forecast = torch.stack([forecast_mean, forecast_std], dim=-1)
    crps_normal = crps_gaussian(gaussian_forecast, true_returns)
    mae_normal = torch.mean(torch.abs(forecast_mean - true_returns))
    print(f"   MAE (point): {mae_normal:.6f}")
    print(f"   CRPS: {torch.mean(crps_normal):.6f}")

    # Model 3: Full ensemble from generative model
    print("\n3. Ensemble Diffusion Model:")
    n_ensemble = 100
    ensemble_forecast = forecast_mean.unsqueeze(-1) + \
                       torch.randn(n_stocks, n_days, n_ensemble) * forecast_std.unsqueeze(-1)
    crps_ensemble = crps_empirical(ensemble_forecast, true_returns)
    mae_ensemble = torch.mean(torch.abs(torch.mean(ensemble_forecast, dim=-1) - true_returns))
    print(f"   MAE (ensemble mean): {mae_ensemble:.6f}")
    print(f"   CRPS: {torch.mean(crps_ensemble):.6f}")
    
    print("\nInterpretation:")
    print("- Lower CRPS indicates better probabilistic forecasts")
    print("- CRPS accounts for both accuracy and calibration")
    print("- For well-calibrated models, CRPS should be lower than MAE")


def plot_crps_calibration():
    """
    Visualize CRPS calibration properties.
    """
    print("\n" + "=" * 60)
    print("CRPS CALIBRATION VISUALIZATION")
    print("=" * 60)
    
    # Generate data
    n_obs = 1000
    true_values = torch.randn(n_obs)
    
    # Test different forecast uncertainties
    std_values = torch.linspace(0.1, 3.0, 20)
    crps_values = []
    
    for std in std_values:
        # Forecast with this uncertainty level
        forecast_params = torch.stack([
            true_values,  # Perfect mean
            torch.ones_like(true_values) * std
        ], dim=-1)
        
        crps = crps_gaussian(forecast_params, true_values)
        crps_values.append(torch.mean(crps).item())
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(std_values.numpy(), crps_values, 'b-', linewidth=2, label='CRPS')
    plt.axhline(y=1/np.sqrt(np.pi), color='r', linestyle='--', 
                label='Optimal CRPS (σ=1)')
    plt.xlabel('Forecast Standard Deviation')
    plt.ylabel('CRPS')
    plt.title('CRPS vs Forecast Uncertainty\n(Perfect mean, varying std)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/berkay/GNN-Diffusion-Stock-Forecasting/demos/crps/crps_calibration.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Calibration plot saved as 'crps_calibration.png'")
    print("Key insight: CRPS is minimized when forecast uncertainty matches true uncertainty")


if __name__ == "__main__":
    print("CRPS (Continuous Ranked Probability Score) for Time-Series Evaluation")
    print("=" * 70)
    
    demonstrate_crps_properties()
    demonstrate_gaussian_vs_empirical()
    stock_forecasting_example()
    plot_crps_calibration()
    
    print("\n" + "=" * 70)
    print("SUMMARY: WHEN TO USE CRPS")
    print("=" * 70)
    print("✓ Evaluating probabilistic/distributional forecasts")
    print("✓ Comparing models with different uncertainty estimates")
    print("✓ Assessing forecast calibration")
    print("✓ Time-series with inherent uncertainty")
    print("✓ Diffusion models and other generative forecasting methods")
    print()
    print("Key advantages over traditional metrics:")
    print("• Rewards well-calibrated uncertainty")
    print("• Proper scoring rule (encourages honest forecasts)")
    print("• Reduces to MAE for deterministic forecasts")
    print("• Sensitive to both bias and dispersion")
