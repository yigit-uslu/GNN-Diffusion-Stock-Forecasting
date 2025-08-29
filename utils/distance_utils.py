import numpy as np
from scipy.stats import wasserstein_distance



def kl_divergence_from_samples(samples1, samples2, bins=100, eps=1e-10):
    """Calculate KL divergence between two sample sets using histograms"""
    # Find common range
    min_val = min(np.min(samples1), np.min(samples2))
    max_val = max(np.max(samples1), np.max(samples2))
    
    # Create histogram bins
    bin_edges = np.linspace(min_val, max_val, bins+1)
    
    # Compute histograms
    hist1, _ = np.histogram(samples1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(samples2, bins=bin_edges, density=True)
    
    # Add small epsilon to avoid log(0)
    hist1 = hist1 + eps
    hist2 = hist2 + eps
    
    # Normalize
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Compute KL divergence
    kl = np.sum(hist1 * np.log(hist1 / hist2))
    
    return kl



def kl_divergence_per_dimension(samples1, samples2, bins=100, eps=1e-10):
    """
    Calculate KL divergence between two sample sets per dimension
    
    Args:
        samples1: Array with shape (n_samples, n_dimensions)
        samples2: Array with shape (m_samples, n_dimensions)
        bins: Number of histogram bins
        eps: Small value to avoid log(0)
        
    Returns:
        Array of KL divergences, one per dimension
    """
    # Make sure inputs are at least 2D arrays
    samples1 = np.atleast_2d(samples1)
    samples2 = np.atleast_2d(samples2)
    
    assert samples1.shape[1] == samples2.shape[1], "Samples must have the same number of dimensions"
    
    n_dims = samples1.shape[1]
    kl_divs = np.zeros(n_dims)
    
    for dim in range(n_dims):
        kl_divs[dim] = kl_divergence_from_samples(
            samples1[:, dim], samples2[:, dim], bins=bins, eps=eps)
    
    return kl_divs


def compute_wasserstein(samples1, samples2):
    """Compute Wasserstein distance between two sample sets"""
    return wasserstein_distance(samples1, samples2)


def wasserstein_per_dimension(samples1, samples2):
    """
    Compute Wasserstein distance between two sample sets per dimension
    
    Args:
        samples1: Array with shape (n_samples, n_dimensions)
        samples2: Array with shape (m_samples, n_dimensions)
        
    Returns:
        Array of Wasserstein distances, one per dimension
    """
    # Make sure inputs are at least 2D arrays
    samples1 = np.atleast_2d(samples1)
    samples2 = np.atleast_2d(samples2)
    
    assert samples1.shape[1] == samples2.shape[1], "Samples must have the same number of dimensions"
    
    n_dims = samples1.shape[1]
    w_dists = np.zeros(n_dims)
    
    for dim in range(n_dims):
        w_dists[dim] = wasserstein_distance(samples1[:, dim], samples2[:, dim])
    
    return w_dists