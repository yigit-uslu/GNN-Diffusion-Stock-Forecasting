
import abc
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


class GeometricRandomWalk(abc.ABC):
    def __init__(self, mu, sigma, dt = 0.01, t0 = 0, t1 = 1):
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.t0 = t0
        self.t1 = t1
        self.N = int((t1 - t0) / dt)
        self.t = np.arange(1, self.N + 1)

    @property
    def drift(self):
        return (self.mu - 0.5 * self.sigma ** 2) * self.dt

    @property
    def diffusion(self):
        return self.sigma * (self.dt ** 0.5)
    

    def brownian_increment(self, num_paths):
        # Generate independent increments (not cumulative!)
        return np.random.normal(0.0, 1.0, size = (num_paths, self.N))
    

    def generate_paths(self, num_paths, initial_log_return = None, return_type: str = 'prices'):
        
        dW = self.brownian_increment(num_paths)
        # Generate log-increments for each time step
        log_increments = self.drift + self.diffusion * dW

        if return_type == 'log-increments':
            return log_increments
        
        else:
            if initial_log_return is not None:
                if np.isscalar(initial_log_return):
                    log_increments = np.column_stack([np.full((num_paths, 1), initial_log_return), log_increments])
                else:
                    if len(initial_log_return) != num_paths:
                        initial_log_return = np.repeat(initial_log_return[0], num_paths, axis = 0)
                    assert len(initial_log_return) == num_paths, "Initial log return length must match number of paths"
                    log_increments = np.column_stack([initial_log_return, log_increments])
            
            else:
                # No initial log return, is like starting from log(1) = 0
                pass

            if return_type == 'log-returns':
                return self.log_increments_to_log_returns(log_increments)
        
            elif return_type == 'prices':
                return self.log_returns_to_returns(self.log_increments_to_log_returns(log_increments))
            
            else:
                raise ValueError("Invalid return_type. Choose from 'prices', 'log-returns', 'log-increments'.")
        
    
    def log_increments_to_log_returns(self, log_increments, time_dim: int = 1):
        return np.cumsum(log_increments, axis=time_dim)
    
    def log_returns_to_returns(self, log_returns):
        return np.exp(log_returns)


    def fit(self, paths, paths_type: str = 'prices'):
        """
        Estimate mu and sigma from observed paths.
        
        Args:
            paths (array): Shape (num_paths, N_time_steps + 1)
            
        Returns:
            dict: Estimated parameters
        """

        if paths_type == 'log-increments':
            log_increments = paths # Assume paths are already log increments
            assert np.isnan(log_increments).sum() == 0, "Input log increments contain NaN values. {} paths: {}, log_increments: {}".format(paths_type, paths, log_increments)

        elif paths_type == 'log-returns':
            # log_increments = paths - np.column_stack([np.zeros(paths.shape[0]), paths[:, 1:]]) # Assume paths are already log returns
            log_increments = np.diff(paths, axis=1)
            assert np.isnan(log_increments).sum() == 0, "Input log increments contain NaN values. {} paths: {}, log_increments: {}".format(paths_type, paths, log_increments)

        elif paths_type == 'prices':
            # Calculate log increments from prices
            log_increments = np.diff(np.log(paths), axis=1) # np.log(paths[:, 1:]) - np.log(paths[:, :-1])
            assert np.isnan(log_increments).sum() == 0, "Input log increments contain NaN values. {} paths: {}, log_increments: {}".format(paths_type, paths, log_increments)

        # Flatten to 1D array
        log_increments_flat = log_increments.flatten()
        
        # Estimate parameters
        mean_log_increment = np.mean(log_increments_flat)
        var_log_increment = np.var(log_increments_flat)

        estimated_sigma = np.sqrt(var_log_increment / self.dt)
        estimated_mu = mean_log_increment / self.dt + 0.5 * estimated_sigma ** 2

        # Validate estimated parameters with NaN checks
        assert not np.isnan(estimated_mu), "Estimated mu is NaN"
        assert not np.isnan(estimated_sigma), "Estimated sigma is NaN"
        assert estimated_sigma >= 0, "Estimated sigma must be non-negative"
        # Note: mu can be negative in GBM, so no assertion needed
        
        return {
            'mu': estimated_mu,
            'sigma': estimated_sigma
        }
    

    def plot_sample_paths(self, paths, num_paths_to_plot=5, save_path=None):
        """
        Plot sample paths for visualization.
        
        Args:
            paths (array): Output from generate_paths
            num_paths_to_plot (int): Number of sample paths to plot
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        # time_grid = np.linspace(self.t0, self.t1,
        #                         len(paths[0]), # self.N + 1
        #                         )
        time_grid = np.linspace(start = self.t0, stop = self.t1, num = self.N + 1)
        # print("Delta_t, t_0, t_1: ", delta_t := time_grid[1] - time_grid[0], time_grid[0], time_grid[-1])


        for i in range(min(num_paths_to_plot, paths.shape[0])):
            plt.plot(time_grid[-len(paths[i]):], paths[i], alpha=0.7, linewidth=1)

        plt.title(f'Geometric Random Walk Sample Paths: μ={self.mu}, σ={self.sigma}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    

class CorrelatedGeometricRandomWalk(GeometricRandomWalk):
    """
    Simulates multiple correlated Geometric Random Walks using a covariance matrix.
    
    For N assets, the dynamics are:
    dlog(S_i) = (mu_i - 0.5*sigma_i^2)*dt + sigma_i * dW_i
    
    where dW follows a multivariate normal with correlation matrix rho.
    The covariance matrix is: Sigma = D * Rho * D
    where D is diagonal matrix of volatilities.
    """
    
    def __init__(self, mu_vector, sigma_vector, correlation_matrix=None, covariance_matrix=None, 
                 dt=0.01, t0=0, t1=1):
        """
        Initialize correlated GRW.
        
        Args:
            mu_vector (array): Vector of drift parameters [mu_1, mu_2, ..., mu_N]
            sigma_vector (array): Vector of volatilities [sigma_1, sigma_2, ..., sigma_N]
            correlation_matrix (array, optional): N x N correlation matrix
            covariance_matrix (array, optional): N x N covariance matrix (alternative to correlation)
            dt (float): Time step
            t0 (float): Start time
            t1 (float): End time
        """
        # Convert to numpy arrays
        self.mu_vector = np.array(mu_vector)
        self.sigma_vector = np.array(sigma_vector)
        self.n_assets = len(self.mu_vector)
        
        # Validate inputs
        assert len(self.sigma_vector) == self.n_assets, "mu and sigma vectors must have same length"
        
        # Set up covariance matrix
        if covariance_matrix is not None:
            self.covariance_matrix = np.array(covariance_matrix)
            assert self.covariance_matrix.shape == (self.n_assets, self.n_assets), \
                "Covariance matrix must be N x N"
            # Extract correlation matrix and volatilities
            std_matrix = np.sqrt(np.diag(self.covariance_matrix))
            self.correlation_matrix = self.covariance_matrix / np.outer(std_matrix, std_matrix)
            # Update sigma_vector from covariance matrix diagonal
            self.sigma_vector = std_matrix
        elif correlation_matrix is not None:
            self.correlation_matrix = np.array(correlation_matrix)
            assert self.correlation_matrix.shape == (self.n_assets, self.n_assets), \
                "Correlation matrix must be N x N"
            # Construct covariance matrix: Sigma = D * Rho * D
            D = np.diag(self.sigma_vector)
            self.covariance_matrix = D @ self.correlation_matrix @ D
        else:
            # Default to identity (independent assets)
            self.correlation_matrix = np.eye(self.n_assets)
            self.covariance_matrix = np.diag(self.sigma_vector ** 2)
        
        # Validate correlation matrix properties
        assert np.allclose(self.correlation_matrix, self.correlation_matrix.T), \
            "Correlation matrix must be symmetric but got {}".format(self.correlation_matrix)
        eigenvals = np.linalg.eigvals(self.correlation_matrix)
        assert np.all(eigenvals >= -1e-8), "Correlation matrix must be positive semi-definite but got eigenvalues {}".format(eigenvals)

        # Validate covariance matrix properties
        assert np.allclose(self.covariance_matrix, self.covariance_matrix.T), \
            "Covariance matrix must be symmetric but got {}".format(self.covariance_matrix)
        eigenvals_cov = np.linalg.eigvals(self.covariance_matrix)
        assert np.all(eigenvals_cov >= -1e-8), "Covariance matrix must be positive semi-definite but got eigenvalues {}".format(eigenvals_cov)


        # Store time parameters
        self.dt = dt
        self.t0 = t0
        self.t1 = t1
        self.N = int((t1 - t0) / dt)
        self.t = np.arange(1, self.N + 1)
        
        # For compatibility with parent class (use first asset's parameters)
        super().__init__(self.mu_vector[0], self.sigma_vector[0], dt, t0, t1)

    @property
    def drift_vector(self):
        """Vector of drift terms for each asset."""
        return (self.mu_vector - 0.5 * self.sigma_vector ** 2) * self.dt

    def correlated_brownian_increments(self, num_paths):
        """
        Generate correlated Brownian motion increments using Cholesky decomposition.
        
        Args:
            num_paths (int): Number of simulation paths
            
        Returns:
            array: Shape (num_paths, N_time_steps, N_assets) of correlated increments
        """
        # Generate independent standard normal random variables
        Z = np.random.normal(0.0, 1.0, size=(num_paths, self.N, self.n_assets))
        
        # Cholesky decomposition of correlation matrix
        try:
            L = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition for positive semi-definite matrices
            eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
            eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Apply correlation: dW = Z @ L^T
        correlated_increments = Z @ L.T
        
        return correlated_increments


    def generate_correlated_paths(self, num_paths, initial_values=None):
        """
        Generate correlated GRW paths for multiple assets.
        
        Args:
            num_paths (int): Number of simulation paths
            initial_values (array, optional): Initial values for each asset (default: all 1.0)
            
        Returns:
            array: Shape (num_paths, N_time_steps + 1, N_assets) of price paths
        """
        if initial_values is None:
            initial_values = np.ones(self.n_assets)
        else:
            initial_values = np.array(initial_values)
            assert len(initial_values) == self.n_assets, \
                "Initial values must match number of assets"
        
        # Generate correlated Brownian increments
        dW = self.correlated_brownian_increments(num_paths)
        
        # Apply GBM dynamics for each asset
        # dlog(S_i) = (mu_i - 0.5*sigma_i^2)*dt + sigma_i * dW_i
        drift_terms = self.drift_vector.reshape(1, 1, -1)  # Broadcast shape
        diffusion_terms = (self.sigma_vector * np.sqrt(self.dt)).reshape(1, 1, -1)
        
        log_increments = drift_terms + diffusion_terms * dW
        
        # Cumulative sum to get log prices
        log_prices = np.cumsum(log_increments, axis=1)
        
        # Add initial log prices and exponentiate
        log_initial = np.log(initial_values).reshape(1, 1, -1)
        log_prices_with_initial = np.concatenate([
            np.repeat(log_initial, num_paths, axis=0), 
            log_prices
        ], axis=1)
        
        paths = np.exp(log_prices_with_initial)
        
        return paths


    def fit(self, paths, is_log_paths=False):
        """
        Estimate mu vector, sigma vector, and correlation matrix from observed paths
        by running object: estimate_parameters method.
        
        Args:
            paths (array): Shape (num_paths, N_time_steps + 1, N_assets)
            
        Returns:
            dict: Estimated parameters
        """
        return self.estimate_parameters(paths, is_log_paths=is_log_paths)
    


    def estimate_parameters(self, paths, is_log_paths=False):
        """
        Estimate mu vector, sigma vector, and correlation matrix from observed paths.
        
        Args:
            paths (array): Shape (num_paths, N_time_steps + 1, N_assets)
            
        Returns:
            dict: Estimated parameters
        """

        if is_log_paths:
            log_returns = paths # Assume paths are already daily log returns
        else:
            # Calculate log returns for each asset
            log_returns = np.log(paths[:, 1:, :] / paths[:, :-1, :])
        
        # Reshape to (num_paths * N_time_steps, N_assets)
        log_returns_flat = log_returns.reshape(-1, self.n_assets)
        
        # Estimate parameters
        mean_log_returns = np.mean(log_returns_flat, axis=0)
        cov_log_returns = np.cov(log_returns_flat, rowvar=False)

        # print("Mean log returns:\n", mean_log_returns)
        # print("Covariance matrix of log returns:\n", cov_log_returns)
        
        # Convert back to GBM parameters
        estimated_sigma_vector = np.sqrt(np.diag(cov_log_returns) / self.dt)
        estimated_mu_vector = mean_log_returns / self.dt + 0.5 * estimated_sigma_vector ** 2
        
        # Estimated correlation matrix
        std_matrix = np.sqrt(np.diag(cov_log_returns))
        estimated_correlation = cov_log_returns / np.outer(std_matrix, std_matrix)
        estimated_covariance = np.diag(estimated_sigma_vector) @ estimated_correlation @ np.diag(estimated_sigma_vector)

        # Assert that the correlation matrix is symmetric and valid
        assert np.allclose(estimated_correlation, estimated_correlation.T), \
            "Estimated correlation matrix must be symmetric but got {}".format(estimated_correlation)
        eigenvals = np.linalg.eigvals(estimated_correlation)
        assert np.all(eigenvals >= -1e-8), "Estimated correlation matrix must be positive semi-definite but got eigenvalues {}".format(eigenvals)


        # Validate covariance matrix properties
        assert np.allclose(estimated_covariance, estimated_covariance.T), \
            "Estimated covariance matrix must be symmetric but got {}".format(estimated_covariance)
        eigenvals_cov = np.linalg.eigvals(estimated_covariance)
        assert np.all(eigenvals_cov >= -1e-8), "Estimated covariance matrix must be positive semi-definite but got eigenvalues {}".format(eigenvals_cov)


        return {
            'mu_vector': estimated_mu_vector,
            'sigma_vector': estimated_sigma_vector,
            'correlation_matrix': estimated_correlation,
            'covariance_matrix': estimated_covariance,
            'log_returns_covariance': cov_log_returns
        }
    
    

    def plot_sample_paths(self, paths, num_paths_to_plot=5, save_path=None):
        """
        Plot sample correlated paths for visualization.
        
        Args:
            paths (array): Output from generate_correlated_paths
            num_paths_to_plot (int): Number of sample paths to plot per asset
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(self.n_assets, 1, figsize=(12, 3*self.n_assets), sharex=True)
        if self.n_assets == 1:
            axes = [axes]
        
        time_grid = np.linspace(self.t0, self.t1, self.N + 1)
        
        for asset_idx in range(self.n_assets):
            ax = axes[asset_idx]
            for path_idx in range(min(num_paths_to_plot, paths.shape[0])):
                ax.plot(time_grid, paths[path_idx, :, asset_idx], alpha=0.7, linewidth=1)
            
            ax.set_ylabel(f'Asset {asset_idx + 1} Price')
            ax.set_title(f'Asset {asset_idx + 1}: μ={self.mu_vector[asset_idx]:.3f}, σ={self.sigma_vector[asset_idx]:.3f}')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def generate_paths(self, num_paths, initial_values=None):
        """Override parent method to generate single asset path (for compatibility)."""
        correlated_paths = self.generate_correlated_paths(num_paths, initial_values=initial_values)

        if initial_values is None:
            return correlated_paths[:, 1:]  # Exclude initial values, generate_correlated_paths adds initial values
        else:
            return correlated_paths



def test_correlated_grw():
    """Test the correlated GRW implementation."""
    print("=" * 60)
    print("TESTING CORRELATED GEOMETRIC RANDOM WALK")
    print("=" * 60)
    
    # Define parameters for 3 assets
    mu_vector = [0.08, 0.12, 0.10]
    sigma_vector = [0.2, 0.25, 0.22]
    
    # Define correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])
    
    # Create correlated GRW
    cgrw = CorrelatedGeometricRandomWalk(
        mu_vector=mu_vector,
        sigma_vector=sigma_vector,
        correlation_matrix=correlation_matrix,
        dt=1e-3,
        t0=0,
        t1=1
    )
    
    # Generate paths
    num_paths = 10000
    paths = cgrw.generate_correlated_paths(num_paths)
    
    print(f"Generated {num_paths} paths for {cgrw.n_assets} assets")
    print(f"Path shape: {paths.shape}")
    print(f"Time steps: {cgrw.N}")
    print()
    
    # Estimate parameters
    estimated = cgrw.estimate_parameters(paths)
    
    # Compare true vs estimated parameters
    print("TRUE vs ESTIMATED PARAMETERS:")
    print("-" * 40)
    print("Drift (mu) parameters:")
    for i, (true_mu, est_mu) in enumerate(zip(mu_vector, estimated['mu_vector'])):
        print(f"  Asset {i+1}: True={true_mu:.4f}, Estimated={est_mu:.4f}, Error={abs(est_mu-true_mu)/true_mu*100:.2f}%")
    
    print("\nVolatility (sigma) parameters:")
    for i, (true_sigma, est_sigma) in enumerate(zip(sigma_vector, estimated['sigma_vector'])):
        print(f"  Asset {i+1}: True={true_sigma:.4f}, Estimated={est_sigma:.4f}, Error={abs(est_sigma-true_sigma)/true_sigma*100:.2f}%")
    
    print(f"\nTrue Correlation Matrix:")
    print(correlation_matrix)
    print(f"\nEstimated Correlation Matrix:")
    print(estimated['correlation_matrix'])
    print(f"\nCorrelation Estimation Errors:")
    correlation_errors = np.abs(correlation_matrix - estimated['correlation_matrix'])
    print(correlation_errors)
    print(f"Max correlation error: {np.max(correlation_errors):.4f}")
    
    # Visualize sample paths
    print(f"\nGenerating visualization of correlated paths...")
    cgrw.plot_sample_paths(paths, num_paths_to_plot=10, 
                          save_path='../demos/grw/correlated_grw_paths.png')
    print("Plot saved as 'correlated_grw_paths.png'")
    
    return cgrw, paths, estimated


if __name__ == "__main__":
    # Test single asset GRW first
    print("TESTING SINGLE ASSET GRW")
    print("=" * 50)
    
    mu = 0.1
    sigma = 0.2
    dt = 1
    T = 25 # trading days
    num_paths = 1000
    
    grw = GeometricRandomWalk(mu, sigma, dt = dt, t0=0, t1=dt * T)

    for path_type in ['prices', 'log-returns', 'log-increments']:
        print(f"\nGenerating paths returning {path_type}...")
        sample_paths = grw.generate_paths(num_paths, return_type=path_type, initial_log_return=0.5)
        print(f"Sample {path_type} paths shape: {sample_paths.shape}")
        # print(sample_paths)

        grw.plot_sample_paths(sample_paths, num_paths_to_plot=10, 
                              save_path=f'../demos/grw/sample_grw_paths_{path_type}.png')

        print(f"\nFitting parameters from generated {path_type} paths...")

        mu_fit, sigma_fit = grw.fit(sample_paths, paths_type=path_type).values()

        print(f"True mu: {mu:.4f}, True sigma: {sigma:.4f}")
        print(f"Fitted mu: {mu_fit:.4f}, Fitted sigma: {sigma_fit:.4f}")

        mu_error = abs(mu_fit - mu) / mu * 100
        sigma_error = abs(sigma_fit - sigma) / sigma * 100
        print(f"Mu estimation error: {mu_error:.2f}%")
        print(f"Sigma estimation error: {sigma_error:.2f}%")



    # paths = grw.generate_paths(num_paths)

    # Add initial value of 1 (S_0 = 1)
    # paths = np.concatenate([np.ones((num_paths, 1)), paths], axis = 1)

    # Calculate log returns and estimate parameters
    # log_returns = np.log(paths[:, 1:] / paths[:, :-1])
    # mean_log_return = np.mean(log_returns)
    # var_log_return = np.var(log_returns)
    # estimated_sigma = np.sqrt(var_log_return / grw.dt)
    # estimated_mu = mean_log_return / grw.dt + 0.5 * estimated_sigma ** 2
    # print(f"True mu: {mu:.6f}, True sigma: {sigma:.6f}")
    # print(f"Estimated mu: {estimated_mu:.6f}, Estimated sigma: {estimated_sigma:.6f}")

    # mu_fit, sigma_fit = grw.fit(sample_paths).values()
    # print(f"True mu: {mu:.6f}, True sigma: {sigma:.6f}")
    # print(f"Fitted mu: {mu_fit:.6f}, Fitted sigma: {sigma_fit:.6f}")
    
    # mu_error = abs(mu_fit - mu) / mu * 100
    # sigma_error = abs(sigma_fit - sigma) / sigma * 100
    # print(f"Mu estimation error: {mu_error:.2f}%")
    # print(f"Sigma estimation error: {sigma_error:.2f}%")
    
    print("\n" + "=" * 70)
    
    # Test correlated GRW
    # test_correlated_grw()

