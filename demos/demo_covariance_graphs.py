#!/usr/bin/env python3
"""
Example script demonstrating how to create edge_index and edge_weight pairs 
from the covariance of closing prices.

This script shows multiple approaches:
1. Using the new utility functions directly
2. Integration with the existing dataset structure
3. Different methods (covariance vs correlation)
4. Different thresholding and normalization options
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Import the new utility functions
from utils.covariance_graph_utils import (
    closing_prices_to_graph, 
    pandas_closing_prices_to_graph,
    example_usage
)
from datasets.utils import get_graph_from_covariance


def demo_covariance_graph_creation():
    """
    Demonstrate various ways to create graphs from covariance of closing prices.
    """
    print("=== Covariance Graph Creation Demo ===\n")
    
    # Method 1: Using synthetic data
    print("1. Creating graph from synthetic closing prices:")
    print("-" * 50)
    
    # Create synthetic closing prices for 5 stocks over 100 time steps
    torch.manual_seed(42)
    num_stocks, num_timestamps = 5, 100
    
    # Generate correlated price movements
    base_returns = torch.randn(num_stocks, num_timestamps) * 0.02  # 2% daily volatility
    correlation_matrix = torch.tensor([
        [1.0, 0.7, 0.3, 0.1, 0.2],
        [0.7, 1.0, 0.4, 0.2, 0.1], 
        [0.3, 0.4, 1.0, 0.6, 0.3],
        [0.1, 0.2, 0.6, 1.0, 0.5],
        [0.2, 0.1, 0.3, 0.5, 1.0]
    ])
    
    # Apply correlation structure
    L = torch.linalg.cholesky(correlation_matrix)
    correlated_returns = torch.mm(L, base_returns)
    
    # Convert to price levels (starting at $100)
    initial_prices = torch.full((num_stocks, 1), 100.0)
    closing_prices = initial_prices * torch.cumprod(1 + correlated_returns, dim=1)
    
    print(f"Closing prices shape: {closing_prices.shape}")
    print(f"Price range: ${closing_prices.min():.2f} - ${closing_prices.max():.2f}")
    
    # Create graphs using different methods
    methods_to_test = [
        ("covariance", {"threshold": 0.1, "normalize": True}),
        ("correlation", {"threshold": 0.3, "normalize": True}),
        ("correlation", {"threshold": 0.5, "normalize": True, "use_absolute": False})
    ]
    
    for method, kwargs in methods_to_test:
        edge_index, edge_weight = closing_prices_to_graph(
            closing_prices, 
            method=method,
            **kwargs
        )
        print(f"\n{method.capitalize()} graph (threshold={kwargs.get('threshold', 0.0)}):")
        print(f"  Edges: {edge_index.shape[1]}")
        print(f"  Edge weights range: [{edge_weight.min():.3f}, {edge_weight.max():.3f}]")
        print(f"  Edge index shape: {edge_index.shape}")
        
        # Show some example edges
        if edge_index.shape[1] > 0:
            print(f"  Example edges (source->target, weight):")
            for i in range(min(5, edge_index.shape[1])):
                src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
                weight = edge_weight[i].item()
                print(f"    {src}->{tgt}: {weight:.3f}")
    
    print("\n" + "="*70 + "\n")
    
    # Method 2: Using real data (if available)
    print("2. Creating graph from real stock data:")
    print("-" * 50)
    
    # Check if we have real data available
    data_path = Path("./datasets/raw/values.csv")
    alt_data_path = Path("./SP100AnalysisWithGNNs/data/SP100/raw/values.csv")
    
    if data_path.exists():
        values_file = str(data_path)
    elif alt_data_path.exists():
        values_file = str(alt_data_path)
    else:
        print("Real stock data not found. Skipping this demo.")
        print(f"Looked for data at: {data_path} and {alt_data_path}")
        return
    
    try:
        # Method 2a: Using the pandas function
        edge_index_real, edge_weight_real = pandas_closing_prices_to_graph(
            values_file,
            method="correlation",
            threshold=0.3,
            normalize=True
        )
        
        print(f"Real data correlation graph:")
        print(f"  Edges: {edge_index_real.shape[1]}")
        print(f"  Edge weights range: [{edge_weight_real.min():.3f}, {edge_weight_real.max():.3f}]")
        
        # Method 2b: Using the integrated dataset function
        x, close_prices, edge_index_integrated, edge_weight_integrated, info = get_graph_from_covariance(
            values_file,
            method="correlation",
            threshold=0.4,
            normalize=True
        )
        
        print(f"\nIntegrated dataset function results:")
        print(f"  Node features shape: {x.shape}")
        print(f"  Close prices shape: {close_prices.shape}")
        print(f"  Edges: {edge_index_integrated.shape[1]}")
        print(f"  Info: {info}")
        
    except Exception as e:
        print(f"Error processing real data: {e}")
        print("This might be due to data format differences.")
    
    print("\n" + "="*70 + "\n")
    
    # Method 3: Demonstrating different threshold effects
    print("3. Effect of different thresholds:")
    print("-" * 50)
    
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.7]
    for threshold in thresholds:
        edge_index, edge_weight = closing_prices_to_graph(
            closing_prices,
            method="correlation",
            threshold=threshold,
            normalize=True
        )
        density = edge_index.shape[1] / (num_stocks * (num_stocks - 1))  # excluding self-loops
        print(f"Threshold {threshold:0.1f}: {edge_index.shape[1]:3d} edges (density: {density:.3f})")


def demo_integration_with_existing_dataset():
    """
    Show how to modify the existing SP100Stocks dataset to use covariance graphs.
    """
    print("=== Integration with Existing Dataset ===\n")
    
    print("To integrate covariance-based graphs with your existing SP100Stocks dataset,")
    print("you can modify the dataset class as follows:\n")
    
    example_code = '''
class SP100StocksWithCovarianceGraph(SP100Stocks):
    """
    Extended SP100Stocks dataset that uses covariance/correlation of closing prices
    to create the graph structure instead of a pre-computed adjacency matrix.
    """
    
    def __init__(self, root: str = "../data/SP100/", values_file_name: str = "values.csv", 
                 past_window: int = 25, future_window: int = 1,
                 target_column_name: str = "NormClose", force_reload: bool = False, 
                 transform: Callable = None,
                 # New parameters for covariance graph
                 graph_method: str = "correlation", cov_method: str = "sample",
                 threshold: float = 0.3, use_absolute: bool = True,
                 remove_self_loops: bool = True, normalize: bool = True):
        
        self.graph_method = graph_method
        self.cov_method = cov_method
        self.threshold = threshold
        self.use_absolute = use_absolute
        self.remove_self_loops = remove_self_loops
        self.normalize = normalize
        
        # Note: We don't need adj_file_name anymore since we compute the graph
        super().__init__(root, values_file_name, None, past_window, future_window,
                        target_column_name, force_reload, transform)
    
    @property
    def raw_file_names(self) -> list[str]:
        # Only need the values file, not the adjacency matrix
        return [self.values_file_name]
    
    def process(self) -> None:
        # Use the new covariance-based graph function
        x, close_prices, edge_index, edge_weight, info_dict = get_graph_from_covariance(
            values_path=osp.join(self.root, f"raw/{self.values_file_name}"),
            target_column_name=self.target_column_name,
            method=self.graph_method,
            cov_method=self.cov_method,
            threshold=self.threshold,
            use_absolute=self.use_absolute,
            remove_self_loops=self.remove_self_loops,
            normalize=self.normalize
        )
        
        # Rest of the processing remains the same...
        self.info = info_dict
        # ... (continue with existing timestamp processing code)
    '''
    
    print(example_code)


if __name__ == "__main__":
    # Run the main demonstration
    demo_covariance_graph_creation()
    
    # Show integration example
    demo_integration_with_existing_dataset()
    
    print("\n=== Summary ===")
    print("You now have several ways to create edge_index and edge_weight from covariance:")
    print("1. Use `closing_prices_to_graph()` for torch tensors")
    print("2. Use `pandas_closing_prices_to_graph()` for CSV files") 
    print("3. Use `get_graph_from_covariance()` for full dataset integration")
    print("4. Customize the SP100Stocks class to use covariance graphs")
    print("\nAll functions support:")
    print("- Covariance or correlation methods")
    print("- Thresholding to control graph sparsity")
    print("- Normalization and absolute value options")
    print("- Self-loop removal")
