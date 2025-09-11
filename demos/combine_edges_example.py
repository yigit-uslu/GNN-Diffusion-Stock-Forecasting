#!/usr/bin/env python3
"""
Practical example showing how to combine edge_index_t_corr and edge_weight_t 
with existing edge_index and edge_weight where edge_weight_t becomes the last 
edge feature dimension.
"""

import torch
import numpy as np
from datasets.utils import combine_edge_features


def demonstrate_exact_combination():
    """
    Demonstrate the exact scenario described in the question.
    """
    print("=== Combining edge_index_t_corr and edge_weight_t with existing graph ===\n")
    
    # Simulate your existing graph (e.g., from correlation or adjacency matrix)
    print("1. Existing graph:")
    print("-" * 30)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.8, 0.6, 0.4, 0.9])  # Single feature (correlation)
    
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_weight shape: {edge_weight.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")
    print(f"Edges: {[(edge_index[0,i].item(), edge_index[1,i].item()) for i in range(edge_index.shape[1])]}")
    
    # Simulate your temporal correlation graph  
    print("\n2. Temporal correlation graph:")
    print("-" * 30)
    edge_index_t_corr = torch.tensor([[0, 2, 4, 1], [2, 4, 0, 3]], dtype=torch.long)
    edge_weight_t = torch.tensor([0.7, 0.5, 0.3, 0.8])  # Temporal correlation weights
    
    print(f"edge_index_t_corr shape: {edge_index_t_corr.shape}")
    print(f"edge_weight_t shape: {edge_weight_t.shape}")
    print(f"Number of temporal edges: {edge_index_t_corr.shape[1]}")
    print(f"Temporal edges: {[(edge_index_t_corr[0,i].item(), edge_index_t_corr[1,i].item()) for i in range(edge_index_t_corr.shape[1])]}")
    
    # Combine them using the utility function
    print("\n3. Combined graph:")
    print("-" * 30)
    combined_edge_index, combined_edge_weight = combine_edge_features(
        edge_index, edge_weight,           # Existing graph
        edge_index_t_corr, edge_weight_t,  # Temporal graph  
        fill_value=0.0                     # Fill missing edges with 0
    )
    
    print(f"combined_edge_index shape: {combined_edge_index.shape}")
    print(f"combined_edge_weight shape: {combined_edge_weight.shape}")
    print(f"Number of combined edges: {combined_edge_index.shape[1]}")
    print(f"Edge features: {combined_edge_weight.shape[1]} (original + temporal)")
    
    print("\n4. Detailed edge breakdown:")
    print("-" * 30)
    print("Edge\t\tOriginal\tTemporal")
    print("(src->tgt)\tWeight\t\tWeight")
    print("-" * 40)
    
    for i in range(combined_edge_index.shape[1]):
        src, tgt = combined_edge_index[0, i].item(), combined_edge_index[1, i].item()
        orig_weight = combined_edge_weight[i, 0].item()
        temp_weight = combined_edge_weight[i, 1].item()
        print(f"{src}->{tgt}\t\t{orig_weight:.3f}\t\t{temp_weight:.3f}")
    
    return combined_edge_index, combined_edge_weight


def demonstrate_with_multi_feature_existing_graph():
    """
    Demonstrate when the existing graph already has multiple edge features.
    """
    print("\n\n=== Case with Multi-Feature Existing Graph ===\n")
    
    # Existing graph with multiple features (e.g., correlation + sector)
    print("1. Existing multi-feature graph:")
    print("-" * 35)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.tensor([
        [0.8, 1.0],  # Edge 0->1: correlation=0.8, sector=1.0 (same sector)
        [0.6, 0.0],  # Edge 1->2: correlation=0.6, sector=0.0 (different sector)  
        [0.4, 1.0]   # Edge 2->0: correlation=0.4, sector=1.0 (same sector)
    ])
    
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_weight shape: {edge_weight.shape}")
    print("Features: [correlation, sector_bonus]")
    
    # Temporal correlation graph
    print("\n2. Temporal correlation graph:")
    print("-" * 35)
    edge_index_t_corr = torch.tensor([[0, 1, 3], [1, 3, 0]], dtype=torch.long)
    edge_weight_t = torch.tensor([0.9, 0.7, 0.5])
    
    print(f"edge_index_t_corr shape: {edge_index_t_corr.shape}")
    print(f"edge_weight_t shape: {edge_weight_t.shape}")
    
    # Combine them
    print("\n3. Combined graph:")
    print("-" * 35)
    combined_edge_index, combined_edge_weight = combine_edge_features(
        edge_index, edge_weight,
        edge_index_t_corr, edge_weight_t,
        fill_value=0.0
    )
    
    print(f"combined_edge_index shape: {combined_edge_index.shape}")
    print(f"combined_edge_weight shape: {combined_edge_weight.shape}")
    print("Features: [correlation, sector_bonus, temporal_correlation]")
    
    print("\n4. Feature breakdown:")
    print("-" * 50)
    print("Edge\t\tCorr\tSector\tTemporal")
    print("(src->tgt)\tWeight\tBonus\tCorr")
    print("-" * 50)
    
    for i in range(combined_edge_index.shape[1]):
        src, tgt = combined_edge_index[0, i].item(), combined_edge_index[1, i].item()
        corr_w = combined_edge_weight[i, 0].item()
        sector_w = combined_edge_weight[i, 1].item() 
        temp_w = combined_edge_weight[i, 2].item()
        print(f"{src}->{tgt}\t\t{corr_w:.3f}\t{sector_w:.3f}\t{temp_w:.3f}")


def practical_usage_in_your_code():
    """
    Show how to use this in your actual codebase.
    """
    print("\n\n=== Practical Usage in Your Code ===\n")
    
    usage_example = '''
# In your dataset processing or model preparation code:

# 1. You have your existing graph (from correlation, adjacency matrix, etc.)
x, close_prices, edge_index, edge_weight, info = get_graph_in_pyg_format(
    values_path="path/to/values.csv",
    adj_path="path/to/adj.npy", 
    target_column_name="NormClose"
)

# 2. You compute temporal correlation graph
edge_index_t_corr, edge_weight_t = compute_temporal_correlation(
    close_prices, window_size=20, threshold=0.3
)

# 3. Combine them to add temporal features as the last dimension
from datasets.utils import combine_edge_features

combined_edge_index, combined_edge_weight = combine_edge_features(
    edge_index, edge_weight,           # Your existing graph
    edge_index_t_corr, edge_weight_t,  # Your temporal graph
    fill_value=0.0                     # Value for missing temporal correlations
)

# 4. Update your data object
data = StocksDataDiffusion(
    x=x,
    edge_index=combined_edge_index,     # Updated edge index
    edge_weight=combined_edge_weight,   # Updated edge weight with temporal features
    close_price=close_prices,
    y=y,
    close_price_y=close_price_y,
    timestamp=timestamp,
    stocks_index=stocks_index,
    info=info
)

# 5. Your model will now receive edge weights with shape (num_edges, num_features)
# where the last feature dimension contains the temporal correlation weights
    '''
    
    print(usage_example)


def temporal_correlation_computation_example():
    """
    Show how you might compute temporal correlation.
    """
    print("\n=== Computing Temporal Correlation ===\n")
    
    code_example = '''
def compute_temporal_correlation(close_prices: torch.Tensor, 
                               window_size: int = 20, 
                               threshold: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute temporal correlation graph using recent price movements.
    
    Args:
        close_prices: Tensor of shape (num_stocks, num_timestamps)
        window_size: Number of recent time steps to use
        threshold: Minimum correlation to create an edge
        
    Returns:
        edge_index_t_corr: Temporal correlation edge indices
        edge_weight_t: Temporal correlation weights
    """
    num_stocks, num_timestamps = close_prices.shape
    
    # Use recent window for temporal correlation
    if num_timestamps >= window_size:
        recent_prices = close_prices[:, -window_size:]
    else:
        recent_prices = close_prices
    
    # Compute correlation matrix for recent period
    temporal_corr = torch.corrcoef(recent_prices)
    temporal_corr.fill_diagonal_(0.0)  # Remove self-loops
    
    # Apply threshold and convert to sparse format
    adj_matrix = torch.abs(temporal_corr) * (torch.abs(temporal_corr) > threshold)
    
    # Convert to edge format
    from torch_geometric.utils import dense_to_sparse
    edge_index_t_corr, edge_weight_t = dense_to_sparse(adj_matrix)
    
    return edge_index_t_corr, edge_weight_t


# Usage in your dataset processing:
edge_index_t_corr, edge_weight_t = compute_temporal_correlation(
    close_prices, window_size=20, threshold=0.3
)
    '''
    
    print(code_example)


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_exact_combination()
    demonstrate_with_multi_feature_existing_graph()
    practical_usage_in_your_code()
    temporal_correlation_computation_example()
    
    print("\n=== Summary ===")
    print("✓ Use combine_edge_features() to merge your graphs")
    print("✓ edge_weight_t becomes the last feature dimension")
    print("✓ Missing edges are filled with your specified fill_value")
    print("✓ Combined graph includes all edges from both graphs")
    print("✓ Resulting edge_weight shape: (num_edges, original_features + 1)")
