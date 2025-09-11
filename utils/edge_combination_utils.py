import torch
from typing import Tuple, Optional
import numpy as np


def combine_edge_graphs(
    edge_index1: torch.Tensor,
    edge_weight1: torch.Tensor,
    edge_index2: torch.Tensor, 
    edge_weight2: torch.Tensor,
    fill_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine two edge graphs by merging their edge indices and stacking edge weights as features.
    
    Args:
        edge_index1: First edge index tensor of shape (2, num_edges1)
        edge_weight1: First edge weights of shape (num_edges1,) or (num_edges1, num_features1)
        edge_index2: Second edge index tensor of shape (2, num_edges2)
        edge_weight2: Second edge weights of shape (num_edges2,) or (num_edges2, num_features2)
        fill_value: Value to use for missing edges in either graph
        
    Returns:
        combined_edge_index: Combined edge index of shape (2, num_combined_edges)
        combined_edge_weight: Combined edge weights of shape (num_combined_edges, total_features)
    """
    
    # Ensure edge weights are 2D
    if edge_weight1.dim() == 1:
        edge_weight1 = edge_weight1.unsqueeze(-1)
    if edge_weight2.dim() == 1:
        edge_weight2 = edge_weight2.unsqueeze(-1)
    
    # Get dimensions
    num_features1 = edge_weight1.shape[1]
    num_features2 = edge_weight2.shape[1]
    total_features = num_features1 + num_features2
    
    # Convert edge indices to tuples for easier comparison
    edges1_set = set(zip(edge_index1[0].tolist(), edge_index1[1].tolist()))
    edges2_set = set(zip(edge_index2[0].tolist(), edge_index2[1].tolist()))
    
    # Find all unique edges
    all_edges = edges1_set.union(edges2_set)
    
    # Create combined edge index
    combined_edge_list = list(all_edges)
    combined_edge_index = torch.tensor(combined_edge_list).t().long()
    
    # Create combined edge weights
    num_combined_edges = len(combined_edge_list)
    combined_edge_weight = torch.full((num_combined_edges, total_features), fill_value, dtype=torch.float32)
    
    # Map original edges to combined indices
    edge_to_idx = {edge: idx for idx, edge in enumerate(combined_edge_list)}
    
    # Fill in weights from first graph
    for i, (src, tgt) in enumerate(zip(edge_index1[0].tolist(), edge_index1[1].tolist())):
        combined_idx = edge_to_idx[(src, tgt)]
        combined_edge_weight[combined_idx, :num_features1] = edge_weight1[i]
    
    # Fill in weights from second graph
    for i, (src, tgt) in enumerate(zip(edge_index2[0].tolist(), edge_index2[1].tolist())):
        combined_idx = edge_to_idx[(src, tgt)]
        combined_edge_weight[combined_idx, num_features1:] = edge_weight2[i]
    
    return combined_edge_index, combined_edge_weight


def combine_temporal_and_static_graphs(
    edge_index_static: torch.Tensor,
    edge_weight_static: torch.Tensor,
    edge_index_temporal: torch.Tensor,
    edge_weight_temporal: torch.Tensor,
    fill_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Specifically combine static (e.g., correlation-based) and temporal graphs.
    
    Args:
        edge_index_static: Static edge index (e.g., from correlation)
        edge_weight_static: Static edge weights
        edge_index_temporal: Temporal edge index (e.g., from time-series correlation)
        edge_weight_temporal: Temporal edge weights
        fill_value: Value for missing edges
        
    Returns:
        combined_edge_index: Combined edge indices
        combined_edge_weight: Combined weights with static weights first, temporal weights last
    """
    return combine_edge_graphs(
        edge_index_static, edge_weight_static,
        edge_index_temporal, edge_weight_temporal,
        fill_value=fill_value
    )


def add_temporal_features_to_existing_graph(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_index_t_corr: torch.Tensor,
    edge_weight_t: torch.Tensor,
    fill_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add temporal correlation features to an existing graph structure.
    This is the function that directly answers your question.
    
    Args:
        edge_index: Existing edge indices of shape (2, num_edges)
        edge_weight: Existing edge weights of shape (num_edges,) or (num_edges, num_features)
        edge_index_t_corr: Temporal correlation edge indices of shape (2, num_temporal_edges)
        edge_weight_t: Temporal edge weights of shape (num_temporal_edges,)
        fill_value: Value to use for edges that don't exist in temporal graph
        
    Returns:
        updated_edge_index: Updated edge indices including all edges from both graphs
        updated_edge_weight: Updated edge weights with temporal weights as last feature dimension
    """
    return combine_edge_graphs(
        edge_index, edge_weight,
        edge_index_t_corr, edge_weight_t,
        fill_value=fill_value
    )


def example_usage():
    """
    Demonstrate how to combine edge graphs.
    """
    print("=== Edge Graph Combination Example ===\n")
    
    # Example 1: Simple case with some overlapping edges
    print("Example 1: Basic combination")
    print("-" * 40)
    
    # First graph: correlation-based
    edge_index1 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # 3 edges
    edge_weight1 = torch.tensor([0.8, 0.6, 0.4])  # correlation weights
    
    # Second graph: temporal correlation
    edge_index2 = torch.tensor([[0, 1, 3], [1, 3, 0]], dtype=torch.long)  # 3 edges (1 overlaps)
    edge_weight2 = torch.tensor([0.7, 0.5, 0.9])  # temporal weights
    
    combined_edge_index, combined_edge_weight = combine_edge_graphs(
        edge_index1, edge_weight1, edge_index2, edge_weight2, fill_value=0.0
    )
    
    print(f"Original graph 1: {edge_index1.shape[1]} edges")
    print(f"Original graph 2: {edge_index2.shape[1]} edges") 
    print(f"Combined graph: {combined_edge_index.shape[1]} edges")
    print(f"Combined edge weight shape: {combined_edge_weight.shape}")
    print(f"Edge features: {combined_edge_weight.shape[1]} (original + temporal)")
    
    print("\nCombined edges and weights:")
    for i in range(combined_edge_index.shape[1]):
        src, tgt = combined_edge_index[0, i].item(), combined_edge_index[1, i].item()
        weights = combined_edge_weight[i].tolist()
        print(f"  Edge {src}->{tgt}: {weights}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: More realistic case with multi-feature weights
    print("Example 2: Multi-feature weights")
    print("-" * 40)
    
    # Graph with multiple edge features (e.g., correlation + sector)
    edge_index_main = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight_main = torch.tensor([
        [0.8, 1.0],  # correlation + sector bonus
        [0.6, 0.0],
        [0.4, 1.0], 
        [0.3, 0.0]
    ])
    
    # Temporal correlation graph
    edge_index_temporal = torch.tensor([[0, 2, 4], [2, 4, 1]], dtype=torch.long)
    edge_weight_temporal = torch.tensor([0.9, 0.7, 0.5])
    
    combined_edge_index, combined_edge_weight = add_temporal_features_to_existing_graph(
        edge_index_main, edge_weight_main, edge_index_temporal, edge_weight_temporal
    )
    
    print(f"Main graph: {edge_index_main.shape[1]} edges with {edge_weight_main.shape[1]} features")
    print(f"Temporal graph: {edge_index_temporal.shape[1]} edges")
    print(f"Combined: {combined_edge_index.shape[1]} edges with {combined_edge_weight.shape[1]} features")
    
    print("\nFeature breakdown:")
    print("  Features 0-1: Original features (correlation, sector)")
    print("  Feature 2: Temporal correlation")
    
    print("\nSample combined edges:")
    for i in range(min(5, combined_edge_index.shape[1])):
        src, tgt = combined_edge_index[0, i].item(), combined_edge_index[1, i].item()
        weights = combined_edge_weight[i].tolist()
        print(f"  Edge {src}->{tgt}: {[f'{w:.2f}' for w in weights]}")


def integrate_with_dataset_utils():
    """
    Show how to integrate this with your existing dataset utilities.
    """
    print("\n=== Integration with Dataset Utils ===\n")
    
    example_code = '''
# In your datasets/utils.py, you can modify get_graph_in_pyg_format or create a new function:

def get_multi_feature_graph(values_path: str, adj_path: str, 
                           target_column_name: str = "NormClose",
                           add_temporal_correlation: bool = True,
                           temporal_window: int = 20,
                           temporal_threshold: float = 0.3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Create a graph with multiple edge features including temporal correlation.
    """
    
    # Get the standard graph
    x, close_prices, edge_index, edge_weight, info_dict = get_graph_in_pyg_format(
        values_path, adj_path, target_column_name
    )
    
    if add_temporal_correlation:
        # Compute temporal correlation on a rolling window
        edge_index_temporal, edge_weight_temporal = compute_temporal_correlation_graph(
            close_prices, window=temporal_window, threshold=temporal_threshold
        )
        
        # Combine with existing graph
        from utils.covariance_graph_utils import add_temporal_features_to_existing_graph
        edge_index, edge_weight = add_temporal_features_to_existing_graph(
            edge_index, edge_weight, edge_index_temporal, edge_weight_temporal
        )
        
        info_dict["Edge_features"] = ["correlation", "temporal_correlation"]
        info_dict["Temporal_window"] = temporal_window
    
    return x, close_prices, edge_index, edge_weight, info_dict


def compute_temporal_correlation_graph(close_prices: torch.Tensor, 
                                     window: int = 20, 
                                     threshold: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute temporal correlation graph using rolling windows.
    """
    num_stocks, num_timestamps = close_prices.shape
    
    # Compute rolling correlations (simplified - you might want more sophisticated temporal analysis)
    if num_timestamps < window:
        # Use all available data if less than window size
        recent_prices = close_prices
    else:
        # Use last 'window' time steps
        recent_prices = close_prices[:, -window:]
    
    # Compute correlation matrix for recent period
    temporal_corr = torch.corrcoef(recent_prices)
    temporal_corr.fill_diagonal_(0.0)  # Remove self-loops
    
    # Convert to edge format with thresholding
    from torch_geometric.utils import dense_to_sparse
    adj_matrix = torch.abs(temporal_corr) * (torch.abs(temporal_corr) > threshold)
    edge_index, edge_weight = dense_to_sparse(adj_matrix)
    
    return edge_index, edge_weight
    '''
    
    print(example_code)


if __name__ == "__main__":
    example_usage()
    integrate_with_dataset_utils()
