"""
Graph pooling methods for node selection in UNet architecture.
Provides degree-based node selection as an alternative to random selection.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import degree
from typing import Tuple, Optional


class NodeDegreeBasedSelection(nn.Module):
    """
    Select nodes based on their degree centrality.
    Nodes with higher connectivity are considered more important.
    This is particularly useful for stock networks where highly connected 
    stocks often have more market influence.
    """
    def __init__(self, ratio: float = 0.5, min_score: Optional[float] = None):
        """
        Args:
            ratio: Fraction of nodes to keep (e.g., 0.5 keeps half the nodes)
            min_score: Minimum degree score threshold (optional)
        """
        super().__init__()
        self.ratio = ratio
        self.min_score = min_score
    
    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select nodes based on degree centrality.
        
        Args:
            edge_index: Edge indices [2, E]
            num_nodes: Total number of nodes
            batch: Batch vector [N] (optional, for batched graphs)
        
        Returns:
            selection_matrix: [N_out, N_in] binary selection matrix
            selected_indices: [N_out] indices of selected nodes
        """
        device = edge_index.device
        
        # Calculate degree for each node
        node_degrees = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        
        if batch is not None:
            # Handle batched graphs
            batch_size = batch.max().item() + 1
            selection_matrices = []
            selected_indices_list = []
            
            for b in range(batch_size):
                mask = batch == b
                batch_degrees = node_degrees[mask]
                
                # Select top-k nodes based on degree for this batch
                num_select = max(1, int(batch_degrees.size(0) * self.ratio))
                
                if self.min_score is not None:
                    # Filter by minimum score first
                    valid_mask = batch_degrees >= self.min_score
                    if valid_mask.sum() > 0:
                        valid_degrees = batch_degrees[valid_mask]
                        valid_indices = torch.nonzero(valid_mask).squeeze()
                        
                        if valid_degrees.size(0) >= num_select:
                            _, top_indices = torch.topk(valid_degrees, num_select)
                            batch_selected = valid_indices[top_indices]
                        else:
                            batch_selected = valid_indices
                    else:
                        # If no nodes meet min_score, fall back to top-k
                        _, batch_selected = torch.topk(batch_degrees, num_select)
                else:
                    _, batch_selected = torch.topk(batch_degrees, num_select)
                
                # Map back to global indices
                global_indices = torch.nonzero(mask).squeeze()[batch_selected]
                selected_indices_list.append(global_indices)
            
            selected_indices = torch.cat(selected_indices_list)
        else:
            # Single graph case
            num_select = max(1, int(num_nodes * self.ratio))
            
            if self.min_score is not None:
                # Filter by minimum score first
                valid_mask = node_degrees >= self.min_score
                if valid_mask.sum() > 0:
                    valid_degrees = node_degrees[valid_mask]
                    valid_indices = torch.nonzero(valid_mask).squeeze()
                    
                    if valid_degrees.size(0) >= num_select:
                        _, top_indices = torch.topk(valid_degrees, num_select)
                        selected_indices = valid_indices[top_indices]
                    else:
                        selected_indices = valid_indices
                else:
                    # If no nodes meet min_score, fall back to top-k
                    _, selected_indices = torch.topk(node_degrees, num_select)
            else:
                _, selected_indices = torch.topk(node_degrees, num_select)
        
        # Create binary selection matrix
        num_selected = selected_indices.size(0)
        selection_matrix = torch.zeros(num_selected, num_nodes, 
                                     device=device, dtype=torch.float)
        selection_matrix[torch.arange(num_selected), selected_indices] = 1.0
        
        return selection_matrix, selected_indices
    
    
    def get_node_scores(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Get degree scores for all nodes (useful for analysis).
        
        Args:
            edge_index: Edge indices [2, E]
            num_nodes: Total number of nodes
            
        Returns:
            node_degrees: [N] degree scores for each node
        """
        return degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)