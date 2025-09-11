import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.nn import TAGConv
import networkx as nx
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_adj(adj_path: str = "datasets/raw/adj.npy"):
    # Load the adjacency matrix from a .npy file
    adj = np.load(adj_path)
    # Convert the adjacency matrix to a PyTorch tensor
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    return adj_tensor


def create_rand_adj(N, sparsity=0.7, add_self_loops=True, remove_self_loops=False):
    # Create a random N x N weighted adjacency matrix
    adj = torch.rand(N, N)
    adj = (adj + adj.t()) / 2  # Make it symmetric
    adj[adj < sparsity] = 0  # Sparsify the matrix

    if add_self_loops:
        adj.fill_diagonal_(1)  # Add self-loops
    
    if remove_self_loops:
        adj.fill_diagonal_(0)  # Remove self-loops

    return adj


def get_pyg_graph(adj_tensor, x=None):
    # Convert the adjacency matrix to a PyG Data object
    edge_index, edge_weight = dense_to_sparse(adj_tensor)
    data = Data(edge_index=edge_index, edge_attr=edge_weight, x=x)
    return data


def display_feature_matrices(data: Data, pos, ax, offset_y=0.15, fontsize=6, precision=1):
    """
    Display feature matrices above each node.
    
    Args:
        data: PyG Data object with x tensor of shape (N, F_in, T)
        pos: Node positions dictionary
        ax: Matplotlib axis
        offset_y: Vertical offset above nodes
        fontsize: Font size for the text
        precision: Number of decimal places to show
    """
    if data.x is None:
        return
    
    x_tensor = data.x.numpy() if hasattr(data.x, 'numpy') else data.x
    N, F_in, T = x_tensor.shape
    
    for node_idx in range(N):
        if node_idx in pos:  # Make sure node exists in position dict
            node_pos = pos[node_idx]
            feature_matrix = x_tensor[node_idx]  # Shape: (F_in, T)
            
            # Format the matrix as a compact string
            matrix_str = format_matrix_simple(feature_matrix, precision=precision)  # Use 0 for integers
            
            # Position text above the node
            text_x = node_pos[0]
            text_y = node_pos[1] + offset_y
            
            # Add text with background for better readability
            ax.text(text_x, text_y, matrix_str, 
                   fontsize=fontsize, 
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.9),
                   family='monospace')  # Monospace for better matrix alignment


def format_matrix_compact(matrix, precision=1, max_cols=None, max_rows=None):
    """
    Format a 2D matrix as a compact string representation.
    
    Args:
        matrix: 2D numpy array of shape (F_in, T)
        precision: Number of decimal places
        max_cols: Maximum columns to display (None for all)
        max_rows: Maximum rows to display (None for all)
    
    Returns:
        str: Formatted matrix string
    """
    F_in, T = matrix.shape
    
    # Limit display size for readability
    display_rows = min(F_in, max_rows) if max_rows else min(F_in, 4)
    display_cols = min(T, max_cols) if max_cols else min(T, 6)
    
    lines = []
    
    # Add matrix bracket opening
    lines.append("⎡")
    
    for i in range(display_rows):
        row_str = ""
        for j in range(display_cols):
            val = matrix[i, j]
            if precision == 0:
                row_str += f"{int(val):3d} "
            else:
                row_str += f"{val:.{precision}f} "
        
        # Add ellipsis if we're truncating columns
        if display_cols < T:
            row_str += "..."
        
        # Add appropriate bracket
        if i == 0 and display_rows == 1:
            lines.append(f"⎢{row_str}⎥")
        elif i == 0:
            lines.append(f"⎢{row_str}⎥")
        elif i == display_rows - 1:
            lines.append(f"⎣{row_str}⎦")
        else:
            lines.append(f"⎢{row_str}⎥")
    
    # Add ellipsis if we're truncating rows
    if display_rows < F_in:
        lines.append("⎢ ... ⎥")
        lines.append("⎦")
    
    return "\n".join(lines)


def format_matrix_simple(matrix, precision=1):
    """
    Simple matrix formatting - more compact for small displays.
    """
    F_in, T = matrix.shape
    
    if F_in == 1 and T <= 6:
        # Single row - display horizontally
        row = matrix[0]
        if precision == 0:
            return "[" + " ".join(f"{int(val)}" for val in row) + "]"
        else:
            return "[" + " ".join(f"{val:.{precision}f}" for val in row) + "]"
    
    elif F_in <= 4 and T <= 4:
        # Small matrix - show compactly
        lines = []
        for i in range(F_in):
            if precision == 0:
                row_str = " ".join(f"{int(val)}" for val in matrix[i])
            else:
                row_str = " ".join(f"{val:.{precision}f}" for val in matrix[i])
            lines.append(f"[{row_str}]")
        return "\n".join(lines)
    
    else:
        # Large matrix - show shape and sample values
        return f"({F_in}×{T})\n[{matrix[0,0]:.{precision}f}...{matrix[-1,-1]:.{precision}f}]"


# Plot the graph using networkx and matplotlib
def plot_graph(data: Data, save_path=None, edge_style="curved", show_feature_matrices=True):
    """
    Plots the pyg_graph using networkx and matplotlib with bent/curved edges.
    
    Args:
        data: PyG Data object
        save_path: Path to save the plot
        edge_style: "curved", "varied", "spring", "outward", "bundled", "force_directed", or "straight"
        show_feature_matrices: Whether to display feature matrices above nodes
    """

    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    edges = list(zip(edge_index[0], edge_index[1], data.edge_attr.numpy()))
    G.add_weighted_edges_from(edges)

    # if data.x is not None:
    #     G = nx.relabel_nodes(G, dict(enumerate(data.x.numpy().flatten())))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    
    # Use different layouts for different edge styles
    if edge_style == "force_directed":
        # Use a layout that naturally spreads nodes for better edge separation
        pos = nx.spring_layout(G, k=2, iterations=50)  # k controls node separation
    else:
        pos = nx.spring_layout(G)
    
    if data.pos is None:
        print("pos.values().shape: ", torch.from_numpy(np.stack(list(pos.values()), axis=0)).shape)
        data.pos = torch.from_numpy(np.stack(list(pos.values()), axis=0))

    else:
        # Convert data.pos to a dictionary for networkx
        pos = {i: data.pos[i].numpy() for i in range(data.pos.shape[0])}

    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
    
    if edge_style == "curved":
        # Method 1: Simple curved edges with fixed curvature
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True,
                              connectionstyle="arc3,rad=0.2", 
                              arrowstyle='-', width=1.5)
    
    elif edge_style == "varied":
        # Method 2: Varied curvature for different edges
        import random
        for edge in G.edges():
            rad = random.uniform(0.1, 0.3)  # Random curvature
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax, arrows=True,
                                  edge_color='gray',
                                  connectionstyle=f"arc3,rad={rad}",
                                  arrowstyle='-', width=1.5)
    
    elif edge_style == "spring":
        # Method 3: Spring-like curved edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True,
                              connectionstyle="arc3,rad=0.3",
                              arrowstyle='-', width=1.5, alpha=0.7)
    
    elif edge_style == "outward":
        # Method 4: Maximum outward bending - automatic collision avoidance
        # Calculate optimal curvature based on edge density and node positions
        for edge in G.edges():
            node1, node2 = edge
            pos1, pos2 = pos[node1], pos[node2]
            
            # Calculate distance and angle between nodes
            dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Base curvature scales with inverse distance (closer nodes = more curve)
            base_rad = min(0.5, 0.3 / max(distance, 0.1))
            
            # Check for nearby parallel edges and increase curvature accordingly
            parallel_count = 0
            for other_edge in G.edges():
                if other_edge != edge:
                    other_node1, other_node2 = other_edge
                    # Check if edges are roughly parallel
                    other_pos1, other_pos2 = pos[other_node1], pos[other_node2]
                    other_dx = other_pos2[0] - other_pos1[0]
                    other_dy = other_pos2[1] - other_pos1[1]
                    
                    # Simple parallel check based on angle similarity
                    if abs(dx * other_dy - dy * other_dx) < 0.1:  # Cross product ~ 0
                        parallel_count += 1
            
            # Increase curvature for parallel edges
            final_rad = base_rad * (1 + parallel_count * 0.2)
            final_rad = min(final_rad, 0.8)  # Cap maximum curvature
            
            # Alternate direction for parallel edges
            direction = 1 if hash(edge) % 2 == 0 else -1
            final_rad *= direction
            
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax, arrows=True,
                                  edge_color='gray',
                                  connectionstyle=f"arc3,rad={final_rad}",
                                  arrowstyle='-', width=1.5)
    
    elif edge_style == "bundled":
        # Method 5: Edge bundling approach - group similar edges
        # This creates natural curves that avoid overlaps
        edge_groups = {}
        for edge in G.edges():
            node1, node2 = edge
            pos1, pos2 = pos[node1], pos[node2]
            
            # Create key based on general direction (quantized angle)
            angle = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])
            angle_key = round(angle / (np.pi/8)) * (np.pi/8)  # Quantize to 8 directions
            
            if angle_key not in edge_groups:
                edge_groups[angle_key] = []
            edge_groups[angle_key].append(edge)
        
        # Draw each group with increasing curvature
        for angle_key, edges in edge_groups.items():
            for i, edge in enumerate(edges):
                # Spread edges in group with different curvatures
                rad = 0.1 + i * 0.15
                direction = 1 if i % 2 == 0 else -1
                final_rad = rad * direction
                
                nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax, arrows=True,
                                      edge_color='gray',
                                      connectionstyle=f"arc3,rad={final_rad}",
                                      arrowstyle='-', width=1.5)
    
    elif edge_style == "force_directed":
        # Method 6: Use force-directed layout + automatic maximum bending
        # This combines better node positioning with intelligent edge curving
        for edge in G.edges():
            node1, node2 = edge
            pos1, pos2 = pos[node1], pos[node2]
            
            # Calculate vector from node1 to node2
            edge_vector = np.array([pos2[0] - pos1[0], pos2[1] - pos1[1]])
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length > 0:
                # Find center point
                center = np.array([(pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2])
                
                # Count how many other edges pass near this edge's path
                interference_count = 0
                for other_edge in G.edges():
                    if other_edge != edge:
                        other_node1, other_node2 = other_edge
                        other_pos1, other_pos2 = pos[other_node1], pos[other_node2]
                        other_center = np.array([(other_pos1[0] + other_pos2[0])/2, 
                                               (other_pos1[1] + other_pos2[1])/2])
                        
                        # Check if edge centers are close (potential interference)
                        if np.linalg.norm(center - other_center) < 0.2:
                            interference_count += 1
                
                # Scale curvature based on interference and edge length
                base_curvature = 0.3 + interference_count * 0.2
                length_factor = min(1.0, 0.5 / max(edge_length, 0.1))
                final_curvature = min(base_curvature * length_factor, 0.8)
                
                # Alternate direction based on edge hash for consistency
                direction = 1 if hash(edge) % 2 == 0 else -1
                final_rad = final_curvature * direction
                
                nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax, arrows=True,
                                      edge_color='gray',
                                      connectionstyle=f"arc3,rad={final_rad}",
                                      arrowstyle='-', width=1.5)
    
    elif edge_style == "auto_max_bend":
        # Method 7: Automatic maximum bending - finds optimal curvature to avoid all overlaps
        def calculate_optimal_curvature(edge, all_edges, positions):
            """Calculate the minimum curvature needed to avoid overlaps."""
            node1, node2 = edge
            pos1, pos2 = positions[node1], positions[node2]
            
            max_curvature = 0.1  # Start with minimal curve
            
            # Check against all other edges
            for other_edge in all_edges:
                if other_edge != edge:
                    other_node1, other_node2 = other_edge
                    other_pos1, other_pos2 = positions[other_node1], positions[other_node2]
                    
                    # Simple geometric intersection check
                    # If edges are close or crossing, increase curvature
                    min_distance = point_to_line_distance(
                        (pos1, pos2), (other_pos1, other_pos2)
                    )
                    
                    if min_distance < 0.15:  # Too close threshold
                        required_curve = 0.3 / max(min_distance, 0.05)
                        max_curvature = max(max_curvature, required_curve)
            
            return min(max_curvature, 0.9)  # Cap at reasonable maximum
        
        def point_to_line_distance(line1, line2):
            """Simple approximation of minimum distance between two line segments."""
            p1, p2 = line1
            p3, p4 = line2
            
            # Use midpoint distance as approximation
            mid1 = [(p1[0] + p2[0])/2, (p1[1] + p2[1])/2]
            mid2 = [(p3[0] + p4[0])/2, (p3[1] + p4[1])/2]
            
            return np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
        
        # Apply optimal curvature to each edge
        for i, edge in enumerate(G.edges()):
            optimal_curve = calculate_optimal_curvature(edge, list(G.edges()), pos)
            
            # Alternate direction for visual separation
            direction = 1 if i % 2 == 0 else -1
            final_rad = optimal_curve * direction
            
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax, arrows=True,
                                  edge_color='gray',
                                  connectionstyle=f"arc3,rad={final_rad}",
                                  arrowstyle='-', width=1.5)
    
    else:  # straight
        # Default straight edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Display feature matrices above nodes
    if show_feature_matrices and data.x is not None:
        data_clone = data.clone()
        data_clone.x = data.x[:, :4, :4]  # Use smaller feature matrix for display
        display_feature_matrices(data_clone, pos, ax, fontsize=8, precision=0)

    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

    return data





def demo_graph_time_conv_block(F_in, F_out, N, T, K):

    seed_everything(seed=42)
    
    # Create output directory
    import os
    os.makedirs("./gtconv", exist_ok=True)

    # Generate a random N x N weighted adjacency matrix
    adj = create_rand_adj(N, sparsity=0.2, add_self_loops=False, remove_self_loops=True)

    # Generate a random graph signal X that is a tensor of shape N x F_in x T
    x = torch.randint(0, 10, (N, F_in, T)) / 1.0 # / 10.0 - 0.5  # Values between -0.5 and 0.5


    # Convert the adjacency matrix to a PyG graph
    data = get_pyg_graph(adj, x)

    # Plot with feature matrices displayed
    print("Plotting graph with feature matrices...")
    data = plot_graph(data, save_path="./gtconv/graph_with_features.png", 
                     edge_style="auto_max_bend", show_feature_matrices=True)
    
    # Also create a version without feature matrices for comparison
    data = plot_graph(data, save_path="./gtconv/graph_clean.png", 
                     edge_style="auto_max_bend", show_feature_matrices=False)
    
    print("Saved graphs: graph_with_features.png and graph_clean.png")

    print("data: ", data)

    with torch.inference_mode(True):
        ### Implement k-hop graph convolution ###
        conv = TAGConv(in_channels=F_in * T, out_channels=F_in * T, K=K, normalize=False,
                       aggr='mean')

        x_flattened = data.x.swapaxes(1, 2).reshape(-1, conv.in_channels)  # Flatten F_in and T for convolution
        x_flattened_conv = conv(x_flattened, data.edge_index, data.edge_attr.squeeze(-1))

        assert x_flattened_conv.shape == (N, conv.out_channels), "Output shape mismatch."

        data_conv = data.clone()
        data_conv.x = x_flattened_conv.reshape(N, T, F_in).swapaxes(1, 2)
        # Also create a version without feature matrices for comparison
        data_conv = plot_graph(data_conv, save_path="./gtconv/graph_after_conv.png", 
                        edge_style="auto_max_bend", show_feature_matrices=True)
        

        # temporal_conv(data_conv, time_dim = 2)


    return 0



if __name__ == "__main__":

    F_in, F_out = 16, 32 # number of input and output features
    T = 10 # number of time steps
    N = 8 # number of nodes
    K = 2 # number of hops

    demo_graph_time_conv_block(F_in, F_out, N, T, K)