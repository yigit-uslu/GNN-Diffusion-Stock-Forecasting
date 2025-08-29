import os 
import pandas as pd 
import numpy as np 
import networkx as nx
from typing import Tuple

from utils.graph_utils import draw_stocks_graph_by_correlation, draw_stocks_graph_by_sector, draw_merged_stocks_graph
from utils.plot_utils import plot_stock_correlation_distribution


def group_SP100_stocks_by_sector(stocks, save_dir = None, save_ext = ".pdf") -> Tuple[np.ndarray, nx.Graph]:
    ## Grouping by sector
    # The first step is to create a graph of the stocks linking every stock that belong to the same activity sector.

    adj_stocks = np.array([
	    [stocks.loc[stock1, 'Sector'] == stocks.loc[stock2, 'Sector'] * (stock1 != stock2) for stock1 in stocks.index] for stock2 in stocks.index
            ]).astype(int)
    
    stocks_graph = nx.from_numpy_array(adj_stocks)
    stocks_graph = nx.relabel_nodes(stocks_graph, dict(enumerate(stocks.index)))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/S&P_100_stocks_graph_by_sector" + save_ext
        draw_stocks_graph_by_sector(stocks_graph, save_path)
        
    return adj_stocks, stocks_graph



def compute_stocks_correlation_graph(fundamentals, save_dir, save_ext = ".pdf", corr_method = "spearman", corr_threshold = .7) -> Tuple[np.ndarray, nx.Graph]:

    fundamentals_corr = fundamentals.transpose().corr(method=corr_method)
    fundamentals_corr = (fundamentals_corr - (fundamentals_corr == 1))  # Remove self-correlation
    fundamentals_corr.head(n=10)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plot_stock_correlation_distribution(fundamentals_corr, save_path = f"{save_dir}/S&P_100_stocks_correlation_distribution" + save_ext)

    fundamentals_corr_np = fundamentals_corr.to_numpy()

    adj_fundamentals_corr = (fundamentals_corr_np * (abs(fundamentals_corr_np) > corr_threshold).astype(int))
    fundamentals_corr_graph = nx.from_numpy_array(adj_fundamentals_corr)
    fundamentals_corr_graph = nx.relabel_nodes(fundamentals_corr_graph, dict(enumerate(fundamentals_corr.index)))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/S&P_100_stocks_graph_by_correlation" + save_ext
        draw_stocks_graph_by_correlation(fundamentals_corr_graph, save_path)

    return fundamentals_corr_np, fundamentals_corr_graph




def merge_stock_relation_graphs(fundamentals_corr_np, stocks, save_dir, save_ext, corr_threshold = .7, sector_bonus = .05) -> Tuple[np.ndarray, nx.Graph]:
    """ 
    Merge sector relationship and the correlation graphs to build the final adjacency matrix. A correlation bonus is given to two stocks sharing the same sector.
    """

    share_sector = pd.get_dummies(stocks[["Sector"]]).transpose().corr().to_numpy().astype(int) - np.eye(len(stocks), dtype=int)
    # abs because GCNConv only accepts positive weights
    adj = abs(fundamentals_corr_np) + share_sector * sector_bonus
    adj = adj * (abs(adj) > corr_threshold)  
    adj = adj / adj.max()

    graph = nx.from_numpy_array(adj)
    graph = nx.relabel_nodes(graph, dict(enumerate(stocks.index)))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/S&P_100_stocks_graph_merged" + save_ext
        draw_merged_stocks_graph(graph, save_path)

    # plt.figure(figsize=(12, 5))
    # nx.draw(graph, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_weight='bold', font_color='black', pos=nx.spring_layout(graph, k=.5))
    # plt.title(f'Stocks Graph (Mean degree: {np.mean([degree for node, degree in graph.degree]):.2f}, Density: {nx.density(graph):.4f})')
    # plt.show()

    return adj, graph


    