from typing import Union
import numpy as np
import pandas as pd
import torch


def get_graph_in_pyg_format(values_path: str, adj_path: str,
							target_column_name: str = "NormClose") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
	"""
	Creates the PyTorch Geometric graph data from the stock price data and adjacency matrix.
	:param values_path: Path of the CSV file containing the stock price data
	:param adj_path: Path of the NumPy file containing the adjacency matrix
	:return: The graph data in PyTorch Geometric format
		# x: Node features (nodes_nb, timestamps_nb, features_nb)
		x: Node features (nodes_nb, features_nb, timestamps_nb)
		close_prices: Close prices (nodes_nb, timestamps_nb)
		edge_index: Edge index (2, edge_nb)
		edge_weight: Edge weight (edge_nb,)
		info: Additional info dictionary
	"""
	values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
	adj = np.load(adj_path)
	nodes_nb, edge_nb = len(adj), np.count_nonzero(adj)
	x = torch.tensor(
		values.drop(columns=["Close"]).to_numpy().reshape((nodes_nb, -1, values.shape[1] - 1)), dtype=torch.float32
	) # shape (nodes_nb, timestamps_nb, features_nb)
	x = x.transpose(1, 2) # shape (nodes_nb, features_nb, timestamps_nb)
	close_prices = torch.tensor(
		values[["Close"]].to_numpy().reshape((nodes_nb, -1)), dtype=torch.float32
	)
	edge_index, edge_weight = torch.zeros((2, edge_nb), dtype=torch.long), torch.zeros((edge_nb,), dtype=torch.float32)
	count = 0
	for i in range(nodes_nb):
		for j in range(nodes_nb):
			if (weight := adj[i, j]) != 0:
				edge_index[0, count], edge_index[1, count] = i, j
				edge_weight[count] = weight
				count += 1


	info_dict = {"Features": values.drop(columns=["Close"]).columns.tolist(),
			  		"Target": target_column_name,
				  	"Num_nodes": nodes_nb
	}

	return x, close_prices, edge_index, edge_weight, info_dict



def get_column_names(values_path_or_df: Union[str, pd.DataFrame]) -> list[str]:
	"""
	Retrieves the column names (features) of the dataset stocks
	"""
	if isinstance(values_path_or_df, str):
		values = pd.read_csv(values_path_or_df).set_index(['Symbol', 'Date'])
	elif isinstance(values_path_or_df, pd.DataFrame):
		values = values_path_or_df
	else:
		raise NotImplementedError(f"Unsupported type for values_path: {type(values_path_or_df)}")

	return values.columns.tolist()


def get_column_idx(values_path_or_df: Union[str, pd.DataFrame], column_name: str) -> int:
	"""
	Retrieves the index of a specific column in the dataset stocks
	"""
	if isinstance(values_path_or_df, str):
		values = pd.read_csv(values_path_or_df).set_index(['Symbol', 'Date'])
	elif isinstance(values_path_or_df, pd.DataFrame):
		values = values_path_or_df
	else:
		raise NotImplementedError(f"Unsupported type for values_path: {type(values_path_or_df)}")

	return values.columns.get_loc(column_name)


def get_stocks_labels(values_path: str) -> list[str]:
	"""
	Retrieves the labels (symbols) of the dataset stocks
	:return: The list of stock labels
	"""
	return pd.read_csv(values_path)["Symbol"].unique().tolist()




	


