import os.path as osp
from typing import Callable

import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
# from notebooks.datasets.utils import get_graph_in_pyg_format
from datasets.utils import combine_edge_features, get_graph_in_pyg_format, get_column_idx
from utils.covariance_graph_utils import target_to_graph


class StocksDataDiffusion(Data):
	def __init__(self, x, edge_index, edge_weight, close_price, y, close_price_y, timestamp, stocks_index, info = None):
		super().__init__(x=x, edge_index=edge_index, edge_weight=edge_weight, close_price=close_price, y=y, close_price_y=close_price_y, timestamp=timestamp, stocks_index=stocks_index)
		if info is not None:
			self.info = info
			# for key in info:
			# 	setattr(self, key, info[key])
			# self.info = info

	def __inc__(self, key, value, *args, **kwargs):
		if key in ["timestamp", "stocks_index"]:
			return 0
		
		if key == "info":
			return 0
		# Don't increment any info-related attributes
		# if key in ["Target", "Features", "Num_nodes"]:
		# 	return 0
		return super().__inc__(key, value, *args, **kwargs)


	def __cat_dim__(self, key, value, *args, **kwargs):
		# Default behavior for specific keys
		# if key in ['adj']:
		# 	return 1 # concatenate along the node dimension

		if key in ["timestamp", "stocks_index"]:
			return 0  # Batch timestamps and stocks_index
		
		# Don't batch info dictionary or any of its keys
		if key == "info":
			return None
		
		if key in ["Features", "Target"]:
			return 0

		# if key in self.info:
		# 	return None  # Don't batch - keeps only first occurrence
		
		return super().__cat_dim__(key, value, *args, **kwargs)

	


class SP100Stocks(Dataset):
	"""
	Stock price data for the S&P 100 companies.
	"""

	def __init__(self, root: str = "../data/SP100/", values_file_name: str = "values.csv", adj_file_name: str = "adj.npy", past_window: int = 25, future_window: int = 1,
			  target_column_name: str = "NormClose",
			  corr_threshold: float = None,
			  force_reload: bool = False, transform: Callable = None):
		self.values_file_name = values_file_name
		self.adj_file_name = adj_file_name
		self.past_window = past_window
		self.future_window = future_window
		self.target_column_name = target_column_name
		self.corr_threshold = corr_threshold
		super().__init__(root, force_reload=force_reload, transform=transform)

	@property
	def raw_file_names(self) -> list[str]:
		return [
			self.values_file_name, self.adj_file_name
		]

	@property
	def processed_file_names(self) -> list[str]:
		return [
			f'timestep_{idx}.pt' for idx in range(len(self))
		]

	def download(self) -> None:
		pass

	def process(self) -> None:
		x, close_prices, edge_index, edge_weight, info_dict = get_graph_in_pyg_format(
			values_path=osp.join(self.root, f"raw/{self.values_file_name}"),
			adj_path=osp.join(self.root, f"raw/{self.adj_file_name}"),
			target_column_name=self.target_column_name
		)

		self.info = info_dict
		target_column_idx = get_column_idx(values_path_or_df=osp.join(self.root, f"raw/{self.values_file_name}"),
											column_name=info_dict["Target"]
											) - 1 # subtract one because we dropped the first column from values
		print("Target column name / index:", f"{info_dict['Target']} / {target_column_idx}")

		
		timestamps = []
		for idx in range(x.shape[2] - self.past_window - self.future_window):
			if self.corr_threshold is not None:	
				# Recompute edge_index and edge_weight based on correlation of closing prices in the current window
				# window_close_prices = close_prices[:, idx:idx + self.past_window]
				past_window_y = x[:, target_column_idx, idx:idx + self.past_window]

				# print(f"Last stock target values in all window: {x[-1, target_column_idx, :]}")

				# print(f"At time index {idx}, past window {info_dict['Target']} is {past_window_y}.")
				edge_index_t_corr, edge_weight_t_corr = target_to_graph(
					past_window_y,
					method="correlation", 
					threshold=self.corr_threshold,
					use_absolute=True,
					remove_self_loops=True,
					normalize=True,
					save_path=self.root + f"/raw/temporal_corr_{idx}.pdf" if idx < 10 else None
				)

				# Make sure none of the edge weights are NaN
				assert not torch.isnan(edge_weight_t_corr).any(), "Edge weights contain {} many NaN values.".format(torch.isnan(edge_weight_t_corr).sum().item())

				print(f"At time index {idx}, computed temporal correlation graph with {edge_weight_t_corr.size(0)} edges based on closing prices in the window [{idx}, {idx + self.past_window}).") if idx < 10 else None

				# Combine your graphs
				combined_edge_index, combined_edge_weight = combine_edge_features(
					edge_index, edge_weight,           # Existing graph
					edge_index_t_corr, edge_weight_t_corr,  # Windowed temporal correlation graph
					fill_value=0.0,                     # Value for missing edges
					debug_print= idx < 10               # Print debug info for first 10 time steps
				)

				assert combined_edge_weight.dim() == 2, "combined_edge_weight should have shape (num_edges, num_features)"
				print(f"Temporal correlation graph added {combined_edge_weight.size(0) - edge_weight.size(0)} additional edges to multi-graph.") if idx < 10 else None

				# Make sure none of the edge weights are NaN
				assert not torch.isnan(combined_edge_weight).any(), "Edge weights contain {} many NaN values.".format(torch.isnan(combined_edge_weight).sum().item())

			else:
				combined_edge_index, combined_edge_weight = edge_index, edge_weight
			
			timestamps.append(
				StocksDataDiffusion(
					x=x[:, :, idx:idx + self.past_window],
					edge_index=combined_edge_index,
					edge_weight=combined_edge_weight,
					close_price=close_prices[:, idx:idx + self.past_window],
					y=x[:, target_column_idx, idx + self.past_window:idx + self.past_window + self.future_window],
					close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
					stocks_index=torch.arange(x.shape[0]),
					timestamp=torch.LongTensor([idx]).repeat(x.shape[0]), # Repeat for each node in the batch
					info=info_dict
				)
			)
		

		# Original code without dynamic graph computation
		# timestamps = [
		# 	StocksDataDiffusion(
		# 		x=x[:, :, idx:idx + self.past_window],
		# 		edge_index=edge_index,
		# 		edge_weight=edge_weight,
		# 		close_price=close_prices[:, idx:idx + self.past_window],
		# 		y=x[:, target_column_idx, idx + self.past_window:idx + self.past_window + self.future_window],
		# 		close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
		# 		stocks_index=torch.arange(x.shape[0]),
		# 		timestamp=torch.LongTensor([idx]).repeat(x.shape[0]), # Repeat for each node in the batch
		# 		info=info_dict
		# 	) for idx in range(x.shape[2] - self.past_window - self.future_window)
		# ]
		for t, timestep in enumerate(timestamps):
			torch.save(
				timestep, osp.join(self.processed_dir, f"timestep_{t}.pt")
			)

	def len(self) -> int:
		values = pd.read_csv(self.raw_paths[0]).set_index(['Symbol', 'Date'])
		return len(values.loc[values.index[0][0]]) - self.past_window - self.future_window

	def get(self, idx: int) -> Data:
		data = torch.load(osp.join(self.processed_dir, f'timestep_{idx}.pt'))
		return data
