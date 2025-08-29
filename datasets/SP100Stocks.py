import os.path as osp
from typing import Callable

import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
# from notebooks.datasets.utils import get_graph_in_pyg_format
from datasets.utils import get_graph_in_pyg_format, get_column_idx


class StocksDataDiffusion(Data):
	def __init__(self, x, edge_index, edge_weight, close_price, y, close_price_y, timestamp):
		super().__init__(x=x, edge_index=edge_index, edge_weight=edge_weight, close_price=close_price, y=y, close_price_y=close_price_y, timestamp=timestamp)
		

	def __inc__(self, key, value, *args, **kwargs):
		if key == "timestamp":
			return 0
		return super().__inc__(key, value, *args, **kwargs)

	def __cat_dim__(self, key, value, *args, **kwargs):
		# Default behavior for specific keys
		if key in ['adj']:
			return 1 # concatenate along the node dimension

		return super().__cat_dim__(key, value, *args, **kwargs)

	


class SP100Stocks(Dataset):
	"""
	Stock price data for the S&P 100 companies.
	The graph data built from the notebooks is used.
	"""

	def __init__(self, root: str = "../data/SP100/", values_file_name: str = "values.csv", adj_file_name: str = "adj.npy", past_window: int = 25, future_window: int = 1,
			  target_column_name: str = "NormClose", force_reload: bool = False, transform: Callable = None):
		self.values_file_name = values_file_name
		self.adj_file_name = adj_file_name
		self.past_window = past_window
		self.future_window = future_window
		self.target_column_name = target_column_name
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

		timestamps = [
			StocksDataDiffusion(
				x=x[:, :, idx:idx + self.past_window],
				edge_index=edge_index,
				edge_weight=edge_weight,
				close_price=close_prices[:, idx:idx + self.past_window],
				y=x[:, target_column_idx, idx + self.past_window:idx + self.past_window + self.future_window],
				close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
				timestamp=idx
			) for idx in range(x.shape[2] - self.past_window - self.future_window)
		]
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
