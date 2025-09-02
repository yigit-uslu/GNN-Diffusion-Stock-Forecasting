from collections import defaultdict
from typing import Union
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Dataset
from datasets.SP100Stocks import SP100Stocks
from torch_geometric.loader import DataLoader

from torch.utils.data.sampler import WeightedRandomSampler

from utils.plot_utils import plot_batched_data
from utils.stock_data_utils import compute_stocks_correlation_graph, group_SP100_stocks_by_sector, merge_stock_relation_graphs


def create_dataloaders(args, arg_groups, accelerator, dataset) -> Union[DataLoader, dict[str, DataLoader]]:

    batch_size_train = arg_groups['CD-train-algo'].x_batch_size * arg_groups["CD-train-algo"].batch_size_diffusion

    # Split the dataset into train/val/test sets
    train_part = arg_groups["dataset"].train_dataset_fraction
    val_part = (1 - train_part) / 2

    train_dataset, val_dataset, test_dataset = dataset[:int(len(dataset) * train_part)], \
        dataset[int(len(dataset) * train_part):int(len(dataset) * (train_part + val_part))], \
            dataset[int(len(dataset) * (train_part + val_part)):]
    

    train_val_dataset = dataset[:int(len(dataset) * val_part)]

    accelerator.print(f"Train dataset: {len(train_dataset)} / {len(dataset)} samples.")
    accelerator.print(f"Validation dataset: {len(val_dataset)} / {len(dataset)} samples.")
    accelerator.print(f"Test dataset: {len(test_dataset)} / {len(dataset)} samples.")

    if batch_size_train > len(train_dataset):
        # Create weights (uniform for simplicity)
        weights = torch.ones(len(train_dataset))
        # Create sampler - key point: num_samples can be any value
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=batch_size_train,  # e.g., 1500 samples per epoch
            replacement=True  # This is crucial!
        )
    else:
        sampler = None

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size_train,
                            sampler=sampler,
                            shuffle=True if sampler is not None else False,
                            pin_memory=True,
                            follow_batch=["y"]),

        "train-val": DataLoader(train_val_dataset, batch_size=len(train_val_dataset), shuffle=False, pin_memory=True, drop_last=True,
                                follow_batch=["y"]),
        "val": DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True, drop_last=True,
                          follow_batch=["y"]),
        "test": DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True, drop_last=True,
                           follow_batch=["y"])
    }

    return dataloaders



def run_create_pyg_dataloaders_pipeline(args, arg_groups, accelerator, dataset):
    dataloaders = create_dataloaders(args, arg_groups, accelerator, dataset)

    # Visualize dataloader evolution for test networks
    data = next(iter(dataloaders["test"]))
    stocks_idx = np.random.choice(100, 4)
    num_features = data.x.shape[1]
    for plot_target in ["Closing_Price", ("y", dataloaders['test'].dataset.target_column_name), "Closing_Price_y"] + \
        [(f'x[{i}]', dataloaders['test'].dataset.info['Features'][i]) for i in range(num_features)]:
        plot_batched_data(data = data.clone(),
                        stocks_idx = stocks_idx,
                        plot_target=plot_target,
                        save_dir = f"{args.experiment_name}/data/logs/test"
                        )
        
    return dataloaders


def create_pyg_graphs(args, arg_groups, accelerator, experiment_name) -> Dataset:
    """
    This function creates PyTorch Geometric (PyG) graphs from the dataset.
    """

    root = f"{args.experiment_name}/data"

    accelerator.print("Loading SP100 stock data...")
    values = pd.read_csv(f'{root}/raw/values.csv').set_index(['Symbol', 'Date'])
    values.head()

    assert len(values.index.get_level_values('Symbol').unique()) == 100, "Expected 100 stocks, got {}".format(len(values.index.get_level_values('Symbol').unique()))

    # Assert there is the same number of dates for each stock
    assert all(values.index.get_level_values('Symbol').value_counts() == len(values.index.get_level_values('Date').unique())), "Not all stocks have the same number of dates."


    dataset = SP100Stocks(root=root,
                          values_file_name="values.csv",
                          adj_file_name="adj.npy",
                          past_window=arg_groups["dataset"].past_window, # 25
                          future_window=arg_groups["dataset"].future_window, # 1
                          target_column_name=arg_groups["dataset"].target_column_name, # "NormClose"
                          )
    
    accelerator.print("SP100Stocks dataset: ", dataset)
    accelerator.print("SP100Stocks dataset[0]: ", dataset[0])
    accelerator.print("SP100Stocks dataset[-1]: ", dataset[-1])

    return dataset



def load_and_move_values(load_dir, save_dir):
    """
    Load and move the values file from the load directory to the save directory.
    """

    values = pd.read_csv(f"{load_dir}/values.csv").set_index(['Symbol', 'Date'])
    values.to_csv(f"{save_dir}/values.csv")
    


def run_create_pyg_dataset_pipeline(args, arg_groups, accelerator, experiment_name) -> Union[Dataset, dict[Dataset]]:

    if args.dataset_name == "S&P 100":

        save_dir = f"{args.experiment_name}/data/raw"
        save_ext = ".pdf"

        stocks = pd.read_csv(f"{args.data_dir}/stocks.csv").set_index('Symbol')
        stocks.head(n=10)
        assert len(stocks) == 100, "Expected 100 stocks, got {}".format(len(stocks))

        adj_stocks, stocks_graph = group_SP100_stocks_by_sector(stocks = stocks,
                                                                save_dir = save_dir,
                                                                save_ext = save_ext
                                                                )
        

        fundamentals = pd.read_csv(f"{args.data_dir}/fundamentals.csv").set_index("Symbol")
        fundamentals.head(n=10)
        assert len(fundamentals) == 100, "Expected 100 stocks, got {}".format(len(fundamentals))

        fundamentals_corr_np, fundamentals_corr_graph = compute_stocks_correlation_graph(fundamentals = fundamentals,
                                                                                      save_dir = save_dir,
                                                                                      save_ext = save_ext,
                                                                                      corr_threshold = arg_groups["dataset"].corr_threshold
                                                                                      )

        merged_adj, merged_graph = merge_stock_relation_graphs(fundamentals_corr_np = fundamentals_corr_np,
                                                   stocks = stocks,
                                                   save_dir = save_dir,
                                                   save_ext = save_ext,
                                                   corr_threshold = arg_groups["dataset"].corr_threshold,
                                                   sector_bonus = arg_groups["dataset"].sector_bonus
                                                   )
        
        # Save the merged adjacency graph to file.
        np.save(f'{save_dir}/adj.npy', merged_adj)

        # This function is here only temporarily because values file is precomputed in notebooks for the time being.
        load_and_move_values(load_dir = args.data_dir,
                            save_dir = save_dir
                            )

        # Create PyG Datasets
        dataset = create_pyg_graphs(args, arg_groups, accelerator, experiment_name)


    else:
        raise ValueError(f"Dataset {args.dataset_name} not recognized.")

    
    # # Your dataset creation logic here
    # dataset = defaultdict(list)
    return dataset
