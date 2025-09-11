from collections import defaultdict
from typing import Union
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Dataset
from datasets.SP100Stocks import SP100Stocks
from torch_geometric.loader import DataLoader

from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset

from utils.plot_utils import plot_batched_data
from utils.stock_data_utils import compute_stocks_correlation_graph, group_SP100_stocks_by_sector, merge_stock_relation_graphs, stack_stock_relation_graphs


def index_chunk_util_fn(dataset_indices, chunk_size, shuffle_chunks=False):
    """
    Split the dataset indices into smaller chunks of a specified size.

    Args:
        dataset_indices (list[int]): The indices of the dataset to split.
        chunk_size (int): The size of each chunk.
        shuffle_chunks (bool): Whether to shuffle the chunks.

    Returns:
        list[list[int]]: A list of dataset chunk indices.
    """
    chunks = [dataset_indices[i:i + chunk_size] for i in range(0, len(dataset_indices), chunk_size)]
    if shuffle_chunks:
        np.random.shuffle(chunks)
    return chunks


def split_dataset_indices(dataset_indices, arg_groups):

    # Split the dataset into train/val/test sets
    chunk_size = (arg_groups["dataset"].future_window + arg_groups["dataset"].past_window) * 2
    chunks = index_chunk_util_fn(dataset_indices=dataset_indices,
                                                chunk_size=chunk_size,
                                                shuffle_chunks=True
    )
    
    num_train_chunks = int(len(chunks) * arg_groups["dataset"].train_dataset_fraction)
    num_val_chunks = int((len(chunks) - num_train_chunks) / 2)
    num_test_chunks = len(chunks) - num_train_chunks - num_val_chunks
    print("Dataset split into {} chunks of size up to {}.".format(len(chunks), chunk_size))

    assert num_train_chunks > 0, "Training set is empty. Please adjust the train_dataset_fraction or chunk_size."
    assert num_val_chunks > 0, "Validation set is empty. Please adjust the train_dataset_fraction or chunk_size."
    assert num_test_chunks > 0, "Test set is empty. Please adjust the train_dataset_fraction or chunk_size."

    train_dataset_indices = torch.LongTensor(sorted([idx for chunk in chunks[:num_train_chunks] for idx in chunk]))
    val_dataset_indices = torch.LongTensor(sorted([idx for chunk in chunks[num_train_chunks:num_train_chunks + num_val_chunks] for idx in chunk]))
    test_dataset_indices = torch.LongTensor(sorted([idx for chunk in chunks[num_train_chunks + num_val_chunks:] for idx in chunk]))
    train_val_dataset_indices = torch.LongTensor(sorted([idx for chunk in chunks[:num_val_chunks] for idx in chunk if idx in train_dataset_indices]))

    assert len(train_dataset_indices) > 0, "Training set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
    assert len(val_dataset_indices) > 0, "Validation set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
    assert len(test_dataset_indices) > 0, "Test set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
    assert len(train_val_dataset_indices) > 0, "Train-Val set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."

    return train_dataset_indices, val_dataset_indices, test_dataset_indices, train_val_dataset_indices


def split_dataset(args, arg_groups, dataset):

    # # Split the dataset into train/val/test sets
    # train_part = arg_groups["dataset"].train_dataset_fraction
    # val_part = (1 - train_part) / 2

    # train_dataset, val_dataset, test_dataset = dataset[:int(len(dataset) * train_part)], \
    #     dataset[int(len(dataset) * train_part):int(len(dataset) * (train_part + val_part)):arg_groups["dataset"].future_window], \
    #         dataset[int(len(dataset) * (train_part + val_part))::arg_groups["dataset"].future_window]
    
    # train_val_dataset = dataset[:int(len(dataset) * val_part):arg_groups["dataset"].future_window]

    
    train_idx, val_idx, test_idx, train_val_idx = \
        split_dataset_indices(dataset_indices = list(range(len(dataset))),
                              arg_groups = arg_groups
        )

    print("Val dataset indices: ", val_idx)
    print("Test dataset indices: ", test_idx)
    print("Train-Val dataset indices: ", train_val_idx)

    # train_dataset = Subset(dataset, train_dataset_indices)
    # val_dataset = Subset(dataset, val_dataset_indices)
    # test_dataset = Subset(dataset, test_dataset_indices)
    # train_val_dataset = Subset(dataset, train_val_dataset_indices)

    # return train_dataset, val_dataset, test_dataset, train_val_dataset


def create_dataloaders(args, arg_groups, accelerator, dataset) -> Union[DataLoader, dict[str, DataLoader]]:

    batch_size_train = arg_groups['CD-train-algo'].x_batch_size * arg_groups["CD-train-algo"].batch_size

    train_idx, val_idx, test_idx, train_val_idx = \
        split_dataset_indices(dataset_indices = list(range(len(dataset))),
                              arg_groups = arg_groups
        )
    
    train_dataset = dataset.index_select(train_idx)
    val_dataset = dataset.index_select(val_idx[::arg_groups["dataset"].future_window])
    test_dataset = dataset.index_select(test_idx[::arg_groups["dataset"].future_window])
    train_val_dataset = dataset.index_select(train_val_idx[::arg_groups["dataset"].future_window])

    # # Split the dataset into train/val/test sets
    # train_part = arg_groups["dataset"].train_dataset_fraction
    # val_part = (1 - train_part) / 2

    # train_dataset, val_dataset, test_dataset = dataset[:int(len(dataset) * train_part)], \
    #     dataset[int(len(dataset) * train_part):int(len(dataset) * (train_part + val_part)):arg_groups["dataset"].future_window], \
    #         dataset[int(len(dataset) * (train_part + val_part))::arg_groups["dataset"].future_window]
    
    # train_val_dataset = dataset[:int(len(dataset) * val_part):arg_groups["dataset"].future_window]



    # accelerator.print('Train dataset: ', train_dataset)
    # accelerator.print("Selected indices from dataset: ", dataset.index_select(torch.LongTensor([0, 2, 3, 4])))
    # accelerator.print('Dataset[selected_indices]: ', dataset[torch.LongTensor([0, 2, 3, 4])])
    # dummy_dataset = dataset[torch.randperm(len(dataset))]

    # dummy_dataloader = DataLoader(dummy_dataset, batch_size=batch_size_train, shuffle=False, pin_memory=True, drop_last=False,
    #                              follow_batch=["y"],
    #                              exclude_keys=["close_price", "close_price_y", "timestamps"])
    
    # for dummy_data in dummy_dataloader:
    #     accelerator.print("Dummy dataloader batch:")
    #     accelerator.print(dummy_data)
    #     accelerator.print("Dummy dataloader batch sizes:")
    #     accelerator.print({key: value.size() for key, value in dummy_data.items() if torch.is_tensor(value)})
    #     accelerator.print("Dummy dataloader batch keys:", dummy_data.keys())
    #     break

    # # Terminate the script 
    # raise ValueError("Terminating the script after printing the datasets for debugging purposes.")

    accelerator.print(f"Train dataset: {len(train_dataset)} / {len(dataset)} samples.")
    accelerator.print(f"Validation dataset: {len(val_dataset)} / {len(dataset)} samples with {arg_groups['dataset'].future_window} delta timesteps in-between.")
    accelerator.print(f"Test dataset: {len(test_dataset)} / {len(dataset)} samples with {arg_groups['dataset'].future_window} delta timesteps in-between.")
    accelerator.print(f"Train-Val dataset: {len(train_val_dataset)} / {len(dataset)} samples with {arg_groups['dataset'].future_window} delta timesteps in-between.")


    if arg_groups["CD-train-algo"].batch_size > 1:
        # Create weights (uniform for simplicity)
        weights = torch.ones(len(train_dataset))
        # Create sampler - key point: num_samples can be any value
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=batch_size_train * 10,  # e.g., 1500 samples per epoch
            replacement=True  # This is crucial!
        )

        accelerator.print("Using weighted random sampler for training dataloader.")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train,
                                       sampler=sampler,
                                       pin_memory=True,
                                       follow_batch=["y"],
                                       exclude_keys=["close_price", "close_price_y", "timestamps"])
    else:
        accelerator.print("Using standard sequential sampler for training dataloader with shuffling.")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train,
                                       shuffle=True,
                                       pin_memory=True,
                                       follow_batch=["y"],
                                       exclude_keys=["close_price", "close_price_y", "timestamps"])

    dataloaders = {
        "train": train_dataloader,             
        "train-val": DataLoader(train_val_dataset, batch_size=len(train_val_dataset), shuffle=False, pin_memory=True, drop_last=True,
                                follow_batch=["y"]),
        "val": DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True, drop_last=True,
                          follow_batch=["y"]),
        "test": DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True, drop_last=True,
                           follow_batch=["y"])
    }

    # if batch_size_train > len(train_dataset):
    #     # Create weights (uniform for simplicity)
    #     weights = torch.ones(len(train_dataset))
    #     # Create sampler - key point: num_samples can be any value
    #     sampler = WeightedRandomSampler(
    #         weights=weights,
    #         num_samples=batch_size_train,  # e.g., 1500 samples per epoch
    #         replacement=True  # This is crucial!
    #     )
    # else:
    #     sampler = None

    # dataloaders = {
    #     "train": DataLoader(train_dataset, batch_size=batch_size_train,
    #                         sampler=sampler,
    #                         shuffle=True if sampler is not None else False,
    #                         pin_memory=True,
    #                         follow_batch=["y"]),

    #     "train-val": DataLoader(train_val_dataset, batch_size=len(train_val_dataset), shuffle=False, pin_memory=True, drop_last=True,
    #                             follow_batch=["y"]),
    #     "val": DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True, drop_last=True,
    #                       follow_batch=["y"]),
    #     "test": DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True, drop_last=True,
    #                        follow_batch=["y"])
    # }
    return dataloaders



def run_create_pyg_dataloaders_pipeline(args, arg_groups, accelerator, dataset):
    dataloaders = create_dataloaders(args, arg_groups, accelerator, dataset)

    # Visualize dataloader evolution for test networks
    data = next(iter(dataloaders["test"]))
    stocks_idx = np.random.choice(100, 4)
    num_features = data.x.shape[1]

    target_column_name = dataloaders['test'].dataset.dataset.target_column_name if isinstance(dataloaders['test'].dataset, Subset) else dataloaders['test'].dataset.target_column_name
    info_features = dataloaders['test'].dataset.dataset.info['Features'] if isinstance(dataloaders['test'].dataset, Subset) else dataloaders['test'].dataset.info['Features']
    
    for plot_target in ["Closing_Price", ("y", target_column_name), "Closing_Price_y"] + \
        [(f'x[{i}]', info_features[i]) for i in range(num_features)]:
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
                          corr_threshold=arg_groups["dataset"].corr_threshold if arg_groups["dataset"].temporal_correlation_graph is True else None, # 0.7
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
        
        if arg_groups["CD-model"].edge_features_nb == 1:

            merged_adj, merged_graph = merge_stock_relation_graphs(fundamentals_corr_np = fundamentals_corr_np,
                                                    stocks = stocks,
                                                    save_dir = save_dir,
                                                    save_ext = save_ext,
                                                    corr_threshold = arg_groups["dataset"].corr_threshold,
                                                    sector_bonus = arg_groups["dataset"].sector_bonus
                                                    )
        
            # Save the merged adjacency graph to file.
            np.save(f'{save_dir}/adj.npy', merged_adj)


        elif arg_groups["CD-model"].edge_features_nb > 1:
            adj_list = []
            delta_thr = 0.05
            for thresh in [arg_groups["dataset"].corr_threshold,
                           arg_groups["dataset"].corr_threshold - delta_thr,
                           arg_groups["dataset"].corr_threshold + delta_thr,
                            arg_groups["dataset"].corr_threshold - 2 * delta_thr,
                            arg_groups["dataset"].corr_threshold + 2 * delta_thr
                           ][:arg_groups["CD-model"].edge_features_nb - 1]:  # -1 because we add the sector graph at the end
                
                fundamentals_corr_np, _ = compute_stocks_correlation_graph(fundamentals = fundamentals,
                                                                        save_dir = save_dir,
                                                                        save_ext = save_ext,
                                                                        corr_threshold = thresh   
                                                                        )
                # Note that fundamentals_corr_np is not thresholded, but the adjacency matrix returned by the function is thresholded.
                adj = fundamentals_corr_np * (abs(fundamentals_corr_np) > thresh)
                adj_list.append(adj)


            stacked_adj, stacked_graph = stack_stock_relation_graphs(adj_list = adj_list[:arg_groups["CD-model"].edge_features_nb - 1] + [adj_stocks],
                                                                    stocks = stocks,
                                                                    save_dir = save_dir,
                                                                    save_ext = save_ext
                                                                    )
            # Save the stacked adjacency graph to file.
            np.save(f'{save_dir}/adj.npy', stacked_adj)

        else:
            raise ValueError("edge_features_nb must be >= 1.")

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
