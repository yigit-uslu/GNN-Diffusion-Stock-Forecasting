import argparse
from collections import defaultdict
import json
import numpy as np
import math
import sys, os

from datasets.utils import get_column_names
# from utils.channel_utils import convert_P_max_and_noise_PSD

NUM_NODES = 100

MAX_LOGGED_NETWORKS = 2 # max number of network configs/graphs logged.
MAX_LOGGED_CLIENTS = 20 # max number of clients logged.

CD_LOG_FREQ = 100 # 500 log diffusion-model evaluations every x epochs.
CD_VAL_FREQ = CD_LOG_FREQ // 10 # evaluate diffusion loss over val and test phases every x epochs.
CD_SAVE_MODEL_FREQ = CD_LOG_FREQ # save diffusion model weights every x epochs.
CD_LOG_MSE_CONDITION = 0.25 # 0.25 log diffusion-model evaluations only after training MSE loss drops below this level
GRADIENT_ACCUMULATION_STEPS = 32
DEBUG_TRAIN = False # True
SINUSOIDAL_TIME_EMBED_MAX_T = 500 # 10000 # maximum time embedding period for sinusoidal time embeddings.

# FIGSIZE = (8, 4)
# FONTSIZE = 8

SA_TRAIN_BATCH_INCLUDE_KEYS_LIST = ['edge_index_l', 'edge_weight_l',
                           'edge_index', 'edge_weight',
                           'weighted_adjacency', 'weighted_adjacency_l',
                           'avg_graph_edge_index_l', 'avg_graph_edge_weight_l',
                           'transmitters_index', 'num_nodes',
                        #    'batch',
                           ]

BATCH_INCLUDE_KEYS_LIST = ['edge_index_l', 'edge_weight_l',
                        #    'avg_graph_edge_index_l', 'avg_graph_edge_weight_l',
                           'transmitters_index', 'num_nodes',
                        #    'batch',
                           ]

BATCH_INCLUDE_KEYS_INSTANTANEOUS_CSI_LIST = ['edge_index', 'edge_weight', 'transmitters_index', 'num_nodes']


def path_str(v):
    # Convert a string path to a valid path
    if isinstance(v, str):
        v = v.replace(" ", "")
        v = v.replace("\n", "")
        # if not os.path.exists(v):
        #     print(f"File path {v} does not exist.")
        os.makedirs(v, exist_ok=True)
        # assert os.path.exists(v), f"File path {v} does not exist."
        return v
    else:
        raise argparse.ArgumentTypeError('String path expected.')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    

def str2intdict(v):
    return str2dict(v, v_type='int')

def str2floatdict(v):
    return str2dict(v, v_type='float')

def str2intlist(v):
    return str2list(v, v_type='int')

def str2floatlist(v):
    return str2list(v, v_type='float')

def str2strlist(v):
    return str2list(v, v_type='str')

def str2list(v, v_type = 'str'):
    # if isinstance(v, None):
    #     return None
    
    v = v.replace(" ", "")
    v = v.replace("\n", "")
    # if len(v) == 0:
    #     return None

    v_delim = v.split(",")
    if len(v_delim) == 1:
        values = [v]
    else:
        values = v_delim

    if v_type == 'str':
        values = [none_or_str(_) for _ in values]
    elif v_type == 'int':
        values = [int(_) for _ in values]
    elif v_type == 'float':
        values = [none_or_float(_) for _ in values]
    else:
        raise NotImplementedError
    
    return values


def channelListParser(v, v_type='int'):

    v = v.replace(" ", "")
    v = v.replace("\n", "")

    # if len(v) == 0:
    #     return None

    v_delim = v.split(",")
    if len(v_delim) == 1:
        values = [v]
    else:
        values = v_delim

    new_values = []
    for value in values:
        v_delim = value.split("-")
        if len(v_delim) == 1:
            value = int(value)
            new_values.append(value)
        
        elif len(v_delim) == 2:
            print("v_delim: ", v_delim)
            value = [int(v) for v in range(int(v_delim[0]), int(v_delim[1]) + 1)]
            new_values.extend(value)

        else:
            new_values.append(int(value))

    print("Parsed channel list: ", new_values)

    return new_values


    

def str2dict(v, v_type='str'):
    keys = ["train", "val", "test"]
    if isinstance(v, str):
        v = v.replace(" ", "")
        v = v.replace("\n", "")
        v_delim = v.split(",")
        if len(v_delim) == 1:
            values = [v] * len(keys)
        else:
            values = v_delim

        if v_type == 'str':
            # pass
            values = [none_or_str(_) for _ in values]
        elif v_type == 'int':
            values = [int(_) for _ in values]
        elif v_type == 'float':
            values = [none_or_float(_) for _ in values]
        else:
            raise NotImplementedError
        
        return dict(zip(keys, values))
    
    
    elif isinstance(v, list):
        values = v
        if v_type == 'str':
            pass
        elif v_type == 'int':
            values = [int(_) for _ in values]
        elif v_type == 'float':
            values = [float(_) for _ in values]
        else:
            raise NotImplementedError
        
        return dict(zip(keys, values))
    
    elif isinstance(v, dict):
        return v
    
    else:
        raise argparse.ArgumentTypeError('Single str or list or dictionary value expected.')    
    

def none_or_str(value):
    if isinstance(value, str):
        if value.lower() == 'none':
            return None
    return value


def none_or_float(value):
    if isinstance(value, str):
        if value.lower() == 'none':
            return None
        else:
            return float(value)
    return value


def none_or_int(value):
    if isinstance(value, str):
        if value.lower() == 'none':
            return None
        else:
            return int(value)
    return value
    
    

def create_dict_from_sys_argv(argv):
    # https://stackoverflow.com/questions/12807539/how-do-you-convert-command-line-args-in-python-to-a-dictionary
    print('argv: ', argv)
    d = defaultdict(list)
    for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in argv)):
        d[k].append(v)
    for k in (k for k in d if len(d[k])==1):
        d[k] = d[k][0]

    return d
    

class LoadFromFile(argparse.Action):
    # def __call__ (self, parser, namespace, values, option_string = None):
    #     with values as f:
    #         # parse arguments in the file and store them in the target namespace
    #         parser.parse_args(f.read().split(), namespace)
    def __call__ (self, parser, namespace, values, option_string=None):
        try:
            with values as f:
                contents = f.read()

            # parse arguments in the file and store them in a blank namespace
            data = parser.parse_args(contents.split(), namespace=None)
            for k, v in vars(data).items():
                # set arguments in the target namespace if they have not been set yet
                if getattr(namespace, k, None) is None:
                    setattr(namespace, k, v)
        except:
            print('Config file could not be read...')
            pass    


def make_parser(parse_known_args=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=path_str, default='./Price-Forecasting-Experiments', help = 'Root path.')
    parser.add_argument('--device', type=str, default='cuda:0', help = 'Computation device.')
    parser.add_argument('--gpu_fraction', type=float, default=1.0, help = "Fraction of GPU memory to use when batch sizes are automatically determined. Set to 1.0 to use all GPU memory.")
    parser.add_argument('--diffusion_policy', type=str, default='price-forecast', choices=['price-forecast'], help = 'Choice of diffusion policy to train.')
    parser.add_argument('--random_seed', type=int, default=42, help = 'Random seed.')
    parser.add_argument('--track', type=str2bool, default=False, help = 'Toggle True to enable Wandb logging.')
    parser.add_argument('--wandb_project_name', type=none_or_str, default=None, help = 'Wandb project name.')
    parser.add_argument('--wandb_group_name', type=none_or_str, default=None, help = 'Wandb run group name.')
    parser.add_argument('--wandb_entity', type=none_or_str, default=None, help = 'Wandb team entity name.')
    parser.add_argument('--log_dataset', type=str2bool, default=True, help = 'Toggle True to log datasets.')
    parser.add_argument('--max_jobs', type=int, default=1, help = "Max number of parallel GPU runs. GPU is split between the processes.")


    ### Accelerator config ###
    accelerator_args = parser.add_argument_group(title='accelerator', description='Config args for Accelerator.') 
    accelerator_args.add_argument('--auto_device_placement', type=str2bool, default=True, help = 'Toggle True to automatically place tensors on available devices.')
    accelerator_args.add_argument('--gradient_accumulation_steps', type=int, default=1, help = 'Number of gradient accumulation steps.')
    accelerator_args.add_argument('--split_batches', type=str2bool, default=True, help = 'Whether batches are split across processes.')
    accelerator_args.add_argument('--sync_with_dataloader', type=str2bool, default=False, help = 'Whether dataloader sync happens.')
    accelerator_args.add_argument('--sync_each_batch', type=str2bool, default=False, help = 'Whether each batch is synced.')
    accelerator_args.add_argument('--load_accelerator_chkpt_path', type=none_or_str, default=None, help = 'Dir for loading accelerator state from a previous run.')


    ### Dataset config ###
    dataset_args = parser.add_argument_group(title='dataset', description='Config args for dataset.')
    dataset_args.add_argument('--dataset_name', type=str, default="S&P 100", choices=["S&P 100"], help="Name of the dataset to use.")
    dataset_args.add_argument('--data_dir', type=path_str, default='./SP100AnalysisWithGNNs/data/SP100/raw', help='Directory to load/store the dataset.')
    dataset_args.add_argument('--corr_threshold', type=float, default=0.7, help = 'Thresholding for the correlation matrix.')
    dataset_args.add_argument('--sector_bonus', type=float, default=0.05, help = 'Bonus for stocks sharing the same sector.')
    dataset_args.add_argument('--past_window', type=int, default=25, help = 'Number of past time steps to consider.') # 25
    dataset_args.add_argument('--future_window', type=int, default=1, help = 'Number of future time steps to predict.')
    dataset_args.add_argument('--target_column_name', type=str, default="DailyLogReturn", help = 'Name of the target column to predict.' + \
    f'Dataset S&P 100: {get_column_names(values_path_or_df = "./SP100AnalysisWithGNNs/data/SP100/raw/values.csv")}')
    dataset_args.add_argument('--train_dataset_fraction', type=float, default=0.7, help = 'Fraction of the dataset to use for training.')


    ### Diffusion process and training algo parameters ###
    CD_args = parser.add_argument_group(title='CD-train-algo', description='Config args for (conditional) diffusion model training algorithm.')
    CD_args.add_argument('--noaccelerate_diffusion', type=str2bool, default=True, help = 'Use Huggingface accelerator.')
    CD_args.add_argument('--lr_diffusion', type = float, default=1e-3, help = 'Primal update step size.')
    CD_args.add_argument('--weight_decay_diffusion', type = float, default=1e-4, help = 'Optimizer weight decay. High weight decay is not good in general.')
    CD_args.add_argument('--lr_sched_gamma_diffusion', type = float, default=0.9, help = 'Learning rate scheduler gamma.')
    CD_args.add_argument('--batch_size_diffusion', type = int, default=64, help = 'Batch size for number of distinct graphs in a batch.') # 4
    CD_args.add_argument('--x_batch_size_diffusion', type = int, default=500, help = 'Batch size for number of node signals per graph in a batch.') # 1000
    CD_args.add_argument('--n_epochs_diffusion', type = int, default=5000, help = 'Number of training epochs.') # 2000
    CD_args.add_argument('--pgrad_clipping_constant_diffusion', type = float, default=1., help = 'Clip the norm of the gradients.')
    CD_args.add_argument('--beta_schedule_diffusion', type = str, default='cosine', choices = ['linear', 'cosine'], help = 'Noise schedule.')
    CD_args.add_argument('--diffusion_steps_diffusion', type = int, default=500, help = 'Number of dffusion steps.')
    CD_args.add_argument('--diffuse_n_samples_diffusion', type = int, default=500, help = 'Number of dffusion steps.')
    CD_args.add_argument('--ema_alpha_diffusion', type = none_or_float, default=0.99, help = 'Learning rate for Exponential Moving Average (EMA) updated on model parameters. No EMA if set to None.')
    CD_args.add_argument('--sampler_diffusion', type = str, default='ddpm', choices=['ddpm', 'ddpm_x0', 'ddim'], help = 'Choice of Diffusion Sampler')
    CD_args.add_argument('--eta_diffusion', type = float, default=0.2, help = 'Stochasticity parameter for the diffusion sampler if DDDIM is used.')


    ### Diffusion model architecture parameters ###
    CD_model_args = parser.add_argument_group(title='CD-model', description='Config args for conditional diffusion model neural network model.')
    CD_model_args.add_argument('--model_diffusion', type = str, default='LeConv', choices = ['LeConv', 'TAGConv'], help = 'Conv. Model architecture.')
    CD_model_args.add_argument('--sinusoidal_time_embed_diffusion', type = str2bool, default=True, help = 'Choose whether sinusoidal time embeddings are used.')
    CD_model_args.add_argument('--use_res_connection_time_embed_diffusion', type = str2bool, default=True, help = 'Choose whether sinusoidal time embeddings are fed to every layer via residual connections.')
    CD_model_args.add_argument('--gnn_backbone_diffusion', type = str, default='gnn-unet', choices = ['gnn', 'simple-gnn', 'resplus-gnn', 'mlp-conditional-gnn', 'mlp-conditional-mlp', 'gnn-conditional-gnn', 'gnn-unet', 'mlp-conditional-graph-transformer', 'graph-transformer-conditional-mlp', 'graph-transformer'], help = 'GNN backbone architectures.')
    CD_model_args.add_argument('--load_model_chkpt_path_diffusion', type = none_or_str, default=None, help = 'Pre-trained model weights load path.') # pre-trained model weights load
    CD_model_args.add_argument('--load_cd_train_chkpt_path_diffusion', type = none_or_str, default=None, help = 'Path to the checkpoint for the CD training.')    
    CD_model_args.add_argument('--n_layers_diffusion', type = int, default=4, help = 'Num of gnn backbone blocks.')
    CD_model_args.add_argument('--n_sublayers_diffusion', type = int, default=2, help = 'Num of features in GNN layers.')
    CD_model_args.add_argument('--hidden_dim_diffusion', type = int, default=64, help = 'Num of features in each layer of a GNN block.')
    CD_model_args.add_argument('--batch_norm_diffusion', type = str2bool, default=True, help = 'Batch normalization in GNN layers.')
    CD_model_args.add_argument('--dropout_rate_diffusion', type = float, default=0.0, help = 'Dropout rate for regularization in GNN layers.')
    CD_model_args.add_argument('--k_hops_diffusion', type = int, default=2, help = 'If applicable, the k-hop neighborhoods are considered for aggregation in each GNN layer.')
    CD_model_args.add_argument('--condition_by_summation_weight_diffusion', type = float, default=None, help = 'If None, conditioning is done by multiplying signal embeddings with time embeddings. If not None, final embedding is the weighted sum of signal embedding with weighted time and any conditioning graph embedding.')
    CD_model_args.add_argument('--use_checkpointing_diffusion', type = str2bool, default=False, help = 'Use gradient checkpointing to run a forward-pass segment for each checkpointed segment during backward propagation (GraphTransformer only)')
    CD_model_args.add_argument('--res_connection_diffusion', type = str, default='skip', choices=['mlp', 'skip'], help = 'Type of residual connection (GraphTransfromer only)')
    CD_model_args.add_argument('--norm_layer_diffusion', type = str, default=None, choices=[None, 'batch', 'layer', 'graph', 'instance', 'group'], help = 'Type of normalization layer (GraphUNet only)')
    CD_model_args.add_argument('--layer_norm_mode_diffusion', type = str, default="node", choices = ['node', 'graph'], help = 'Mode of operation if layer normalization is used.')
    CD_model_args.add_argument('--conv_layer_normalize_diffusion', type = str2bool, default=False, help = 'If True, apply spectral normalization to edge weights in convolutional layers.')
    CD_model_args.add_argument('--pool_layer_diffusion', type = str, default='topk', choices=['topk', 'spectral', 'centrality'], help = 'Type of pooling layer (GraphUNet only)')
    CD_model_args.add_argument('--pool_ratio_diffusion', type = float, default=0.5, help = 'Pooling ratio.')
    CD_model_args.add_argument('--pool_multiplier_diffusion', type = float, default=1., help = 'Post-multiply pooled features to rescale them.')
    CD_model_args.add_argument('--sum_skip_connection_diffusion', type = str2bool, default=True, help = 'Whether sum or concatenate skip connections with block outputs.')
    CD_model_args.add_argument('--activation_diffusion', type = str, default='tanh', choices=['tanh', 'relu', 'leaky-relu'], help = 'Type of activation (GraphTransformer only)')
    CD_model_args.add_argument('--attn_self_loop_fill_value_diffusion', type = str, default="max", help = 'How to fill self loops with GATv2Conv.')
    CD_model_args.add_argument('--apply_gcn_norm_diffusion', type = str2bool, default=False, help = 'Whether gcn norm is applied.')
    CD_model_args.add_argument('--attn_num_heads_diffusion', type = int, default=8, help = 'Number of graph-attention heads (GraphTransformer only). If set to NOne, graph-attention layers are replaced with graph-convolutional layers.')

    # Load an experiment from file
    # parser.add_argument('--file', type=open, action=LoadFromFile, help = 'Load a configuration from a file.') 
    parser.add_argument('--config_file', type=none_or_str, default=None, help = 'Revert to this config file for default values if not None.')

    if parse_known_args:
        args, unknown_args = parser.parse_known_args()
        print("\n Known parsed args: ", args, "\n")
        print("\n Unknown parsed args: ", unknown_args, "\n")
    else:
        args = parser.parse_args()
        print("\n Parsed args: ", args, "\n")


    # Revert to the config file for default values optionally
    if args.config_file is not None:
        try:
            with open(args.config_file) as f:
                print(f'Loading default arguments from file {args.config_file}...')
                file_args_dict = json.load(f)
                print('Default arguments loaded successfully.')
                parser_args_dict = create_dict_from_sys_argv(sys.argv[1:])
                # print('parser_args_dict: ', parser_args_dict)
                for k,v in parser_args_dict.items():
                    parser_args_dict[k] = type(vars(args)[k])(v)
                # print('parser_args_dict after casting: ', parser_args_dict)
                file_args_dict.update({k: v for k, v in parser_args_dict.items() if v is not None})  # Update if v is not None
                args = argparse.Namespace(**file_args_dict)
                # default_args_dict.update({k: v for k, v in vars(args).items() if v is not None})  # Update if v is not None
                # args = argparse.Namespace(**default_args_dict)

                print('args: ', args)

        except:
            print('Config file could not be read...')
            pass   


    arg_groups={}
    for group in parser._action_groups:
        # if group.title in ['options', 'positional arguments']:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}

        if group.title in ['CD-train-algo', 'CD-model']:
            keys = list(group_dict.keys()).copy()
            for key in keys:
                new_key = key.removesuffix('_diffusion')
                group_dict[new_key] = group_dict.pop(key)
      
        arg_groups[group.title]=argparse.Namespace(**group_dict)


    # Assert arguments
    if arg_groups["dataset"].dataset_name == "S&P 100":
        assert arg_groups["dataset"].target_column_name in get_column_names(values_path_or_df="./SP100AnalysisWithGNNs/data/SP100/raw/values.csv"), f"Target column name {arg_groups['dataset'].target_column_name} is not in the values file."

    print('args: ', args)

    # for key in arg_groups:
    #     if key in ['DR-train-algo', 'DR-model']:
    #         for kkey in arg_groups[]

    
    return args, arg_groups