import io
import torch
import matplotlib.pyplot as plt
import wandb
from PIL import Image
from models.StockForecastGNN import StockForecastDiffusionGNN
from models.StockForecastTemporalConvGNN import StockForecastDiffusionTemporalConvGNN
from models.eval import get_probabilistic_errors, get_regression_errors
from utils.plot_utils import plot_regression_errors_deprecated, plot_regression


def count_parameters(model):
    """
    Count the total number of trainable parameters in a torch.nn.Module.
    
    Args:
        model (torch.nn.Module): The model to count parameters for
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_detailed(model, accelerator=None):
    """
    Count and print detailed parameter information for a torch.nn.Module.
    
    Args:
        model (torch.nn.Module): The model to analyze
        accelerator: Accelerator object for distributed printing (optional)
        
    Returns:
        int: Total number of trainable parameters
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print_fn = print # accelerator.print if accelerator is not None else print
    
    print_fn(f"=" * 60)
    print_fn(f"MODEL PARAMETER SUMMARY")
    print_fn(f"=" * 60)
    print_fn(f"Trainable parameters: {trainable_params:,}")
    print_fn(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print_fn(f"Total parameters: {total_params:,}")
    print_fn(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
    print_fn(f"=" * 60)
    
    return trainable_params


def get_diffusion_model_architecture(args, n_features_in, n_features_out, num_nodes, diffusion_steps,
                                     timesteps_cond = None,
                                     n_features_cond = None
                                     ):
    print(f"Creating diffusion model with architecture: {args.gnn_backbone}, n_features_in = {n_features_in}, n_features_out = {n_features_out}, num_nodes = {num_nodes}, diffusion_steps = {diffusion_steps}")

    if args.gnn_backbone == 'resplus-gnn':
        model = StockForecastDiffusionGNN(nfeatures=n_features_in, nunits=args.hidden_dim, nblocks=args.n_layers, nsteps = diffusion_steps,
                                          num_features_cond=n_features_cond,
                                          num_timesteps_cond=timesteps_cond,
                             norm = args.norm_layer, conv_layer_normalize = args.conv_layer_normalize, k_hops = args.k_hops,
                             use_res_connection_time_embed = args.use_res_connection_time_embed,
                             dropout_rate = args.dropout_rate,
                             norm_kws = {'layer_norm_mode': args.layer_norm_mode},
                             conv_model = args.model,
                             edge_features_nb = args.edge_features_nb,
                             aggr_list = args.aggr_list,
                             conv_batch_norm = args.conv_batch_norm
                            #  pool_ratio = args.pool_ratio,
                            #  pool_layer = args.pool_layer,
                            #  pool_multiplier = args.pool_multiplier,
                            #  sum_skip_connection = args.sum_skip_connection,
                            #  use_checkpointing = args.use_checkpointing,
                            #  attn_self_loop_fill_value = args.attn_self_loop_fill_value,
                            #  apply_gcn_norm=args.apply_gcn_norm
                            )

    elif args.gnn_backbone == 'temporal-resplus-gnn':
        model = StockForecastDiffusionTemporalConvGNN(nfeatures=n_features_in, nunits=args.hidden_dim, nblocks=args.n_layers, nsteps = diffusion_steps,
                                          num_features_cond=n_features_cond,
                                          num_timesteps_cond=timesteps_cond,
                             norm = args.norm_layer, conv_layer_normalize = args.conv_layer_normalize, k_hops = args.k_hops,
                             use_res_connection_time_embed = args.use_res_connection_time_embed,
                             dropout_rate = args.dropout_rate,
                             norm_kws = {'layer_norm_mode': args.layer_norm_mode},
                             conv_model = args.model,
                             edge_features_nb = args.edge_features_nb,
                             aggr_list = args.aggr_list,
                             conv_batch_norm = args.conv_batch_norm

        )

    return model


    

def create_cd_model(accelerator, args,
                    diffusion_steps = 100,
                    n_features_in = 1, n_features_out = 1, n_features_cond = None,
                    timesteps_cond = None,
                    device = 'cpu', num_nodes = 100):
    '''
    Create and initialize the (conditional) diffusion model.
    '''

    model = get_diffusion_model_architecture(args,
                                             n_features_in=n_features_in,
                                             n_features_out=n_features_out,
                                             n_features_cond=n_features_cond,
                                             timesteps_cond=timesteps_cond,
                                             num_nodes=num_nodes, diffusion_steps=diffusion_steps)
    
    # Print detailed parameter information right after model initialization
    count_parameters_detailed(model)

    is_trained = False
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = accelerator.device
    model = model.to(device)


    if hasattr(args, 'load_model_chkpt_path') and args.load_model_chkpt_path is not None:
        try:
            # with accelerator.main_process_first():
            state_dict = torch.load(args.load_model_chkpt_path)
            # Print model's state_dict
            accelerator.print("Loaded state_dict:")
            for param_tensor in state_dict:
                accelerator.print(param_tensor, "\t", state_dict[param_tensor].size())

            print("Model's state_dict:")
            for param_tensor in model.state_dict():
                accelerator.print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            
            # useful to unwrap only if you use the load function after making your model go through prepare()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(args.load_model_chkpt_path))

            accelerator.print('Loading pretrained diffusion model weights is successful.')
            is_trained = True
        except:
            accelerator.print(f'Loading model state dict from path {args.load_model_chkpt_path} failed.\nDefault weight initialization is applied.')

    return model, is_trained



def log_plot_regression(preds, target, observations, timestamps, metric, **kwargs):
    """
    Plot regression for the specified stock over a given time period.
    """

    fig, _ = plot_regression(preds = preds,
                             target = target,
                             observations = observations,
                             timestamps = timestamps,
                             metric=metric, fig = None, axs = None, **kwargs)
    
    # fig.savefig("./demos/debug_plot.png")

    buf = io.BytesIO()
    fig.savefig(buf, format = "png")
    buf.seek(0)
    plt.close(fig)

    log_key = f"regression-plot-{metric}"
    log_item = wandb.Image(Image.open(buf))

    return {log_key: log_item}


def log_plot_regression_errors_deprecated(preds, target, metric, stocks_idx = None, **kwargs):
    """
    This log/plot function will be deprecated soon. Use log_plot_regression_errors instead.
    """

    fig, _ = plot_regression_errors_deprecated(preds, target, metric=metric, stocks_idx=stocks_idx, fig = None, axs = None, **kwargs)

    buf = io.BytesIO()
    fig.savefig(buf, format = "png")
    buf.seek(0)
    plt.close(fig)

    log_key = f"regression-plot-{metric}"
    log_item = wandb.Image(Image.open(buf))

    return {log_key: log_item}




def log_probabilistic_error():

    err_dict = {}  
    
    
    
    return err_dict



def log_regression_errors(preds, target, grw_pred, cgrw_pred,
                          pred_log_returns, target_log_returns, grw_pred_log_returns, cgrw_pred_log_returns,
                          target_metric = "Close",
                          **kwargs):

    err_metrics = {}  


    # Mean regression errors for target metric
    print("Mean regression errors for {}: ".format(target_metric))
    err_metrics.update({f"{k}-{target_metric}_gdm": v for k,v in get_regression_errors(preds, target).items()})
    err_metrics.update({f"{k}-{target_metric}_grw": v for k, v in get_regression_errors(grw_pred, target).items()})
    # err_metrics.update({f"{k}_cgrw": v for k, v in get_regression_errors(cgrw_pred, target).items()})

    # Mean regression errors for Log Returns
    print("Mean regression errors for Log Returns of {}: ".format("Log Returns"))
    err_metrics.update({f"{k}-Log Returns_gdm": v for k,v in get_regression_errors(pred_log_returns, target_log_returns).items()})
    err_metrics.update({f"{k}-Log Returns_grw": v for k, v in get_regression_errors(grw_pred_log_returns, target_log_returns).items()})
    # err_metrics.update({f"{k}_cgrw": v for k, v in get_regression_errors(cgrw_pred, target).items()})


    # Regression error for mean prediction of target metric
    print("Regression errors for mean prediction of {}: ".format(target_metric))
    mean_pred = preds.mean(dim = -1, keepdim=True) # [num_timestamps, num_stocks, future_window, 1]
    mean_pred_grw = grw_pred.mean(dim = -1, keepdim=True) # [num_timestamps, num_stocks, future_window, 1]
    # mean_pred_cgrw = cgrw_pred.mean(dim = -1, keepdim=True) # [num_timestamps, num_stocks, future_window, 1]
    err_metrics.update({f"{k}-{target_metric}_mean_pred_gdm": v for k, v in get_regression_errors(mean_pred, target).items()})
    err_metrics.update({f"{k}-{target_metric}_mean_pred_grw": v for k, v in get_regression_errors(mean_pred_grw, target).items()})
    # err_metrics.update({f"{k}_mean_pred_cgrw": v for k, v in get_regression_errors(mean_pred_cgrw, target).items()})

    # Regression error for mean prediction of Log Returns
    print("Regression errors for mean prediction of {}: ".format("Log Returns"))
    mean_pred = pred_log_returns.mean(dim = -1, keepdim=True) # [num_timestamps, num_stocks, future_window, 1]
    mean_pred_grw = grw_pred_log_returns.mean(dim = -1, keepdim=True) # [num_timestamps, num_stocks, future_window, 1]
    # mean_pred_cgrw = cgrw_pred.mean(dim = -1, keepdim=True) # [num_timestamps, num_stocks, future_window, 1]
    err_metrics.update({f"{k}-Log Returns_mean_pred_gdm": v for k, v in get_regression_errors(mean_pred, target_log_returns).items()})
    err_metrics.update({f"{k}-Log Returns_mean_pred_grw": v for k, v in get_regression_errors(mean_pred_grw, target_log_returns).items()})
    # err_metrics.update({f"{k}_mean_pred_cgrw": v for k, v in get_regression_errors(mean_pred_cgrw, target).items()})



    # Probabilistic error metrics for target metric
    err_metrics.update({f"{k}-{target_metric}_gdm": v for k, v in get_probabilistic_errors(preds = torch.permute(preds, (0, -1, 2, 1)),
                                                    target = torch.permute(target, (0, -1, 2, 1)).squeeze(1)
                                                    ).items()
                            })

    err_metrics.update({f"{k}-{target_metric}_grw": v for k, v in get_probabilistic_errors(preds = torch.permute(grw_pred, (0, -1, 2, 1)),
                                                                                target = torch.permute(target, (0, -1, 2, 1)).squeeze(1)
                                                                                ).items()
                            })
    
    # err_metrics.update({f"{k}_cgrw": v for k, v in get_probabilistic_errors(preds = torch.permute(cgrw_pred, (0, -1, 2, 1)),
    #                                                                            target = torch.permute(target, (0, -1, 2, 1)).squeeze(1)
    #                                                                            ).items()
    #                         })


    # Probabilistic error metrics for Log Returns
    err_metrics.update({f"{k}-Log Returns_gdm": v for k, v in get_probabilistic_errors(preds = torch.permute(pred_log_returns, (0, -1, 2, 1)),
                                                    target = torch.permute(target_log_returns, (0, -1, 2, 1)).squeeze(1)
                                                    ).items()
                            })

    err_metrics.update({f"{k}-Log Returns_grw": v for k, v in get_probabilistic_errors(preds = torch.permute(grw_pred_log_returns, (0, -1, 2, 1)),
                                                                                target = torch.permute(target_log_returns, (0, -1, 2, 1)).squeeze(1)
                                                                                ).items()
                            })
    # err_metrics.update({f"{k}_cgrw": v for k, v in get_probabilistic_errors(preds = torch.permute(cgrw_pred, (0, -1, 2, 1)),
    #                                                                            target = torch.permute(target, (0, -1, 2, 1)).squeeze(1)
    #                                                                            ).items()
    #                         })
    
    
    
    return err_metrics