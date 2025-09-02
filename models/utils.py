import io
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from PIL import Image
from models.StockForecastGNN import StockForecastDiffusionGNN
from utils.plot_utils import plot_regression_errors


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
                             norm_kws = {'layer_norm_mode': args.layer_norm_mode}
                            #  pool_ratio = args.pool_ratio,
                            #  pool_layer = args.pool_layer,
                            #  pool_multiplier = args.pool_multiplier,
                            #  sum_skip_connection = args.sum_skip_connection,
                            #  use_checkpointing = args.use_checkpointing,
                            #  attn_self_loop_fill_value = args.attn_self_loop_fill_value,
                            #  apply_gcn_norm=args.apply_gcn_norm
                            )
        
    else:
        raise NotImplementedError(f"Model architecture {args.gnn_backbone} is not implemented.")
    

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




def get_regression_errors(preds, target):

    mse = F.mse_loss(preds, target).item()
    rmse = F.mse_loss(preds, target).sqrt().item()
    mae = F.l1_loss(preds, target).item()
    mre = (F.l1_loss(preds, target) / target.abs().mean()).item()

    return {"mse": mse, "rmse": rmse, "mae": mae, "mre": mre}



def log_plot_regression_errors(preds, target, metric, stocks_idx = None):

    fig, _ = plot_regression_errors(preds, target, metric=metric, stocks_idx=stocks_idx, fig = None, axs = None)

    buf = io.BytesIO()
    fig.savefig(buf, format = "png")
    buf.seek(0)
    plt.close(fig)

    log_key = f"regression-plot-{metric}"
    log_item = wandb.Image(Image.open(buf))

    return {log_key: log_item}