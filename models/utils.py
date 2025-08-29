import torch
from models.StockForecastGNN import StockForecastDiffusionGNN


def get_diffusion_model_architecture(args, n_features, num_nodes, diffusion_steps):
    print(f"Creating diffusion model with architecture: {args.gnn_backbone}, n_features = {n_features}, num_nodes = {num_nodes}, diffusion_steps = {diffusion_steps}")

        
    if args.gnn_backbone == 'resplus-gnn':
        model = StockForecastDiffusionGNN(nfeatures=n_features, nunits=args.hidden_dim, nblocks=args.n_layers, nsteps = diffusion_steps,
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


    

def create_cd_model(accelerator, args, n_features = 1, diffusion_steps = 100, device = 'cpu', num_nodes = 100):
    '''
    Create and initialize the (conditional) diffusion model.
    '''
    # from utils.model_utils import PrimalGNN

    # model = PrimalGNN(conv_model=args.model, P_max=0.01, num_features_list=[n_features] + args.num_features_list, k_hops = args.k_hops, batch_norm = args.batch_norm, dropout_rate = args.dropout_rate)
    
    # model = DiffusionGNN(nfeatures=n_features, nunits=128, nblocks = 4, nsteps=diffusion_steps, nsubblocks=3)


    model = get_diffusion_model_architecture(args, n_features=n_features, num_nodes=num_nodes, diffusion_steps=diffusion_steps)
    
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