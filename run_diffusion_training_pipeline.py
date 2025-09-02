import copy
from typing import Tuple, Union
from torch_geometric.data import Batch, Data

import torch
from core.StockForecastDiffusionLearner import StockPriceForecastDiffusionLearnerWrapper

from run_create_pyg_dataset_pipeline import run_create_pyg_dataloaders_pipeline
from models.utils import create_cd_model
from torch_geometric.loader import DataLoader


from collections import defaultdict
import copy
from core.Diffusion import ConditionalDiffusionLearner


def run_diffusion_training(args, arg_groups, accelerator,
                           experiment_name,
                           dataset, 
                           cd_model: torch.nn.Module = None):
    

    device = accelerator.device


    ### Create dataloaders ###
    cd_dataloaders = run_create_pyg_dataloaders_pipeline(args=args,
                                                        arg_groups=arg_groups,
                                                        accelerator=accelerator,
                                                        dataset=dataset
                                                        )

    accelerator.print("Created dataloaders for diffusion training.")                                               


    ### Create Diffusion Learner ###
    cd_learner = StockPriceForecastDiffusionLearnerWrapper(config=arg_groups['dataset'],
                                                            diffusion_config=arg_groups['CD-train-algo'],
                                                            device=device
                                                            )
    
    accelerator.print(f"Created diffusion learner of class {cd_learner.__class__.__name__} for stock price forecasting.")


    # Create and train a diffusion model policy to forecast stock prices
    cd_model, is_cd_model_trained = create_cd_model(accelerator=accelerator,
                                                    args=arg_groups['CD-model'],
                                                    n_features_in=dataset.future_window,
                                                    n_features_out=dataset.future_window,
                                                    n_features_cond=len(dataset.info["Features"]), timesteps_cond=dataset.past_window,
                                                    diffusion_steps=arg_groups['CD-train-algo'].diffusion_steps,
                                                    device=device,
                                                    num_nodes=dataset.info["Num_nodes"]
                                                    )
    
    assert cd_model is not None, "Conditional diffusion model is not created successfully."
    assert is_cd_model_trained is False, "Conditional diffusion model is already trained. Please check the model path or the training configuration."
    
    accelerator.print("arg_groups[CD-train-algo]: ", arg_groups['CD-train-algo'])
    
    n_iters_cd_per_epoch = 10
    cd_model = cd_learner.train(
        accelerator=accelerator,
        args=args, arg_groups=arg_groups,
        # sa_model=model,
        cd_model=cd_model,
        cd_optimizer=None,
        cd_lr_sched=None,
        cd_dataloaders=cd_dataloaders,
        n_iters_cd_per_epoch=n_iters_cd_per_epoch,
        n_epochs=[arg_groups['CD-train-algo'].n_epochs // (n_iters_cd_per_epoch)],
        device=device,
        # save_sa_train_chkpt_path = f"{args.experiment_name}/sa-models",
        save_train_chkpt_path = f"{args.experiment_name}/cd-models",
        save_test_metrics_path = f"{args.experiment_name}/data/logs",
        load_train_chkpt_path = args.load_cd_train_chkpt_path_diffusion,
        )

    return cd_model, cd_learner

# def run_diffusion_training(args, arg_groups, accelerator,
#                             experiment_name,
#                             # channel_dataset,
#                             dataset,
#                             sa_model: torch.nn.Module = None,
#                             # baseline_metrics
#                             ) -> Tuple[torch.nn.Module, Union[InterferenceDiffusionLearnerWrapper]]:
    
#     accelerator.print("Running diffusion training pipeline...")

#     # Check if the channel_dataloader is a dictionary of WirelessDataset objects
#     assert isinstance(channel_dataloader, dict) and 'train' in channel_dataloader and isinstance(channel_dataloader['train'], DataLoader) \
#     and isinstance(channel_dataloader['train'].dataset, WirelessDataset), \
#         "channel_dataloader should be a dictionary with 'train' key containing a DataLoader with WirelessDataset."
    
#     # Test iterating over the channel_dataloader to ensure it batches correctly.
#     for batch_idx, (data, sample_idx) in enumerate(channel_dataloader['train']):
#         assert isinstance(sample_idx, torch.LongTensor), f"Expected sample_idx to be a LongTensor, got {sample_idx} of type {type(sample_idx)}."
#         assert isinstance(data, (Data, Batch)), f"Expected Data or Batch, got data {data} of type {type(data)}."

#     # channel_dataset = {k: v.dataset for k, v in channel_dataloader.items()}

#     # # Create a separate dataloader for SA-training if train_networks_list is not None
#     # sa_channel_dataloader = defaultdict(list)
#     # if args.expert_policy == 'state-augmented' and arg_groups['SA-train-algo'].train_networks_list is not None and len(arg_groups['SA-train-algo'].train_networks_list):
#     #     for phase, dataset in channel_dataset.items():
#     #         if phase == 'train':
#     #             data_list, id_list = list(zip(*list(channel_dataloader[phase].dataset)))
#     #             data_list = [x for x, i in sorted(zip(data_list, id_list), key=lambda pair: pair[1]) if i in arg_groups["SA-train-algo"].train_networks_list]

#     #             sa_channel_dataloader[phase] = create_channel_dataloader(data_list=data_list, #
#     #                                                                      batch_size=arg_groups['RRM'].batch_size_channels,
#     #                                                                      shuffle=(phase == 'train')
#     #                                                                      )
#     #         else:
#     #             sa_channel_dataloader[phase] = create_channel_dataloader(data_list=channel_dataset[phase], #
#     #                                                                      batch_size=arg_groups['RRM'].batch_size_channels,
#     #                                                                      shuffle=(phase == 'train')
#     #                                                                      )
                
#     # else:
#     #     for phase, dataset in channel_dataset.items():
#     #         sa_channel_dataloader[phase] = create_channel_dataloader(data_list=dataset, #
#     #                                                                  batch_size=arg_groups['RRM'].batch_size_channels,
#     #                                                                  shuffle=(phase == 'train')
#     #                                                                  )
        
#     device = args.device
#     if args.diffusion_policy == 'interference-power':
#         diffusion_policy = InterferenceDiffusionLearnerWrapper(config=arg_groups['SA-train-algo'],
#                                                                   channel_config=arg_groups['RRM'],
#                                                                   diffusion_config=arg_groups['CD-train-algo'],
#                                                                   device=device
#                                                                   )

#     elif args.diffusion_policy == 'state-augmented-power-allocation':
#         diffusion_policy = PowerAllocationDiffusionLearnerWrapper(config=arg_groups['SA-train-algo'],
#                                                                   channel_config=arg_groups['RRM'],
#                                                                   diffusion_config=arg_groups['CD-train-algo'],
#                                                                   device=device
#                                                                   )
#     else:
#         raise ValueError(f"Unknown diffusion policy: {args.diffusion_policy}. Please check the configuration.")

#     # if any([not len(sa_channel_dataloader[phase].dataset) == len(channel_dataloader[phase].dataset) for phase in sa_channel_dataloader]):
#     #     accelerator.print('SA-channel dataloaders dataset and channel dataloaders dataset have different lengths.')
#     #     args_copy = copy.deepcopy(args)
#     #     for phase in args_copy.num_channels:
#     #         args_copy.num_channels[phase] = len(sa_channel_dataloader[phase].dataset)

#     #     accelerator.print("args.num_channels: ", args.num_channels)
#     #     accelerator.print("args_copy.num_channels: ", args_copy.num_channels)

#     #     # expert_policy_loggers = make_sa_loggers(args=args_copy, log_path=f'{args.experiment_name}/SA-train-logs')
    
#     # else:
#     #     pass
#         # expert_policy_loggers = make_sa_loggers(args=args, log_path=f'{args.experiment_name}/SA-train-logs')



#     # # Subsample the channel dataset (if needed) to create a diffusion dataset
#     # diffusion_dataloader = defaultdict(list)
#     # if arg_groups['CD-train-algo'].train_networks_list is not None and len(arg_groups['CD-train-algo'].train_networks_list):
#     #     for phase, dataset in channel_dataset.items():
#     #         if phase == 'train':
#     #             data_list, id_list = list(zip(*list(channel_dataloader[phase].dataset)))
#     #             data_list = [x for x, i in sorted(zip(data_list, id_list), key=lambda pair: pair[1]) if i in arg_groups["CD-train-algo"].train_networks_list]
                
#     #             diffusion_dataloader[phase] = create_channel_dataloader(data_list=data_list, #
#     #                                                                      batch_size=arg_groups['RRM'].batch_size_channels,
#     #                                                                      shuffle=(phase == 'train')
#     #                                                                      )
#     #         else:
#     #             diffusion_dataloader[phase] = create_channel_dataloader(data_list=channel_dataset[phase], #
#     #                                                                      batch_size=arg_groups['RRM'].batch_size_channels,
#     #                                                                      shuffle=(phase == 'train')
#     #                                                                      )
                
#     #     args_copy = copy.deepcopy(args)
#     #     for phase in args_copy.num_channels:
#     #         args_copy.num_channels[phase] = len(diffusion_dataloader[phase].dataset)

#     #     accelerator.print("args.num_channels: ", args.num_channels)
#     #     accelerator.print("args_copy.num_channels: ", args_copy.num_channels)

#     #     expert_policy_loggers = make_sa_dataset_loggers(args=args_copy, log_path=f'{args.experiment_name}/SA-dataset-logs')
                
#     # else:
#     #     for phase, dataset in channel_dataset.items():
#     #         diffusion_dataloader[phase] = create_channel_dataloader(data_list=dataset, #
#     #                                                                  batch_size=arg_groups['RRM'].batch_size_channels,
#     #                                                                  shuffle=(phase == 'train')
#     #                                                                  )
            

#     # Create and train a diffusion model policy to imitate state-augmented policies.
#     cd_model, is_cd_model_trained = create_cd_model(accelerator=accelerator,
#                                                     args=arg_groups['CD-model'],
#                                                     n_features=1,
#                                                     diffusion_steps=arg_groups['CD-train-algo'].diffusion_steps,
#                                                     device=device,
#                                                     n_clients=args.n
#                                                     )
    
#     assert cd_model is not None, "Conditional diffusion model is not created successfully."
#     assert is_cd_model_trained is False, "Conditional diffusion model is already trained. Please check the model path or the training configuration."
    
#     accelerator.print("arg_groups[CD-train-algo]: ", arg_groups['CD-train-algo'])
    
#     n_iters_cd_per_epoch = 10
#     cd_model = diffusion_policy.train(
#         accelerator=accelerator,
#         args=args, arg_groups=arg_groups,
#         # sa_model=model,
#         cd_model=cd_model,
#         cd_optimizer=None,
#         cd_lr_sched=None,
#         channel_dataloader=channel_dataloader,
#         n_iters_cd_per_epoch=n_iters_cd_per_epoch,
#         n_epochs=[arg_groups['CD-train-algo'].n_epochs // (n_iters_cd_per_epoch)],
#         loggers = None,
#         sa_model=sa_model,
#         device=device,
#         # save_sa_train_chkpt_path = f"{args.experiment_name}/sa-models",
#         save_train_chkpt_path = f"{args.experiment_name}/cd-models",
#         save_test_metrics_path = f"{args.experiment_name}/data",
#         load_train_chkpt_path = args.load_cd_train_chkpt_path_diffusion,
#         )


#     return cd_model, diffusion_policy




# def run_diffusion_testing(args, arg_groups, accelerator,
#                             experiment_name,
#                             # channel_dataset,
#                             channel_dataloader,
#                             sa_model: torch.nn.Module = None,
#                             # baseline_metrics
#                             ) -> Tuple[torch.nn.Module, Union[InterferenceDiffusionLearnerWrapper]]:
    
#     accelerator.print("Running diffusion testing pipeline...")

#     # Check if the channel_dataloader is a dictionary of WirelessDataset objects
#     assert isinstance(channel_dataloader, dict) and 'test' in channel_dataloader and isinstance(channel_dataloader['train'], DataLoader) \
#     and isinstance(channel_dataloader['test'].dataset, WirelessDataset), \
#         "channel_dataloader should be a dictionary with 'train' key containing a DataLoader with WirelessDataset."
    
#     # Test iterating over the channel_dataloader to ensure it batches correctly.
#     for batch_idx, (data, sample_idx) in enumerate(channel_dataloader['test']):
#         assert isinstance(sample_idx, torch.LongTensor), f"Expected sample_idx to be a LongTensor, got {sample_idx} of type {type(sample_idx)}."
#         assert isinstance(data, (Data, Batch)), f"Expected Data or Batch, got data {data} of type {type(data)}."

        
#     device = args.device
#     if args.diffusion_policy == 'interference-power':
#         diffusion_policy = InterferenceDiffusionLearnerWrapper(config=arg_groups['SA-train-algo'],
#                                                                   channel_config=arg_groups['RRM'],
#                                                                   diffusion_config=arg_groups['CD-train-algo'],
#                                                                   device=device
#                                                                   )

#     elif args.diffusion_policy == 'state-augmented-power-allocation':
#         diffusion_policy = PowerAllocationDiffusionLearnerWrapper(config=arg_groups['SA-train-algo'],
#                                                                   channel_config=arg_groups['RRM'],
#                                                                   diffusion_config=arg_groups['CD-train-algo'],
#                                                                   device=device
#                                                                   )
#     else:
#         raise ValueError(f"Unknown diffusion policy: {args.diffusion_policy}. Please check the configuration.")

            

#     # Create and train a diffusion model policy to imitate state-augmented policies.
#     cd_model, is_cd_model_trained = create_cd_model(accelerator=accelerator,
#                                                     args=arg_groups['CD-model'],
#                                                     n_features=1,
#                                                     diffusion_steps=arg_groups['CD-train-algo'].diffusion_steps,
#                                                     device=device,
#                                                     n_clients=args.n
#                                                     )
    
#     assert cd_model is not None, "Conditional diffusion model is not created successfully."
#     assert is_cd_model_trained is False, "Conditional diffusion model is already trained. Please check the model path or the training configuration."
    
#     accelerator.print("arg_groups[CD-train-algo]: ", arg_groups['CD-train-algo'])
    
#     n_iters_cd_per_epoch = 10
#     cd_model = diffusion_policy.test(
#         accelerator=accelerator,
#         args=args, arg_groups=arg_groups,
#         # sa_model=model,
#         cd_model=cd_model,
#         cd_optimizer=None,
#         cd_lr_sched=None,
#         channel_dataloader=channel_dataloader,
#         n_iters_cd_per_epoch=n_iters_cd_per_epoch,
#         n_epochs=[arg_groups['CD-train-algo'].n_epochs // (n_iters_cd_per_epoch)],
#         loggers = None,
#         sa_model=sa_model,
#         device=device,
#         # save_sa_train_chkpt_path = f"{args.experiment_name}/sa-models",
#         save_train_chkpt_path = f"{args.experiment_name}/cd-models",
#         save_test_metrics_path = f"{args.experiment_name}/data",
#         load_train_chkpt_path = args.load_cd_train_chkpt_path_diffusion,
#         )
    
#     accelerator.print("Diffusion model testing completed.")


#     return cd_model, diffusion_policy




# def run_diffusion_transferability(args, arg_groups, accelerator,
#                             experiment_name,
#                             # channel_dataset,
#                             channel_dataloader,
#                             sa_model: torch.nn.Module = None,
#                             # baseline_metrics
#                             ) -> Tuple[torch.nn.Module, Union[InterferenceDiffusionLearnerWrapper]]:

#     accelerator.print("Running diffusion transferability pipeline...")

#     # Check if the channel_dataloader is a dictionary of WirelessDataset objects
#     assert isinstance(channel_dataloader, dict) and 'test' in channel_dataloader and isinstance(channel_dataloader['train'], DataLoader) \
#     and isinstance(channel_dataloader['test'].dataset, WirelessDataset), \
#         "channel_dataloader should be a dictionary with 'train' key containing a DataLoader with WirelessDataset."
    
#     # Test iterating over the channel_dataloader to ensure it batches correctly.
#     for batch_idx, (data, sample_idx) in enumerate(channel_dataloader['test']):
#         assert isinstance(sample_idx, torch.LongTensor), f"Expected sample_idx to be a LongTensor, got {sample_idx} of type {type(sample_idx)}."
#         assert isinstance(data, (Data, Batch)), f"Expected Data or Batch, got data {data} of type {type(data)}."

        
#     device = args.device
#     if args.diffusion_policy == 'interference-power':
#         diffusion_policy = InterferenceDiffusionLearnerWrapper(config=arg_groups['SA-train-algo'],
#                                                                   channel_config=arg_groups['RRM'],
#                                                                   diffusion_config=arg_groups['CD-train-algo'],
#                                                                   device=device
#                                                                   )

#     elif args.diffusion_policy == 'state-augmented-power-allocation':
#         diffusion_policy = PowerAllocationDiffusionLearnerWrapper(config=arg_groups['SA-train-algo'],
#                                                                   channel_config=arg_groups['RRM'],
#                                                                   diffusion_config=arg_groups['CD-train-algo'],
#                                                                   device=device
#                                                                   )
#     else:
#         raise ValueError(f"Unknown diffusion policy: {args.diffusion_policy}. Please check the configuration.")

            

#     # Create and train a diffusion model policy to imitate state-augmented policies.
#     cd_model, is_cd_model_trained = create_cd_model(accelerator=accelerator,
#                                                     args=arg_groups['CD-model'],
#                                                     n_features=1,
#                                                     diffusion_steps=arg_groups['CD-train-algo'].diffusion_steps,
#                                                     device=device,
#                                                     n_clients=args.n
#                                                     )
    
#     assert cd_model is not None, "Conditional diffusion model is not created successfully."
#     assert is_cd_model_trained is False, "Conditional diffusion model is already trained. Please check the model path or the training configuration."
    
#     accelerator.print("arg_groups[CD-train-algo]: ", arg_groups['CD-train-algo'])
    
#     n_iters_cd_per_epoch = 10
#     cd_model = diffusion_policy.test_transferability(
#         accelerator=accelerator,
#         args=args, arg_groups=arg_groups,
#         # sa_model=model,
#         cd_model=cd_model,
#         cd_optimizer=None,
#         cd_lr_sched=None,
#         channel_dataloader=channel_dataloader,
#         n_iters_cd_per_epoch=n_iters_cd_per_epoch,
#         n_epochs=[arg_groups['CD-train-algo'].n_epochs // (n_iters_cd_per_epoch)],
#         loggers = None,
#         sa_model=sa_model,
#         device=device,
#         # save_sa_train_chkpt_path = f"{args.experiment_name}/sa-models",
#         save_train_chkpt_path = f"{args.experiment_name}/cd-models",
#         save_test_metrics_path = f"{args.experiment_name}/data",
#         load_train_chkpt_path = args.load_cd_train_chkpt_path_diffusion,
#         )
    
#     accelerator.print("Diffusion model transferability test completed.")


#     return cd_model, diffusion_policy







# # def run_diffusion_training(args, arg_groups, accelerator,
# #                            experiment_name,
# #                            diffusion_dataloader,
# #                            destandardize,
# #                            baseline_metrics = None,
# #                            ):
    
# #     # if arg_groups['CD-train-algo'].train_networks_list is not None and len(arg_groups['CD-train-algo'].train_networks_list):

# #     #     for phase, dataset in diffusion_dataloader.items():
# #     #         print(f"[DEBUG] phase: {phase}, dataset: {dataset}")
# #     #         if phase in ['train']:
# #     #             data_list, id_list = list(zip(*list(diffusion_dataloader[phase].dataset)))
# #     #             data_list = [x for x, i in sorted(zip(data_list, id_list), key=lambda pair: pair[1]) if i in arg_groups["CD-train-algo"].train_networks_list]
        
# #     #             diffusion_dataloader[phase] = DataLoader(dataset = data_list,
# #     #                                                     batch_size=args.batch_size_diffusion,
# #     #                                                     shuffle=(phase == 'train')
# #     #                                                     )
# #     #         else:
# #     #             pass
                
# #     #     args_copy = copy.deepcopy(args)
# #     #     for phase in args_copy.num_channels:
# #     #         print(f"[DEBUG] phase: {phase}, len(diffusion_dataloader[phase].dataset): {len(diffusion_dataloader[phase].dataset)}")
# #     #         args_copy.num_channels[phase] = len(diffusion_dataloader[phase].dataset)

# #     #     cd_loggers = make_cd_loggers(args=args_copy, log_path=f'{args.experiment_name}/CD-logs', inv_transform = destandardize)
    
# #     # else:
# #     #     cd_loggers = make_cd_loggers(args=args, log_path=f'{args.experiment_name}/CD-logs', inv_transform = destandardize)

# #     if any([not len(diffusion_dataloader[phase].dataset) == args.num_channels[phase] for phase in diffusion_dataloader]):
# #         accelerator.print('Diffusion dataloader and original channel dataloader have different lengths.')
# #         args_copy = copy.deepcopy(args)
# #         for phase in args_copy.num_channels:
# #             args_copy.num_channels[phase] = len(diffusion_dataloader[phase].dataset)

# #         accelerator.print("args.num_channels: ", args.num_channels)
# #         accelerator.print("args_copy.num_channels: ", args_copy.num_channels)

# #         cd_loggers = make_cd_loggers(args=args_copy, log_path=f'{args.experiment_name}/CD-train-logs', inv_transform=destandardize, baseline_metrics=baseline_metrics)
    
# #     else:
# #         cd_loggers = make_cd_loggers(args=args, log_path=f'{args.experiment_name}/CD-train-logs', inv_transform=destandardize, baseline_metrics=baseline_metrics)


# #     ############ Initialize/load a conditional diffusion model ################
# #     cd_device = args.device
# #     cd_model, is_cd_model_trained = create_cd_model(accelerator=accelerator,
# #                                                     args=arg_groups['CD-model'],
# #                                                     n_features=1,
# #                                                     diffusion_steps=arg_groups['CD-train-algo'].diffusion_steps,
# #                                                     device=cd_device,
# #                                                     n_clients=args.n
# #                                                     )
    
# #     if args.expert_policy in ['state-augmented', 'ITLinQ', 'WMMSE']:
# #         ''' Instantiate a diffusion model training pipeline to mimic a state-augmented primal-dual training algorithm.'''
# #         cd_learner = ConditionalDiffusionLearner(config = arg_groups['CD-train-algo'],
# #                                                 device = cd_device
# #                                                 )
        
# #     elif args.expert_policy == 'uniform-random':
# #         ''' Instantiate a constrained diffusion model training pipeline to convert a uniform-random policy to a feasible and near-optimal data distribution. '''
# #         from core.StateAugmentation import Obj, Constraints
# #         obj = Obj()
# #         constraints = Constraints(r_min=args.r_min, n_constraints=args.n)
# #         cd_learner = ConstrainedConditionalDiffusionLearner(config=arg_groups['CD-train-algo'],
# #                                                             obj=obj,
# #                                                             constraints=constraints,
# #                                                             device=cd_device,
# #                                                             noise_var = arg_groups['RRM'].noise_var
# #                                                             )
# #     else:
# #         raise ValueError


# #     # all_variables = prepare_everything(args=args, arg_groups=arg_groups, device = device)
# #     accelerator.wait_for_everyone()
# #     # accelerator.print('all_variables: ', all_variables)

# #     accelerator.print("**********************************************************************************\n\n\n\n")
# #     accelerator.print("\n\n\n\n**********************************************************************************")

# #     if not is_cd_model_trained:
# #         lambdas = 0.
# #         cd_model, _ = cd_learner.train(accelerator=accelerator,
# #                                        model=cd_model,
# #                                     #    dataloader=diffusion_dataloader['train'] if not args.train_on_single_network_diffusion else diffusion_dataloader['test'],
# #                                        dataloader={'train': diffusion_dataloader['train'], 'val': diffusion_dataloader['val']} if (not args.train_on_single_network_diffusion) or True else diffusion_dataloader['test'],
# #                                        n_epochs=arg_groups['CD-train-algo'].n_epochs,
# #                                        lr = arg_groups['CD-train-algo'].lr,
# #                                        device=cd_device,
# #                                     #    loggers=cd_loggers['train'],
# #                                        loggers={'train': cd_loggers['train'], 'val': cd_loggers['val']},
# #                                        P_max = arg_groups['RRM'].P_max,
# #                                        noise_var = arg_groups['RRM'].noise_var,
# #                                        lambdas=lambdas, # used only by ConstrainedConditionalDiffusionLearner
# #                                        lr_dual=1., # used only by ConstrainedConditionalDiffusionLearner
# #                                        inv_transform = destandardize,
# #                                        graph_token_drop_prob = arg_groups['CD-train-algo'].graph_token_drop_prob,
# #                                        graph_instantaneous_csi_prob = arg_groups['CD-train-algo'].graph_instantaneous_csi_prob,
# #                                        ema_alpha = arg_groups['CD-train-algo'].ema_alpha,
# #                                        save_cd_train_chkpt_path=f"{args.experiment_name}/models",
# #                                        load_accelerator_chkpt_path=args.load_accelerator_chkpt_path,
# #                                        )
# #         is_cd_model_trained = True
        
# #         # save trained model weights
# #         # torch.save(cd_model.state_dict(), f'{args.experiment_name}/models/cd_model_{experiment_name}.pt')
# #         accelerator.wait_for_everyone()
# #         if accelerator.is_main_process:
# #             accelerator.save(accelerator.unwrap_model(cd_model).state_dict(), f'{args.experiment_name}/models/cd_model_{experiment_name}.pt')

# #             # Save loggers to file
# #             for phase in cd_loggers:
# #                 if cd_loggers[phase] is not None and len(cd_loggers[phase]):
# #                     save_obj = create_logger_dict(cd_loggers[phase])
# #                     save_path = cd_loggers[phase][0].log_path
# #                     save_logger_object(obj=save_obj, filename=save_path + '/loggers.pkl')
# #                     print(f'{phase} loggers have been saved and pickled successfully.')
        
# #         # accelerator.end_training()


# #     return cd_model, cd_learner, cd_loggers