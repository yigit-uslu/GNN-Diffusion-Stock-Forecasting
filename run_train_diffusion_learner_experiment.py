
import os, json, sys
from run_create_pyg_dataset_pipeline import run_create_pyg_dataloaders_pipeline, run_create_pyg_dataset_pipeline
# from run_diffusion_training_pipeline import run_diffusion_training
from run_diffusion_training_pipeline import run_diffusion_training
from utils.general_utils import seed_everything

from utils.accelerator_utils import DummyAccelerator, SilentAccelerator
from utils.general_utils import create_folders_and_dirs, seed_everything
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, DataLoaderConfiguration, ProjectConfiguration


def run_train_diffusion_learner_experiment(args, arg_groups, experiment_root_name):

    seed_everything(args.random_seed)
    experiment_name = experiment_root_name + f"_seed_{args.random_seed}"
    # experiment_name = make_experiment_name(args)

    # args.experiment_name = f'{args.root}/SA-Experiments/{experiment_name}'
    args.experiment_name = f'{args.root}/{experiment_name}'

    
    if args.noaccelerate_diffusion:
        print("Using dummy accelerator...")
        accelerator = DummyAccelerator(device = "cuda:0")

    else:

        ### Init accelerate ###
        config = DataLoaderConfiguration(split_batches=arg_groups['accelerator'].split_batches)
        plugin = GradientAccumulationPlugin(num_steps=arg_groups['accelerator'].gradient_accumulation_steps,
                                            sync_with_dataloader=arg_groups['accelerator'].sync_with_dataloader,
                                            sync_each_batch=arg_groups['accelerator'].sync_each_batch)
        
        project = ProjectConfiguration(project_dir=args.experiment_name,
                                    logging_dir=f"{args.experiment_name}/CD-checkpoints",
                                        automatic_checkpoint_naming=True, total_limit=100
                                        )

        print(f'Accelerator config: ', arg_groups['accelerator'])
        accelerator = Accelerator(dataloader_config=config,
                                device_placement=arg_groups['accelerator'].auto_device_placement,
                                gradient_accumulation_plugin=plugin,
                                project_config=project,
                                log_with="wandb" if args.track else None
                                )
    
    # Make it silent
    # accelerator = SilentAccelerator(accelerator, silent=True)
    
    
    # Initialise your wandb run, passing wandb parameters and any config information
    accelerator.init_trackers(
        project_name=f"{args.wandb_project_name}",
        config=vars(args),
        init_kwargs={"wandb": {"entity": None,
                               "group": args.wandb_group_name,
                               "name": args.experiment_name, # experiment_root_name
                               "save_code": True}} if args.track else None
        )

    if accelerator.is_local_main_process: # do once
        create_folders_and_dirs(args.experiment_name)

        ### Save the parsed configuration ###
        with open(f'{args.experiment_name}/config.json', 'w') as f:
            json.dump(vars(args), f, indent = 6)

        for key, value in arg_groups.items():
            if key not in ['options', 'positional_arguments']:
                with open(f"{args.experiment_name}/{key}_config.json", "w") as f:
                    json.dump(vars(value), f, indent=6)


    with open(f"{args.experiment_name}/console_logs.txt", 'w') as sys.stdout:
        print("CUDA visible devices: ", os.environ["CUDA_VISIBLE_DEVICES"])
        device = accelerator.device
        print(f"accelerator process: {accelerator.process_index}\t accelerator device: {accelerator.device}")

        # Accelerator.device is not JSON-serializable
        args.device = device


        ### Create datasets ###
        dataset = run_create_pyg_dataset_pipeline(args=args,
                                arg_groups=arg_groups,
                                accelerator=accelerator,
                                experiment_name=experiment_name
                                )

        if args.diffusion_policy in ['price-forecast']:

            accelerator.print(f"Running diffusion training for diffusion policy {args.diffusion_policy}...")
            cd_model, cd_learner = run_diffusion_training(args = args,
                                                            arg_groups=arg_groups,
                                                            accelerator=accelerator,
                                                            experiment_name=experiment_name,
                                                            dataset=dataset,
                                                            )
            accelerator.print(f"Finished diffusion training for diffusion policy {args.diffusion_policy}.")


        else:
            raise ValueError(f"Unknown diffusion policy: {args.diffusion_policy}. "
                             f"Please choose from ['price-forecast']. ")

        

        return 0