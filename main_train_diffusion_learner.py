

from argparse import Namespace
import os
import json
from core.config import make_parser
from run_train_diffusion_learner_experiment import run_train_diffusion_learner_experiment
from utils.general_utils import make_experiment_name

# from utils.wandb utils import WANDB_API_KEY

def accelerated_train_diffusion_learner(**kwargs):
    if 'args' in kwargs:
        print('Copying cmdline args from dictionary passed to main()...')
        args = kwargs.get('args')
        if not isinstance(args, Namespace):
            args = Namespace(**args)
    elif 'file' in kwargs:
        file = kwargs.get('config_file')
        with open(file) as f:
            print(f'Loading cmdline arguments from file {file}...')
            args_dict = json.load(f)
            args = Namespace(**args_dict)
    else:
        print('Parsing cmdline arguments...')
        args = make_parser()

    if isinstance(args, tuple):
        arg_groups = args[1]
        args = args[0]


    args.pid = os.getpid()
    print(f"Process ID: {args.pid}")

    assert args.ema_alpha_diffusion is not None, "When using accelerate, ema_alpha_diffusion should be set to sth other than None."

    # if not args.nowandb:
    #     os.environ['WANDB_API_KEY'] = WANDB_API_KEY

    print('Random seed: ', args.random_seed)
    print('Root path: ', args.root)

    if hasattr(args, "max_jobs") and not args.max_jobs == 1:
        pass
        # torch.cuda.set_per_process_memory_fraction(1 / args.max_jobs, device=0)  # Limit GPU 0 to 50% memory
    else:
        pass
# torch.cuda.set_per_process_memory_fraction(0.8, device=1)  # Limit GPU 1 to 80%

    experiment_root_name = make_experiment_name(args)

    n_runs = 1
    for _ in range(n_runs):
        # run_train_dual_regressor_experiment(args=args, arg_groups=arg_groups, experiment_root_name=experiment_root_name,
        #                                     # run_diffusion_pipeline = (args.run_expert_policy_only_flag == False)
        #                                     )
        run_train_diffusion_learner_experiment(args=args, arg_groups=arg_groups, experiment_root_name=experiment_root_name)




if __name__ == "__main__":
    # main()
    accelerated_train_diffusion_learner()