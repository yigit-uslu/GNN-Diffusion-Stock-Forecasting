from argparse import Namespace
from collections import defaultdict
import os, copy
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import TensorDataset, Subset
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import abc
from utils.distance_utils import wasserstein_per_dimension
from utils.ema import EMA
from utils.diffusion_model_utils import save_cd_train_chkpt, snr_weighting
from utils.logger_utils import create_logger_dict, save_logger_object
from utils.sde_utils import VPSDE
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch_scatter import scatter
from accelerate import Accelerator
import wandb

from core.config import CD_LOG_FREQ, CD_SAVE_MODEL_FREQ, CD_LOG_MSE_CONDITION, CD_VAL_FREQ, DEBUG_TRAIN, BATCH_INCLUDE_KEYS_LIST, BATCH_INCLUDE_KEYS_INSTANTANEOUS_CSI_LIST, MAX_LOGGED_NETWORKS
from core.config import MAX_LOGGED_CLIENTS



def get_gpu_memory_info():

    # Get the current device
    device = torch.device("cuda")

    # Memory allocated by tensors
    allocated_memory = torch.cuda.memory_allocated(device)
    # Reserved memory (memory that is managed by the caching allocator)
    reserved_memory = torch.cuda.memory_reserved(device)
    
    print(f"Memory Allocated: {allocated_memory / 1024**2:.2f} MB")
    print(f"Memory Reserved: {reserved_memory / 1024**2:.2f} MB")


class DiffusionLearner(abc.ABC):
    def __init__(self, device, **kwargs):
        self.device = device
        self.beta_schedule = kwargs.get('beta_schedule', 'cosine')
        self.diffusion_steps = kwargs.get('diffusion_steps', 500)
        self.beta_min = kwargs.get('beta_min', 0.1)
        self.beta_max = kwargs.get('beta_max', 20)
        self.cosine_s = kwargs.get('cosine_s', 0.008)
        self.prior = torch.randn

        if self.beta_schedule == 'linear':
            self.betas = torch.linspace(self.beta_min / self.diffusion_steps, self.beta_max / self.diffusion_steps, self.diffusion_steps)
            self.alphas = 1 - self.betas
            self.baralphas = torch.cumprod(self.alphas, dim = 0)
            self.sde = VPSDE(N=self.diffusion_steps, beta_min = self.beta_min, beta_max = self.beta_max, beta_schedule = 'linear')

        elif self.beta_schedule == 'cosine':
            # Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
            s = self.cosine_s
            timesteps = torch.tensor(range(0, self.diffusion_steps), dtype=torch.float32).to(self.device)
            schedule = torch.cos((timesteps / self.diffusion_steps + s) / (1 + s) * torch.pi / 2)**2

            self.baralphas = schedule / schedule[0]
            # betas = 1 - baralphas / torch.concatenate([baralphas[0:1], baralphas[0:-1]])
            self.betas = 1 - self.baralphas / torch.cat([self.baralphas[0:1], self.baralphas[0:-1]])
            self.alphas = 1 - self.betas

            self.sde = None

        else:
            raise NotImplementedError
        

        self.batch_include_keys = {'train': BATCH_INCLUDE_KEYS_LIST,
                                   'eval': BATCH_INCLUDE_KEYS_LIST
                                   }
        
        self.batch_include_keys_instantaneous_csi = {'train': BATCH_INCLUDE_KEYS_INSTANTANEOUS_CSI_LIST,
                                                     'eval': BATCH_INCLUDE_KEYS_LIST
                                   }
        

    def sample_prior(self, sample_size):
        return self.prior(size=sample_size) # e.g., N(0, I)
    

    def generate_random_timesteps(self, num_steps, size, device, weights = None):
        try:
            if weights is None:
                timesteps = torch.randint(0, num_steps, size=size).to(device=device)

            else:
                assert len(weights) == num_steps, f"Length of weights {len(weights)} must be equal to num_steps {num_steps}."
                timesteps = torch.multinomial(weights, num_samples=size[0], replacement=True).to(device=device)
                timesteps = timesteps.view(size)

        except:
            timesteps = torch.randint(0, num_steps, size=size).to(device=device)
            print(f"Failed to generate random timesteps with weights = {weights}.")

        return timesteps

        
    def noise(self, Xbatch, t):
        # eps = torch.randn(size=Xbatch.shape).to(Xbatch.device)
        device = Xbatch.device
        t = t.to(device)
        eps = self.sample_prior(sample_size=Xbatch.shape).to(device)
        noised = (self.baralphas.to(device)[t] ** 0.5).repeat(1, Xbatch.shape[1]) * Xbatch + ((1 - self.baralphas.to(device)[t]) ** 0.5).repeat(1, Xbatch.shape[1]) * eps
        return noised, eps
    

    def score_fnc_to_noise_pred(self, timestep, score, noise_pred = None):
        '''
        Convert score function to noise pred function and vice versa.
        '''
        sqrt_1m_alphas_cumprod = (1 - self.baralphas[timestep]).pow(0.5)
        if noise_pred is None:
            noise_pred = -sqrt_1m_alphas_cumprod * score
        else:
            score = -1 / sqrt_1m_alphas_cumprod  * noise_pred
        return score, noise_pred
    

    def accelerated_sample(self, model, data, sampler = 'ddpm', nsamples = 100, device = 'cpu', **kwargs):
        n_graphs_per_batch = kwargs.get('n_graphs_per_batch', None)
        eta = kwargs.get('eta', 0.2)

        if sampler == 'ddpm':
            return self.accelerated_sample_ddpm(model, data, nsamples = nsamples, device = device, n_graphs_per_batch = n_graphs_per_batch)
        elif sampler == 'ddpm_x0':
            return self.accelerated_sample_ddpm_x0(model, data, nsamples = nsamples, device = device, ngraphs_per_batch = n_graphs_per_batch)
        elif sampler == 'ddim':
            return self.accelerated_sample_ddim(model, data, nsamples = nsamples, device = device, n_graphs_per_batch = n_graphs_per_batch, eta=eta)
        else:
            raise ValueError(f"Sampler {sampler} is not implemented.")
        

    def batch_from_data_list(self, data_list, exclude_keys, remap_keys = None, batch_size = None):
        data_batch = Batch.from_data_list(data_list=data_list,
                                          exclude_keys=exclude_keys)
        
        if remap_keys is not None:
            tau = None
            print("Training with instantaneous CSI.")
            # print(f"remap_keys: {remap_keys}")
            for key in remap_keys:
                try:
                    if key in data_batch.keys:
                        print(f"Remapping key {key} to {remap_keys[key]}")
                        
                        if isinstance(data_batch[key], list):
                            data_batch[key] = data_batch[key][0]
                        # if key in ['edge_index', 'edge_weight']:
                        #     tau = torch.randint(0, len(data_batch[key]), size = (1,)).squeeze().item() if tau is None else tau
                        #     data_batch[remap_keys[key]] = data_batch[key][tau].clone()
                        # else:
                        data_batch[remap_keys[key]] = data_batch[key].clone()
                        
                        del data_batch[key]

                except:
                    if key in data_batch.keys():
                        print(f"Remapping key {key} to {remap_keys[key]}")
                        
                        if isinstance(data_batch[key], list):
                            data_batch[key] = data_batch[key][0]

                        data_batch[remap_keys[key]] = data_batch[key].clone()
                        
                        del data_batch[key]
                    
            print(f"data_batch keys: {data_batch.keys()}")
        
        # if batch_size is not None:
        #     data_batch.batch = torch.remainder(data_batch.batch, batch_size)

        return data_batch

    

    def accelerated_sample_ddpm(self, model, data, nsamples = 100, device = 'cpu', n_graphs_per_batch = None):
        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)""" 
        model.eval()
        with torch.no_grad():
            
            x_batch = self.sample_prior(sample_size=(nsamples, *data.x.shape[1:])).to(device) # [n_samples, lambdas]



            # data_batch = Batch.from_data_list([Data(edge_index=data.edge_index_l, edge_weight = data.edge_weight_l, transmitters_index = data.transmitters_index, num_nodes = data.num_nodes)] * len(x_batch))
            
            try:
                data_batch = self.batch_from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
                                                exclude_keys=[key for key in data.keys() if key not in self.batch_include_keys['eval']],
                                                batch_size = n_graphs_per_batch if n_graphs_per_batch is not None else data.num_graphs
                                                )
            except:
                data_batch = self.batch_from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
                                                exclude_keys=[key for key in data.keys if key not in self.batch_include_keys['eval']],
                                                batch_size = n_graphs_per_batch if n_graphs_per_batch is not None else data.num_graphs
                                                )
        
            # data_batch = Batch.from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
            #                                     exclude_keys=[key for key in data.keys() if key not in self.batch_include_keys['eval']])
            
            
            x = x_batch.view(-1, x_batch.shape[-1])

            # xt = [x.cpu()]
            xt = [x.clone()]
            scoret = []
            for t in tqdm(range(self.diffusion_steps-1, 0, -1)):
                
                predicted_noise = model(x, torch.full([x.shape[0], 1], t).to(device), data_batch)
                if isinstance(predicted_noise, tuple):
                    predicted_noise = predicted_noise[0]

                # See DDPM paper between equations 11 and 12
                x = 1 / (self.alphas[t] ** 0.5) * (x - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * predicted_noise)
                if t > 1:
                    # See DDPM paper section 3.2.
                    # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                    variance = self.betas[t]
                    std = variance ** (0.5)
                    x += std * self.sample_prior(sample_size=x.shape).to(device)
                # xt += [x.cpu()]
                xt += [x.clone()]

                score, _ = self.score_fnc_to_noise_pred(timestep=t, score = None, noise_pred=predicted_noise)
                # scoret += [score.cpu()]
                scoret += [score.clone()]

        return x, {'x_t': xt,
                    'score_t': scoret,
                    'batch': data.batch.clone()
                #    'batch': data.batch.cpu()
                    }
    

    def sample_ddpm(self, model, data, nsamples = 100, device = 'cpu', n_graphs_per_batch = None):
        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)""" 
        model.eval()
        with torch.no_grad():
            
            x_batch = self.sample_prior(sample_size=(nsamples, *data.x.shape[1:])).to(device)
            # data_batch = Batch.from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
            #                                     exclude_keys=[key for key in data.keys() if key not in self.batch_include_keys['eval']])

            data_batch = self.batch_from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
                                                exclude_keys=[key for key in data.keys() if key not in self.batch_include_keys['eval']],
                                                batch_size = n_graphs_per_batch if n_graphs_per_batch is not None else data.num_graphs
                                                )
            
            # if n_graphs_per_batch is not None:
            #     data_batch.batch = torch.remainder(data_batch.batch, n_graphs_per_batch)
            
            x = x_batch.view(-1, x_batch.shape[-1])

            xt = [x.cpu()]
            scoret = []
            for t in tqdm(range(self.diffusion_steps-1, 0, -1)):
                
                predicted_noise = model(x, torch.full([x.shape[0], 1], t).to(device), data_batch)
                if isinstance(predicted_noise, tuple):
                    predicted_noise = predicted_noise[0]

                # See DDPM paper between equations 11 and 12
                x = 1 / (self.alphas[t] ** 0.5) * (x - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * predicted_noise)
                if t > 1:
                    # See DDPM paper section 3.2.
                    # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                    variance = self.betas[t]
                    std = variance ** (0.5)
                    x += std * self.sample_prior(sample_size=x.shape).to(device)
                xt += [x.cpu()]

                score, _ = self.score_fnc_to_noise_pred(timestep=t, score = None, noise_pred=predicted_noise)
                scoret += [score.cpu()]
            return x, {'x_t': xt, 'score_t': scoret, 'batch': data.batch.cpu()}
        
        

    def accelerated_sample_ddpm_x0(self, model, data, nsamples = 100, device = 'cpu', ngraphs_per_batch = None):
        """Sampler that uses the equations in DDPM paper to predict x0, then use that to predict x_{t-1}
        
        This is how DDPM is implemented in HuggingFace Diffusers, to allow working with models that predict
        x0 instead of the noise. It is also how we explain it in the Mixture of Diffusers paper.
        """
        model.eval()
        with torch.no_grad():
            
            x_batch = self.sample_prior(sample_size=(nsamples, *data.x.shape[1:])).to(device)
            # data_batch = Batch.from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
            #                                     exclude_keys=[key for key in data.keys() if key not in self.batch_include_keys['eval']])
            data_batch = self.batch_from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
                                                exclude_keys=[key for key in data.keys() if key not in self.batch_include_keys['eval']],
                                                batch_size = ngraphs_per_batch if ngraphs_per_batch is not None else data.num_graphs
                                                )

            # if n_graphs_per_batch is not None:
            #     data_batch.batch = torch.remainder(data_batch.batch, n_graphs_per_batch)
            
            x = x_batch.view(-1, x_batch.shape[-1])

            xt = [x.clone()]
            scoret = []
            for t in tqdm(range(self.diffusion_steps-1, 0, -1)):
                
                predicted_noise = model(x, torch.full([x.shape[0], 1], t).to(device), data_batch)
                if isinstance(predicted_noise, tuple):
                    predicted_noise = predicted_noise[0]

                # Predict original sample using DDPM Eq. 15
                x0 = (x - (1 - self.baralphas[t]) ** (0.5) * predicted_noise) / self.baralphas[t] ** (0.5)

                # Predict previous sample using DDPM Eq. 7
                c0 = (self.baralphas[t-1] ** (0.5) * self.betas[t]) / (1 - self.baralphas[t])
                ct = self.alphas[t] ** (0.5) * (1 - self.baralphas[t-1]) / (1 - self.baralphas[t])
                x = c0 * x0 + ct * x

                # Add noise
                if t > 1:
                    # Instead of variance = betas[t] the Stable Diffusion implementation uses this expression
                    variance = (1 - self.baralphas[t-1]) / (1 - self.baralphas[t]) * self.betas[t]
                    variance = torch.clamp(variance, min=1e-20)
                    std = variance ** (0.5)
                    x += std * self.sample_prior(sample_size=x.shape).to(device)

                xt += [x.clone()]

                score, _ = self.score_fnc_to_noise_pred(timestep=t, score = None, noise_pred=predicted_noise)
                scoret += [score.clone()]
        # return x, {'x_t': xt, 'score_t': scoret, 'batch': data.batch.cpu()}
        return x, {'x_t': xt,
                       'score_t': scoret,
                       'batch': data.batch.clone()
                    #    'batch': data.batch.cpu()
                       }
        
        

    def accelerated_sample_ddim(self, model, data, nsamples = 100, device = 'cpu', n_graphs_per_batch = None, eta = 0.2):
        """Sampler following the Denoising Diffusion Implicit Models method by Song et al.
        https://arxiv.org/abs/2010.02502

        Args:
            eta: Controls the stochasticity of the sampler. eta = 0 is DDIM deterministic, eta = 1 recovers DDPM stochastic sampling
        """
        model.eval()
        with torch.no_grad():
            
            x_batch = self.sample_prior(sample_size=(nsamples, *data.x.shape[1:])).to(device)
            # data_batch = Batch.from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
            #                                 exclude_keys=[key for key in data.keys() if key not in self.batch_include_keys['eval']])
            
            data_batch = self.batch_from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
                                                exclude_keys=[key for key in data.keys() if key not in self.batch_include_keys['eval']],
                                                batch_size = n_graphs_per_batch if n_graphs_per_batch is not None else data.num_graphs
                                                )
            
            x = x_batch.view(-1, x_batch.shape[-1])

            xt = [x.clone()]
            scoret = []
            for t in tqdm(range(self.diffusion_steps-1, 0, -1)):
                
                predicted_noise = model(x, torch.full([x.shape[0], 1], t).to(device), data_batch)
                if isinstance(predicted_noise, tuple):
                    predicted_noise = predicted_noise[0]

                # Predict x0 from xt and noise
                x0_pred = (x - ((1 - self.baralphas[t])**0.5) * predicted_noise) / (self.baralphas[t]**0.5)

                # Get the coefficient for x0 prediction
                alpha_next = self.baralphas[t-1] if t > 1 else 1.0
                c1 = eta * ((1 - alpha_next) / (1 - self.baralphas[t]) * (1 - self.baralphas[t] / alpha_next))**0.5
                c2 = ((1 - alpha_next) - c1**2)**0.5

                # Get xt-1 by combining the predicted x0 and noise
                x = (alpha_next)**0.5 * x0_pred + c2 * predicted_noise
                if t > 1:
                    x = x + c1 * self.sample_prior(sample_size=x.shape).to(device)

                xt += [x.clone()]
                score, _ = self.score_fnc_to_noise_pred(timestep=t, score=None, noise_pred=predicted_noise) 
                scoret += [score.clone()]

        # return x, {'x_t': xt, 'score_t': scoret, 'batch': data.batch.cpu()}
        return x, {'x_t': xt,
                        'score_t': scoret,
                        'batch': data.batch.clone()
                        #    'batch': data.batch.cpu()
                        }


        
    # Implement a training routine.
    @abc.abstractmethod
    def train(self, model, dataloader, n_epochs, lr, device, loggers):
        pass



class ConditionalDiffusionLearner(DiffusionLearner):
    def __init__(self, config, device = 'cpu'):
        print('Initializing the conditional diffusion-learner.')
        self.config = config
        self.device = device
        super(ConditionalDiffusionLearner, self).__init__(device=device, kwargs=vars(config) if isinstance(config, Namespace) else config)
        self.loss_fn = nn.MSELoss(reduction = 'none') # loss function for the training routine
        self.optim_weight_decay = config.weight_decay if hasattr(config, 'weight_decay') else 0.0
        self.clip_grad_norm_constant = config.pgrad_clipping_constant if hasattr(config, 'pgrad_clipping_constant') else None


    def train(self, accelerator, model, dataloader, n_epochs = 100, lr = 1e-1, P_max = 0.01, noise_var = 0., inv_transform = None, device = 'cpu', loggers = None, **kwargs):
        per_epoch_grad_only = kwargs.get('per_epoch_grad_only', False)
        graph_token_drop_prob = kwargs.get('graph_token_drop_prob', 0.1)
        graph_instantaneous_csi_prob = kwargs.get('graph_instantaneous_csi_prob', 0.0)
        ema_alpha = kwargs.get('ema_alpha', 0.99)
        use_checkpointing = kwargs.get('use_checkpointing', False)
        save_cd_train_chkpt_path = kwargs.get('save_cd_train_chkpt_path', None)
        load_accelerator_chkpt_path = kwargs.get('load_accelerator_chkpt_path', None)

        sampler = kwargs.get('sampler', 'ddpm')
        sampler_eta = kwargs.get('sampler_eta', 0.2)

        print('Training the conditional diffusion model.')
        q = graph_token_drop_prob if 'conditional' in model.prototype_name else None # probability of null token
        print(f'Graph token drop probability q = {q}')
        
        model = model.to(device)
        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=self.optim_weight_decay)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=self.optim_weight_decay)  # AdamW with weight decay


        T_0=10
        T_mult=2
        gamma=0.9
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max = n_epochs // 10, eta_min=lr * 1e-2)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0, T_mult=T_mult, eta_min=lr * 1e-3)

        warmup_period = T_0

        # Define LambdaLR to apply decay to restart values
        def decay_lambda(epoch):
            if epoch < warmup_period:
                return (epoch + 1) / warmup_period
            else:
                epoch -= warmup_period
                restart_count = 0
                T_cur = T_0  # Current restart period
                while epoch >= T_cur:
                    epoch -= T_cur
                    T_cur *= T_mult
                    restart_count += 1
                return gamma ** restart_count  # Apply decay factor per restart

        # Wrap the scheduler
        decay_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lambda)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[cosine_scheduler, decay_scheduler], milestones=[0])

        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
        lr_schedule_milestone = CD_LOG_FREQ
        # scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=lr_schedule_milestone)
        # scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
        # scheduler = optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler1, scheduler2], milestones=[lr_schedule_milestone])
        
        scaler = GradScaler()


        def make_cd_log_criterion(n_epochs, cd_log_freq):

            checkpoints = [cd_log_freq * i for i in [1, 2, 3, 4, 5, 6, 8, 11, 15, 21, 32, 48, 64, 96, 128, 192, 256, 512] if cd_log_freq * i < n_epochs]

            def thunk(epoch):
                return (epoch + 1) in checkpoints or (epoch + 1) == n_epochs
            return thunk

        cd_log_criterion = make_cd_log_criterion(n_epochs=n_epochs, cd_log_freq=CD_LOG_FREQ)
        cd_val_criterion = make_cd_log_criterion(n_epochs=n_epochs, cd_log_freq=CD_VAL_FREQ)

        if accelerator is not None:
            accelerator.print('Accelerated diffusion model training...')
            model, optimizer, dataloader['train'], scheduler = accelerator.prepare(model, optimizer, dataloader['train'], scheduler)
            for phase in dataloader:
                if phase not in ['train']:
                    dataloader[phase] = accelerator.prepare(dataloader[phase])

            model = model.to(device)

        else:
            print("Accelerator is not running...")

        if ema_alpha is not None:
            # ema = EMA(ema_alpha)
            # ema.register(model)

            ema = EMA(parameters=model.parameters(),
                      ema_max_decay=ema_alpha,
                      ema_inv_gamma=1.0
                      )
            print(f'EMA({ema_alpha}) model is registered.')

        else:
            ema = None

        if wandb.run is not None:
            # wandb_tracker = accelerator.get_tracker("wandb")
            # wandb.define_metric("epoch")
            wandb.define_metric("GDM-algo/train_epoch_*", step_metric="epoch", goal="minimize", summary="best")
            wandb.define_metric("GDM-sampler/*", step_metric="epoch", summary="last")
            wandb.define_metric("GDM-algo-losses-t-sweep/train_epoch_*", step_metric="epoch", goal="minimize", summary="best")
            wandb.define_metric("GDM-algo-weights-t-sweep/epoch_*", step_metric="epoch", goal='minimize', summary='best')
            wandb.define_metric("GDM-algo/lr*", step_metric="epoch", summary="last")

            wandb.watch(model, log_freq=CD_LOG_FREQ, log="all", log_graph=True, idx=1)

        t_weights = None
        loss_t_weights = torch.ones(self.diffusion_steps, device=device)
        # Compute SNR-dependent weights
        try:
            loss_t_weights = snr_weighting(alphas_cumprod=self.baralphas, mode="log_snr")
            print(f"SNR-dependent weights: {loss_t_weights}")

        except:
            print("Failed to compute SNR-dependent weights.")


        # Try loading accelerator state
        if load_accelerator_chkpt_path is not None:
            accelerator.print(f"Accelerator load state chkpt path is found at {load_accelerator_chkpt_path}")
            try:
                load_dir = f"{load_accelerator_chkpt_path}"
                accelerator.load_state(input_dir=load_dir)

                accelerator.print(f"Accelerator state is loaded successfully from {load_dir}")

            except:
                accelerator.print(f"Failed to load accelerator state from {load_dir}")


        # Register diffusion model for checkpointing
        accelerator.register_for_checkpointing(model, optimizer, scheduler)

        # Save the starting state
        # save_dir = f"{accelerator.project_dir}/checkpoints/GDM-training"
        # os.makedirs(save_dir, exist_ok=True)
        accelerator.save_state()

        for epoch in tqdm(range(n_epochs), disable=not accelerator.is_local_main_process):

            run_eval = False
            avg_loss = defaultdict(float) # ['train' and 'val']
            avg_loss_simplified = defaultdict(float)
            # avg_t_loss = defaultdict(float)

            for phase in dataloader: # train and val phases

                if phase not in ['train'] and not cd_val_criterion(epoch): # skip validation phase
                    continue

                if phase == 'train':
                    model.train()
                    model.zero_grad()
                else:
                    model.eval()

                epoch_loss = epoch_loss_simplified = steps = epoch_pgrad_norm = 0

                all_losses = torch.zeros(len(dataloader[phase].dataset), dtype = torch.float32, device = device) # track losses across all networks
                # all_simplified_losses = torch.zeros_like(all_losses)
                n_time_samples = 10
                all_t_losses = torch.zeros(n_time_samples, dtype = torch.float32, device = torch.device("cpu")) # track losses across all time steps
    
                for data, graph_idx in dataloader[phase]:
                    
                    with torch.set_grad_enabled(phase == 'train'):                    
                        with accelerator.accumulate(model):
                            # torch.cuda.empty_cache()
                        
                        # Run the forward pass with autocasting
                        # with torch.autocast(device_type='cuda', dtype=torch.float16):
                        # with torch.autocast(enabled=False, device_type=device, dtype=torch.float32):


                            if epoch < 10 and phase == 'train':
                                accelerator.print('GPU utilization before data.to(device)')
                                get_gpu_memory_info()

                                # Print out graph index to assert that dataloader is shuffling properly.
                                accelerator.print(f"Phase = {phase}\tEpoch = {epoch}\tgraph_idx={graph_idx}")
                                accelerator.print("dataloader.sampler: ", dataloader[phase].sampler.__class__.__name__)

                            # data = data.to(device)
                            graph_idx = graph_idx.to(device)


                            if phase == 'train':
                                x_batch_size = self.config.x_batch_size
                            else: # use less samples for val and test phases
                                val_and_test_phase_sample_decrease = 0.2
                                x_batch_size = int(min(self.config.x_batch_size, val_and_test_phase_sample_decrease * self.config.x_batch_size // max(1, len(dataloader[phase].dataset) // len(dataloader['train'].dataset))))

                                                    
                            try:
                                batch_include_keys = self.batch_include_keys['train'] 
                                if np.random.random() < graph_instantaneous_csi_prob: # probability of instantaneous CSI
                                    batch_include_keys = batch_include_keys + ['edge_index_t', 'edge_weight_t', 't'] # self.batch_include_keys_instantaneous_csi['train'] # self.batch_include_keys['train']

                                # accelerator.print("data.keys: ", data.keys())
                                # accelerator.print("data.edge_index_t: ", data.edge_index_t)
                                # accelerator.print("data.edge_weight_t: ", data.edge_weight_t)
                                # accelerator.print("data.t: ", data.t)
                                        
                                

                                data_batch = self.batch_from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(x_batch_size)],
                                                                exclude_keys=[key for key in data.keys() if key not in batch_include_keys],
                                                                remap_keys = None, #remap_keys,
                                                                batch_size = data.num_graphs
                                                                )
                            except:
                                data_batch = self.batch_from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(x_batch_size)],
                                                            exclude_keys=[key for key in data.keys if key not in batch_include_keys],
                                                            remap_keys = None, # remap_keys,
                                                            batch_size = data.num_graphs
                                                            )
                                
                            
                            
                            # Move data batch to GPU
                            data_batch = data_batch.to(device)
                            if (epoch == 0 or (epoch + 1) % CD_LOG_FREQ == 0) and phase == 'train':
                                accelerator.print('GPU utilization after data_batch.to(device)')
                                get_gpu_memory_info()


                            n_graphs_per_batch = len(graph_idx)
                            X_batch = data.x.squeeze(-1) # [T, (B x n), 1] -> (T, B x n)
                            X_batch = torch.stack([x.view(n_graphs_per_batch, -1) for x in X_batch], dim = 0) # T, B, n
                            X_graph_id = data.batch
                            
                            # Get different sample indices for each graph to avoid correlation between samples
                            x_idx = torch.randint(low=0, high=len(data.x), size = (x_batch_size, n_graphs_per_batch)).to(X_batch.device)
                            X_batch = torch.cat([X_batch[x_idx[:, i], i] for i in range(x_idx.shape[1])], dim = 1) # T, n x B
                            X_batch = X_batch.view(-1, 1)

                            assert X_batch.device == torch.device('cpu'), f"X_batch.device: {X_batch.device}"
                            # X_batch = X_batch.view(-1, X_batch.shape[-1])

                            n = data_batch.num_nodes // data_batch.num_graphs
                            timesteps = self.generate_random_timesteps(num_steps = self.diffusion_steps, size = [len(X_batch) // n, 1], device = X_batch.device, weights=t_weights)
                            timesteps = timesteps.repeat_interleave(repeats=n, dim = 0)
                            noised, eps = self.noise(X_batch, timesteps)

                            if q is not None:
                                predicted_noise = model(noised.to(device), timesteps.to(device), data_batch, conditioning = np.random.random() > q, return_attention_weights = DEBUG_TRAIN, debug_forward_pass = DEBUG_TRAIN)
                            else:
                                predicted_noise = model(noised.to(device), timesteps.to(device), data_batch, return_attention_weights = DEBUG_TRAIN, debug_forward_pass = DEBUG_TRAIN)

                            if isinstance(predicted_noise, tuple):
                                for i in range(len(predicted_noise), 0, -1):
                                    if i == 3:
                                        model_debug_data = predicted_noise[i-1]
                                    elif i == 2:
                                        attn_weights = predicted_noise[i-1]
                                    elif i == 1:
                                        predicted_noise = predicted_noise[i-1]
                                    else:
                                        pass
                                # attn_weights = predicted_noise[1]
                                # predicted_noise = predicted_noise[0]
                            else:
                                attn_weights = None
                                model_debug_data = None

                            # print('Step: ', steps, '\tgraph_idx: ', graph_idx)

                            all_loss_terms = self.loss_fn(predicted_noise.view(-1, n), eps.view(-1, n).to(device)).mean(dim = -1)
                            all_graph_idx = graph_idx.repeat(x_batch_size,).to(device)
                            scatter(src=all_loss_terms, index=all_graph_idx, dim=0, reduce='mean', out=all_losses) # compute avg loss over each graph separately
                            
                            if loss_t_weights is None:
                                loss = all_loss_terms.mean()
                                simplified_loss = loss.clone()
                            else:
                                loss_t_weights = snr_weighting(alphas_cumprod=self.baralphas[timesteps[::n]], mode="log_snr").view_as(all_loss_terms)


                                # accelerator.print("loss_t_weights.min, max, isnan: ", loss_t_weights.min(), loss_t_weights.max(), torch.any(torch.isnan(loss_t_weights))) if epoch == 0 else None

                                loss_t_weights = loss_t_weights / loss_t_weights.sum()

                                # accelerator.print("loss_t_weights: ", loss_t_weights.shape) if epoch == 0 else None
                                accelerator.print("all_loss_terms: ", all_loss_terms.shape) if epoch == 0 else None
                                loss = (all_loss_terms * loss_t_weights).sum()

                                simplified_loss = all_loss_terms.mean()

                            tt = timesteps[::n].view(-1)
                            dtimesteps = torch.ones_like(tt) * self.diffusion_steps // n_time_samples
                            all_time_idx = torch.div(tt, dtimesteps, rounding_mode='floor').long()
                            
                            # scatter(src=all_loss_terms, index=all_time_idx, dim=0, reduce='mean', out=all_t_losses) # compute avg loss over each time step separately
                            scatter(src=all_loss_terms.clone().detach().cpu(), index=all_time_idx, dim=0, reduce='mean', out=all_t_losses) # compute avg loss over each time step separately

                            # Dynamically update the weights for the timesteps
                            if phase == 'train' and (epoch + 1) >= 1.1 * n_epochs:
                                t_weights = all_t_losses.repeat_interleave(repeats=(self.diffusion_steps // n_time_samples)) / all_t_losses.sum()
                        
                        
                        if not per_epoch_grad_only and phase == 'train':
                            optimizer.zero_grad()
                            
                            if accelerator is not None:
                                accelerator.backward(loss)
                            else:
                                loss.backward()     

                            if self.clip_grad_norm_constant is not None and accelerator is not None:
                                # Perform gradient clipping
                                # torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm_constant) # 1

                                try:
                                    # min_norm = 0.05 * self.clip_grad_norm_constant
                                    min_norm = 0.005 * self.clip_grad_norm_constant
                                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)

                                    if total_norm < min_norm:
                                        scale_factor = min_norm / (total_norm + 1e-6)  # Avoid division by zero
                                        for p in model.parameters():
                                            if p.grad is not None:
                                                p.grad.data.mul_(scale_factor)  # Scale up small gradients
                                except:
                                    print(f"Skipping failed min-clipping norms of gradients.")

                                accelerator.clip_grad_norm_(model.parameters(), self.clip_grad_norm_constant)

                                params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                                pgrad_norm = np.sqrt(np.sum([p.grad.norm().item()**2 for p in params]))
                                epoch_pgrad_norm += pgrad_norm

                            optimizer.step()

                            # scaler.scale(loss).backward()
                            # # scaler.step() first unscales the gradients of the optimizer's assigned params.
                            # scaler.step(optimizer=optimizer)

                            if ema is not None:
                                # Update the exponential moving average
                                ema.step(model.parameters())
                                # ema.update(model)

                            model.zero_grad()

                            # scale = scaler.get_scale()
                            # # Update the scale function for next iteration
                            # scaler.update()
                            # skip_lr_sched = (scale > scaler.get_scale())
                            skip_lr_sched = False

                    # accelerator.print('accelerator.gather(loss): ', accelerator.gather(loss))
                    # accelerator.print('accelerator process index: ', accelerator.process_index)
                    # accelerator.print('Steps: ', steps, '\tAll losses: ', all_losses)
                    
                    # epoch_loss += loss
                    epoch_loss += accelerator.gather(loss).mean()
                    epoch_loss_simplified += accelerator.gather(simplified_loss).mean()
                    steps += 1

                    # print('steps: ', steps)
                    # data_batch = data_batch.cpu()
                    del data_batch
                    torch.cuda.empty_cache()
                
                avg_loss[phase] = epoch_loss / steps
                avg_loss_simplified[phase] = epoch_loss_simplified / steps
                avg_pgrad_norm = epoch_pgrad_norm / steps

                ### gather individual network losses also across networks ###
                all_losses_gathered = accelerator.gather(all_losses) # [num_graphs] -> [num_graphs * num_processes]
                accelerator.print('accelerator.gather(all_losses): ', all_losses_gathered)

                process_tensor = torch.tensor([accelerator.process_index]).to(device=device)
                num_processes = len(accelerator.gather(process_tensor))
                all_losses = all_losses_gathered.view(num_processes, -1).sum(dim = 0)
                accelerator.print('accelerator.gather(all_losses).sum_across_all_processes: ', all_losses)

                if all_t_losses is not None:
                    ### gather average losses across different timesteps ###
                    all_t_losses_gathered = accelerator.gather(all_t_losses) # [num_timesteps] -> [num_timesteps * num_processes]

                    accelerator.print('accelerator.gather(all_t_losses): ', all_t_losses_gathered)

                    process_tensor = torch.tensor([accelerator.process_index]).to(device=device)
                    num_processes = len(accelerator.gather(process_tensor))
                    all_t_losses = all_t_losses_gathered.view(num_processes, -1).sum(dim = 0)
                    accelerator.print('accelerator.gather(all_t_losses).sum_across_all_processes: ', all_t_losses)

                if per_epoch_grad_only and phase == 'train':

                    # loss = self.loss_fn(all_pred.view(-1, n).to(device), all_eps.view(-1, n).to(device))
                    optimizer.zero_grad()

                    if accelerator is not None:
                        accelerator.backward(avg_loss[phase])
                    else:
                        avg_loss[phase].backward()

                    optimizer.step()

                    if ema is not None:
                        ema.update(model)
                    # epoch_loss += loss
                    # steps += 1
                    model.zero_grad()

                epoch_variables = {'diffusion-loss': avg_loss[phase],
                                    'diffusion-pgrad-norm': avg_pgrad_norm,
                                    'diffusion-all-networks-loss': all_losses.tolist(),
                                    'model_attn_weights': attn_weights,
                                    'model_debug_data': model_debug_data,
                                    'model_layer_norms': (epoch, model_debug_data['model_layer_norms']) if model_debug_data is not None else (epoch, None)
                                    }
                
                try:
                    accelerator.log({f"GDM-algo/{phase}_epoch_loss": avg_loss[phase].mean().item(),
                                     f"GDM-algo/{phase}_epoch_simplified_loss": avg_loss_simplified[phase].mean().item(),
                                     f"GDM-algo/{phase}_epoch_pgrad_norm": avg_pgrad_norm,
                                     f"GDM-algo/{phase}_epoch_loss_max":  all_losses.max().item(),
                                     f"GDM-algo/{phase}_epoch_loss_min":  all_losses.min().item(),
                                     "epoch": epoch
                    })
                    accelerator.print(f'GDM-algo/{phase}_epoch_loss logged successfully.')

                    if all_t_losses is not None and DEBUG_TRAIN:
                        accelerator.log({f"GDM-algo-losses-t-sweep/{phase}_epoch_t_loss": all_t_losses.mean().item(),
                                        f"GDM-algo-losses-t-sweep/{phase}_epoch_t_loss_max":  all_t_losses.max().item(),
                                        f"GDM-algo-losses-t-sweep/{phase}_epoch_t_loss_min":  all_t_losses.min().item(),
                                        f"GDM-algo-losses-t-sweep/{phase}_epoch_t_loss_argmin":  torch.argmin(all_t_losses).item(),
                                        f"GDM-algo-losses-t-sweep/{phase}_epoch_t_loss_argmax":  torch.argmax(all_t_losses).item(),
                                        f"GDM-algo-losses-t-sweep/{phase}_epoch_t_0_loss":  all_t_losses[0].item(),
                                        f"GDM-algo-losses-t-sweep/{phase}_epoch_t_1/2_loss":  all_t_losses[len(all_t_losses) // 2].item(),
                                        f"GDM-algo-losses-t-sweep/{phase}_epoch_t_1_loss":  all_t_losses[-1].item(),
                                        "epoch": epoch
                                        })
                        accelerator.print(f'GDM-algo/{phase}_epoch_t_loss logged successfully.')

                    
                    if epoch == 0 or (epoch + 1) % CD_LOG_FREQ == 0 and DEBUG_TRAIN:    
                        try:
                            accelerator.print('Logging average diffusion losses across timesteps as a chart to Wandb...')               
            
                            fig, ax = plt.subplots()
                            ax.plot(np.arange(len(all_t_losses)), all_t_losses.detach().cpu().numpy(), label = f'Epoch = {epoch}')
                            ax.set_xlabel('Timesteps')
                            ax.set_ylabel('Diffusion loss')
                            wandb.log({f"plots/GDM-algo-t-losses", wandb.Image(fig)}, step = epoch)
                            plt.close(fig)

                            # # accelerator.log({f"charts/GDM-algo-t-losses-epoch-{epoch}": fig})
                            # ### Log all_t_losses as a plot to Wandb and have these overlaid each epoch we log
                            # accelerator.print('Logging average diffusion losses across timesteps as a plot to Wandb...')
                            # wandb.log({f"plots/GDM-algo-t-losses": wandb.Plotly(fig)}, step = epoch)

                            # plt.close(fig)
                        
                        except:
                            accelerator.print("Failed to log all_t_losses to wandb through Accelerator.")
                            

                    if t_weights is not None and phase == 'train' and DEBUG_TRAIN:
                        ttw = t_weights.unique()
                        accelerator.log({f"GDM-algo-weights-t-sweep/epoch_t_weight": ttw.mean().item(),
                                        f"GDM-algo-weights-t-sweep/epoch_t_weight_max":  ttw.max().item(),
                                        f"GDM-algo-weights-t-sweep/epoch_t_weight_min":  ttw.min().item(),
                                        f"GDM-algo-weights-t-sweep/epoch_t_weight_argmin":  torch.argmin(ttw).item(),
                                        f"GDM-algo-weights-t-sweep/epoch_t_weight_argmax":  torch.argmax(ttw).item(),
                                        f"GDM-algo-weights-t-sweep/epoch_t_0_weight":  ttw[0].item(),
                                        f"GDM-algo-weights-t-sweep/epoch_t_1/2_weight":  ttw[len(ttw) // 2].item(),
                                        f"GDM-algo-weights-t-sweep/epoch_t_1_weight":  ttw[-1].item(),
                                        "epoch": epoch
                                        })
                        
                        accelerator.print(f'GDM-algo/epoch_t_weights logged successfully.')
                    
                    # wandb.log({f"Histograms/{phase}_epoch_t_loss": wandb.Histogram(all_t_losses.tolist())}, global_step = epoch)
                    # accelerator.print(f'GDM-algo/{phase}_epoch_t_loss Histogram logged successfully.')
                    # wandb.log({f"Histograms/{phase}_epoch_loss": wandb.Histogram(all_losses.tolist())}, global_step = epoch)
                    # accelerator.print(f'GDM-algo/{phase}_epoch_loss Histogram logged successfully.')                    

                except:
                    accelerator.print("Failed to log all metrics to wandb through Accelerator.")
                    pass
                
                if phase == 'train' and (epoch + 1) % CD_SAVE_MODEL_FREQ == 0:
                    log_path = [logger.log_path for logger in loggers[phase] if logger.log_metric == 'model_weights'][0]
                    # Save model weights
                    if accelerator is None:
                        torch.save(model.state_dict(), f'{log_path}/cd_model_state_dict_epoch_{epoch}.pt')
                        print(f'Saving model weights at epoch {epoch} is successful.')
                    else:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), f'{log_path}/cd_model_state_dict_epoch_{epoch}.pt')
                        accelerator.print(f'Saving model weights at epoch {epoch} is successful.')

                
                if accelerator.is_main_process:
                    # accelerator.print("Diffusion training logging!")
                    if loggers[phase] is not None:
                        for logger in loggers[phase]:
                            if logger.log_metric in ['diffusion-loss', 'diffusion-pgrad-norm']: # log losses and model gradients
                                logger.update_data(epoch_variables)
                                logger()

                            elif logger.log_metric in ['diffusion-all-networks-loss']: # log losses for all networks
                                logger.update_data(epoch_variables)
                                logger()

                            ### Using a logger to save accelerated model does not work ###
                            # elif logger.log_metric in ['model_weights']: # save model weights checkpoint
                            #     if (epoch + 1) % CD_SAVE_MODEL_FREQ == 0:
                            #         logger(accelerator.unwrap_model(model).state_dict(), epoch = epoch)

                            elif logger.log_metric in ['model_attn_weights']:
                                # if (epoch + 1) % CD_LOG_FREQ == 0:
                                if cd_log_criterion(epoch):
                                    logger.update_data(epoch_variables)
                                    logger()

                            elif logger.log_metric in ['model_debug_data']:
                                # if (epoch + 1) % CD_LOG_FREQ == 0:
                                if cd_log_criterion(epoch):
                                    logger.update_data(epoch_variables)
                                    logger(n_graphs_per_batch = n_graphs_per_batch)

                            elif logger.log_metric in ["model_layer_norms"]:
                                logger.update_data(epoch_variables)
                                # if (epoch + 1) % CD_LOG_FREQ == 0:
                                if cd_log_criterion(epoch):
                                    # logger.update_data(epoch_variables)
                                    logger()
                    
                if (epoch + 1) % 10 == 0:
                    accelerator.print(f"Epoch {epoch}, phase = {phase}, loss = {avg_loss[phase]}")

                if not skip_lr_sched and phase == 'train':
                    
                    # if scheduler.T_cur == 0 and epoch !=0:  # to prevent decrease on the first epoch
                    #     accelerator.print('Lr scheduler is resetting with decreased base_lr.')
                    #     new_base_lr = max(scheduler.eta_min * 10, scheduler.base_lrs[0] * restart_base_lr_decay_factor)
                    #     scheduler.base_lrs[0] = new_base_lr

                    scheduler.step()
                    try:
                        current_lr = optimizer.param_groups[0]['lr']
                        accelerator.log({f"GDM-algo/lr": current_lr,
                                         "epoch": epoch
                                        })
                        # accelerator.log({f"GDM-algo/lr-0": scheduler.get_last_lr()[0],
                        #                  f"GDM-algo/lr-1": scheduler.get_last_lr()[-1],
                        #                  "epoch": epoch},
                        #                  global_step = epoch
                        #                 )
                    except:
                        pass

                # if avg_loss['train'] < CD_LOG_MSE_CONDITION:
                if avg_loss_simplified['train'] < CD_LOG_MSE_CONDITION: # unweighted loss is a reliable metric for running eval phase
                    run_eval = True

                # if (epoch + 1) % CD_LOG_FREQ == 0 and inv_transform is not None:
                if cd_log_criterion(epoch) and inv_transform is not None:
                    accelerator.print(f"Verifying permutation equivariance of the architecture on {phase} data.")
                    self.verify_perm_equivariance(model=model, dataloader=dataloader[phase], device=device)

                    if run_eval:
                        accelerator.print(f'Running eval phase on {phase} data.')
                        eval_graphs_idx = torch.arange(0, min(len(dataloader[phase].dataset), MAX_LOGGED_NETWORKS))

                        if eval_graphs_idx is not None:
                            # From dataloader[phase], create a new dataloader with only the eval_graphs_idx
                            eval_dataset = Subset(dataloader[phase].dataset, eval_graphs_idx)
                            accelerator.print(f'Creating eval dataset with {len(eval_dataset)} graphs.')
                            eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=dataloader[phase].collate_fn)

                            accelerator.print("Passing eval_dataloader to self.accelerated_eval().")

                            eval_loggers = [logger for logger in loggers[phase] if not (hasattr(logger, 'network_id') and logger.network_id not in eval_graphs_idx.tolist())]

                            accelerator.print("Passing eval_loggers to self.accelerated_eval().")

                            eval_metrics = self.accelerated_eval(
                                                accelerator=accelerator,
                                                model=model,
                                                dataloader=eval_dataloader,
                                                nsamples=self.config.diffuse_n_samples,
                                                device=device,
                                                loggers=eval_loggers,
                                                P_max=P_max,
                                                noise_var = noise_var,
                                                inv_transform = inv_transform,
                                                sampler=sampler,
                                                eta=sampler_eta
                                                )
                            
                        else:
                            eval_metrics = self.accelerated_eval(
                                            accelerator=accelerator,
                                            model=model,
                                            dataloader=dataloader[phase],
                                            nsamples=self.config.diffuse_n_samples,
                                            device=device,
                                            loggers=loggers[phase],
                                            P_max=P_max,
                                            noise_var = noise_var,
                                            inv_transform = inv_transform,
                                            sampler=sampler,
                                            eta=sampler_eta
                                            )

                        accelerator.wait_for_everyone() # trying to debug why EMA update breaks with accelerate.

                        try:
                            eval_metrics = {f'GDM-sampler/{phase}_{key}': eval_metrics[key] for key in eval_metrics}
                            eval_metrics.update({"epoch": epoch})
                            accelerator.log(eval_metrics)
                        except:
                            accelerator.print("Logging eval metrics from diffusion sampler failed.")


                        if accelerator.is_main_process and (epoch + 1) % CD_SAVE_MODEL_FREQ == 0:
                            if save_cd_train_chkpt_path is not None:
                                save_cd_train_chkpt(save_path=save_cd_train_chkpt_path,
                                                    model=accelerator.unwrap_model(model),
                                                    optimizer=accelerator.unwrap_model(optimizer),
                                                    lr_sched=accelerator.unwrap_model(scheduler),
                                                    epoch=epoch,
                                                    accelerator=accelerator
                                                    )
                                
                                # accelerator.save_state()
                                accelerator.print(f'Accelerator state is successfully saved at GDM-train-algo, epoch = {epoch}.')
                                
                            # Save loggers to file
                            for phase in loggers:
                                if loggers[phase] is not None and len(loggers[phase]):
                                    save_obj = create_logger_dict(loggers[phase])
                                    save_path = loggers[phase][0].log_path
                                    save_logger_object(obj=save_obj,
                                                    #    filename=save_path + f'/loggers_epoch_{epoch}.pkl',
                                                       filename=save_path + f'/loggers.pkl'
                                                       )
                                    print(f'{phase} loggers have been saved and pickled successfully.')

                        
                    else:
                        accelerator.print(f'Skipping eval phase on {phase} data because loss = ' + str(avg_loss['train'].item()) + f' < {CD_LOG_MSE_CONDITION} = CD_LOG_MSE_CONDITION failed.')

        accelerator.save_state()
        accelerator.print('Diffusion training completed successfully.')

        return model, epoch_loss
    
    

    def verify_perm_equivariance(self, model, dataloader, nsamples = 1, device = 'cpu'):
        " Verify permutation equivariance of the architecture."

        return

        model = model.to(device)
        model = model.eval()

        with torch.no_grad():
            for data, graph_idx in dataloader:
                data = data.to(device)

                x_batch = self.sample_prior(sample_size=(nsamples, *data.x.shape[1:])).to(device)
                # data_batch = Batch.from_data_list([Data(edge_index=data.edge_index_l, edge_weight = data.edge_weight_l, transmitters_index = data.transmitters_index, num_nodes = data.num_nodes)] * len(x_batch))
                data_batch = Batch.from_data_list(data_list=[temp for temp in data.to_data_list() for _ in range(len(x_batch))],
                                                    exclude_keys=[key for key in data.keys() if key not in BATCH_INCLUDE_KEYS_LIST])
                
                x = x_batch.view(-1, x_batch.shape[-1])
                t = self.diffusion_steps - 1 # pick a random timestep

                x_perm = x.clone()
                predicted_noise = model(x, torch.full([x.shape[0], 1], t).to(device), data_batch)
                if isinstance(predicted_noise, tuple):
                    predicted_noise = predicted_noise[0]

                perm = torch.randperm(x.size(0)).to(device=x.device) # Random permutation of node indices
                perm = torch.arange(x.size(0)).to(device=x.device)
                perm[0] = 1
                perm[1] = 2
                perm[2] = 0

                x_perm = torch.zeros_like(x)
                batch_perm = torch.zeros_like(data_batch.batch)
                transmitters_index = torch.zeros_like(data_batch.transmitters_index)
                
                x_perm[perm] = x
                batch_perm[perm] = data_batch.batch
                transmitters_index[perm] = data_batch.transmitters_index
                edge_index_l = perm[data_batch.edge_index_l]

                data_batch.batch = batch_perm
                data_batch.transmitters_index = transmitters_index
                data_batch.edge_index_l = edge_index_l

                predicted_noise_perm = model(x_perm, torch.full([x_perm.shape[0], 1], t).to(device), data_batch)
                if isinstance(predicted_noise_perm, tuple):
                    predicted_noise_perm = predicted_noise_perm[0]

                predicted_noise_perm_corrected = predicted_noise_perm[perm]

                print("Pred noise: ", predicted_noise)
                print("Pred nosie perm corrected: ", predicted_noise_perm_corrected)
                # Compare the outputs (they should match if the model is equivariant)
                assert torch.allclose(predicted_noise, predicted_noise_perm_corrected), "The model is not permutation equivariant!"

    

    def accelerated_eval(self, accelerator, model, dataloader, nsamples = 100, device = 'cpu', loggers = None, P_max = 0.01, noise_var = 0., inv_transform = None,
                         sampler = 'ddpm',
                         eta = 0.2,
                         eval_fn = None
                         ):
        """
        Eval method adapted for multi-gpu/process evaluation.
        """

        model = model.to(device)
        model = model.eval()

        all_variables = defaultdict(list)
        all_graph_idx = []
        for data, graph_idx in dataloader:
            data = data.to(device)
            graph_idx = graph_idx.to(device)

            Xgen, Xgen_hist = self.accelerated_sample(
                                               model=model,
                                            #    model=accelerator.unwrap_model(model),
                                               data=data,
                                               sampler=sampler,
                                               nsamples=nsamples,
                                               device=device,
                                               n_graphs_per_batch = len(graph_idx),
                                               eta = eta
                                               )
            
            if inv_transform is not None:
                pgen = P_max * inv_transform(Xgen)
                porig = P_max * inv_transform(data.x)
            else:
                pgen = P_max * Xgen
                porig = P_max * data.x
            pgen.data.clamp_(min = 0., max = P_max)

            pgen_rates = self.eval_rates(p=pgen, data=data, noise_var=noise_var)
            pgen_avg_rates = pgen_rates.mean(dim = 0)
            pgen_std_rates = pgen_rates.std(dim = 0)

            porig_rates = self.eval_rates(p = porig, data=data, noise_var=noise_var)
            porig_avg_rates = porig_rates.mean(dim = 0)
            porig_std_rates = porig_rates.std(dim = 0)

            graph_idx_gathered = accelerator.gather(graph_idx)
            accelerator.print('Accelerator process id: ', accelerator.process_index, '\tGraph idx: ', graph_idx, '\tGathered graph idx: ', graph_idx_gathered)
            all_graph_idx.extend(graph_idx_gathered.tolist())
            # all_graph_idx.extend(graph_idx.tolist())

            """ Swap axes from [nsamples, graph, ...] shape to [graph, nsamples ...], gather subbatch of graphs from different processes in the first dimension, and revert to axis swapping. """
            # x = torch.moveaxis(data.x, source=1, destination=0)
            # Xgen = torch.moveaxis(Xgen, source=1, destination=0)

            x_gathered = accelerator.gather(data.x.clone())
            Xgen_gathered = accelerator.gather(Xgen)

            print("data.x.shape / device: ", data.x.shape, data.x.device)
            print('x_gathered.shape / device: ', x_gathered.shape, x_gathered.device)

            print("Xgen.shape / device: ", Xgen.shape, Xgen.device)
            print('Xgen_gathered.shape / device: ', Xgen_gathered.shape, Xgen_gathered.device)


            pgen_gathered = accelerator.gather(pgen)
            porig_gathered = accelerator.gather(porig)
            print("pgen_gathered.shape / device: ", pgen_gathered.shape, pgen_gathered.device)

            pgen_rates_gathered = accelerator.gather(pgen_rates)
            porig_rates_gathered = accelerator.gather(porig_rates)
            print("pgen_rates_gathered.shape / device: ", pgen_rates_gathered.shape, pgen_rates_gathered.device)


            # x_gathered = torch.moveaxis(x_gathered, source=0, destination=1)
            # Xgen_gathered = torch.moveaxis(Xgen_gathered, source=0, destination=1)
            """  """

            """ Avg rates are gathered automatically across the first dimension. """
            porig_avg_rates_gathered = accelerator.gather(porig_avg_rates)
            pgen_avg_rates_gathered = accelerator.gather(pgen_avg_rates)

            porig_std_rates_gathered = accelerator.gather(porig_std_rates)
            pgen_std_rates_gathered = accelerator.gather(pgen_std_rates)

            accelerator.print("porig_avg_rates_gathered.shape: ", porig_avg_rates_gathered.shape)
            accelerator.print("pgen_avg_rates_gathered.shape: ", pgen_avg_rates_gathered.shape)

            n = data.num_nodes // data.num_graphs

            # x_t = torch.stack(Xgen_hist['x_t'], dim = 0).squeeze(-1)
            # score_t = torch.stack(Xgen_hist['score_t'], dim = 0).squeeze(-1)

            """ Swap axes from [time, nsamples x graph] shape to [graph x nsamples, time ...], gather subbatch of graphs from different processes in the first dimension, and revert to axis swapping. """
            x_t = torch.stack(Xgen_hist['x_t'], dim = 0).squeeze(-1)
            score_t = torch.stack(Xgen_hist['score_t'], dim = 0).squeeze(-1)

            x_t = torch.moveaxis(x_t, source=1, destination=0)
            score_t = torch.moveaxis(score_t, source=1, destination=0)

            x_t_gathered = accelerator.gather(x_t)
            score_t_gathered = accelerator.gather(score_t)

            x_t_gathered = torch.moveaxis(x_t_gathered, source=0, destination=1)
            score_t_gathered = torch.moveaxis(score_t_gathered, source=0, destination=1)


            accelerator.print('Accelerator process id: ', accelerator.process_index, '\tx_t_gathered.shape: ', x_t_gathered.shape)
            """          """
            
            # variables = {'diffusion-sampler': [(data.x.squeeze(-1)[:, idx*n:(idx+1)*n], Xgen.squeeze(-1).view(nsamples, -1)[:, idx*n:(idx+1)*n]) for idx in range(len(data))],
            #              'opt-problem': [(porig_avg_rates.squeeze(-1).view(len(data), -1)[_], pgen_avg_rates.squeeze(-1).view(len(data), -1)[_]) for _ in range(len(data))],
            #              'diffusion-sampler-hist': [(x_t.view(x_t.shape[0], nsamples, -1)[:, :, idx*n:(idx+1)*n], score_t.view(score_t.shape[0], nsamples, -1)[:, :, idx*n:(idx+1)*n]) for idx in range(len(data))]
            #              }

            process_tensor = torch.tensor([accelerator.process_index]).to(device=device)
            num_processes = len(accelerator.gather(process_tensor))
            eff_len_data = len(data) * num_processes

            print("porig_avg_rates_gathered.squeeze(-1).view(eff_len_data, -1).shape: ", porig_avg_rates_gathered.squeeze(-1).view(eff_len_data, -1).shape)
            print("pgen_avg_rates_gathered.squeeze(-1).view(eff_len_data, -1).shape: ", pgen_avg_rates_gathered.squeeze(-1).view(eff_len_data, -1).shape)

            print("x_gathered.device: ", x_gathered.device)
            print("pgen_avg_rates_gathered.device: ", pgen_avg_rates_gathered.device)
            print("porig_avg_rates_gathered.device: ", porig_avg_rates_gathered.device)

            accelerator.wait_for_everyone()

            variables = {'diffusion-sampler': [(x_gathered.squeeze(-1)[:, idx*n:(idx+1)*n].cpu(), Xgen_gathered.squeeze(-1).view(nsamples, -1)[:, idx*n:(idx+1)*n].cpu()) for idx in range(eff_len_data)],
                         'Ps': [(porig_gathered.squeeze(-1)[:, idx*n:(idx+1)*n].cpu(), pgen_gathered.squeeze(-1).view(nsamples, -1)[:, idx*n:(idx+1)*n].cpu()) for idx in range(eff_len_data)],
                         'rates': [(porig_rates_gathered.squeeze(-1)[:, idx*n:(idx+1)*n].cpu(), pgen_rates_gathered.squeeze(-1)[:, idx*n:(idx+1)*n].cpu()) for idx in range(eff_len_data)],
                         'opt-problem': [(porig_avg_rates_gathered.squeeze(-1).view(eff_len_data, -1)[idx].cpu(), pgen_avg_rates_gathered.squeeze(-1).view(eff_len_data, -1)[idx].cpu()) for idx in range(eff_len_data)],
                         'diffusion-sampler-hist': [(x_t_gathered.view(x_t_gathered.shape[0], nsamples, -1)[:, :, idx*n:(idx+1)*n].cpu(), score_t_gathered.view(score_t_gathered.shape[0], nsamples, -1)[:, :, idx*n:(idx+1)*n].cpu()) for idx in range(eff_len_data)]
                         }
            
            # Update the dictionary with standard deviation of rates
            variables.update({'opt-problem-std': [(porig_std_rates_gathered.squeeze(-1).view(eff_len_data, -1)[idx].cpu(), pgen_std_rates_gathered.squeeze(-1).view(eff_len_data, -1)[idx].cpu()) for idx in range(eff_len_data)]})

            # Update the dictionary with Wasserstein distance between the two distributions
            variables.update({'diffusion-sampler-wass': [(  wasserstein_per_dimension( porig_gathered.squeeze(-1)[:, idx*n:(idx+1)*n].cpu() / P_max, pgen_gathered.squeeze(-1).view(nsamples, -1)[:, idx*n:(idx+1)*n].cpu() / P_max ), ) for idx in range(eff_len_data)]})
            
            for key in variables:
                all_variables[key].extend(variables[key])

        # for key in all_variables:
        #     all_variables[key] = accelerator.gather(all_variables[key])
        # for key in all_variables:
        #     print(f'self.eval -> key = {key}, len(all_graph_idx) = {len(all_graph_idx)}, len(all_variables[key]) = {len(all_variables[key])}')
        #     print(f'All graph idx = {all_graph_idx}')
        #     all_variables[key] = [all_variables[key][idx] for idx in all_graph_idx]
        accelerator.wait_for_everyone()

        eval_metrics = {}
        if accelerator.is_main_process:
            for key in all_variables:
                print(f'self.eval -> key = {key}, len(all_graph_idx) = {len(all_graph_idx)}, len(all_variables[key]) = {len(all_variables[key])}')
                print(f'All graph idx = {all_graph_idx}')
                all_variables[key] = [all_variables[key][idx] for idx in all_graph_idx]

                new_list = [None] * len(all_graph_idx)
                # Rearrange all_variables[key] based on all_graph_idx list
                for i in range(len(all_graph_idx)):
                    new_list[all_graph_idx[i]] = all_variables[key][i]

                all_variables[key] = [_ for _ in new_list]

                try:
                    # if eval_fn is None:
                    if key in ['opt-problem']:
                        eval_metrics.update({f"eval-obj-global": np.array([all_variables[key][id][1].cpu() for id in range(len(all_variables[key]))]).mean().item(),
                                            f"expert-obj-global": np.array([all_variables[key][id][0].cpu() for id in range(len(all_variables[key]))]).mean().item(),
                                            f"eval-constraints-1-percentile-global": np.percentile(np.array([all_variables[key][id][1].cpu() for id in range(len(all_variables[key]))]), 1).item(),
                                            f"expert-constraints-1-percentile-global": np.percentile(np.array([all_variables[key][id][0].cpu() for id in range(len(all_variables[key]))]), 1).item(),
                                            f"eval-constraints-5-percentile-global": np.percentile(np.array([all_variables[key][id][1].cpu() for id in range(len(all_variables[key]))]), 5).item(),
                                            f"expert-constraints-5-percentile-global": np.percentile(np.array([all_variables[key][id][0].cpu() for id in range(len(all_variables[key]))]), 5).item()
                        })
                        
                        for graph_id in range(len(all_graph_idx)):
                            eval_metrics.update({f"eval-obj-graph-{graph_id}": all_variables[key][graph_id][1].cpu().mean().item(),
                                                f"expert-obj-graph-{graph_id}": all_variables[key][graph_id][0].cpu().mean().item(),
                                                f"expert-constraints-1-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][0].cpu().numpy(), 1).item(),
                                                f"eval-constraints-1-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][1].cpu().numpy(), 1).item(),
                                                f"expert-constraints-5-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][0].cpu().numpy(), 5).item(),
                                                f"eval-constraints-5-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][1].cpu().numpy(), 5).item()
                                                })
                            
                    elif key in ['opt-problem-std']:
                        for graph_id in range(len(all_graph_idx)):
                            eval_metrics.update({f"eval-rates-std-1-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][1].cpu().numpy(), 1).item(),
                                                f"expert-rates-std-1-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][0].cpu().numpy(), 1).item(),
                                                f"eval-rates-std-50-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][1].cpu().numpy(), 50).item(),
                                                f"expert-rates-std-50-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][0].cpu().numpy(), 50).item(),
                                                f"eval-rates-std-99-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][1].cpu().numpy(), 99).item(),
                                                f"expert-rates-std-99-percentile-graph-{graph_id}": np.percentile(all_variables[key][graph_id][0].cpu().numpy(), 99).item()
                            })

                    # else:
                        # pass
                                                
                            
                except:
                    accelerator.print("Computing diffusion-sampler metrics to a log dict failed in self.accelerated_eval().")

            
            if eval_fn is not None:
                eval_dict = eval_fn(all_variables)
                eval_metrics.update({"eval_fn": eval_dict})
            else:
                eval_metrics.update({"eval_fn": None})

            if loggers is not None:
                for logger in loggers:
                    if logger.log_metric in ['diffusion-sampler', 'diffusion-sampler-hist']:
                        logger.update_data(all_variables)
                        for i in range(0, min((MAX_LOGGED_CLIENTS // 2) * 2, n), 2):
                            logger(scatter_client_idx = [i, (i+1) % n])

                    if logger.log_metric in ['opt-problem']:
                        logger.update_data(all_variables)
                        logger()
        
        return eval_metrics
            

    def eval(self, accelerator, model, dataloader, nsamples = 100, device = 'cpu', loggers = None, P_max = 0.01, noise_var = 0., inv_transform = None):
        model = model.to(device)
        model = model.eval()

        all_variables = defaultdict(list)
        all_graph_idx = []
        for data, graph_idx in dataloader:
            data = data.to(device)

            Xgen, Xgen_hist = self.sample_ddpm(
                                               model=model,
                                            #    model=accelerator.unwrap_model(model),
                                               data=data,
                                               nsamples=nsamples,
                                               device=device,
                                               n_graphs_per_batch = len(graph_idx)
                                               )
            
            if inv_transform is not None:
                pgen = P_max * inv_transform(Xgen)
                porig = P_max * inv_transform(data.x)
            else:
                pgen = P_max * Xgen
                porig = P_max * data.x
            pgen.data.clamp_(min = 0., max = P_max)

            pgen_rates = self.eval_rates(p=pgen, data=data, noise_var=noise_var)
            pgen_avg_rates = pgen_rates.mean(dim = 0)

            porig_rates = self.eval_rates(p = porig, data=data, noise_var=noise_var)
            porig_avg_rates = porig_rates.mean(dim = 0)

            all_graph_idx.extend(graph_idx.tolist())

            n = data.num_nodes // data.num_graphs

            x_t = torch.stack(Xgen_hist['x_t'], dim = 0).squeeze(-1)
            # eps_t = torch.stack(Xgen_hist['eps_t'], dim = 0).squeeze(-1)
            score_t = torch.stack(Xgen_hist['score_t'], dim = 0).squeeze(-1)
            
            variables = {'diffusion-sampler': [(data.x.squeeze(-1)[:, idx*n:(idx+1)*n], Xgen.squeeze(-1).view(nsamples, -1)[:, idx*n:(idx+1)*n]) for idx in range(len(data))],
                         'opt-problem': [(porig_avg_rates.squeeze(-1).view(len(data), -1)[_], pgen_avg_rates.squeeze(-1).view(len(data), -1)[_]) for _ in range(len(data))],
                         'diffusion-sampler-hist': [(x_t.view(x_t.shape[0], nsamples, -1)[:, :, idx*n:(idx+1)*n], score_t.view(score_t.shape[0], nsamples, -1)[:, :, idx*n:(idx+1)*n]) for idx in range(len(data))]
                         }
            
            for key in variables:
                all_variables[key].extend(variables[key])

        for key in all_variables:
            all_variables[key] = accelerator.gather(all_variables[key])
        for key in all_variables:
            print(f'self.eval -> key = {key}, len(all_graph_idx) = {len(all_graph_idx)}, len(all_variables[key]) = {len(all_variables[key])}')
            print(f'All graph idx = {all_graph_idx}')
            all_variables[key] = [all_variables[key][idx] for idx in all_graph_idx]

        if accelerator.is_main_process:
            if loggers is not None:
                for logger in loggers:
                    if logger.log_metric in ['diffusion-sampler', 'diffusion-sampler-hist']:
                        logger.update_data(all_variables)
                        for i in range(0, n, 2):
                            logger(scatter_client_idx = [i, (i+1) % n])

                    if logger.log_metric in ['opt-problem']:
                        logger.update_data(all_variables)
                        logger()


def weighted_MSE(input, target, weights = torch.tensor([1.])):
    weights = (weights / weights.mean()).to(device = input.device) # normalize weights
    loss = nn.MSELoss(reduction='none')
    weighted_loss = loss(input, target) * weights
    return weighted_loss.mean()


class ConstrainedConditionalDiffusionLearner(ConditionalDiffusionLearner):
    def __init__(self, config, obj, constraints, device = 'cpu', noise_var = 0.):
        print('Initializing the constrained conditional diffusion-learner.')
        self.config = config
        self.device = device
        super(ConstrainedConditionalDiffusionLearner, self).__init__(device=device, config=config)

        self.tau = 10. # kl-regularization
        
        def constraint_fnc(p, data, batch_dim = 0, keep_batch_dim = True, average = True, indicator = False):
            all_rates = self.eval_rates(p=p, data=data, noise_var=noise_var)

            if indicator:
                all_slacks = F.softplus(constraints(all_rates)) # proxy for indicator
                if average:
                    slacks = all_slacks.mean(dim = batch_dim, keepdim = keep_batch_dim)
                else:
                    slacks = all_slacks

            else:
                if average:
                    rates = all_rates.mean(dim = batch_dim, keepdim = keep_batch_dim) # avg_rates
                else:
                    rates = all_rates

                slacks = constraints(rates)

            return slacks
        
        
        def obj_fnc(p, data, batch_dim = 0, keep_batch_dim = True, average = True):
            all_rates = self.eval_rates(p=p, data=data, noise_var=noise_var)
            if average:
                rates = all_rates.mean(dim = batch_dim, keepdim = keep_batch_dim) # avg_rates
            else:
                rates = all_rates

            obj_value = obj(rates, node_dim = batch_dim + 1)
            return obj_value
        
        self.constraints = constraint_fnc
        self.obj = obj_fnc

        def get_weights(x_0, lambdas, data, node_dim = -2):
            MIN_WEIGHT, MAX_WEIGHT = 1e-10, 1e10
            obj_values = self.obj(x_0, data, batch_dim=node_dim - 1, average=False).unsqueeze(-1)
            constraint_values = self.constraints(x_0, data, batch_dim = node_dim - 1, average=False, indicator=True)
            
            lagrangian_values = obj_values + (lambdas * constraint_values).sum(dim = node_dim, keepdim = True)
            weights = (-self.tau * (lagrangian_values - lagrangian_values.mean())).exp().data.clamp_(min = MIN_WEIGHT, max = MAX_WEIGHT)
            
            assert not torch.any(torch.isnan(weights)), 'nan values encountered.'
            assert not torch.any(weights == float('inf')), 'Inf values encountered.'

            # weights = torch.ones_like(weights)

            if weights.quantile(.95).item() > 20:
                pass
                # print('Max (95% percentile) weight: ', weights.quantile(.95).item())

            weights = weights / weights.sum()
            weights.data.clamp_(min = 0, max = 1)

            return weights


        def loss_fn(eps, eps_pred, weights): # assume lambdas = [G, n, 1]
            loss = weighted_MSE(input = eps_pred, target=eps, weights=weights)
            return loss

        self.get_weights = get_weights
        self.loss_fn = loss_fn # nn.MSELoss() # loss function for the training routine


    def get_optimal_lambdas(self, dataloader, lambdas_0 = 0., n_epochs = 100, lr_dual = 0.1, device = 'cpu', loggers = None):
        print('Compute optimal dual multipliers...')
        num_graphs = len(dataloader)
        n = dataloader.dataset[0][0].num_nodes

        if lambdas_0 is None:
            lambdas_0 = 1.

        if isinstance(lambdas_0, float):
            lambdas_0 = lambdas_0 * torch.ones((num_graphs, n), dtype = torch.float32, device = device)
        # elif isinstance(lambdas_0, np.array):
            # lambdas_0 = torch.from_numpy(lambdas_0).to(dtype = torch.float32, device=device).view(num_graphs, n)


        for epoch in tqdm(range(n_epochs)):
            for data, graph_idx in dataloader:
                pass

        return lambdas_0    
        


    def train(self, model, dataloader, lambdas, n_epochs = 100, lr = 1e-1, lr_dual = 0.1, P_max = 0.01, device = 'cpu', loggers = None):
        print('Training the conditional diffusion model with constraints.')
        q = 0.0 # probability of null token
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_epochs)

        lambdas = self.get_optimal_lambdas(dataloader=dataloader, lambdas_0=lambdas)

        for epoch in tqdm(range(n_epochs)):
            epoch_loss = steps = 0

            for data, graph_idx in dataloader:
                model.zero_grad()
                data = data.to(device)

                n = data.num_nodes // data.num_graphs

                lambdas_batch = lambdas[graph_idx]

                X_batch = data.x # [T, (B * n), 1]
                X_batch = dataloader.dataset[0][0].transform_x(torch.rand((self.config.x_batch_size, *X_batch.shape[1:])), invert = False).to(device)

                ### Add weighted sampler here ###
                x_0_detransformed = P_max * dataloader.dataset[0][0].transform_x(X_batch.view(-1, n, X_batch.shape[-1]), invert = True).data.clamp_(min = 0, max = 1.)
                # loss = nn.MSELoss()(predicted_noise.view(-1, n).to(device), eps.view(-1, n).to(device))
                weights = self.get_weights(x_0=x_0_detransformed.to(device),
                                           lambdas=lambdas_batch.unsqueeze(-1).to(device),
                                           data=data, # data_batch
                                           node_dim=-2
                                           ) # [T, B, 1]
                
                del x_0_detransformed
                
                weights = weights.squeeze().cpu()
                weighted_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
                X_batch_dataset = TensorDataset(X_batch.cpu())
                X_batch_dataloader = DataLoader(dataset=X_batch_dataset, batch_size=self.config.x_batch_size, sampler=weighted_sampler)
                X_batch = next(iter(X_batch_dataloader))[0].to(device)
                ### Add weighted sampler here ###

                del X_batch_dataset, X_batch_dataloader

                # X_batch = dataloader.dataset[0][0].transform_x(torch.rand_like(X_batch), invert = False) # a dirty patch to resample from uniform distribution
                # x_idx = torch.randint(low=0, high=X_batch.shape[0], size = (self.config.x_batch_size,)).to(X_batch.device)
                # X_batch = X_batch[x_idx]

                X_batch = (X_batch - X_batch.mean()) / X_batch.std()

                dithering_noise_sigma = 0.01
                X_batch = X_batch + dithering_noise_sigma * torch.randn_like(X_batch)

                data_batch = Batch.from_data_list([Data(edge_index=data.edge_index_l, edge_weight = data.edge_weight_l, transmitters_index = data.transmitters_index, num_nodes = data.num_nodes)] * len(X_batch))
                X_batch = X_batch.view(-1, X_batch.shape[-1])

                timesteps = torch.randint(0, self.diffusion_steps, size=[len(X_batch), 1]).to(X_batch.device)
                noised, eps = self.noise(X_batch, timesteps)

                if np.random.random() < q:
                    data_batch = None

                predicted_noise = model(noised.to(device), timesteps.to(device), data_batch)

                # x_0_detransformed = P_max * dataloader.dataset[0][0].transform_x(X_batch.view(-1, n, X_batch.shape[-1]), invert = True).data.clamp_(min = 0, max = 1.)
                loss = nn.MSELoss()(predicted_noise.view(-1, n).to(device), eps.view(-1, n).to(device))
                # weights = self.get_weights(x_0=x_0_detransformed.to(device),
                #                            lambdas=lambdas_batch.unsqueeze(-1).to(device),
                #                            data=data, # data_batch
                #                            node_dim=-2
                #                            )
                
                # loss = self.loss_fn(eps=eps.view(-1, n), eps_pred=predicted_noise.view(-1, n), weights=weights)
                # loss = self.loss_fn(predicted_noise.view(-1, data.num_nodes), eps.view(-1, data.num_nodes).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                steps += 1
                # with torch.no_grad():
                #     lambdas_batch += lr_dual * 

            epoch_variables = {'diffusion-loss': epoch_loss / steps}
            if loggers is not None:
                for logger in loggers:
                    if logger.log_metric in ['diffusion-loss']:
                        logger.update_data(epoch_variables)
                        logger()
                    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch} loss = {epoch_loss / steps}")
            scheduler.step()

        return model, epoch_loss

            

            

        