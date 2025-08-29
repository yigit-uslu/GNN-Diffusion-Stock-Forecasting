from collections import defaultdict
import abc
import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm, PowerNorm, TwoSlopeNorm
import seaborn as sns
import os
import re
from PIL import Image
import torch
import pickle
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, from_scipy_sparse_matrix, to_dense_adj
from torch_geometric.data import Data
from utils.graph_utils import visualize_graph, rescale_edge_weights
from utils.plot_helper_utils import hdr_plot_style
from scipy import sparse
from core.config import MAX_LOGGED_NETWORKS

MAX_CURVES_PER_PLOT = 8
NUM_NODES = 20



from utils.plot_utils import create_repeating_colormap
from utils.style_utils import *


# def save_logger_object(obj, filename):
#     with open(filename, 'wb') as outp:  # Overwrites any existing file.
#         pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


class Logger(abc.ABC):
    def __init__(self, data, log_metric, log_path):
        self.data = data
        self.log_metric = log_metric
        self.log_path = log_path

    # @abc.abstractmethod
    def update_data(self, new_data):
        self.data.append(new_data)

    @abc.abstractmethod
    def __call__(self):
        pass

    def save_logs_as_gifs(self, **kwargs):
        pass

    def raise_log_error(self):
        print(f"{self.__repr__} failed to save log.")

    # @abc.abstractmethod
    def get_save_data(self):
        return {'data': self.data,
                'log_metric': self.log_metric,
                'log_path': self.log_path
                }


# class SALagrangianLogger(Logger):
#     def __init__(self, data, log_path):
#         super(SALagrangianLogger, self).__init__(data=data, log_metric='Lagrangian', log_path=log_path)

#     def update_data(self, new_data):
#         data = new_data[self.log_metric].mean().item()
#         super().update_data(data)

#     def __call__(self):
#         L = np.array(self.data)
#         iters = np.arange(len(L))

#         if self.log_path is not None:

#             fig, ax = plt.subplots(1, 1, figsize = (8, 4))
#             ax.plot(iters, L, '-r')
#             ax.set_xlabel('Iteration')
#             ax.set_ylabel(self.log_metric)
#             ax.grid(True)
#             fig.tight_layout()

#             os.makedirs(f"{self.log_path}", exist_ok=True)
#             plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)
#             plt.close(fig)

class SALagrangianLogger(Logger):
    def __init__(self, data, log_path):
        super(SALagrangianLogger, self).__init__(data=data, log_metric='Lagrangian', log_path=log_path)

    # def update_data(self, new_data, avg_data = True):
    #     # debug_print(f"log_metric: " + self.log_metric)
    #     # debug_print(f"data[log_metric]: {new_data[self.log_metric]}")
    #     data = new_data[self.log_metric]
    #     if isinstance(data, list):
    #         data = np.array(data)

    #     if not isinstance(data, float) and avg_data:
    #         data = data.mean().item()

    #     super().update_data(data)
        
    def update_data(self, new_data):
        # debug_print(f"log_metric: " + self.log_metric)
        # debug_print(f"data[log_metric]: {new_data[self.log_metric]}")
        data = new_data[self.log_metric]
        if isinstance(data, list):
            data = np.array(data)

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # if not isinstance(data, float) and avg_data:
        #     data = data.mean().item()
        super().update_data(data)


    def __call__(self, avg_data = True, dim_labels = ['epochs', 'lambdas']):

        try:
            # print('self.data: ', self.data)
            L = np.array(self.data)

            if avg_data and len(L.shape) >= 2:
                L = np.mean(L, axis = -1)

            if len(L.shape) == 1: # avg Lagrangian or pgrad norm
                iters = np.arange(len(L)) + 1

                if self.log_path is not None:

                    fig, ax = plt.subplots(1, 1)
                    ax.plot(iters, L, '-r')
                    ax.set_xlabel('Epoch (k)')
                    ax.set_ylabel(self.log_metric)
                    ax.grid(True)
                    # fig.tight_layout()

                    save_dir = f'{self.log_path}'
                    os.makedirs(save_dir, exist_ok=True)

                    plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)
                    plt.close(fig)

            else: # individual Lagrangians
                n_iters, n_samples = L.shape
                # debug_print(f"L.shape: {L.shape}")

                max_curves_per_plot = 16
                legend_row_length = 8
                n_cols_legend = (max_curves_per_plot + legend_row_length-1) // legend_row_length

                for i in range(0, n_samples, max_curves_per_plot):
                    start = i 
                    end = min(i + max_curves_per_plot, n_samples)

                    fig, ax = plt.subplots(1, 1)

                    if dim_labels[1] == 'lambdas':
                        if end-start == 1: # single curve
                            ax.plot(np.arange(n_iters) + 1, L[:, start:end], linestyle = ':', label = r'$\mathcal{{L}}(\mathbf{{H}}_0, \mathbf{{\lambda}}_{' + str(start) + '})$')
                        else:
                            ax.plot(np.arange(n_iters) + 1, L[:, start:end], linestyle = ':', label = [r'$\mathcal{{L}}(\mathbf{{H}}_0, \mathbf{{\lambda}}_{' + str(s) + '})$' for s in list(range(start, end))])
                    
                    elif dim_labels[1] == 'graphs':
                        if end-start == 1: # single curve
                            ax.plot(np.arange(n_iters) + 1, L[:, start:end], linestyle = ':', label = r'$\mathcal{{L}}(\mathbf{{H}}_{' + str(start) + '})$')
                        else:
                            ax.plot(np.arange(n_iters) + 1, L[:, start:end], linestyle = ':', label = [r'$\mathcal{{L}}(\mathbf{{H}}_{' + str(s) + '})$' for s in list(range(start, end))])
                    
                    ax.set_xlabel('Epoch (k)')
                    ax.set_ylabel(self.log_metric)

                    if len(L) > 100: # Adjust y-axis limits for large number of iterations.
                        ymax = max([0.9 * np.max(L[-100:]), 1.1 * np.max(L[-100:])]) # Account for the sign of Lagrangian
                        ymin = min([0.9 * np.min(L[-100:]), 1.1 * np.min(L[-100:])])
                        ax.set_ylim([ymin, ymax])


                    ax.grid(True)
                    ax.legend(loc = 'best', ncol = n_cols_legend)
                    # fig.tight_layout()

                    save_dir = f'{self.log_path}/{self.log_metric}'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f'{save_dir}/Lagrangian_vs_{dim_labels[1]}_{start}-{end}.pdf', dpi = 300)
                    plt.close(fig)

        except:
            self.raise_log_error()



class SAModelLogger(Logger):
    def __init__(self, data, log_path):
        super(SAModelLogger, self).__init__(data = data, log_metric='model_weights', log_path = log_path)

    def update_data(self, new_data):
        pass

    def __call__(self, model_state_dict, epoch = None):

        try:
            os.makedirs(f"{self.log_path}", exist_ok=True)
            
            # save trained model weights
            if epoch is not None:
                torch.save(model_state_dict, f'{self.log_path}/sa_model_state_dict_epoch_{epoch}.pt')
            else:
                torch.save(model_state_dict, f'{self.log_path}/sa_model_state_dict.pt')

        except:
            self.raise_log_error()


class PgradLogger(SALagrangianLogger):
    def __init__(self, data, log_path):
        Logger.__init__(self, data=data, log_metric='pgrad_norm', log_path = log_path)
        # super(PgradLogger, self).__init__(data=data, log_metric = 'pgrad_norm', log_path = log_path)


class SASamplerLogger(Logger):
    def __init__(self, data, log_path, network_id = 0, client_idx = None):
        super(SASamplerLogger, self).__init__(data = data, log_metric = 'Ps', log_path=log_path)
        self.network_id = network_id
        self.client_idx = client_idx

    def update_data(self, new_data, network_dim, client_dim):

        data = new_data[self.log_metric]


        if self.network_id is None:
            all_data = []
            for network_id in range(data.shape[network_dim]):
                all_data.append(torch.index_select(data.detach().cpu(), dim = network_dim, index=torch.tensor(network_id)))
            data = torch.cat(all_data, dim = client_dim) # append these nodes
        
        else:
            if isinstance(data, torch.Tensor):
                data = torch.index_select(data.detach().cpu(), dim = network_dim, index=torch.tensor(self.network_id))
            else:
                data = data[self.network_id].detach().cpu().unsqueeze(0)

        # if isinstance(data, torch.Tensor):
        #     data = torch.index_select(data.detach().cpu(), dim = network_dim, index=torch.tensor(self.network_id))
        # else:
        #     data = data[self.network_id].detach().cpu().unsqueeze(0)

        if self.client_idx is not None:
            data = torch.index_select(data, dim = client_dim, index=torch.tensor(self.client_idx))
        data.squeeze_(dim = network_dim).numpy()
        self.data = data
        # self.data = new_data[self.log_metric][self.network_id]


    def __call__(self, scatter_client_idx = [0, 1]):

        try:
            if self.log_path is not None:

                if scatter_client_idx is None:
                    L = np.array(self.data)
                    
                    n_iters, n_networks = L.shape
                    max_curves_per_plot = MAX_CURVES_PER_PLOT
                    n_cols_legend = (max_curves_per_plot + 3) // 4

                    for i in range(0, n_networks, max_curves_per_plot):
                        start = i 
                        end = min(i + max_curves_per_plot, n_networks)

                        fig, ax = plt.subplots(1, 1)
                        ax.plot(np.arange(n_iters), L[:, start:end], linestyle = ':', marker = 'd', label = [f'{s}' for s in list(range(start, end))])
                        ax.set_xlabel('Timestep (t)')
                        ax.set_ylabel(self.log_metric)
                        ax.grid(True)
                        ax.legend(loc = 'best', ncol = n_cols_legend)
                        # fig.tight_layout()

                        save_dir = f'{self.log_path}/{self.log_metric}'
                        os.makedirs(save_dir, exist_ok=True)
                        plt.savefig(f'{save_dir}/network_{self.network_id}_clients_{start}-{end}.pdf', dpi = 300)
                        plt.close(fig)


                else:
                    X_gen = [self.data.cpu()]
                    labels = ['Power alloc. expert policy']
                    title = ['Scatter Plot of Expert Policy Samples']

                    # if self.transform is not None:
                    #     X_orig = (X_orig, self.transform(X_orig))
                    #     X_gen = (X_gen, self.transform(X_gen))

                    #     title = (title[0], title[0] + ' (destandardized data)')

                    n_cols = len(X_gen)
                    fig, axs = plt.subplots(1, 1, figsize = (6 * n_cols, 6), squeeze = False)
                    
                    for i, ax in enumerate(axs.flat):
                        # Generate an array of indices to color the points
                        cindices = np.arange(len(X_gen[i]))
                        norm = plt.Normalize(cindices.min(), cindices.max())
                        # Choose a colormap
                        cmap = plt.get_cmap('coolwarm')

                        ax.scatter(x = X_gen[i][:, scatter_client_idx[0]], y = X_gen[i][:, scatter_client_idx[1]],
                                c=cindices, cmap=cmap, norm=norm,
                                s = 5.0, marker = 'o', label = labels[0]
                                )
                        
                        ax.grid(True)
                        ax.set_xlabel(f'Client {scatter_client_idx[0]}')
                        ax.set_ylabel(f'Client {scatter_client_idx[1]}')
                        ax.set_title(title[i])
                        ax.set_xlim([-0.05, 1])
                        ax.set_ylim([-0.05, 1])

                        # Create a mapable object
                        mappable = cm.ScalarMappable(norm = norm, cmap = cmap)
                        mappable.set_array(cindices)

                        # Add a colorbar
                        plt.colorbar(mappable=mappable, label = 'Timestep (t)', ax = ax)
                        
                        ax.legend(loc = 'best')

                    # fig.tight_layout()

                    save_dir = f'{self.log_path}/{self.log_metric}'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f'{save_dir}/network_{self.network_id}_client_idx_{scatter_client_idx}.pdf', dpi = 300)
                    plt.close(fig=fig)

        except:
            self.raise_log_error()


# class SASamplerLogger(Logger):
#     def __init__(self, data, log_path, network_id = 0, client_idx = None):
#         super(SASamplerLogger, self).__init__(data = data, log_metric = 'Ps', log_path=log_path)
#         self.network_id = network_id
#         self.client_idx = client_idx

#     def update_data(self, new_data, network_dim, client_dim):
#         data = torch.index_select(new_data[self.log_metric].detach().cpu(), dim = network_dim, index=torch.tensor(self.network_id))
#         if self.client_idx is not None:
#             data = torch.index_select(data, dim = client_dim, index=torch.tensor(self.client_idx))
#         data.squeeze_(dim = network_dim).numpy()
#         self.data = data
#         # self.data = new_data[self.log_metric][self.network_id]

#     def __call__(self, scatter_client_idx = [0, 1]):

#         if self.log_path is not None:

#             if scatter_client_idx is None:
#                 L = np.array(self.data)
#                 iters = np.arange(len(L))

#                 fig, ax = plt.subplots(1, 1, figsize = (8, 4))
#                 ax.plot(iters, L, linestyle = '-', marker = 'd')
#                 ax.set_xlabel('Timestep (t)')
#                 ax.set_ylabel(self.log_metric)
#                 ax.grid(True)
#                 fig.tight_layout()

#                 save_dir = f'{self.log_path}/{self.log_metric}'
#                 os.makedirs(save_dir, exist_ok=True)
#                 plt.savefig(f'{save_dir}/network_{self.network_id}.pdf', dpi = 300)
#                 plt.close()


#             else:
#                 X_gen = [self.data.cpu()]
#                 labels = ['Power alloc. expert policy']
#                 title = ['Scatter Plot of Expert Policy Samples']

#                 # if self.transform is not None:
#                 #     X_orig = (X_orig, self.transform(X_orig))
#                 #     X_gen = (X_gen, self.transform(X_gen))

#                 #     title = (title[0], title[0] + ' (destandardized data)')

#                 n_cols = len(X_gen)
#                 fig, axs = plt.subplots(1, 1, figsize = (6 * n_cols, 6), squeeze = False)
                
#                 for i, ax in enumerate(axs.flat):
#                     # Generate an array of indices to color the points
#                     cindices = np.arange(len(X_gen[i]))
#                     norm = plt.Normalize(cindices.min(), cindices.max())
#                     # Choose a colormap
#                     cmap = plt.get_cmap('coolwarm')

#                     ax.scatter(x = X_gen[i][:, scatter_client_idx[0]], y = X_gen[i][:, scatter_client_idx[1]],
#                             c=cindices, cmap=cmap, norm=norm,
#                             s = 5.0, marker = 'o', label = labels[0]
#                             )
                    
#                     ax.grid(True)
#                     ax.set_xlabel(f'Client {scatter_client_idx[0]}')
#                     ax.set_ylabel(f'Client {scatter_client_idx[1]}')
#                     ax.set_title(title[i])
#                     ax.set_xlim([-0.05, 1])
#                     ax.set_ylim([-0.05, 1])

#                     # Create a mapable object
#                     mappable = cm.ScalarMappable(norm = norm, cmap = cmap)
#                     mappable.set_array(cindices)

#                     # Add a colorbar
#                     plt.colorbar(mappable=mappable, label = 'Timestep (t)', ax = ax)
                    
#                     ax.legend(loc = 'best')

#                 fig.tight_layout()

#                 save_dir = f'{self.log_path}/{self.log_metric}'
#                 os.makedirs(save_dir, exist_ok=True)
#                 plt.savefig(f'{save_dir}/network_{self.network_id}_client_idx_{scatter_client_idx}.pdf', dpi = 300)
#                 plt.close()

class SALambdasLogger(Logger):
    def __init__(self, data, log_path, network_id = 0, client_idx = None):
        super(SALambdasLogger, self).__init__(data=data, log_metric='lambdas', log_path=log_path)
        self.network_id = network_id
        self.client_idx = client_idx

    def update_data(self, new_data, network_dim = 0, client_dim = 1):
        data = new_data[self.log_metric]
        if isinstance(data, torch.Tensor):
            data = torch.index_select(data.detach().cpu(), dim = network_dim, index=torch.tensor(self.network_id))
        else:
            data = data[self.network_id].detach().cpu().unsqueeze(network_dim)
        
        if self.client_idx is not None:
            data = torch.index_select(data, dim = client_dim, index=torch.tensor(self.client_idx))
        data.squeeze_(dim = network_dim).numpy()
        self.data = data
        # super().update_data(data)

    def __call__(self, scatter_client_idx = [0, 1], epoch = None, save_pdf = True):

        try:

            if self.log_path is not None:

                if scatter_client_idx is None:
                    L = np.array(self.data)
                    n_iters, n_networks = L.shape

                    max_curves_per_plot = MAX_CURVES_PER_PLOT
                    n_cols_legend = (max_curves_per_plot + 3) // 4

                    for i in range(0, n_networks, max_curves_per_plot):
                        start = i 
                        end = min(i + max_curves_per_plot, n_networks)


                        fig, ax = plt.subplots(1, 1)
                        ax.plot(np.arange(n_iters), L[:, start:end], linestyle = ':', marker = 'd', label = [f'{s}' for s in list(range(start, end))])
                        ax.set_xlabel('Iteration (k)')
                        ax.set_ylabel(self.log_metric)
                        ax.grid(True)
                        ax.legend(loc = 'best', ncol = n_cols_legend)
                        # fig.tight_layout()

                        save_dir = f'{self.log_path}/{self.log_metric}'
                        os.makedirs(save_dir, exist_ok=True)

                        if save_pdf:
                            fformat = 'pdf'
                        else:
                            fformat = 'png'

                        if epoch is None:
                            plt.savefig(f'{save_dir}/network_{self.network_id}_clients_{start}-{end}.{fformat}', dpi = 300)
                        else:
                            plt.savefig(f'{save_dir}/network_{self.network_id}_epoch_{epoch}_clients_{start}-{end}.{fformat}', dpi = 300)

                        plt.close(fig)

                
                else:
                    X_gen = [self.data.cpu()]
                    labels = ['Dual multipliers']
                    title = ['Scatter Plot of Expert Policy Samples']

                    n_cols = len(X_gen)
                    fig, axs = plt.subplots(1, 1, figsize = (6 * n_cols, 6), squeeze = False)
                    
                    for i, ax in enumerate(axs.flat):
                        # Generate an array of indices to color the points
                        cindices = np.arange(len(X_gen[i]))
                        norm = plt.Normalize(cindices.min(), cindices.max())
                        # Choose a colormap
                        cmap = plt.get_cmap('coolwarm')

                        # Scatter plot
                        ax.scatter(x = X_gen[i][:, scatter_client_idx[0]], y = X_gen[i][:, scatter_client_idx[1]],
                                c=cindices, cmap=cmap, norm=norm,
                                s = 5.0, marker = 'o', label = labels[0]
                                )
                        
                        ax.grid(True)
                        ax.set_xlabel(f'Client {scatter_client_idx[0]}')
                        ax.set_ylabel(f'Client {scatter_client_idx[1]}')
                        ax.set_title(title[i])
                        ax.set_xlim([-0.05, 1])
                        ax.set_ylim([-0.05, 1])

                        # Create a mapable object
                        mappable = cm.ScalarMappable(norm = norm, cmap = cmap)
                        mappable.set_array(cindices)

                        # Add a colorbar
                        plt.colorbar(mappable=mappable, label = 'Iteration (k)', ax = ax)
                        ax.legend(loc = 'best')

                    save_dir = f'{self.log_path}/{self.log_metric}'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f'{save_dir}/network_{self.network_id}_client_idx_{scatter_client_idx}.pdf', dpi = 300)
                    plt.close(fig=fig)

        except:
            self.raise_log_error()


    def save_logs_as_gif(self, delete_images = True):

        network_id = self.network_id
        folder_path = f"{self.log_path}/{self.log_metric}"


        if not os.path.isdir(folder_path):
            return

        # Gather all numbered .png files logged for the specified network
        jpg_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".png") and f"network_{network_id}" in f],
            key=lambda x: SALambdasLogger.assign_file_number(x)
            )

        # for jpg_file in jpg_files:
        #     print(jpg_file)
            # print(jpg_file, "\t", get_epoch_number(jpg_file, pad_to_n_digits=2), "\t", get_client_number(jpg_file, pad_to_n_digits=2), "\n")

        # Ensure there are images to process
        if not jpg_files:
            # print("No PNG files found in the specified folder.")
            return

        # Open the images and store them in a list
        images = [Image.open(os.path.join(folder_path, file)) for file in jpg_files]
        n_epochs = len(np.unique([int(SALambdasLogger.get_epoch_number(file)) for file in jpg_files]))
        durations_list = [200 if not (i + 1) % n_epochs == 0 else 1000 for i in range(len(jpg_files))]

        # Create the GIF
        output_gif_path = os.path.join(folder_path, f"network_{network_id}.gif")
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=durations_list,  # Duration for each frame in milliseconds
            loop=0  # Infinite loop
        )
        print(f"GIF saved as {output_gif_path}")

        if delete_images:
            # Delete the original JPG files
            for file in jpg_files:
                os.remove(os.path.join(folder_path, file))
            print("Original PNG files deleted.")


    @staticmethod
    def get_epoch_number(string, pad_to_n_digits = 1):

        # Pattern to extract text between square brackets
        pattern = r"epoch_(.*?)_clients"

        match = re.search(pattern, string)
        if match:
            epoch_number = match.group(1)  # Group 1 contains the matched text
            if len(epoch_number) < pad_to_n_digits:
                epoch_number = "0" * (pad_to_n_digits - len(epoch_number)) + epoch_number
                # print(epoch_number)
            # print(epoch_number) 

        return epoch_number

    @staticmethod
    def get_client_number(string, pad_to_n_digits = 2):

        # Pattern to extract text between square brackets
        pattern = r"clients_(.*?)-"

        match = re.search(pattern, string)
        if match:
            clients_start_idx = match.group(1)  # Group 1 contains the matched text
            # print(epoch_number) 

            if len(clients_start_idx) < pad_to_n_digits:
                clients_start_idx = "0" * (pad_to_n_digits - len(clients_start_idx)) + clients_start_idx


        pattern = r"-(.*?).png"
        match = re.search(pattern, string)
        if match:
            clients_end_idx = match.group(1)  # Group 1 contains the matched text
            # print(epoch_number) 

            if len(clients_end_idx) < pad_to_n_digits:
                clients_end_idx = "0" * (pad_to_n_digits - len(clients_end_idx)) + clients_end_idx
        
        clients_idx = clients_start_idx + clients_end_idx
        return clients_idx

    @staticmethod
    def assign_file_number(string, pad_to_n_digits = 3):
        return int(SALambdasLogger.get_client_number(string, pad_to_n_digits) + SALambdasLogger.get_epoch_number(string, pad_to_n_digits))



class SALagrangianOverTimeLogger(Logger):
    def __init__(self, data, log_path, network_id = 0):
        super(SALagrangianOverTimeLogger, self).__init__(data=data, log_metric='lagrangian-over-iterations', log_path=log_path)
        self.network_id = network_id
        # self.client_idx = client_idx

    def update_data(self, new_data, network_dim = 0):
        data = new_data[self.log_metric]
        if isinstance(data, torch.Tensor):
            data = torch.index_select(data.detach().cpu(), dim = network_dim, index=torch.tensor(self.network_id))
        else:
            data = data[self.network_id].detach().cpu().unsqueeze(network_dim)
        
        # if self.client_idx is not None:
        #     data = torch.index_select(data, dim = client_dim, index=torch.tensor(self.client_idx))
        data.squeeze_(dim = network_dim).numpy()
        self.data = data
        # super().update_data(data)

    def __call__(self, epoch = None, save_pdf = True):

        try:

            print("self.data.shape: ", self.data.shape)
            L = np.array(self.data)
            iters = np.arange(len(L)) + 1

            if self.log_path is not None:

                fig, ax = plt.subplots(1, 1)
                ax.plot(iters, L, '-db')
                ax.set_xlabel(r'Iteration ($k$), Timestep ($kT_0$)')
                ax.set_ylabel(self.log_metric)
                ax.grid(True)
                # fig.tight_layout()

                save_dir = f'{self.log_path}/{self.log_metric}'
                os.makedirs(save_dir, exist_ok=True)

                if save_pdf:
                    fformat = 'pdf'
                else:
                    fformat = 'png'

                if epoch is None:
                    plt.savefig(f'{save_dir}/network_{self.network_id}.{fformat}', dpi = 300)
                else:
                    plt.savefig(f'{save_dir}/network_{self.network_id}_epoch-{epoch}.{fformat}', dpi = 300)

        except:
            self.raise_log_error()


    def save_logs_as_gif(self, delete_images = True):

        network_id = self.network_id
        folder_path = f"{self.log_path}/{self.log_metric}"

        if not os.path.isdir(folder_path):
            return

        # Gather all numbered .png files logged for the specified network
        jpg_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".png") and f"network_{network_id}" in f],
            key=lambda x: SALagrangianOverTimeLogger.assign_file_number(x)
            )

        # for jpg_file in jpg_files:
        #     print(jpg_file)
            # print(jpg_file, "\t", get_epoch_number(jpg_file, pad_to_n_digits=2), "\t", get_client_number(jpg_file, pad_to_n_digits=2), "\n")

        # Ensure there are images to process
        if not jpg_files:
            print("No PNG files found in the specified folder.")
            return

        # Open the images and store them in a list
        images = [Image.open(os.path.join(folder_path, file)) for file in jpg_files]
        n_epochs = len(np.unique([int(SALagrangianOverTimeLogger.get_epoch_number(file)) for file in jpg_files]))
        # durations_list = [200 if not (i + 1) % n_epochs == 0 else 1000 for i in range(len(jpg_files))]

        # Create the GIF
        output_gif_path = os.path.join(folder_path, f"network_{network_id}.gif")
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=200, # durations_list,  # Duration for each frame in milliseconds
            loop=0  # Infinite loop
        )
        print(f"GIF saved as {output_gif_path}")

        if delete_images:
            # Delete the original JPG files
            for file in jpg_files:
                os.remove(os.path.join(folder_path, file))
            print("Original PNG files deleted.")


    @staticmethod
    def get_epoch_number(string, pad_to_n_digits = 1):

        # Pattern to extract text between square brackets
        pattern = r"epoch-(.*?).png"

        match = re.search(pattern, string)
        if match:
            epoch_number = match.group(1)  # Group 1 contains the matched text
            if len(epoch_number) < pad_to_n_digits:
                epoch_number = "0" * (pad_to_n_digits - len(epoch_number)) + epoch_number
                # print(epoch_number)
            # print(epoch_number) 

        return epoch_number

    
    @staticmethod
    def assign_file_number(string, pad_to_n_digits = 3):
        return int(SALagrangianOverTimeLogger.get_epoch_number(string, pad_to_n_digits))


# class SALambdasLogger(Logger):
#     def __init__(self, data, log_path, network_id = 0, client_idx = None):
#         super(SALambdasLogger, self).__init__(data=data, log_metric='lambdas', log_path=log_path)
#         self.network_id = network_id
#         self.client_idx = client_idx

#     def update_data(self, new_data, network_dim = 0, client_dim = 1):
#         data = torch.index_select(new_data[self.log_metric].detach().cpu(), dim = network_dim, index=torch.tensor(self.network_id))
#         if self.client_idx is not None:
#             data = torch.index_select(data, dim = client_dim, index=torch.tensor(self.client_idx))
#         data.squeeze_(dim = network_dim).numpy()
#         self.data = data
#         # super().update_data(data)

#     def __call__(self, scatter_client_idx = [0, 1], epoch = None, save_pdf = True):


#         if self.log_path is not None:

#             if scatter_client_idx is None:
#                 L = np.array(self.data)
#                 iters = np.arange(len(L))

#                 fig, ax = plt.subplots(1, 1, figsize = (8, 4))
#                 ax.plot(iters, L, linestyle = '-', marker = 'd')
#                 ax.set_xlabel('Iteration (k)')
#                 ax.set_ylabel(self.log_metric)
#                 ax.grid(True)
#                 fig.tight_layout()

#                 save_dir = f'{self.log_path}/{self.log_metric}'
#                 os.makedirs(save_dir, exist_ok=True)

#                 if save_pdf:
#                     fformat = 'pdf'
#                 else:
#                     fformat = 'png'

#                 if epoch is None:
#                     plt.savefig(f'{save_dir}/network_{self.network_id}.{fformat}', dpi = 300)
#                 else:
#                     plt.savefig(f'{save_dir}/network_{self.network_id}-{epoch}.{fformat}', dpi = 300)

#                 plt.close()

            
#             else:
#                 X_gen = [self.data.cpu()]
#                 labels = ['Dual multipliers']
#                 title = ['Scatter Plot of Expert Policy Samples']

#                 n_cols = len(X_gen)
#                 fig, axs = plt.subplots(1, 1, figsize = (6 * n_cols, 6), squeeze = False)
                
#                 for i, ax in enumerate(axs.flat):
#                     # Generate an array of indices to color the points
#                     cindices = np.arange(len(X_gen[i]))
#                     norm = plt.Normalize(cindices.min(), cindices.max())
#                     # Choose a colormap
#                     cmap = plt.get_cmap('coolwarm')

#                     # Scatter plot
#                     ax.scatter(x = X_gen[i][:, scatter_client_idx[0]], y = X_gen[i][:, scatter_client_idx[1]],
#                                c=cindices, cmap=cmap, norm=norm,
#                                s = 5.0, marker = 'o', label = labels[0]
#                                )
                    
#                     ax.grid(True)
#                     ax.set_xlabel(f'Client {scatter_client_idx[0]}')
#                     ax.set_ylabel(f'Client {scatter_client_idx[1]}')
#                     ax.set_title(title[i])
#                     ax.set_xlim([-0.05, 1])
#                     ax.set_ylim([-0.05, 1])

#                     # Create a mapable object
#                     mappable = cm.ScalarMappable(norm = norm, cmap = cmap)
#                     mappable.set_array(cindices)

#                     # Add a colorbar
#                     plt.colorbar(mappable=mappable, label = 'Iteration (k)', ax = ax)

#                     ax.legend(loc = 'best')

#                 fig.tight_layout()

#                 save_dir = f'{self.log_path}/{self.log_metric}'
#                 os.makedirs(save_dir, exist_ok=True)
#                 plt.savefig(f'{save_dir}/network_{self.network_id}_client_idx_{scatter_client_idx}.pdf', dpi = 300)
#                 plt.close()


class SARatesLogger(SASamplerLogger):
    def __init__(self, data, log_path, network_id = 0, client_idx = None, r_min = None):
        # super(SARatesLogger, self).__init__(data = data, log_metric = 'rates', log_path=log_path)
        # self.network_id = network_id
        # self.client_idx = client_idx
        super(SARatesLogger, self).__init__(data = data, log_path=log_path, network_id=network_id, client_idx=client_idx)
        self.log_metric = 'rates'
        self.r_min = r_min


    def barplot_opt_problem(self, avg_rates, metric_names = ['Expert policy'], percentiles = [1, 5, 10, 25, 50]):

        if not isinstance(avg_rates, list):
            avg_rates = [avg_rates]

        if not isinstance(metric_names, list):
            metric_names = [metric_names]

        if not isinstance(percentiles, list):
            percentiles = [percentiles]

        assert len(metric_names) == len(avg_rates), f"{len(metric_names)} many metrics but {len(avg_rates)} many rate-data has been passed."

        #### Bar Plot of the Optimization Problem ####
        # percentiles = [1, 5, 10]
        # percentile_labels = ['Obj'] + [f"C {i}%" for i in percentiles]

        all_metrics = []
        all_labels = []
        for avg_rate in avg_rates:

            obj_value = avg_rate.mean(axis=1).squeeze()

            labels = ['Obj (mean rate)']
            metrics = [obj_value.item()]
            for percentile in percentiles:
                labels.append(f'{percentile}-percentile rate')
                constraint_values = np.percentile(avg_rate.squeeze(), q=percentile)
                metrics.append(constraint_values.item())

            all_metrics.append(metrics)
            all_labels.append(labels)
        # metric_names = ['Expert policy']

        if self.log_path is not None:

            fig, ax = plt.subplots(1, 1, figsize = (18, 6))

            histtypes = ['stepfilled', 'step', 'bar', 'barstacked']
            # Pick a random histype
            histtype = np.random.choice(histtypes)
            
            for i, (avg_rate, metric_name) in enumerate(zip(avg_rates, metric_names)):
                ax.hist(avg_rate.cpu().flatten(), bins = 100, alpha = 0.5, label = metric_name, density = True, cumulative=True, histtype=histtype)

            if self.r_min is not None:
                ax.axvline(x=self.r_min, ymin=0, ymax=1, color='r', linestyle='--', label = r"$r_{\min}$")
                
            ax.set_xlabel('Ergodic Rate')
            ax.set_ylabel('CDF')
            ax.legend(loc = 'best')
            ax.set_title('Rate Histogram')

            save_dir = f'{self.log_path}/opt-problem'
            os.makedirs(save_dir, exist_ok=True)

            if self.network_id is not None:
                plt.savefig(f"{save_dir}/histogram_network_{self.network_id}.pdf", dpi=300)
            else:
                plt.savefig(f"{save_dir}/histogram_global.pdf", dpi=300)

            plt.close(fig)


            fig, ax = plt.subplots(1, 1, figsize = (3 + len(percentiles), 3))

            for metric_name, metrics in zip(metric_names, all_metrics):
                ax.plot(np.arange(1, 1 + len(metrics)), metrics, label = metric_name)

            if self.r_min is not None:
                ax.axhline(y=self.r_min, xmin=0, xmax=1, color='r', linestyle='--', label = r"$r_{\min}$")

            
            ax.set_xticks(ticks = range(1, 1 + len(all_metrics[0])), labels=all_labels[0], fontsize = 8)
            
            ymin, ymax = ax.get_ylim()
            ymin = 0.0
            ymax = np.ceil(2 * ymax) / 2
            dy = 0.5
            yticks = np.arange(ymin, ymax + dy, dy)
            ax.set_yticks(ticks=yticks, labels=[f"{tick:.1f}" for tick in yticks], fontsize = 8)
            
            ax.tick_params(axis='x', labelrotation=45)
            ax.legend(fontsize = 6, loc = 'best')
            
            # fig.tight_layout()

            # if len(metric_names) == 1:
            #     save_dir = f"{self.log_path}/opt-problem/{metric_names[0]}"
            # else:
            save_dir = f"{self.log_path}/opt-problem"
            os.makedirs(save_dir, exist_ok=True)

            if self.network_id is not None:
                plt.savefig(f"{save_dir}/network_{self.network_id}.pdf", dpi=300)
            else:
                plt.savefig(f"{save_dir}/global.pdf", dpi=300)
            plt.close(fig)

    def get_avg_rates(self):
        L = np.array(self.data)
        n_iters, n_clients = L.shape
        avg_rates = L.mean(axis=0, keepdims=True)
        return avg_rates


    def __call__(self, scatter_client_idx=[0, 1]):

        try:
            super().__call__(scatter_client_idx)

            avg_rates = self.get_avg_rates()

            self.barplot_opt_problem(avg_rates=avg_rates)

        except:
            self.raise_log_error()




# class SAOptProblemLogger(Logger):
#     def __init__(self, data, log_path, network_id = 0, r_min = 1.2, n_constraints = 24):
#         super(SAOptProblemLogger, self).__init__(data = data, log_metric = 'opt-problem', log_path=log_path)
#         self.network_id = network_id
#         self.obj = Obj()
#         self.constraints = Constraints(r_min = r_min, n_constraints=n_constraints)

#     def update_data(self, new_data):
#         if self.network_id is not None:
#             self.data = new_data[self.log_metric][self.network_id] # (Different state-augmented initializations)
#         else:
#             temp_data = new_data[self.log_metric]
#             all_data = []
#             for network_id in range(len(temp_data)):
#                 all_data.append(torch.stack(temp_data[network_id]))
            
#             all_data = torch.stack(all_data, dim = 1) # [n_metrics, n_network] x data
#             self.data = [all_data[_] for _ in range(len(all_data))]


#     def __call__(self, metric_names = None, labels = None, save_pdf = True, epoch = None):
#         # metric_names = ['Zero init', 'Optimal init']
#         if metric_names is None:
#             metric_names = ['Expert policy']


#         if self.network_id is not None:
#             if labels is None:
#                 labels = ['Obj'] + [f'C#{i}' for i in range(self.constraints.n_constraints)]

#             metrics = []
#             # all_obj = []
#             # all_constraints = []
#             for avg_rates in self.data:
#                 obj_value = -self.obj(avg_rates).item() / self.constraints.n_constraints
#                 constraints_value = (-self.constraints(avg_rates)).tolist()
#                 # all_obj.append(obj_value.item())
#                 # all_constraints.append(constraints_value)
#                 metrics.append([obj_value] + constraints_value)

#             if self.log_path is not None:
#                 fig, ax = plt.subplots(1, 1, figsize = (len(labels), 4))
#                 ax = grouped_barplot(labels=labels, metrics=metrics, metric_names=metric_names, xlabel = None, ylabel = None, title = 'Opt Problem Bar Plot', axs=ax)
#                 fig.tight_layout()

#                 save_dir = f'{self.log_path}/{self.log_metric}'
#                 os.makedirs(save_dir, exist_ok=True)
#                 plt.savefig(f'{save_dir}/network_{self.network_id}.pdf', dpi = 300)
#                 plt.close()

#         else:
#             q = 0.99
#             metrics = []

#             if labels is None:
#                 labels = ['Obj (mean)'] + [f'Worst constraint (q = {q} - quantile)']

#             for avg_rates in self.data:
#                 obj_value = -self.obj(avg_rates).mean(dim = 0).item() / self.constraints.n_constraints # average across all networks
#                 constraints_value = -torch.quantile(self.constraints(avg_rates).view(-1), q = q).item()
#                 # all_obj.append(obj_value.item())
#                 # all_constraints.append(constraints_value)
#                 metrics.append([obj_value] + [constraints_value])

#             if self.log_path is not None:
#                 fig, ax = plt.subplots(1, 1, figsize = (8, 4))
#                 ax = grouped_barplot(labels=labels, metrics=metrics, metric_names=metric_names, xlabel = None, ylabel = None, title = 'Opt Problem Bar Plot', axs=ax)
#                 ax.tick_params(axis='x', labelrotation=0)
#                 fig.tight_layout()

#                 save_dir = f'{self.log_path}/{self.log_metric}'
#                 os.makedirs(save_dir, exist_ok=True)

#                 if save_pdf:
#                     fformat = 'pdf'
#                 else:
#                     fformat = 'png'

#                 if epoch is None:
#                     plt.savefig(f'{save_dir}/global_metrics.{fformat}', dpi = 300)
#                 else:
#                     plt.savefig(f'{save_dir}/global_metrics-{epoch}.{fformat}', dpi = 300)
#                 plt.close()



class RegressionLossLogger(SALagrangianLogger):
    def __init__(self, data, log_path):
        Logger.__init__(self, data=data, log_metric='regression_loss', log_path=log_path)

class DRLambdasLogger(Logger):
    def __init__(self, data, log_path, network_id = 0, client_idx = None):
        super(DRLambdasLogger, self).__init__(data=data, log_metric='dr-lambdas', log_path=log_path)
        self.network_id = network_id
        self.client_idx = client_idx

    def update_data(self, new_data, network_dim = 0, client_dim = 1):
        data = torch.index_select(new_data[self.log_metric].detach().cpu(), dim = network_dim, index=torch.tensor(self.network_id))
        if self.client_idx is not None:
            data = torch.index_select(data, dim = client_dim, index=torch.tensor(self.client_idx))
        data.squeeze_(dim = network_dim).numpy()
        self.data = data
        # super().update_data(data)

    def __call__(self):
        """
        Plot lambda regression as training progresses.
        """
        L = np.array(self.data) # [n_clients, 2] -> [lambda_hat vs lambda_star]
        client_idx = np.arange(len(L))

        if self.log_path is not None:
            fig, ax = plt.subplots(1, 1)
            ax.plot(client_idx, L, linestyle = '-', marker = 'd', label = ['Pred.', 'Target'])
            ax.set_xlabel(r'Dual multiplier ($\lambda_i$)')
            ax.set_ylabel(self.log_metric)
            ax.grid(True)
            ax.set_ylim([None, 1])
            ax.legend(loc = 'best')
            # fig.tight_layout()

            save_dir = f'{self.log_path}/{self.log_metric}'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/network_{self.network_id}.pdf', dpi = 300)
            plt.close(fig=fig)


class DRNodeFeaturesLogger(DRLambdasLogger):
    def __init__(self, data, log_path, network_id = 0, client_idx = None, features_list = None):
        DRLambdasLogger.__init__(self, data=data, log_path=log_path, network_id=network_id, client_idx=client_idx)
        self.log_metric = 'dr-node-features'
        self.features_list = features_list

    def __call__(self):
        """
        Plot regression features.
        """
        L = np.array(self.data) # [n_clients, n_features]
        client_idx = np.arange(len(L))

        if self.log_path is not None:

            fig, ax = plt.subplots(1, 1)
            ax.plot(client_idx, L, linestyle = '-', marker = 'd', label = self.features_list)
            ax.set_xlabel(r'Dual multiplier ($\lambda_i$)')
            ax.set_ylabel(self.log_metric)
            ax.grid(True)
            ax.legend(loc = 'best')
            # fig.tight_layout()

            save_dir = f'{self.log_path}/{self.log_metric}'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/network_{self.network_id}.pdf', dpi = 300)
            plt.close(fig=fig)



class DRNodeFeaturesvsLambdasScatterLogger(DRLambdasLogger):
    def __init__(self, data, log_path, network_id = 0, client_idx = None, features_list = None):
        DRLambdasLogger.__init__(self, data=data, log_path=log_path, network_id=network_id, client_idx=client_idx)
        self.log_metric = 'dr-node-features-vs-lambdas-scatter'
        self.features_list = features_list

    def __call__(self):
        """
        Scatter plot regression features vs lambdas.
        """

        SCATTER_POINT_SIZE = 0.5
        L = np.array(self.data) # [n_clients, n_features + 1]
        n_features = L.shape[-1] - 1
        client_idx = np.arange(len(L))

        if self.log_path is not None:

            fig, axs = plt.subplots(1, n_features, figsize = (6 * n_features, 4))

            for i, ax in enumerate(axs.flat):
                ax.scatter(x = L[:, -1], y = L[:, i], marker = 'd', s = SCATTER_POINT_SIZE)
                # ax.plot(client_idx, L, linestyle = '-', marker = 'd', label = self.features_list)
                ax.set_xlabel(r'Dual multiplier ($\lambda_i$)')
                ax.set_ylabel(self.features_list[i])
                ax.grid(True)
                # ax.legend(loc = 'best')

            # fig.tight_layout()

            save_dir = f'{self.log_path}/{self.log_metric}'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/network_{self.network_id}.pdf', dpi = 300)
            plt.close(fig=fig)


class NormalizedLogChannelMatrixLogger(Logger):
    def __init__(self, data, log_path, network_id = 0):
        super(NormalizedLogChannelMatrixLogger, self).__init__(data=data, log_metric='normalized-log-channel-matrix', log_path=log_path)
        self.network_id = network_id


    def update_data(self, new_data):
        dataset = new_data[self.network_id]
        edge_indices_l, edge_weights_l = dataset.edge_index_l, dataset.edge_weight_l
        channel_matrix = to_scipy_sparse_matrix(edge_index=edge_indices_l, edge_attr=edge_weights_l).toarray()

        self.data = [channel_matrix, dataset.transmitters_index]


    def __call__(self):
        """
        Heatmap of normalized log channel matrix
        """
        interference_axis = 0
        L = np.array(self.data[0]) # [n_clients x n_clients]
        S = np.diagonal(L)
        ISR = np.sum(L, axis = interference_axis) / np.diagonal(L) - 1
        L_sorted = np.sort(L, axis = interference_axis)[::-1]
        maxISR = L_sorted[1] / np.diagonal(L)

        reverse_ISR = np.sum(L, axis = 1 - interference_axis) / np.diagonal(L) - 1
        reverse_L_sorted = np.sort(L, axis = 1 - interference_axis)[::-1]
        reverse_maxISR = reverse_L_sorted[1] / np.diagonal(L)

        edge_index_l, edge_weight_l = from_scipy_sparse_matrix(sparse.csr_matrix(L))
        edge_index_l_normalized, edge_weight_l_normalized = get_laplacian(edge_index=edge_index_l, edge_weight=edge_weight_l, normalization="sym")
        L_sym_normalized = to_scipy_sparse_matrix(edge_index=edge_index_l_normalized, edge_attr=edge_weight_l_normalized).toarray()

        L_normalized = np.eye(L_sym_normalized.shape[0]) - L_sym_normalized
        L_normalized = L_sym_normalized

        transmitters_index = self.data[1]

        if self.log_path is not None:

            fig, axs = plt.subplots(1, 4, figsize = (24, 6))

            # Define custom colormap: gray → red
            colors = [(0.5, 0.5, 0.5), (1, 0, 0)]  # RGB: Gray to Red
            cmap = mcolors.LinearSegmentedColormap.from_list("gray_to_red", colors)

            sns.heatmap(data=L, cmap = cmap, annot=False, cbar=True, ax=axs[0])
            axs[0].grid(True)
            axs[0].set_title('Log-normalized channel matrix')


            cmap = 'coolwarm'
            sns.heatmap(data=L_normalized, cmap = cmap, annot=False, cbar=True, ax=axs[1])
            axs[1].grid(True)
            axs[1].set_title('Log-normalized channel matrix (With spectral normalization)')

            axs[2].plot(np.arange(S.shape[0]), S, '--o', label = 'Signal strength')
            axs[2].plot(np.arange(ISR.shape[0]), ISR, '--o', label = 'Total rx interference / signal')
            axs[2].plot(np.arange(maxISR.shape[0]), maxISR, '--d', label = 'Max rx interference / signal')
            axs[2].plot(np.arange(reverse_ISR.shape[0]), reverse_ISR, '--*', label = 'Total tx interference / signal')
            axs[2].plot(np.arange(reverse_maxISR.shape[0]), reverse_maxISR, '--h', label = 'Max tx interference / signal')

            axs[2].set_xlabel('Node (i)')
            axs[2].set_ylabel('Proxy for interference/signal ratios')
            axs[2].grid(True)
            axs[2].legend(loc = 'best')

            axs[3].plot(np.arange(transmitters_index.shape[0]), transmitters_index, '--or', label = 'Serving transmitters index.')
            axs[3].grid(True)
            axs[3].legend(loc = 'best')

            save_dir = f'{self.log_path}/{self.log_metric}'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/network_{self.network_id}.pdf', dpi = 300)
            plt.close(fig=fig)



class TxRxLogger(Logger):
    def __init__(self, data, log_path, network_id = 0):
        super(TxRxLogger, self).__init__(data=data, log_metric='tx-rx-locs', log_path=log_path)
        self.network_id = network_id

    def update_data(self, new_data):
        self.data = new_data[self.log_metric][self.network_id]
        
    def __call__(self):
        """
        Visualize the tx-rx pairs
        """
        
        if self.log_path is not None:

            fig, ax = plt.subplots(1, 1, figsize = (8, 8))
            ax = self.plot_tx_rx_locs(tx_rx_locs={"tx": self.data["tx"], "rx": self.data["rx"], "associations": self.data["associations"]},
                                      channel_gains=self.data["H_l"], ax=ax, plt_kws={"cbar_label": "Channel Gains",
                                                                                      "color_norm": "log",
                                                                                      "cmap": plt.get_cmap('coolwarm_r'),
                                                                                      "center_node": "tx"})
            # plt.savefig(f'./ChannelPerturbTest/subnetwork_{id}.png', dpi = 300)
            # plt.close(fig)
            save_dir = f'{self.log_path}/{self.log_metric}'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/network_{self.network_id}_channel_gains.pdf', dpi = 300)
            plt.close(fig=fig)

 
            P_max, noise = self.data["P_max"], self.data["noise_var"]
            p = P_max * np.ones(self.data["tx"].shape[0])

            h = self.data["H_l"]
            print("h.shape: ", h.shape)
            
            h_power_adjusted = np.expand_dims(p, axis = 1) * h
            
            # signal = np.zeros_like(p)
            signal_temp = h_power_adjusted[self.data["associations"]]
            # signal = np.array([signal_temp[rx_idx] for rx_idx in self.data["associations"]]).reshape(*p.shape)
            
            associated_rx = np.array([np.where(row)[0][0] if np.any(row) else -1 for row in self.data["associations"]])
            signal = np.array([x for x, i in sorted(zip(signal_temp, associated_rx), key=lambda pair: pair[1])]).reshape(*p.shape)

            print("signal.max(): ", np.max(signal), 'signal.min(): ', np.min(signal))
            interference = np.sum(h_power_adjusted, axis = 0) - signal
            print("interference.max(): ", np.max(interference), 'interference.min(): ', np.min(interference))
            fr_log_sinr = np.log2(1. + signal / (interference + noise))

            print("FR log SINR min: ", np.min(fr_log_sinr))
            print("FR log SINR max: ", np.max(fr_log_sinr))
            fr_log_sinr_temp = np.zeros_like(self.data["associations"], dtype = np.float32)
            fr_log_sinr_temp[self.data["associations"]] = fr_log_sinr
            fr_log_sinr = fr_log_sinr_temp

            fig, ax = plt.subplots(1, 1, figsize = (8, 8))
            ax = self.plot_tx_rx_locs(tx_rx_locs={"tx": self.data["tx"], "rx": self.data["rx"], "associations": self.data["associations"]},
                                      channel_gains=fr_log_sinr, ax=ax, plt_kws={"cbar_label": "Full-reuse log (1 + SINR)",
                                                                                 "color_norm": "linear",
                                                                                 "center_node": "rx",
                                                                                 "vcenter": self.data["r_min"],
                                                                                 "cmap": plt.get_cmap('coolwarm_r')})
            # plt.savefig(f'./ChannelPerturbTest/subnetwork_{id}.png', dpi = 300)
            # plt.close(fig)
            save_dir = f'{self.log_path}/{self.log_metric}'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/network_{self.network_id}_FR_log_one_plus_sinr.pdf', dpi = 300)
            plt.close(fig=fig)


    def plot_tx_rx_locs(self, tx_rx_locs, channel_gains = None, ax = None, xmin = None, xmax = None, ymin = None, ymax = None, plt_kws = {}):
        if ax is None:
            fig, ax = plt.subplots()

        color_norm_fnc = plt_kws.get('color_norm', "linear")
        center_node = plt_kws.get('center_node', "tx")
        cmap = plt_kws.get('cmap', plt.get_cmap('coolwarm'))
        vcenter = plt_kws.get('vcenter', None)

        tx_locs = tx_rx_locs['tx']
        rx_locs = tx_rx_locs['rx']

        associations = tx_rx_locs['associations'] if 'associations' in tx_rx_locs else np.eye(N = tx_locs.shape[0], M = rx_locs.shape[0], dtype=bool)
        # print("Associations: ", associations)
        # Get the column index of the first True value for each row

        associated_rx = np.array([np.where(row)[0][0] if np.any(row) else -1 for row in associations])
        # print("associated_rx: ", associated_rx)

        # channel_gains = 10 * np.random.rand(*associations.shape) + 1

        # Define colormap
        # cmap = plt.get_cmap('viridis')
        # colors_tx = cmap(np.linspace(0, 1, tx_rx_locs['tx'].shape[0]))
        # colors_rx = cmap(np.linspace(0, 1, tx_rx_locs['rx'].shape[0]))
        if channel_gains is None:
            colors_tx = create_repeating_colormap(tx_locs.shape[0], n_distinct_colors=10)
            colors_rx = create_repeating_colormap(rx_locs.shape[0], n_distinct_colors=10)

        else:
            color_data = channel_gains[associations]
            print("Color data min: ", np.min(color_data)),
            print("Color data max: ", np.max(color_data))


            if color_norm_fnc == "linear":
                if vcenter is not None:

                    try:
                        # Create centered norm
                        vmin = np.min(color_data)
                        vmax = np.max(color_data)

                        # Ensure vmin and vmax are different from center to avoid division by zero
                        if vmin == vcenter:
                            vmin -= 1e-10
                        if vmax == vcenter:
                            vmax += 1e-10

                        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

                    except: # if v_center is not within the range of color_data
                        norm = plt.Normalize(vmin=np.min(color_data), vmax=np.max(color_data))
                else:
                    # Normalize channel gains for color mapping
                    norm = plt.Normalize(vmin=np.min(color_data), vmax=np.max(color_data))
            
            elif color_norm_fnc == "log":
                # Normalize channel gains for color mapping
                norm = LogNorm(vmin=np.min(color_data), vmax=np.max(color_data))

            elif color_norm_fnc == "power":
                # gamma < 1 emphasizes lower values, gamma > 1 emphasizes higher values
                norm = PowerNorm(gamma=0.5, vmin=np.min(color_data), vmax=np.max(color_data))
            
            cmap = cmap

            # Get channel gain and map to color
            # colors = np.array([cmap(norm(channel_gains[i, rx_idx])) for i, rx_idx in enumerate(associated_rx)])
            colors = cmap(norm(color_data))
            # color = cmap(norm(gain))
            colors_tx = colors
            colors_rx = colors

        
        ax.scatter(tx_locs[:, 0], tx_locs[:, 1], label='Tx', marker = 'd', c=colors_tx, s = 20)
        ax.scatter(rx_locs[associated_rx, 0], rx_locs[associated_rx, 1], label='Rx', marker = 'x', c=colors_rx, s = 20)
        

        # Draw edges between associated TX-RX pairs
        if channel_gains is not None:

            # Calculate max radius for scaling
            max_radius = min(np.abs(xmax-xmin), np.abs(ymax-ymin)) / 20 if xmin is not None and xmax is not None and ymin is not None and ymax is not None else 250
            
            for i, rx_idx in enumerate(associated_rx):
                if rx_idx >= 0:
                    # Get coordinates
                    tx_x, tx_y = tx_locs[i]
                    rx_x, rx_y = rx_locs[rx_idx]
                    
                    # Get channel gain and map to color
                    # gain = channel_gains[i, rx_idx] if channel_gains is not None else 1.0
                    gain = color_data[i]
                    color = cmap(norm(gain))
                    
                    # Draw line
                    ax.plot([tx_x, rx_x], [tx_y, rx_y], '-', color=color, linewidth=1.0)

                    # Draw circle around transmitter with radius proportional to channel gain
                    # Normalize radius between 0.01*max_radius and max_radius
                    radius = max_radius * norm(gain) + 1e-10
                    # radius = max_radius * (gain - np.min(channel_gains)) / (np.max(channel_gains) - np.min(channel_gains) + 1e-10)
                    radius = max(radius, 0.05 * max_radius)  # Ensure minimum visibility

                    if center_node == "tx":
                        circle_origin = (tx_x, tx_y)
                    elif center_node == "rx":
                        circle_origin = (rx_x, rx_y)
                    else:
                        raise ValueError(f"Invalid center node: {center_node}")
                    
                    circle = Circle(circle_origin, radius, fill=True, color=color, alpha=0.2, linestyle='-', linewidth = 0.01)
                    ax.add_patch(circle)
    

        # Add colorbar if channel gains are available
        if channel_gains is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # plt.colorbar(sm, ax=ax, label='Channel Gains')

            # Create a smaller colorbar with these options:
            # shrink: Makes the colorbar smaller (0.5 = 50% of original size)
            # aspect: Controls the ratio of long to short dimensions
            # pad: Controls spacing between plot and colorbar
            cbar_label = 'Channel Gains' if plt_kws.get('cbar_label') is None else plt_kws.get('cbar_label')
            cbar = plt.colorbar(sm, ax=ax, label=cbar_label, shrink=0.75, aspect=10, pad=0.05)
            
            # Optionally make the label and ticks smaller
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label(cbar_label, size=14)

            
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.legend()
        return ax



class WirelessGraphLogger(Logger):
    def __init__(self, data, log_path, network_id = 0):
        super(WirelessGraphLogger, self).__init__(data=data, log_metric='wireless-graph', log_path=log_path)
        self.network_id = network_id

    def update_data(self, new_data):
        dataset = new_data[self.network_id]
        self.data = {'node_pos': dataset.pos[0],
                     'edge_index': dataset.edge_index_l,
                     'edge_weight': dataset.edge_weight_l,
                    }
        
    def __call__(self, with_labels = False):
        """
        Visualize the constructed graph.
        """
        
        if self.log_path is not None:

            fig, ax = plt.subplots(1, 1, figsize = (8, 8))

            ### remove small weight edges ###
            edge_weight = rescale_edge_weights(self.data['edge_weight'], max_edge_weight=10., scale='exponential')
            thresh = torch.quantile(edge_weight, 0.5)
            kept_edges_filter = edge_weight > thresh
            edge_index = self.data['edge_index'][:, kept_edges_filter]
            edge_weight = edge_weight[kept_edges_filter]


            # Creating a PyG Data object
            data = Data(pos=self.data['node_pos'],
                        edge_index=edge_index,
                        edge_weight=edge_weight
                        )

            # Visualize the graph with edge weights
            visualize_graph(data, ax=ax, curved=True, with_labels=with_labels, node_size=200, font_size=12)

            # fig.tight_layout()

            save_dir = f'{self.log_path}/{self.log_metric}'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/network_{self.network_id}.pdf', dpi = 300)
            plt.close(fig=fig)



class DiffusionLossLogger(SALagrangianLogger):
    def __init__(self, data, log_path):
        Logger.__init__(self, data=data, log_metric='diffusion-loss', log_path=log_path)


class DiffusionPGradLogger(PgradLogger):
    def __init__(self, data, log_path):
        Logger.__init__(self, data=data, log_metric='diffusion-pgrad-norm', log_path=log_path)


class DiffusionNetworkLossLogger(SALagrangianLogger):
    def __init__(self, data, log_path):
        Logger.__init__(self, data=data, log_metric='diffusion-all-networks-loss', log_path=log_path)

    def update_data(self, new_data):
        data = new_data[self.log_metric]
        self.data.append(data)

    def __call__(self):

        try: 
            L = np.array(self.data) # [n_iters, n_networks]
            n_iters, n_networks = L.shape

            max_curves_per_plot = MAX_CURVES_PER_PLOT # 16
            n_cols_legend = (max_curves_per_plot + 7 // 8)

            if self.log_path is not None:

                for i in range(0, n_networks, max_curves_per_plot):

                    start = i
                    end = min(i + max_curves_per_plot, n_networks)

                    fig, ax = plt.subplots(1, 1)
                    ax.plot(np.arange(n_iters), L[:, start:end], '-', label = [f'{s}' for s in list(range(start, end))])
                    ax.set_ylim([0.0, 1.2])
                    ax.set_xlabel('Epoch (k)')
                    ax.set_ylabel(self.log_metric)
                    ax.grid(True)
                    ax.legend(loc = 'best', ncol = 2)
                    # fig.tight_layout()

                    plt.savefig(f'{self.log_path}/{self.log_metric}-{start}-{end}.pdf', dpi = 300)
                    plt.close(fig)

        except:
            self.raise_log_error()



class DiffusionSamplerLogger(Logger):
    def __init__(self, data, log_path, network_id = 0, transform = None, **kwargs):
        super(DiffusionSamplerLogger, self).__init__(data = data, log_metric = 'diffusion-sampler', log_path=log_path)
        self.network_id = network_id
        self.transform = transform
        
        self.split_subfigures = kwargs.get('split_subfigures', False)
        self.verbose_labels = kwargs.get('verbose_labels', True)

    def get_save_data(self):
        return {'data': self.data,
                'log_metric': self.log_metric,
                'log_path': self.log_path,
                'network_id': self.network_id
                }

    def update_data(self, new_data):
        self.data = new_data[self.log_metric][self.network_id] # (X_orig, Xgen) = 2 x Tensor(nsamples, nfeatures)

        # print(f"new_data[{self.log_metric}][{self.network_id}]: ", new_data[self.log_metric][self.network_id])

    def __call__(self, scatter_client_idx = None):

        try:
            X_orig = self.data[0].cpu()
            X_gen = self.data[1].cpu()

            # print(f'Network id: {self.network_id}\tScatter idx: {scatter_client_idx}\tX_orig.shape = {X_orig.shape}\tX_gen.shape: {X_gen.shape}')

            labels = ['Expert policy', 'Diffusion policy']
            title = ['Scatter Plot of Diffusion Sampling']

            if self.transform is not None:
                X_orig = (X_orig, self.transform(X_orig))
                X_gen = (X_gen, self.transform(X_gen))

                title = (title[0], title[0] + ' (normalized)')

            if self.log_path is not None:

                if not self.split_subfigures:
                    n_cols = len(X_orig)
                    fig, axs = plt.subplots(1, n_cols, figsize = (6 * n_cols, 6), squeeze = False)
                    
                    for i, ax in enumerate(axs.flat):
                        # print(f"i: {i}, client_idx = {scatter_client_idx[0], scatter_client_idx[1]}")
                        x_gen, y_gen = X_gen[i][:, scatter_client_idx[0]], X_gen[i][:, scatter_client_idx[1]]
                        x_orig, y_orig = X_orig[i][:, scatter_client_idx[0]], X_orig[i][:, scatter_client_idx[1]]

                        ax.scatter(x = x_orig, y = y_orig, marker = 'o', label = labels[0])
                        ax.scatter(x = x_gen, y = y_gen, marker = '1', label = labels[1])
                        ax.grid(True)

                        if self.verbose_labels:
                            ax.set_xlabel(f'Client {scatter_client_idx[0]}')
                            ax.set_ylabel(f'Client {scatter_client_idx[1]}')
                            ax.set_title(title[i])

                        if labels[i] == 'Diffusion policy':
                            min_x_val, max_x_val = min(0., x_gen.quantile(0.05)), max(1., x_gen.quantile(0.95))
                            min_y_val, max_y_val = min(0., y_gen.quantile(0.05)), max(1., y_gen.quantile(0.95))
                            
                            ax.set_xlim([min_x_val, max_x_val])
                            ax.set_ylim([min_y_val, max_y_val])

                        ax.legend(loc = 'best')

                    # fig.tight_layout()

                    save_dir = f'{self.log_path}/{self.log_metric}'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f'{save_dir}/network_{self.network_id}_client_idx_{scatter_client_idx}.pdf', dpi = 300)
                    plt.close(fig=fig)

                else:
                    for use_hdr in [False]:
                        for i in range(len(X_orig)):
                            hdr_plot_style() if use_hdr else None
                            fig, ax = plt.subplots(1, 1)
                        
                            x_gen, y_gen = X_gen[i][:, scatter_client_idx[0]], X_gen[i][:, scatter_client_idx[1]]
                            x_orig, y_orig = X_orig[i][:, scatter_client_idx[0]], X_orig[i][:, scatter_client_idx[1]]

                            ax.scatter(x = x_orig, y = y_orig, marker = 'o', label = labels[0], facecolors = 'None', edgecolors="C{}".format(0))
                            ax.scatter(x = x_gen, y = y_gen, marker = '1', label = labels[1], c = "C{}".format(1))
                            ax.grid(True)

                            if self.verbose_labels:
                                ax.set_xlabel(f'Client {scatter_client_idx[0]}')
                                ax.set_ylabel(f'Client {scatter_client_idx[1]}')
                                ax.set_title(title[i])

                            if labels[i] == 'Diffusion policy':
                                min_x_val, max_x_val = min(0., x_gen.quantile(0.05)), max(1., x_gen.quantile(0.95))
                                min_y_val, max_y_val = min(0., y_gen.quantile(0.05)), max(1., y_gen.quantile(0.95))
                                
                                ax.set_xlim([min_x_val, max_x_val])
                                ax.set_ylim([min_y_val, max_y_val])

                            ax.legend(loc = 'best')
                            # fig.tight_layout()

                            save_dir = f'{self.log_path}/{self.log_metric}'
                            os.makedirs(save_dir, exist_ok=True)
                            plt.savefig(f'{save_dir}/network_{self.network_id}_client_idx_{scatter_client_idx}_hdr_{use_hdr}-{i}.pdf', dpi = 300)
                            # plt.savefig(f'{save_dir}/network_{self.network_id}_client_idx_{scatter_client_idx}_hdr_{use_hdr}-{i}.jpg', dpi = 300)
                            plt.close(fig)

                            hdr_plot_style(activate=False) if use_hdr else None

        except:
            self.raise_log_error()


class DiffusionSamplerHistLogger(DiffusionSamplerLogger):
    def __init__(self, data, log_path, network_id = 0, transform = None, **kwargs):
        super(DiffusionSamplerHistLogger, self).__init__(data = data, log_path=log_path, network_id=network_id, transform=transform, **kwargs)
        self.log_metric = 'diffusion-sampler-hist'
        self.plot_timesteps = kwargs.get('timestep', None)

    # def update_data(self, new_data):
    #     self.data = new_data[self.log_metric][self.network_id] # (X_orig, Xgen) = 2 x Tensor(nsamples, nfeatures)

    def get_save_data(self):
        return super().get_save_data()

    def __call__(self, scatter_client_idx = [0, 1]):

        try:
            X_t = self.data[0].cpu()
            Score_t = self.data[1].cpu()

            timesteps = [0, len(X_t) // 2]
            dt = 50

            # print(f'x_t: {X_t.shape}, score_t: {Score_t.shape}')

            if self.transform is not None:
                X_t = (X_t, self.transform(X_t))
                Score_t = (Score_t, Score_t) # normalized score does not change under affine transforms

            if self.log_path is not None:

                for timestep in timesteps:
                    if not self.split_subfigures:
                        n_cols = len(X_t)

                        hdr_plot_style(activate=True)

                        fig, axs = plt.subplots(1, n_cols, figsize = (6 * n_cols, 6), squeeze=False)

                        for i, ax in enumerate(axs.flat):
                            x_t, y_t = X_t[i][-1, :, scatter_client_idx[0]], X_t[i][-1, :, scatter_client_idx[1]] # Langevin dynamics destination
                            xx_t, yy_t = X_t[i][-(2+dt+timestep), :, scatter_client_idx[0]], X_t[i][-(2+dt+timestep), :, scatter_client_idx[1]] # gradient origins

                            scores = Score_t[i][-(1 + dt + timestep), :, :]

                            scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                            scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)

                            score_x, score_y = scores_log1p[:, scatter_client_idx[0]], scores_log1p[:, scatter_client_idx[1]]

                            ax.scatter(x = x_t, y = y_t, alpha = 0.5, color='red', edgecolor='white', s=40)
                            
                            xx = np.stack((xx_t, yy_t), axis = 0)
                            scores = np.stack((score_x, score_y), axis = 0)
                            ax.quiver(*xx, *scores, width = 0.002, color = "white")

                            if self.verbose_labels:
                                ax.set_xlabel(f'Client {scatter_client_idx[0]}')
                                ax.set_ylabel(f'Client {scatter_client_idx[1]}')
                                ax.set_title(f'Timestep = {timestep}')

                        # fig.tight_layout()

                        save_dir = f'{self.log_path}/{self.log_metric}'
                        os.makedirs(save_dir, exist_ok=True)
                        plt.savefig(f'{save_dir}/network_{self.network_id}_client_idx_{scatter_client_idx}_score_field_timestep_{timestep}.pdf', dpi = 300)
                        plt.close(fig)

                        hdr_plot_style(activate=False)

                    else:
                        for i in range(len(X_t)):
                            hdr_plot_style(activate=True)

                            fig, ax = plt.subplots(1, 1)

                            x_t, y_t = X_t[i][-1, :, scatter_client_idx[0]], X_t[i][-1, :, scatter_client_idx[1]] # Langevin dynamics destination
                            xx_t, yy_t = X_t[i][-(2+dt+timestep), :, scatter_client_idx[0]], X_t[i][-(2+dt+timestep), :, scatter_client_idx[1]] # gradient origins

                            scores = Score_t[i][-(1 + dt + timestep), :, :]

                            scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                            scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)

                            score_x, score_y = scores_log1p[:, scatter_client_idx[0]], scores_log1p[:, scatter_client_idx[1]]

                            ax.scatter(x = x_t, y = y_t, alpha = 0.5, color='red', edgecolor='white', s=40)
                            
                            xx = np.stack((xx_t, yy_t), axis = 0)
                            scores = np.stack((score_x, score_y), axis = 0)
                            ax.quiver(*xx, *scores, width = 0.002, color = "white")

                            if self.verbose_labels:
                                ax.set_xlabel(f'Client {scatter_client_idx[0]}')
                                ax.set_ylabel(f'Client {scatter_client_idx[1]}')
                                ax.set_title(f'Timestep = {timestep}')

                            # fig.tight_layout()
                            save_dir = f'{self.log_path}/{self.log_metric}'
                            os.makedirs(save_dir, exist_ok=True)
                            plt.savefig(f'{save_dir}/network_{self.network_id}_client_idx_{scatter_client_idx}_score_field_timestep_{timestep}-{i}.pdf', dpi = 300)
                            plt.close(fig)
                            hdr_plot_style(activate=False)

        except:
            self.raise_log_error()


class CDModelLogger(SAModelLogger):
    def __init__(self, data, log_path):
        super(CDModelLogger, self).__init__(data = data, log_path = log_path)

    def __call__(self, model_state_dict, epoch = None, accelerator = None):

        try:
            # save trained model weights
            if epoch is not None:
                if accelerator is None:
                    torch.save(model_state_dict, f'{self.log_path}/cd_model_state_dict_epoch_{epoch}.pt')
                    print(f'Saving model weights at epoch {epoch} is successful.')
                else:
                    accelerator.save(model_state_dict, f'{self.log_path}/cd_model_state_dict_epoch_{epoch}.pt')
                    accelerator.print(f'Saving model weights at epoch {epoch} is successful.')
            else:
                if accelerator is None:
                    torch.save(model_state_dict, f'{self.log_path}/cd_model_state_dict.pt')
                    print('Saving model weights is successful.')
                else:
                    accelerator.save(model_state_dict, f'{self.log_path}/cd_model_state_dict.pt')
                    accelerator.print('Saving model weights is successful.')

        except:
            self.raise_log_error()


# def stack_dicts_by_first_key(dict_list):
#     grouped = defaultdict(list)
    
#     for d in dict_list:
#         # Assume the first key is the one we want to group by
#         first_key = next(iter(d))  # Get the first key of the dictionary
#         grouped[d[first_key]].append(d)  # Group by the value of the first key
    
#     return grouped
from collections import defaultdict
def stack_dicts_by_first_key(dict_list):
    grouped = defaultdict(list)
    
    for d_list in dict_list:
        for d in d_list:
            grouped[d["layer"]].append(d["norm"])
            # # Assume the first key is the one we want to group by
            # first_key = next(iter(d))  # Get the first key of the dictionary
            # grouped[d[first_key]].append(d)  # Group by the value of the first key
    
    return grouped


class CDModelLayerNormsLogger(Logger):
    def __init__(self, data, log_path):
        super(CDModelLayerNormsLogger, self).__init__(data=data, log_metric='model_layer_norms', log_path=log_path)

    # def update_data(self, data_list):
    #     self.data = data_list[self.log_metric]
    def update_data(self, new_data):
        data = new_data[self.log_metric]
        self.data.append(data)

    def __call__(self):

        try:

            if self.data is None or len(self.data) == 0:
                return 0

            max_curves_per_plot = 8 # 16
            n_cols_legend = (max_curves_per_plot + 15 // 16)

            if isinstance(self.data[0], tuple):
                epochs, dict_list = zip(*self.data)
                epochs = np.array(epochs)
            else:
                epochs = None
                dict_list = self.data

            grouped = stack_dicts_by_first_key(dict_list=dict_list)
            # print("grouped: ", grouped)

            fig, ax = plt.subplots(1, 1)
            start = 0
            end = 0

            max_index = len(grouped)
            for index, (layer, norms) in enumerate(grouped.items()):
                if epochs is None:
                    epochs = np.arange(len(norms))

                ax.plot(epochs, norms, '-', label = f"Layer {layer}")
                ax.set_xlabel('Epoch (k)')
                ax.set_ylabel(self.log_metric)
                ax.grid(True)
                ax.legend(loc = 'best', ncol = 2)
                # fig.tight_layout()

                if (index + 1) % max_curves_per_plot == 0 or (index + 1) == max_index:
                    plt.savefig(f'{self.log_path}/{self.log_metric}-{start}-{end}.pdf', dpi = 300)
                    plt.close(fig)

                    # New figure for the next subset of curves
                    fig, ax = plt.subplots(1, 1)

                    start = index
                    end = index
                else:
                    end += 1

        except:
            self.raise_log_error()
            


class CDModelAttnWeightsLogger(Logger):
    def __init__(self, data, log_path):
        super(CDModelAttnWeightsLogger, self).__init__(data=data, log_metric='model_attn_weights', log_path=log_path)

    def update_data(self, data_list):
        self.data = data_list[self.log_metric]

    def __call__(self):

        try:

            if self.data is None or len(self.data) == 0:
                return 0
            
            attn_weights_list_list = self.data

            n_graphs = 2
            n, m = len(attn_weights_list_list), len(attn_weights_list_list[0])

            for k, data_list in enumerate(attn_weights_list_list):
                for j, data_tuple in enumerate(data_list):
                    
                    layer_name, attn_weights = data_tuple
                    print('layer name: ', layer_name)

                    edge_indices, attn_scores = attn_weights
                    num_heads = attn_scores.shape[-1]

                    fig, axs = plt.subplots(1, num_heads, figsize = (num_heads * 6, 6))

                    for i, ax in enumerate(axs):
                        # ith head only
                        adj = to_scipy_sparse_matrix(edge_indices.detach().cpu(), attn_scores[:, i].detach().cpu()).toarray() 
                        adj = adj[:n_graphs*NUM_NODES, :n_graphs*NUM_NODES]
                        sns.heatmap(data=adj, cmap = 'coolwarm', annot=False, cbar=True, ax=ax)
                        ax.grid(True)
                        ax.set_title(layer_name + f'_head_{i}')

                    # fig.tight_layout()
                    save_dir = f'{self.log_path}/{self.log_metric}'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f'{save_dir}/block_{k}_{layer_name}.pdf', dpi = 300)
                    plt.close(fig)

        except:
            self.raise_log_error()



class CDModelDebugLogger(Logger):
    def __init__(self, data, log_path):
        super(CDModelDebugLogger, self).__init__(data=data, log_metric='model_debug_data', log_path=log_path)

    def update_data(self, data_list):
        self.data = data_list[self.log_metric]

    def __call__(self, n_graphs_per_batch = 1):

        try:
            if self.data is None or len(self.data) == 0:
                return 0
            
            edge_indices, edge_weights, batches = self.data['edge_indices'], self.data['edge_weights'], self.data['batches']

            n_subgraphs = 4
            n_graphs = 4

            for k, (e, w, batch) in enumerate(zip(edge_indices, edge_weights, batches)):
                layer_depth = k

                # try:
                #     all_adj = to_dense_adj(edge_index = e, edge_attr=w, batch=batch)
                # except:
                if n_graphs_per_batch is not None:
                    batch = torch.remainder(batch, n_graphs_per_batch)
                all_adj = to_dense_adj(edge_index = e, edge_attr=w, batch=batch)
                
                # num_graphs = int(batch.max()) + 1
                # num_nodes = len(batch) // num_graphs
                num_nodes = NUM_NODES
                num_subgraphs = len(batch) // num_nodes
                num_graphs = int(batch.max()) + 1

                print('layer_depth: ', layer_depth)
                print('num_subgraphs: ', num_subgraphs)
                print('num_nodes: ', num_nodes)

                n_graphs = min(n_graphs, num_graphs)
                n_subgraphs = min(n_subgraphs, num_subgraphs)

                fig, axs = plt.subplots(n_graphs, 1, figsize = (6, 6 * n_graphs), squeeze=False)
                for i, ax in enumerate(axs.flat):
                    adj = all_adj[i, :n_subgraphs*num_nodes, :n_subgraphs*num_nodes]
                    sns.heatmap(data=adj, cmap = 'coolwarm', annot=False, cbar=True, ax=ax)
                    ax.grid(True)
                    ax.set_title(f'UNet depth = {layer_depth}, Graph #{i}')

                # fig.tight_layout()

                save_dir = f'{self.log_path}/{self.log_metric}'
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(f'{save_dir}/unet_depth_{layer_depth}.pdf', dpi = 300)
                plt.close(fig)

        except:
            self.raise_log_error()



def grouped_barplot(labels, metrics, metric_names, xlabel, ylabel, title, axs = None):
    # Number of groups
    num_groups = len(labels)
    num_metrics = len(metrics)
    
    # Width of each bar
    bar_width = min(0.2, 0.75 / (num_metrics))
    
    # Create a list of positions for bars
    index = np.arange(num_groups)
    
    if axs is None:
        # Plotting grouped bars
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        
        # Plot each metric's bars
        for i in range(num_metrics):
            plt.bar(index + (1/2 + i - num_metrics / 2) * bar_width, metrics[i], bar_width, label=metric_names[i])
        # plt.bar(index, metric2, bar_width, label=metric_names[1])
        # plt.bar(index + bar_width, metric3, bar_width, label=metric_names[2])
        
        # Add labels, title and legend
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(index, labels, rotation = 90)
        plt.legend()
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        ax = plt.gca()
        return ax
    else:
        # Plot each metric's bars
        for i in range(num_metrics):
            axs.bar(index + (1/2 + i - num_metrics / 2) * bar_width, metrics[i], bar_width, label=metric_names[i])
            # Add labels, title and legend
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            axs.set_title(title)
            axs.set_xticks(index, labels)
            axs.tick_params(axis='x', labelrotation=90)
            axs.legend()
            # axs.tight_layout()

        return axs

# # Example usage:
# labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
# labels = [f'Constraint #{i}' for i in range(len(labels))]
# metric1 = [10, 15, 7, 10]
# metric2 = [12, 14, 6, 8]
# metric3 = [8, 11, 9, 12]
# metric4 = [8, 11, 9, 9]
# metrics = [metric1, metric2, metric3, metric4] * 2
# metric_names = ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4'] * 2
# xlabel = 'Constraints'
# ylabel = 'constraint slacks'
# title = 'Grouped Bar Plot of Metrics'

# fig, axs = plt.subplots(1, 1, figsize = (10, 6))
# # axs = None

# axs = grouped_barplot(labels, metrics, metric_names, xlabel, ylabel, title, axs)
# axs.grid(True)


PHASES = ["train", "val", "test"]
def make_cd_loggers(args, log_path, inv_transform = None, baseline_metrics = None):

    # phases = ['train', 'test']
    loggers = defaultdict(list)

    if not args.log_CD:
        for phase in PHASES:
            loggers[phase] = None
        return loggers

    for phase in PHASES:
        updated_log_path = log_path + f'/{phase}'
        # os.makedirs(updated_log_path, exist_ok=True) # toggled off to avoid accelerate's multi-folder logging

        if phase == 'train':
            # pass
            # logger = DiffusionLossLogger(data = [], log_path=updated_log_path)
            # loggers[phase].append(logger)

            logger = DiffusionPGradLogger(data = [], log_path=updated_log_path)
            loggers[phase].append(logger)

            logger = CDModelLogger(data = [], log_path=updated_log_path)
            loggers[phase].append(logger)

            logger = CDModelLayerNormsLogger(data = [], log_path=updated_log_path)
            loggers[phase].append(logger)

            logger = CDModelAttnWeightsLogger(data = [], log_path=updated_log_path)
            loggers[phase].append(logger)

            # logger = CDModelDebugLogger(data = [], log_path=updated_log_path)
            # loggers[phase].append(logger)


            for network_id in range(args.num_channels[phase])[:min(MAX_LOGGED_NETWORKS, args.num_channels[phase])]:
                logger = DiffusionSamplerLogger(data = [],
                                                log_path=updated_log_path,
                                                network_id=network_id,
                                                transform = inv_transform,
                                                split_subfigures = False,
                                                verbose_labels = True
                                                )
                loggers[phase].append(logger)

                logger = DiffusionSamplerHistLogger(data = [],
                                                    log_path=updated_log_path,
                                                    network_id=network_id,
                                                    transform = inv_transform,
                                                    split_subfigures = False,
                                                    verbose_labels = True
                                                    )
                loggers[phase].append(logger)

                

        elif phase in ['val', 'test']:
            # pass
            for network_id in range(args.num_channels[phase])[:min(MAX_LOGGED_NETWORKS, args.num_channels[phase])]:
                logger = DiffusionSamplerLogger(data = [], log_path=updated_log_path, network_id=network_id, transform = inv_transform,
                                                split_subfigures = False, verbose_labels = True)
                loggers[phase].append(logger)

                logger = DiffusionSamplerHistLogger(data = [], log_path=updated_log_path, network_id=network_id, transform = inv_transform,
                                                    split_subfigures = False, verbose_labels = True)
                loggers[phase].append(logger)

        else:
            raise Exception
        

        if phase in ['train', 'val']:
            logger = DiffusionLossLogger(data = [], log_path=updated_log_path)
            loggers[phase].append(logger)

            logger = DiffusionNetworkLossLogger(data = [], log_path=updated_log_path)
            loggers[phase].append(logger)

        if len(loggers[phase]) == 0:
            loggers[phase] = None

    return loggers



def save_logger_object(obj, filename):
    if os.path.exists(filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    else:
        print(f'Loggers could not be saved because {filename} does not exist.')



def create_logger_dict(loggers):
    log_metrics = set([logger.log_metric for logger in loggers]) # unique log metrics
    save_obj = []
    for log_metric in log_metrics:
        # There are multiple loggers with the same log metric but defined for different networks
        temp = [logger.get_save_data() for logger in loggers if logger.log_metric == log_metric]
        save_obj.append((log_metric, temp))
    # save_obj = dict([(logger.log_metric, logger) for logger in sa_loggers[phase]])
    save_obj = dict(save_obj)

    return save_obj