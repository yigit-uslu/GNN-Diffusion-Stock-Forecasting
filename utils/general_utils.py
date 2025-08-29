import numpy as np
import torch
import random
import os, time
from core.config import DEBUG_TRAIN


# from core.model import ConditionalLinearModel, GraphDiffusionModel, DiffusionModel

def debug_print(msg, flag = DEBUG_TRAIN):
    if flag == DEBUG_TRAIN:
        print(msg)
    else:
        pass


def seed_everything(seed):
    # set the random seed
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def assert_same_dtype_and_size(tensor1, tensor2):
    # Check if both tensors have the same data type
    assert tensor1.dtype == tensor2.dtype, f"Data types do not match: {tensor1.dtype} vs {tensor2.dtype}"
    
    # Check if both tensors have the same size
    assert tensor1.size() == tensor2.size(), f"Sizes do not match: {tensor1.size()} vs {tensor2.size()}"


def find_substring_index(string, substring):
    return string.index(substring) + len(substring) - 1


def make_experiment_name(args):
    # Create experiment name
    experiment_name = f'{time.time()}'

    return experiment_name



def create_folders_and_dirs(root_path):

    # create folders to save the data and results
    os.makedirs(f'{root_path}/data', exist_ok=True) 
    for phase in ['train', 'val', 'test']: 
        os.makedirs(f'{root_path}/data/{phase}', exist_ok=True)    

    os.makedirs(f'{root_path}/results', exist_ok=True)  
    os.makedirs(f'{root_path}/models', exist_ok=True)
    os.makedirs(f"{root_path}/datasets", exist_ok=True)



def fit_polynomial(x, y, degree):
    """
    Fits a polynomial to the given data points and returns the coefficients.

    Parameters:
    x (numpy.ndarray): The x-coordinates of the data points.
    y (numpy.ndarray): The y-coordinates of the data points.
    degree (int): The degree of the polynomial to fit.

    Returns:
    numpy.ndarray: The coefficients of the fitted polynomial, ordered from highest to lowest degree.
    """
    # Fit a polynomial of the given degree to the data
    coefficients = np.polyfit(x, y, degree)
    
    return coefficients



def evaluate_polynomial(coefficients, x_values):
    """
    Evaluates a polynomial at given x-values.

    Parameters:
    coefficients (numpy.ndarray): The coefficients of the polynomial, ordered from highest to lowest degree.
    x_values (numpy.ndarray): The x-coordinates where the polynomial should be evaluated.

    Returns:
    numpy.ndarray: The evaluated y-values of the polynomial.
    """
    # Evaluate the polynomial at the given x-values
    y_values = np.polyval(coefficients, x_values)
    
    return y_values



def make_logger(debug = False):
    if debug is True:
        def logger(x):
            print(x)
    else:
        def logger(x):
            pass

    return logger



def make_loss_fn(eval_type):
    if eval_type == 'l1':
        loss_fn = torch.nn.L1Loss(reduction='none')
    elif eval_type == 'l2':
        loss_fn = torch.nn.MSELoss(reduction='none')
    elif eval_type == 'huber':
        loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    else:
        raise NotImplementedError

    def weighted_loss_fn(input, target, weights = None):
        weights = 1. if weights is None else weights
        return torch.mean(weights * loss_fn(input, target))
    
    return weighted_loss_fn
    
    


def make_dual_transform_fnc(eval_type):
    if eval_type == 'softmax':
        def dual_transform(mu):
            return torch.nn.functional.softmax(mu, dim = -1)
    elif eval_type == 'L2_normalization':
        def dual_transform(mu):
            return torch.nn.functional.normalize(mu, p=2, dim = -1)
    elif eval_type == 'none' or eval_type is None:
        def dual_transform(mu):
            return mu
    else:
        raise NotImplementedError
    
    return dual_transform



def compute_hist_bins(a, axis = None, **kwargs):
    nbins = kwargs.get('nbins', 10)
    binWidth = kwargs.get('binWidth', 0.1)

    bins = 0 + np.arange(nbins + 1) * binWidth # np.linspace(start=0, retstep=binWidth, num=nbins + 1).tolist()
    hist = []
    for i in range(len(bins) - 1):
        temp_hist = np.mean((a >= bins[i]) * (a < bins[i+1]), axis = axis)
        hist.append(temp_hist)
    hist = np.stack(hist, axis=axis)
    # hist = hist * 1/hist.sum(axis=axis, keepdims = True)
    # hist, binEdges = np.histogram(a=p, bins=bins, density=False, axis = axis, weights=np.ones_like(p) / )
    return hist



def sample_n_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''
  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)



def convert_defaultdict_to_list_of_dicts(dd):
    list_of_dicts = []
    for key, value_list in dd.items():
        list_of_dicts.append({key: value_list})
    return list_of_dicts

