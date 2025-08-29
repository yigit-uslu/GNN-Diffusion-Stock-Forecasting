

import abc
import torch
from contextlib import contextmanager
import wandb
from accelerate import Accelerator



def unwrap_model(model):
    """
    Unwrap model with type checking for safety.
    """
    if isinstance(model, torch._dynamo.OptimizedModule):
        return model._orig_mod
    elif hasattr(model, '_orig_mod'):  # More general check
        return model._orig_mod
    else:
        return model


def pass_through(*args, **kwargs):
    """
    Returns all arguments unchanged.
    
    Args:
        *args: Any positional arguments
        **kwargs: Any keyword arguments
        
    Returns:
        The same arguments in the same structure they were provided
    """
    # Handle different cases based on what was passed
    if len(args) == 1 and not kwargs:
        # Single positional argument
        return args[0]
    elif args and not kwargs:
        # Multiple positional arguments
        return args
    elif not args and kwargs:
        # Only keyword arguments
        return kwargs
    else:
        # Both positional and keyword arguments
        return args, kwargs


# Define a dummy accelerator class with no effect
class DummyAccelerator(abc.ABC):
    def __init__(self, device):
        self.device = device

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def wait_for_everyone(self):
        pass

    @contextmanager
    def accumulate(self, *args, **kwargs):
        # Don't do anything
        try:
            yield None
        finally:
            pass


    def prepare(self, *args, **kwargs):
        # Return all arguments as they are
        return pass_through(*args, **kwargs)
    
    def gather(self, input):
        if isinstance(input, torch.Tensor):
            return input
        elif isinstance(input, list) or isinstance(input, tuple):
            return torch.stack(input, dim = 0)
    
    def log(self, *args, **kwargs):
        if wandb.run is not None:
            # Log to wandb if it is initialized
            wandb.log(*args, **kwargs)
        else:
            pass


    def unwrap_model(self, model, keep_torch_compile=False):
        """
        Unwraps the model if it is wrapped by the accelerator.
        If keep_torch_compile is True, it will not unwrap the model.
        """
        
        return Accelerator().unwrap_model(model, keep_torch_compile=keep_torch_compile)
    
    @property
    def is_main_process(self):
        return True
    
    @property
    def is_local_main_process(self):
        return True
    
    @property
    def process_index(self):
        return 0
    


# Define a silent accelerator class that suppresses output
# This class is used to make the accelerator silent, meaning it won't print any output
# when the `silent` parameter is set to True.
class SilentAccelerator(abc.ABC):
    def __init__(self, accelerator, silent=False):
        self.accelerator = accelerator
        self.silent = silent
    
    def print(self, *args, **kwargs):
        if not self.silent:
            self.accelerator.print(*args, **kwargs)
    
    def __getattr__(self, name):
        # Forward all other method calls to the original accelerator
        return getattr(self.accelerator, name)