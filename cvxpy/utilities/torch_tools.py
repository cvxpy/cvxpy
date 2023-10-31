import numpy as np
import torch


def select_module(values):
    """
    This function returns torch if values[0] is torch.Tensor, and numpy otherwise.
    This function is useful when the function name and the signature of a function
    is similar in numpy asn in torch.
    """

    return torch if isinstance(values[0], torch.Tensor) else np


def any_module(condition, module):
    """
    This is a helper function that returns any on either np.array. or torch.tensors.
    """
    if module is torch:
        return torch.any(condition).item()
    return np.any(condition)