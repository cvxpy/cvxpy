"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
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