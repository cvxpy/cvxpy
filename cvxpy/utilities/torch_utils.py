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

from enum import Enum

import numpy as np
import torch
from scipy.sparse import coo_matrix, issparse

VAR_TYPE = Enum("VAR_TYPE", "VARIABLE_PARAMETER CONSTANT EXPRESSION")

class VariablesDict():
    """ Helper class that contains a dictionary from a non-constant leaf to an index in *args,
    with a safe add method.
    If a list is given, will create the dictionary in the order of the list."""
    def __init__(self, provided_vars_list: list=[]):
        self.vars_dict = dict()
        for var in provided_vars_list:
            self.add_var(var)
            
    def add_var(self, var):
        """
        var is expected to be either a cp.Variable or a cp.Parameter, but is not enforced.
        """
        if var not in self.vars_dict:
            self.vars_dict[var] = len(self.vars_dict)

    def has_type_in_keys(self, look_type: type):
        """
        This function returns True if one of the keys of this variable is of a certain type,
        and False otherwise
        """
        for var in self.vars_dict.keys():
            if isinstance(var, look_type):
                return True
        return False

def gen_tensor(value, dtype=torch.float64) -> torch.Tensor:
    """This function generates a tensor from an np.array or a sparse matrix.
    If the input is a sparse matrix, a sparse tensor is generated."""
    if not issparse(value):
        return torch.tensor(value, dtype=dtype)
    value_coo = coo_matrix(value)
    vals = value_coo.data
    inds = np.vstack((value_coo.row, value_coo.col))
    i = torch.LongTensor(inds)
    v = torch.FloatTensor(vals)
    return torch.sparse.FloatTensor(i, v, torch.Size(value_coo.shape)).to(dtype)